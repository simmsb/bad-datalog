use std::{collections::HashMap, fmt::Display, mem::MaybeUninit};

use itertools::Itertools;
use smallvec::{smallvec, SmallVec};

use crate::{
    datalog::{DBBackedRelation, Variable},
    query::{Binding, QueryBuilder, QueryClause, RHS},
    storage::Value,
};

#[derive(Default, Debug)]
pub struct QueryPlan {
    pub binding_metas: HashMap<Binding, BindingMeta>,
    var_count: usize,
    pub var_metas: HashMap<usize, VariableMeta>,
    pub inversions: HashMap<&'static str, usize>,
    pub opcodes: Vec<OpCode>,
    final_binding_positions: Vec<Option<usize>>,
}

impl QueryPlan {
    pub(crate) fn plan_for(query: &QueryBuilder) -> Self {
        let mut plan = QueryPlan::default();

        plan.get_metadata(query);
        plan.generate_vars(query);
        plan.insert_final_binding_positons();

        plan
    }
}

impl QueryPlan {
    fn get_metadata(&mut self, query: &QueryBuilder) {
        for clause in &query.clauses {
            match clause {
                QueryClause::Pattern(lhs, attr, rhs) => {
                    self.metadata_visit_pattern(*lhs, attr, rhs);
                }
            }
        }
    }

    fn metadata_visit_pattern(&mut self, lhs: Binding, _attr: &'static str, rhs: &RHS) {
        self.binding_metas.entry(lhs).or_default().onlhs = true;

        match rhs {
            RHS::Bnd(binding) => {
                let meta = self.binding_metas.entry(*binding).or_default();

                assert!(
                    !meta.onrhs,
                    "bindings cannot appear on the rhs twice yet (implicit equality constraints)"
                );

                meta.onrhs = true;
            }
            _ => {}
        }
    }
}

impl QueryPlan {
    fn generate_vars(&mut self, query: &QueryBuilder) {
        for clause in &query.clauses {
            match clause {
                QueryClause::Pattern(lhs, attr, rhs) => {
                    self.vars_visit_pattern(*lhs, attr, rhs);
                }
            }
        }
    }

    fn vars_visit_pattern(&mut self, lhs: Binding, attr: &'static str, rhs: &RHS) {
        match rhs {
            RHS::Bnd(rhs_binding) => {
                let inner_vid = if let Some(rhs_final_vid) =
                    self.binding_metas.get(rhs_binding).unwrap().final_variable
                {
                    // the rhs of this variable is also on the lhs (it can't
                    // be on the rhs somewhere else)
                    //
                    // i.e.
                    //
                    // ?b :book_name ?n
                    // ?b :book_author ?ba
                    // ?r :review_author ?ra
                    //
                    // -- current query
                    //
                    // ?r :review_book ?b
                    //
                    // here, we'll probably have ?b currently be in a
                    // variable of (?b, (?n, ?a)), and ?r will be set as the
                    // relation (?r, ?ra)
                    //
                    // we want to join ?b it with ?r
                    //
                    // to do this we need to first invert (?r, ?b) into (?b,
                    // ?r), then we can join it with (?b, (?n, ?ba)) to get
                    // (?r, (?b, ?n, ?ba)), finally this can then be joined
                    // (?r, ?ra) to get (?r, (?ra, ?b, ?n, ?ba))

                    // generate the inversion variable
                    //
                    // (?r, ?b) -> (?b, ?r)
                    let inv_vid = self.add_inversion(attr, lhs, *rhs_binding);

                    // generate the join with the rhs
                    //
                    // (?b, ?r) joined with (?b, (?n, ?ba)) into (?r, (?b, ?n, ?ba))
                    let inner_vid = self.new_var(VarType::Var);
                    let inner_vmeta = self.var_metas.get_mut(&inner_vid).unwrap();
                    inner_vmeta.binding_positions.insert(lhs, None);
                    inner_vmeta.binding_positions.insert(*rhs_binding, Some(0));

                    // add in the bindings from the rhs variable
                    let rhs_meta_keys = self
                        .var_metas
                        .get(&rhs_final_vid)
                        .unwrap()
                        .binding_positions
                        .keys()
                        .cloned()
                        .filter(|b| *b != lhs && b != rhs_binding)
                        .collect_vec();

                    self.var_metas
                        .get_mut(&inner_vid)
                        .unwrap()
                        .binding_positions
                        .extend(rhs_meta_keys.into_iter().zip((1u8..).map(Some)));

                    self.push_join_op(inner_vid, inv_vid, rhs_final_vid);

                    // seemingly we don't need to update the final vars for the join
                    // makes sense I think
                    // self.update_final_variables(inner_vid);
                    // maybe we need to add the last_var for the inversion

                    inner_vid
                } else {
                    // if there's no funky stuff going on with the RHS then we
                    // just set the inner vid to be the relation
                    // don't update the final position though, that's done in the next step

                    self.var_for_attr(attr, lhs, *rhs_binding)
                };

                if let Some(previous_lhs_vid) = self.binding_metas.get(&lhs).unwrap().final_variable
                {
                    // this binding has already been seen on the lhs, we need to
                    // join it

                    let previous_binding_positions = self
                        .var_metas
                        .get(&previous_lhs_vid)
                        .unwrap()
                        .binding_positions
                        .clone();

                    let inner_binding_positions = self
                        .var_metas
                        .get(&inner_vid)
                        .unwrap()
                        .binding_positions
                        .clone();

                    let out_vid = self.new_var(VarType::Var);
                    let out_vmeta = self.var_metas.get_mut(&out_vid).unwrap();

                    out_vmeta.binding_positions = previous_binding_positions;

                    let mut n = out_vmeta.binding_positions.len() as u8 - 1;
                    for b in inner_binding_positions.keys().cloned() {
                        if !out_vmeta.binding_positions.contains_key(&b) {
                            out_vmeta.binding_positions.insert(b, Some(n));
                            n += 1;
                        }
                    }

                    self.update_final_variables(out_vid);

                    self.push_join_op(out_vid, previous_lhs_vid, inner_vid);
                } else {
                    self.update_final_variables(inner_vid);
                }
            }
            _ => todo!(),
        }
    }

    fn add_inversion(&mut self, src: &'static str, lhs: Binding, rhs: Binding) -> usize {
        let inv_vid = if let Some(&vid) = self.inversions.get(src) {
            vid
        } else {
            let vid = self.new_var(VarType::Var);
            self.inversions.insert(src, vid);
            vid
        };

        let inv_vmeta = self.var_metas.get_mut(&inv_vid).unwrap();
        inv_vmeta.binding_positions.insert(rhs, None);
        inv_vmeta.binding_positions.insert(lhs, Some(0));

        inv_vid
    }

    /// update all the final variables of the bindings in the given variable to point to the variable
    fn update_final_variables(&mut self, vid: usize) {
        let bindings = self
            .var_metas
            .get(&vid)
            .unwrap()
            .binding_positions
            .keys()
            .cloned()
            .collect_vec();

        for b in bindings {
            self.update_final_variable(b, vid);
        }
    }

    fn update_final_variable(&mut self, binding: Binding, vid: usize) {
        self.binding_metas.get_mut(&binding).unwrap().final_variable = Some(vid);
    }

    fn push_join_op(&mut self, dst: usize, lhs: usize, rhs: usize) {
        assert!(lhs != rhs, "what's going on here?");

        let lhs_fetch_method = self.fetch_method_for(dst, lhs);
        let rhs_fetch_method = self.fetch_method_for(dst, rhs);

        let key_binding = self.var_metas.get(&lhs).unwrap().key_binding();
        let key_fetch_method = match self
            .var_metas
            .get(&dst)
            .unwrap()
            .binding_positions
            .get(&key_binding)
        {
            Some(Some(n)) => ReadType::ToPosition(*n),
            Some(None) => ReadType::AsIndex,
            None => ReadType::Ignore,
        };

        let out_len = self.var_metas.get(&dst).unwrap().binding_positions.len() - 1;

        self.opcodes.push(OpCode::JoinInto {
            dst,
            lhs,
            lhs_fetch_method,
            rhs,
            rhs_fetch_method,
            key_fetch_method,
            out_len,
        })
    }

    fn fetch_method_for(&self, dst: usize, src: usize) -> FetchMethod {
        let dst_meta = self.var_metas.get(&dst).unwrap();
        let src_meta = self.var_metas.get(&src).unwrap();

        dst_meta.fetch_method_from(src_meta)
    }

    fn var_for_attr(
        &mut self,
        name: &'static str,
        lhs_binding: Binding,
        rhs_binding: Binding,
    ) -> usize {
        let var_id = self.new_var(VarType::Relation(name));

        self.var_metas
            .get_mut(&var_id)
            .unwrap()
            .binding_positions
            .insert(lhs_binding, None);
        self.var_metas
            .get_mut(&var_id)
            .unwrap()
            .binding_positions
            .insert(rhs_binding, Some(0));

        var_id
    }

    fn new_var(&mut self, type_: VarType) -> usize {
        let var_id = self.var_count;
        self.var_count += 1;

        self.var_metas.insert(var_id, VariableMeta::new(type_));

        var_id
    }
}

impl QueryPlan {
    fn insert_final_binding_positons(&mut self) {
        if !self
            .binding_metas
            .values()
            .map(|m| m.final_variable)
            .all_equal()
        {
            panic!("Query has disjoint sections, if you really want this perform the queries separately and do the cartesian product yourself.");
        }

        let final_vid = self
            .binding_metas
            .values()
            .next()
            .expect("no bindings?")
            .final_variable
            .expect("no variables?");

        let meta = self.var_metas.get(&final_vid).unwrap();

        self.final_binding_positions
            .resize(meta.binding_positions.len(), None);

        for (binding, position) in &meta.binding_positions {
            self.final_binding_positions[binding.0] = position.map(|idx| idx as usize);
        }
    }
}

enum VarOrRel {
    Var(Variable<SmallVec<[Value; 8]>>),
    Rel(DBBackedRelation<Value>),
}

impl VarOrRel {
    fn var(&self) -> Option<&Variable<SmallVec<[Value; 8]>>> {
        match self {
            VarOrRel::Var(v) => Some(v),
            _ => None,
        }
    }

    fn rel(&self) -> Option<&DBBackedRelation<Value>> {
        match self {
            VarOrRel::Rel(v) => Some(v),
            _ => None,
        }
    }
}

impl QueryPlan {
    pub fn execute(&self, db: sled::Db) -> Vec<ResultRow> {
        let things = (0..self.var_count)
            .map(|vid| match self.var_metas.get(&vid).unwrap().type_ {
                VarType::Var => VarOrRel::Var(Variable::<SmallVec<[Value; 8]>>::new()),
                VarType::Relation(name) => {
                    VarOrRel::Rel(DBBackedRelation::from_tree(db.open_tree(name).unwrap()))
                }
            })
            .collect_vec();

        let _vars = things
            .iter()
            .filter_map(|x| match x {
                VarOrRel::Var(v) => Some(v),
                VarOrRel::Rel(_) => None,
            })
            .collect_vec();

        todo!()
    }
}

#[derive(Debug)]
enum VarType {
    Var,
    Relation(&'static str),
}

#[derive(Debug)]
pub struct VariableMeta {
    type_: VarType,
    // the positions of bindings in this variable
    //
    // i.e. if we had the query:
    //     ?e :attr0 ?x
    //     ?e :attr1 ?y
    // then the final variable for ?e would be: (idx, [?x, ?y])
    //
    // which would result in binding_positions being {?e: None, ?x: 0, ?y: 1}
    binding_positions: HashMap<Binding, Option<u8>>,
}

impl VariableMeta {
    fn new(type_: VarType) -> Self {
        Self {
            type_,
            binding_positions: Default::default(),
        }
    }

    fn fetch_method_from(&self, src: &Self) -> FetchMethod {
        match &src.type_ {
            VarType::Var => {
                let translation_map = src
                    .binding_positions
                    .iter()
                    .filter_map(|(b, p)| p.map(|i| (i, b)))
                    .collect::<HashMap<_, _>>();

                // binding_positions really shouldn't be a hashmap lol

                let translation_table = (0..(translation_map.len() as u8))
                    .map(|i| {
                        let b = translation_map.get(&i).unwrap();

                        match self.binding_positions.get(b) {
                            Some(Some(n)) => ReadType::ToPosition(*n),
                            Some(None) => ReadType::AsIndex,
                            None => ReadType::Ignore,
                        }
                    })
                    .collect();

                FetchMethod {
                    table: translation_table,
                }
            }
            VarType::Relation(_) => {
                let src_value_binding = src
                    .binding_positions
                    .iter()
                    .find_map(|(b, p)| p.map(|_| b))
                    .expect("binding positions had no value bindings?");

                match self.binding_positions.get(src_value_binding) {
                    Some(Some(n)) => FetchMethod {
                        table: smallvec![ReadType::ToPosition(*n)],
                    },
                    Some(None) => FetchMethod {
                        table: smallvec![ReadType::AsIndex],
                    },
                    None => FetchMethod {
                        table: smallvec![ReadType::Ignore],
                    },
                }
            }
        }
    }

    fn key_binding(&self) -> Binding {
        self.binding_positions
            .iter()
            .filter(|(_b, i)| i.is_none())
            .map(|(b, _)| *b)
            .next()
            .unwrap()
    }
}

#[derive(Default, Debug)]
pub struct BindingMeta {
    onlhs: bool,
    onrhs: bool,
    // the final variable this binding ends up stored in
    final_variable: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub enum ReadType {
    Ignore,
    AsIndex,
    ToPosition(u8),
}

impl ReadType {
    fn perform(
        &self,
        src_val: Value,
        idx_out: &mut MaybeUninit<u64>,
        val_out: &mut [MaybeUninit<Value>],
    ) {
        match self {
            ReadType::Ignore => {}
            ReadType::AsIndex => {
                idx_out.write(src_val.u().unwrap());
            }
            ReadType::ToPosition(idx) => {
                val_out[*idx as usize].write(src_val);
            }
        }
    }
}

#[derive(Debug)]
pub struct FetchMethod {
    // translation table from source data to rhs data
    table: SmallVec<[ReadType; 8]>,
}

impl FetchMethod {
    fn perform_on_row(
        &self,
        src: &SmallVec<[Value; 8]>,
        idx_out: &mut MaybeUninit<u64>,
        val_out: &mut [MaybeUninit<Value>],
    ) {
        assert_eq!(self.table.len(), src.len());

        for (read, val) in self.table.iter().zip(src) {
            read.perform(val.clone(), idx_out, val_out);
        }
    }

    fn perform_on_scalar(
        &self,
        src: &Value,
        idx_out: &mut MaybeUninit<u64>,
        val_out: &mut [MaybeUninit<Value>],
    ) {
        assert_eq!(self.table.len(), 1);

        self.table[0].perform(src.clone(), idx_out, val_out);
    }
}

#[derive(Debug)]
pub enum OpCode {
    JoinInto {
        dst: usize,
        lhs: usize,
        lhs_fetch_method: FetchMethod,
        rhs: usize,
        rhs_fetch_method: FetchMethod,
        key_fetch_method: ReadType,
        out_len: usize,
    },
}

impl Display for OpCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn format_readtype(
            f: &mut std::fmt::Formatter<'_>,
            src: &str,
            src_idx: usize,
            rt: ReadType,
        ) -> std::fmt::Result {
            match rt {
                ReadType::Ignore => Ok(()),
                ReadType::AsIndex => write!(f, "o.idx <- {}.{}, ", src, src_idx),
                ReadType::ToPosition(n) => write!(f, "o.{} <- {}.{}, ", n, src, src_idx),
            }
        }

        fn format_fetchmethod(
            f: &mut std::fmt::Formatter<'_>,
            src: &str,
            fm: &FetchMethod,
        ) -> std::fmt::Result {
            write!(f, "[")?;
            for (s, i) in fm.table.iter().zip(0..) {
                format_readtype(f, src, i, *s)?;
            }
            write!(f, "]")?;

            Ok(())
        }

        match self {
            OpCode::JoinInto {
                dst,
                lhs,
                lhs_fetch_method,
                rhs,
                rhs_fetch_method,
                key_fetch_method,
                out_len: _,
            } => {
                write!(f, "o:{} <- l:{} <> r:{} ", dst, lhs, rhs)?;
                format_fetchmethod(f, "l", lhs_fetch_method)?;
                write!(f, "; ")?;
                format_fetchmethod(f, "r", rhs_fetch_method)?;
                write!(f, "; ")?;
                match key_fetch_method {
                    ReadType::Ignore => (),
                    ReadType::AsIndex => write!(f, "{}.idx <- in.idx", dst)?,
                    ReadType::ToPosition(n) => write!(f, "{}.{} <- in.idx", dst, n)?,
                };
            }
        }

        Ok(())
    }
}

impl OpCode {
    fn execute(&self, things: &[VarOrRel]) {
        match self {
            OpCode::JoinInto {
                dst,
                lhs,
                lhs_fetch_method,
                rhs,
                rhs_fetch_method,
                key_fetch_method,
                out_len,
            } => {
                let dst_var = things[*dst].var().unwrap();
                let lhs_thing = &things[*lhs];
                let rhs_thing = &things[*rhs];

                // hmmmm
                match (lhs_thing, rhs_thing) {
                    (VarOrRel::Var(lhs), VarOrRel::Var(rhs)) => {
                        dst_var.from_join(lhs, rhs, |k, l, r| {
                            let mut out_v =
                                SmallVec::<[MaybeUninit<Value>; 8]>::with_capacity(*out_len);
                            out_v.resize_with(*out_len, || MaybeUninit::uninit());

                            let mut out_k = MaybeUninit::<u64>::uninit();

                            lhs_fetch_method.perform_on_row(l, &mut out_k, out_v.as_mut_slice());
                            rhs_fetch_method.perform_on_row(r, &mut out_k, out_v.as_mut_slice());
                            key_fetch_method.perform(Value::U(k), &mut out_k, out_v.as_mut_slice());

                            let out_k = unsafe { out_k.assume_init() };
                            let out_v = out_v
                                .into_iter()
                                .map(|x| unsafe { x.assume_init() })
                                .collect();

                            (out_k, out_v)
                        });
                    }
                    (VarOrRel::Var(lhs), VarOrRel::Rel(rhs)) => {
                        dst_var.from_join(lhs, rhs, |k, l, r| {
                            let mut out_v =
                                SmallVec::<[MaybeUninit<Value>; 8]>::with_capacity(*out_len);
                            out_v.resize_with(*out_len, || MaybeUninit::uninit());

                            let mut out_k = MaybeUninit::<u64>::uninit();

                            lhs_fetch_method.perform_on_row(l, &mut out_k, out_v.as_mut_slice());
                            rhs_fetch_method.perform_on_scalar(r, &mut out_k, out_v.as_mut_slice());
                            key_fetch_method.perform(Value::U(k), &mut out_k, out_v.as_mut_slice());

                            let out_k = unsafe { out_k.assume_init() };
                            let out_v = out_v
                                .into_iter()
                                .map(|x| unsafe { x.assume_init() })
                                .collect();

                            (out_k, out_v)
                        });
                    }
                    (VarOrRel::Rel(lhs), VarOrRel::Var(rhs)) => {
                        dst_var.from_join(lhs, rhs, |k, l, r| {
                            let mut out_v =
                                SmallVec::<[MaybeUninit<Value>; 8]>::with_capacity(*out_len);
                            out_v.resize_with(*out_len, || MaybeUninit::uninit());

                            let mut out_k = MaybeUninit::<u64>::uninit();

                            lhs_fetch_method.perform_on_scalar(l, &mut out_k, out_v.as_mut_slice());
                            rhs_fetch_method.perform_on_row(r, &mut out_k, out_v.as_mut_slice());
                            key_fetch_method.perform(Value::U(k), &mut out_k, out_v.as_mut_slice());

                            let out_k = unsafe { out_k.assume_init() };
                            let out_v = out_v
                                .into_iter()
                                .map(|x| unsafe { x.assume_init() })
                                .collect();

                            (out_k, out_v)
                        });
                    }
                    (VarOrRel::Rel(lhs), VarOrRel::Rel(rhs)) => {
                        dst_var.from_join(lhs, rhs, |k, l, r| {
                            let mut out_v =
                                SmallVec::<[MaybeUninit<Value>; 8]>::with_capacity(*out_len);
                            out_v.resize_with(*out_len, || MaybeUninit::uninit());

                            let mut out_k = MaybeUninit::<u64>::uninit();

                            lhs_fetch_method.perform_on_scalar(l, &mut out_k, out_v.as_mut_slice());
                            rhs_fetch_method.perform_on_scalar(r, &mut out_k, out_v.as_mut_slice());
                            key_fetch_method.perform(Value::U(k), &mut out_k, out_v.as_mut_slice());

                            let out_k = unsafe { out_k.assume_init() };
                            let out_v = out_v
                                .into_iter()
                                .map(|x| unsafe { x.assume_init() })
                                .collect();

                            (out_k, out_v)
                        });
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct ResultRow<'a> {
    idx_val: Value,
    inner: SmallVec<[Value; 8]>,
    relocs: &'a Vec<Option<usize>>,
}

impl<'a> ResultRow<'a> {
    pub fn fetch(&self, binding: Binding) -> &Value {
        if let Some(idx) = self.relocs[binding.0] {
            &self.inner[idx]
        } else {
            &self.idx_val
        }
    }
}
