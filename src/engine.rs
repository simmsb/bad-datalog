use std::{collections::HashMap, fmt::Display, mem::MaybeUninit, rc::Rc};

use itertools::Itertools;
use smallvec::{smallvec, SmallVec};

use crate::{
    datalog::{any_changed, DBBackedRelation, Variable, VariableMeta},
    query::{AsFilter, Binding, QueryBuilder, QueryClause, RHS},
    storage::Value,
};

pub struct QueryPlan {
    pub binding_metas: HashMap<Binding, BindingMeta>,
    max_binding: usize,
    var_count: usize,
    pub var_metas: HashMap<usize, VarMeta>,
    pub pre_inversions: HashMap<&'static str, usize>,
    pub pre_filters: Vec<(&'static str, usize, Box<dyn Fn(&Value) -> bool>)>,
    pub opcodes: Vec<OpCode>,
    final_vid: usize,
    final_binding_positions: Vec<Option<usize>>,
}

impl QueryPlan {
    pub(crate) fn plan_for(query: &QueryBuilder) -> Self {
        let mut plan = QueryPlan {
            binding_metas: Default::default(),
            max_binding: query.var_id,
            var_count: 0,
            var_metas: Default::default(),
            pre_inversions: Default::default(),
            pre_filters: Default::default(),
            opcodes: Default::default(),
            final_vid: 0,
            final_binding_positions: Default::default(),
        };

        plan.get_metadata(query);
        plan.construct_plan(query);

        for i in 0..plan.max_binding {
            if !plan.binding_metas.contains_key(&Binding(i)) {
                panic!("Unused binding: {}", i);
            }
        }

        plan.insert_final_binding_positons();

        plan
    }
}

impl QueryPlan {
    fn get_metadata(&mut self, query: &QueryBuilder) {
        for clause in &query.clauses {
            if let QueryClause::Pattern(lhs, attr, rhs) = clause {
                self.metadata_visit_pattern(*lhs, attr, rhs);
            }
        }
    }

    fn metadata_visit_pattern(&mut self, lhs: Binding, _attr: &'static str, rhs: &RHS) {
        self.binding_metas.entry(lhs).or_default().onlhs = true;

        if let RHS::Bnd(binding) = rhs {
            let meta = self.binding_metas.entry(*binding).or_default();

            assert!(
                !meta.onrhs,
                "bindings cannot appear on the rhs twice yet (implicit equality constraints)"
            );

            meta.onrhs = true;
        }
    }
}

impl QueryPlan {
    fn construct_plan(&mut self, query: &QueryBuilder) {
        for clause in &query.clauses {
            match clause {
                QueryClause::Pattern(lhs, attr, rhs) => {
                    self.vars_visit_pattern(*lhs, attr, rhs);
                }
                QueryClause::Filter(asfilter) => {
                    self.process_filter(asfilter.as_ref());
                }
            }
        }
    }

    fn process_filter(&mut self, filter: &dyn AsFilter) {
        let src_vid = self.latest_vid_with(&filter.bindings());
        let positions = self.binding_positions_for(src_vid);

        let fn_ = filter.as_filter(&positions);

        let dst_vid = self.new_var(VarType::Var);

        let binding_positions = self
            .var_metas
            .get(&src_vid)
            .unwrap()
            .binding_positions
            .clone();

        let meta = self.var_metas.get_mut(&dst_vid).unwrap();
        meta.binding_positions = binding_positions;

        self.opcodes.push(OpCode::Filter {
            dst: dst_vid,
            src: src_vid,
            fn_,
        });

        self.update_final_variables(dst_vid);
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
                    let inv_vid = self.add_preinversion(attr, lhs, *rhs_binding);

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

                self.do_join_with_previous(lhs, inner_vid);
            }
            x => {
                let rhs = match x {
                    RHS::Str(s) => Value::S(Rc::new(s.to_string())),
                    RHS::UInt(i) => Value::U(*i),
                    RHS::IInt(i) => Value::I(*i),
                    _ => unreachable!(),
                };

                let fn_: Box<dyn Fn(&Value) -> bool> = Box::new(move |v| v == &rhs);

                let vid = self.add_prefilter(attr, lhs, fn_);

                self.do_join_with_previous(lhs, vid);
            }
        }
    }

    fn do_join_with_previous(&mut self, lhs: Binding, inner_vid: usize) {
        if let Some(previous_vid) = self.binding_metas.get(&lhs).unwrap().final_variable {
            // this binding has already been seen on the lhs or rhs, we
            // need to join it

            let was_on_lhs = self
                .var_metas
                .get(&previous_vid)
                .unwrap()
                .binding_positions
                .get(&lhs)
                .unwrap()
                .is_none();

            let lhs_vid = if was_on_lhs {
                previous_vid
            } else {
                // in this case the lhs variable was on the rhs the last
                // time it was seen, for example:
                //
                // ?e :review_book ?x
                // ?e :review_score ?s
                //
                // -- current query
                //
                // ?x :book_name ?n
                //
                // for this, we'll have (?e, (?x, ?s)) which we'll then
                // map into (?x, (?e, ?s)) so that it can then be joined
                // with (?x, ?n) resulting in (?x, (?e, ?s, ?n)) and
                // then we'll be done because we wanted the lhs to be ?x
                // anyway

                let mut prev_binding_positions = self
                    .var_metas
                    .get(&previous_vid)
                    .unwrap()
                    .binding_positions
                    .clone();

                let prev_key = prev_binding_positions
                    .iter()
                    .filter(|(_k, v)| v.is_none())
                    .map(|(k, _)| k)
                    .next()
                    .cloned()
                    .unwrap();

                if let VarType::Relation(attr) = self.var_metas.get(&previous_vid).unwrap().type_ {
                    // for variables, we can just prep an inversion and use that

                    self.add_preinversion(attr, prev_key, lhs)
                } else {
                    let new_key_pos = prev_binding_positions.get(&lhs).unwrap().expect("how???");
                    prev_binding_positions.insert(lhs, None);
                    prev_binding_positions.insert(prev_key, Some(new_key_pos));

                    let inner_vid = self.new_var(VarType::Var);

                    let inner_vmeta = self.var_metas.get_mut(&inner_vid).unwrap();
                    inner_vmeta.binding_positions = prev_binding_positions;

                    self.push_remap_op(inner_vid, previous_vid);

                    // now we have (?x, (?e, ?s)) and want to join it with (?x, ?n)

                    inner_vid
                }
            };

            let previous_binding_positions = self
                .var_metas
                .get(&lhs_vid)
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
                if let std::collections::hash_map::Entry::Vacant(e) =
                    out_vmeta.binding_positions.entry(b)
                {
                    e.insert(Some(n));
                    n += 1;
                }
            }

            self.update_final_variables(out_vid);

            self.push_join_op(out_vid, lhs_vid, inner_vid);
        } else {
            self.update_final_variables(inner_vid);
        }
    }

    fn add_temp_binding(&mut self) -> Binding {
        let id = self.max_binding;
        self.max_binding += 1;
        let binding = Binding(id);
        self.binding_metas.insert(
            binding,
            BindingMeta {
                onlhs: false,
                onrhs: true,
                final_variable: None,
            },
        );
        binding
    }

    fn add_preinversion(&mut self, src: &'static str, lhs: Binding, rhs: Binding) -> usize {
        let inv_vid = if let Some(&vid) = self.pre_inversions.get(src) {
            vid
        } else {
            let vid = self.new_var(VarType::Var);
            self.pre_inversions.insert(src, vid);
            vid
        };

        let inv_vmeta = self.var_metas.get_mut(&inv_vid).unwrap();
        inv_vmeta.binding_positions.insert(rhs, None);
        inv_vmeta.binding_positions.insert(lhs, Some(0));

        inv_vid
    }

    fn add_prefilter(
        &mut self,
        src: &'static str,
        lhs: Binding,
        fn_: Box<dyn Fn(&Value) -> bool>,
    ) -> usize {
        let vid = self.new_var(VarType::Var);
        // TODO: we can improve this quite a lot by having an option for range
        // prefilters that take advantage of the orderedness of data
        self.pre_filters.push((src, vid, fn_));

        let tmp_rhs = self.add_temp_binding();

        let vmeta = self.var_metas.get_mut(&vid).unwrap();
        vmeta.binding_positions.insert(lhs, None);
        vmeta.binding_positions.insert(tmp_rhs, Some(0));

        self.binding_metas.get_mut(&tmp_rhs).unwrap().final_variable = Some(vid);

        vid
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

    fn push_remap_op(&mut self, dst: usize, src: usize) {
        let fetch_method = self.fetch_method_for(dst, src);

        let key_binding = self.var_metas.get(&src).unwrap().key_binding();
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

        self.opcodes.push(OpCode::Remap {
            dst,
            src,
            fetch_method,
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

        self.var_metas.insert(var_id, VarMeta::new(type_));

        var_id
    }
}

#[derive(Clone, Copy)]
pub enum PositionOf {
    NotHere,
    Index,
    Position(u8),
}

impl QueryPlan {
    fn latest_vid_with(&self, required: &[Binding]) -> usize {
        let vid = self
            .binding_metas
            .get(&required[0])
            .unwrap()
            .final_variable
            .expect("binding has not been seen yet");
        let meta = self.var_metas.get(&vid).unwrap();

        for b in required {
            if !meta.binding_positions.contains_key(b) {
                panic!("cannot create a filter for {:?} satisfying every variable at the current position", required);
            }
        }

        vid
    }

    fn binding_positions_for(&self, vid: usize) -> Vec<PositionOf> {
        let meta = self.var_metas.get(&vid).unwrap();

        let mut positions = vec![PositionOf::NotHere; meta.binding_positions.len()];

        for (binding, position) in &meta.binding_positions {
            positions[binding.0] = position.map_or(PositionOf::Index, PositionOf::Position);
        }

        positions
    }

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

        self.final_vid = final_vid;

        self.final_binding_positions
            .resize(self.binding_metas.len(), None);

        for (binding, position) in &meta.binding_positions {
            self.final_binding_positions[binding.0] = position.map(|idx| idx as usize);
        }
    }
}

#[derive(Debug)]
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

    fn into_var(self) -> Option<Variable<SmallVec<[Value; 8]>>> {
        match self {
            VarOrRel::Var(v) => Some(v),
            _ => None,
        }
    }

    fn into_rel(self) -> Option<DBBackedRelation<Value>> {
        match self {
            VarOrRel::Rel(v) => Some(v),
            _ => None,
        }
    }
}

impl QueryPlan {
    pub fn execute(&self, db: sled::Db) -> (Vec<ResultRow>, ExecutionStats) {
        let mut execution_stats = ExecutionStats::default();
        let things = (0..self.var_count)
            .map(|vid| match self.var_metas.get(&vid).unwrap().type_ {
                VarType::Var => VarOrRel::Var(Variable::<SmallVec<[Value; 8]>>::new()),
                VarType::Relation(name) => {
                    VarOrRel::Rel(DBBackedRelation::from_tree(db.open_tree(name).unwrap()))
                }
            })
            .collect_vec();

        let vars = things
            .iter()
            .filter_map(|x| match x {
                VarOrRel::Var(v) => Some(v),
                VarOrRel::Rel(_) => None,
            })
            .collect_vec();

        for (name, dst_vid) in &self.pre_inversions {
            let var = things[*dst_vid].var().unwrap();

            let rel = DBBackedRelation::<Value>::from_tree(db.open_tree(name).unwrap());
            var.insert_data(
                rel.into_iter()
                    .map(|(k, v)| (v.u().unwrap(), SmallVec::from_elem(Value::U(k), 1))),
            );
        }

        for (name, dst_vid, fn_) in &self.pre_filters {
            let var = things[*dst_vid].var().unwrap();

            let rel = DBBackedRelation::<Value>::from_tree(db.open_tree(name).unwrap());
            var.insert_data(
                rel.into_iter()
                    .filter(|(_k, v)| fn_(v))
                    .map(|(k, v)| (k, SmallVec::from_elem(v, 1))),
            );
        }

        for op in &self.opcodes {
            let s = op.execute(&things);
            execution_stats = execution_stats.combine(s);
        }

        while any_changed(vars.iter().map(|v| *v as &dyn VariableMeta)) {
            for op in &self.opcodes {
                let s = op.execute(&things);
                execution_stats = execution_stats.combine(s);
            }
        }

        let r = { things }
            .remove(self.final_vid)
            .into_var()
            .unwrap()
            .into_relation();
        let r = r
            .into_iter()
            .map(|(idx, row)| ResultRow::from_rel_row(idx, row, &self.final_binding_positions))
            .collect_vec();

        (r, execution_stats)
    }
}

#[derive(Debug)]
enum VarType {
    Var,
    Relation(&'static str),
}

#[derive(Debug)]
pub struct VarMeta {
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

impl VarMeta {
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
    Remap {
        dst: usize,
        src: usize,
        fetch_method: FetchMethod,
        key_fetch_method: ReadType,
        out_len: usize,
    },
    Filter {
        dst: usize,
        src: usize,
        fn_: Box<dyn for<'a> Fn(u64, &'a [Value]) -> bool>,
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
            OpCode::Remap {
                dst,
                src,
                fetch_method,
                key_fetch_method,
                out_len: _,
            } => {
                write!(f, "o:{} <- s:{} ", dst, src)?;
                format_fetchmethod(f, "s", fetch_method)?;
                write!(f, "; ")?;
                match key_fetch_method {
                    ReadType::Ignore => (),
                    ReadType::AsIndex => write!(f, "{}.idx <- in.idx", dst)?,
                    ReadType::ToPosition(n) => write!(f, "{}.{} <- in.idx", dst, n)?,
                };
            }
            OpCode::Filter { dst, src, fn_ } => {
                write!(f, "o:{} <- s:{} if {:p}", dst, src, fn_)?;
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ExecutionStats {
    pub ops_performed: usize,
}

impl ExecutionStats {
    #[must_use]
    pub fn combine(self, other: Self) -> Self {
        Self {
            ops_performed: self.ops_performed + other.ops_performed,
        }
    }
}

impl OpCode {
    fn execute(&self, things: &[VarOrRel]) -> ExecutionStats {
        let ops_performed = match self {
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
                            out_v.resize_with(*out_len, MaybeUninit::uninit);

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
                        })
                    }
                    (VarOrRel::Var(lhs), VarOrRel::Rel(rhs)) => {
                        dst_var.from_join(lhs, rhs, |k, l, r| {
                            let mut out_v =
                                SmallVec::<[MaybeUninit<Value>; 8]>::with_capacity(*out_len);
                            out_v.resize_with(*out_len, MaybeUninit::uninit);

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
                        })
                    }
                    (VarOrRel::Rel(lhs), VarOrRel::Var(rhs)) => {
                        dst_var.from_join(rhs, lhs, |k, l, r| {
                            let mut out_v =
                                SmallVec::<[MaybeUninit<Value>; 8]>::with_capacity(*out_len);
                            out_v.resize_with(*out_len, MaybeUninit::uninit);

                            let mut out_k = MaybeUninit::<u64>::uninit();

                            lhs_fetch_method.perform_on_scalar(r, &mut out_k, out_v.as_mut_slice());
                            rhs_fetch_method.perform_on_row(l, &mut out_k, out_v.as_mut_slice());
                            key_fetch_method.perform(Value::U(k), &mut out_k, out_v.as_mut_slice());

                            let out_k = unsafe { out_k.assume_init() };
                            let out_v = out_v
                                .into_iter()
                                .map(|x| unsafe { x.assume_init() })
                                .collect();

                            (out_k, out_v)
                        })
                    }
                    (VarOrRel::Rel(lhs), VarOrRel::Rel(rhs)) => {
                        dst_var.from_join_rel(lhs, rhs, |k, l, r| {
                            let mut out_v =
                                SmallVec::<[MaybeUninit<Value>; 8]>::with_capacity(*out_len);
                            out_v.resize_with(*out_len, MaybeUninit::uninit);

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
                        })
                    }
                }
            }
            OpCode::Remap {
                dst,
                src,
                fetch_method,
                key_fetch_method,
                out_len,
            } => {
                let dst_var = things[*dst].var().unwrap();
                let src_thing = &things[*src];

                match src_thing {
                    VarOrRel::Var(src) => dst_var.from_map(src, |k, v| {
                        let mut out_v =
                            SmallVec::<[MaybeUninit<Value>; 8]>::with_capacity(*out_len);
                        out_v.resize_with(*out_len, MaybeUninit::uninit);

                        let mut out_k = MaybeUninit::<u64>::uninit();

                        fetch_method.perform_on_row(v, &mut out_k, out_v.as_mut_slice());
                        key_fetch_method.perform(Value::U(k), &mut out_k, out_v.as_mut_slice());

                        let out_k = unsafe { out_k.assume_init() };
                        let out_v = out_v
                            .into_iter()
                            .map(|x| unsafe { x.assume_init() })
                            .collect();

                        (out_k, out_v)
                    }),
                    VarOrRel::Rel(_) => {
                        panic!(
                            "we should be planning pre-inversions instead of remapping relations"
                        )
                    }
                }
            }
            OpCode::Filter { dst, src, fn_ } => {
                let dst_var = things[*dst].var().unwrap();
                let src_thing = &things[*src];

                match src_thing {
                    VarOrRel::Var(src) => dst_var.from_filter(src, |k, v| {
                        if fn_(k, v) {
                            Some((k, v.clone()))
                        } else {
                            None
                        }
                    }),
                    VarOrRel::Rel(_) => {
                        panic!("we should be planning pre-filters instead of filtering relations")
                    }
                }
            }
        };

        ExecutionStats { ops_performed }
    }
}

#[derive(Debug)]
pub struct ResultRow<'a> {
    idx_val: Value,
    inner: SmallVec<[Value; 8]>,
    relocs: &'a [Option<usize>],
}

impl<'a> ResultRow<'a> {
    pub fn fetch(&self, binding: Binding) -> &Value {
        if let Some(idx) = self.relocs[binding.0] {
            &self.inner[idx]
        } else {
            &self.idx_val
        }
    }

    fn from_rel_row(idx: u64, inner: SmallVec<[Value; 8]>, relocs: &'a [Option<usize>]) -> Self {
        ResultRow {
            idx_val: Value::U(idx),
            inner,
            relocs,
        }
    }
}
