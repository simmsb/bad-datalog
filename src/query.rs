// use std::{any::TypeId, collections::HashMap, marker::PhantomData};

use itertools::Itertools;

use crate::{
    engine::{PositionOf, QueryPlan},
    storage::Value,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Binding(pub(crate) usize);

// pub(crate) struct BindingMeta {
// }

#[derive(Default)]
pub struct QueryBuilder {
    pub(crate) var_id: usize,
    // pub(crate) bindings: HashMap<Binding, BindingMeta>,
    pub(crate) clauses: Vec<QueryClause>,
}

impl QueryBuilder {
    pub fn binding(&mut self) -> Binding {
        let id = self.var_id;
        self.var_id += 1;
        Binding(id)
    }

    pub fn clause(&mut self, clause: QueryClause) -> &mut Self {
        self.clauses.push(clause);
        self
    }

    pub fn pattern(&mut self, lhs: Binding, attr: &'static str, rhs: RHS) -> &mut Self {
        self.clause(QueryClause::Pattern(lhs, attr, rhs))
    }

    pub fn filter<P, F>(&mut self, pat: P, fn_: F) -> &mut Self
    where
        P: FilterPattern + 'static,
        F: Clone + for<'a> Fn(P::Output) -> bool + 'static,
    {
        self.clause(QueryClause::Filter(Box::new(PackedFilterPattern {
            pat,
            fn_,
        })))
    }

    pub fn plan(&self) -> QueryPlan {
        QueryPlan::plan_for(self)
    }
}

#[derive(Debug)]
pub enum RHS {
    Str(&'static str),
    UInt(u64),
    IInt(i64),
    Bnd(Binding),
}

pub trait FilterPattern {
    type Output;
    type Indexes: 'static;

    fn bindings(&self) -> Vec<Binding>;
    fn prep(&self, positions: &[PositionOf]) -> Self::Indexes;
    fn extract(idxs: &Self::Indexes, idx: u64, vals: &[Value]) -> Self::Output;
}

impl FilterPattern for Binding {
    type Output = Value;
    type Indexes = Option<usize>;

    fn bindings(&self) -> Vec<Binding> {
        vec![*self]
    }

    fn prep(&self, positions: &[PositionOf]) -> Self::Indexes {
        match positions.get(self.0).unwrap() {
            PositionOf::Index => None,
            PositionOf::Position(x) => Some(*x as usize),
            PositionOf::NotHere => panic!("{:?} doesn't exist yet", self),
        }
    }

    fn extract(idxs: &Self::Indexes, idx: u64, vals: &[Value]) -> Self::Output {
        if let Some(pos) = idxs {
            vals[*pos].clone()
        } else {
            Value::U(idx)
        }
    }
}

macro_rules! ignore_ident{
    ($id:ident, $($t:tt)*) => {$($t)*};
}

macro_rules! doit {
    ($dummy:ident, ) => {};
    ($dummy:ident, $($Y:ident,)*) => {
        doit!($($Y,)*);
        impl FilterPattern for ($(ignore_ident!($Y, Binding),)*) {
            type Output = ($(ignore_ident!($Y, Value),)*);
            type Indexes = ($(ignore_ident!($Y, Option<usize>),)*);

            fn bindings(&self) -> Vec<Binding> {
                use tuple::TupleElements;

                self.elements().cloned().collect_vec()
            }

            fn prep(&self, positions: &[PositionOf]) -> Self::Indexes {
                let ($($Y,)*) = self;

                $(
                    let $Y = match positions.get($Y.0).unwrap() {
                        PositionOf::Index => None,
                        PositionOf::Position(x) => Some(*x as usize),
                        PositionOf::NotHere => panic!("{:?} doesn't exist yet", self),
                    };
                )*

                ($($Y,)*)
            }


            fn extract(idxs: &Self::Indexes, idx: u64, vals: &[Value]) -> Self::Output {
                let ($($Y,)*) = idxs;

                $(
                    let $Y = if let Some(pos) = $Y {
                        vals[*pos].clone()
                    } else {
                        Value::U(idx)
                    };
                )*

                ($($Y,)*)
            }
        }
    }
}

doit!(dummy, a, b, c, d, e, f, g, h, i, j, k, l,);

struct PackedFilterPattern<P, F> {
    pat: P,
    fn_: F,
}

pub trait AsFilter {
    fn bindings(&self) -> Vec<Binding>;
    fn as_filter(&self, positions: &[PositionOf]) -> Box<dyn for<'a> Fn(u64, &'a [Value]) -> bool>;
}

impl<P, F> AsFilter for PackedFilterPattern<P, F>
where
    P: FilterPattern,
    F: Clone + for<'a> Fn(P::Output) -> bool + 'static,
{
    fn bindings(&self) -> Vec<Binding> {
        self.pat.bindings()
    }

    fn as_filter(&self, positions: &[PositionOf]) -> Box<dyn for<'a> Fn(u64, &'a [Value]) -> bool> {
        let indexes = self.pat.prep(positions);
        let fn_ = self.fn_.clone();

        Box::new(move |idx, vals| fn_(<P as FilterPattern>::extract(&indexes, idx, vals)))
    }
}

pub enum QueryClause {
    Pattern(Binding, &'static str, RHS),
    Filter(Box<dyn AsFilter>),
}
