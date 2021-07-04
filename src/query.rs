// use std::{any::TypeId, collections::HashMap, marker::PhantomData};

use crate::engine::QueryPlan;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Binding(pub(crate) usize);

// pub(crate) struct BindingMeta {
// }

#[derive(Default, Debug)]
pub struct QueryBuilder {
    var_id: usize,
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

    pub fn plan(&self) -> QueryPlan {
        QueryPlan::plan_for(self)
    }
}

#[derive(Debug)]
pub enum RHS {
    Str(String),
    Int(u64),
    Bnd(Binding),
}

#[derive(Debug)]
pub enum QueryClause {
    Pattern(Binding, &'static str, RHS),
}
