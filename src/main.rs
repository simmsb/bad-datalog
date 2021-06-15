use dlog::{Pattern, QueryBuilder};

fn main() {
    let mut b = QueryBuilder::new();

    let e = b.fresh_var();
    let e2 = b.fresh_var();
    let v = b.fresh_var();
    let v2 = b.fresh_var();
    b.add_constraint(e, "name", Pattern::Variable(v));
    b.add_constraint(e2, "name", Pattern::Variable(v2));
    b.add_constraint(e, "friend", Pattern::Variable(e2));

    // let plan = b.plan_query();

    // println!("plan: {:#?}", plan);
}
