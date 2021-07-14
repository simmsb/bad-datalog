#![warn(clippy::all)]


use dlog::{Database, QueryBuilder, RHS, object};
use itertools::Itertools;
use tempfile::tempdir;

// fn books() {
//     let book_names = EphemerealRelation::<Rc<String>>::from_iter([
//         (100, "foo".to_owned().into()),
//         (101, "bar".to_owned().into()),
//         (102, "baz".to_owned().into()),
//         (103, "blah".to_owned().into()),
//     ]);

//     let review_users = EphemerealRelation::<Rc<String>>::from_iter([
//         (0, "reviewer_0".to_owned().into()),
//         (1, "reviewer_1".to_owned().into()),
//         (2, "reviewer_0".to_owned().into()),
//         (3, "reviewer_1".to_owned().into()),
//     ]);

//     let review_scores = EphemerealRelation::<u64>::from_iter([(0, 22), (1, 44), (2, 12), (3, 6)]);

//     // review --> book
//     let review_books =
//         EphemerealRelation::<u64>::from_iter([(0, 100), (1, 100), (2, 101), (3, 103)]);

//     // let's do the query:
//     //   ?b :book_name ?t
//     //   ?r :review_book ?b
//     //   ?r :review_user ?u
//     //   ?r :review_score ?s

//     // t: book_names
//     // b: review_books
//     // u: review_user
//     // s: review_score
//     //
//     // r is not on rhs, so is not an initialised
//     // b is used on lhs and rhs, so it's inverse is created to be used on the lhs

//     let t = book_names;
//     let b = review_books;
//     let u = review_users;
//     let s = review_scores;

//     let b_inv = Variable::new();
//     b_inv.insert_data(b.into_iter().map(|(k, v)| (v, k)));

//     // step 0: build (?r, (?b, ?t)) via (b_inv: ?b ?r) x (t: ?b ?t)
//     let stage_0 = Variable::new();

//     // step 1: build (?r, (?b, ?t, ?u)))
//     let stage_1 = Variable::new();

//     // step 2: build (?r, (?b, ?t, ?u, ?s))))
//     let stage_2 = Variable::new();

//     while any_changed([
//         b_inv.as_dyn(),
//         stage_0.as_dyn(),
//         stage_1.as_dyn(),
//         stage_2.as_dyn(),
//     ]) {
//         stage_0.from_join(&b_inv, &t, |b, r, t| (*r, (b, t.clone())));
//         stage_1.from_join(&stage_0, &u, |r, (b, t), u| (r, (*b, t.clone(), u.clone())));
//         stage_2.from_join(&stage_1, &s, |r, (b, t, u), s| {
//             (r, (*b, t.clone(), u.clone(), *s))
//         });
//     }

//     let out = stage_2.into_relation();

//     for (idx, (b, t, u, s)) in out {
//         println!(
//             "review: {}, book: {}, book name: {}, reviewer: {}, score: {}",
//             idx, b, t, u, s
//         );
//     }
// }

// fn from_join_ex() {
//     let v1 = Variable::<u64>::new();
//     v1.insert_data([(0, 1), (1, 2)]);
//     v1.insert_data([(1, 0), (2, 1)]);

//     while any_changed([v1.as_dyn()]) {
//         v1.from_join(&v1, &v1, |_k, &a, &b| (a, b));
//     }

//     let r = v1.into_relation();

//     for (a, b) in r {
//         println!("{} is reachable from {}", b, a);
//     }
// }

fn test_query() {
    let tmp = tempdir().unwrap();
    let sled_db = sled::open(tmp.path()).unwrap();

    let mut db = Database::from_sled(sled_db.clone());

    let book_foo = db.add_object(&object!(
        book_name: "foo",
        book_price: 100,
    ));

    let book_bar = db.add_object(&object!(
        book_name: "bar",
        book_price: 101,
    ));

    let _book_baz = db.add_object(&object!(
        book_name: "baz",
        book_price: 102,
    ));

    let book_blah = db.add_object(&object!(
        book_name: "blah",
        book_price: 103,
    ));

    db.add_object(&object!(
        review_book: book_foo,
        review_user: "reviewer_0",
        review_score: 10,
    ));

    db.add_object(&object!(
        review_book: book_foo,
        review_user: "reviewer_1",
        review_score: 100,
    ));

    db.add_object(&object!(
        review_book: book_bar,
        review_user: "reviewer_0",
        review_score: 102,
    ));

    db.add_object(&object!(
        review_book: book_blah,
        review_user: "reviewer_1",
        review_score: 99,
    ));

    let mut builder = QueryBuilder::default();

    let b = builder.binding();
    let p = builder.binding();
    let t = builder.binding();
    let r = builder.binding();
    // let u = builder.binding();
    let s = builder.binding();

    builder
        .pattern(r, "review_book", RHS::Bnd(b))
        .pattern(r, "review_user", RHS::Str("reviewer_0"))
        .pattern(r, "review_score", RHS::Bnd(s))
        .pattern(b, "book_name", RHS::Bnd(t))
        .pattern(b, "book_price", RHS::Bnd(p))
        .filter(p, |p_v| p_v.i().unwrap() > 100)
        ;

    let plan = builder.plan();

    println!("bindings: ");

    for (binding, meta) in &plan.binding_metas {
        println!("{:?}: {:?}", binding, meta);
    }

    println!("\nvariables: ");

    for (vid, meta) in plan.var_metas.iter().sorted_by_key(|(k, _v)| *k) {
        println!("{}: {:?}", vid, meta);
    }

    println!("\nopcodes: ");

    for (attr, vid) in &plan.pre_inversions {
        println!("{} <- {}", vid, attr);
    }

    for op in &plan.opcodes {
        println!("{}", op);
    }

    let results = plan.execute(sled_db);

    println!("results: {:?}", results);

    for row in results {
        println!(
            "b: {}, p: {}, t: {}, r: {}, s: {}",
            row.fetch(b),
            row.fetch(p),
            row.fetch(t),
            row.fetch(r),
            row.fetch(s),
        )
    }
}

fn main() {
    // for f in [books, from_join_ex, test_query] {
    //     f()
    // }
    test_query();
}
