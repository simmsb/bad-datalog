use std::rc::Rc;

use dlog::*;

fn books() {
    let book_names = Relation::<Rc<String>>::from_iter([
        (100, "foo".to_owned().into()),
        (101, "bar".to_owned().into()),
        (102, "baz".to_owned().into()),
        (103, "blah".to_owned().into()),
    ]);

    let review_users = Relation::<Rc<String>>::from_iter([
        (0, "reviewer_0".to_owned().into()),
        (1, "reviewer_1".to_owned().into()),
        (2, "reviewer_0".to_owned().into()),
        (3, "reviewer_1".to_owned().into()),
    ]);

    let review_scores = Relation::<u64>::from_iter([(0, 22), (1, 44), (2, 12), (3, 6)]);

    // review --> book
    let review_books = Relation::<u64>::from_iter([(0, 100), (1, 100), (2, 101), (3, 103)]);

    // let's do the query:
    //   ?b :book_name ?t
    //   ?r :review_book ?b
    //   ?r :review_user ?u
    //   ?r :review_score ?s

    // t: book_names
    // b: review_books
    // u: review_user
    // s: review_score
    //
    // r is not on rhs, so is not an initialised
    // b is used on lhs and rhs, so it's inverse is created to be used on the lhs

    let t = book_names;
    let b = review_books;
    let u = review_users;
    let s = review_scores;

    let b_inv = Variable::new();
    b_inv.insert_data(b.into_iter().map(|(k, v)| (v, k)));

    // step 0: build (?r, (?b, ?t)) via (b_inv: ?b ?r) x (t: ?b ?t)
    let stage_0 = Variable::new();

    // step 1: build (?r, (?b, ?t, ?u)))
    let stage_1 = Variable::new();

    // step 2: build (?r, (?b, ?t, ?u, ?s))))
    let stage_2 = Variable::new();

    while any_changed([
        b_inv.as_dyn(),
        stage_0.as_dyn(),
        stage_1.as_dyn(),
        stage_2.as_dyn(),
    ]) {
        stage_0.from_join(&b_inv, &t, |b, r, t| (*r, (b, t.clone())));
        stage_1.from_join(&stage_0, &u, |r, (b, t), u| (r, (*b, t.clone(), u.clone())));
        stage_2.from_join(&stage_1, &s, |r, (b, t, u), s| {
            (r, (*b, t.clone(), u.clone(), *s))
        });
    }

    let out = stage_2.into_relation();

    for (idx, (b, t, u, s)) in out {
        println!(
            "review: {}, book: {}, book name: {}, reviewer: {}, score: {}",
            idx, b, t, u, s
        );
    }
}

fn from_join_ex() {
    let v1 = Variable::<u64>::new();
    v1.insert_data([(0, 1), (1, 2)]);
    v1.insert_data([(1, 0), (2, 1)]);

    while any_changed([v1.as_dyn()]) {
        v1.from_join(&v1, &v1, |k, &a, &b| (a, b));
    }

    let r = v1.into_relation();

    for (a, b) in r {
        println!("{} is reachable from {}", b, a);
    }
}

fn main() {
    for f in [books, from_join_ex] {
        f()
    }
}
