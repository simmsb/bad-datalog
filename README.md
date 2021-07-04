# Simple datalog database and query planner

Something I've been messing around with: Trying to implement a datalog database
with a query api similar to datomic.

The datalog implementation is pretty much the same as
[datafrog](https://github.com/rust-lang/datafrog)'s implementation, but with
support for file-backed databases.


## Example

``` rust
let tmp = tempdir().unwrap();
let sled_db = sled::open(tmp.path()).unwrap();

let mut db = Database::from_sled(sled_db.clone());

// add some data

let book_foo = db.add_object(&object!(
    book_name: "foo",
    book_price: 100,
));

let book_bar = db.add_object(&object!(
    book_name: "bar",
    book_price: 100,
));

let _book_baz = db.add_object(&object!(
    book_name: "baz",
    book_price: 100,
));

let book_blah = db.add_object(&object!(
    book_name: "blah",
    book_price: 100,
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

// put together a query

let mut builder = QueryBuilder::default();

let b = builder.binding();
let p = builder.binding();
let t = builder.binding();
let r = builder.binding();
let u = builder.binding();
let s = builder.binding();

builder
    .pattern(b, "book_name", RHS::Bnd(t))
    .pattern(b, "book_price", RHS::Bnd(p))
    .pattern(r, "review_book", RHS::Bnd(b))
    .pattern(r, "review_user", RHS::Bnd(u))
    .pattern(r, "review_score", RHS::Bnd(s));

let plan = builder.plan();
let results = plan.execute(sled_db);

for row in results {
    println!(
        "b: {}, p: {}, t: {}, r: {}, u: {}, s: {}",
        row.fetch(b),
        row.fetch(p),
        row.fetch(t),
        row.fetch(r),
        row.fetch(u),
        row.fetch(s),
    )
}

// b: 0, p: 100, t: foo, r: 4, u: reviewer_0, s: 10
// b: 0, p: 100, t: foo, r: 5, u: reviewer_1, s: 100
// b: 1, p: 100, t: bar, r: 6, u: reviewer_0, s: 102
// b: 3, p: 100, t: blah, r: 7, u: reviewer_1, s: 99
```
