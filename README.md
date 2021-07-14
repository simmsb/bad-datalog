# Simple datalog database and query planner

Something I've been messing around with: Trying to implement a datalog database
with a query api similar to datomic.

The datalog implementation is pretty much the same as
[datafrog](https://github.com/rust-lang/datafrog)'s implementation, but with
support for file-backed databases.


## Example

```rust
use dlog::{Database, QueryBuilder, RHS, object};
use tempfile::tempdir;

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

// b: 1, p: 101, t: bar, r: 6, s: 102
```
