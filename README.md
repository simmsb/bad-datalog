# Simple datalog database and query planner

Something I've been messing around with: Trying to implement a datalog database
with a query api similar to datomic.

The datalog implementation is pretty much the same as
[datafrog](https://github.com/rust-lang/datafrog)'s implementation, but with
support for file-backed databases.
