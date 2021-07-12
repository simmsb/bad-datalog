use std::{fmt::Display, rc::Rc};

use serde::{Deserialize, Serialize};
use sled::Db;

pub struct Database {
    inner: Db,
}

impl Database {
    pub fn from_sled(db: Db) -> Self {
        Database { inner: db }
    }

    /// Add a new object to the database, returning the ID assigned to it
    ///
    /// The object shouldn't exist in the database already.
    pub fn add_object(&mut self, object: &Object) -> u64 {
        assert!(object.id.is_none(), "can't insert an inserted object");

        let id = self.inner.generate_id().unwrap();
        let id_view = &id.to_be_bytes();

        for (k, v) in &object.inner {
            let tree = self.inner.open_tree(k).unwrap();

            tree.insert(id_view, v.as_slice()).unwrap();
        }

        id
    }
}

#[derive(Default, Debug, Clone)]
pub struct Object {
    id: Option<u64>,
    inner: Vec<(&'static str, Vec<u8>)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Value {
    S(Rc<String>),
    I(i64),
    U(u64),
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::S(s) => write!(f, "{}", s),
            Value::I(i) => write!(f, "{}", i),
            Value::U(u) => write!(f, "{}", u),
        }
    }
}

impl Value {
    pub fn u(self) -> Option<u64> {
        match self {
            Value::U(n) => Some(n),
            _ => None,
        }
    }

    pub fn i(self) -> Option<i64> {
        match self {
            Value::I(n) => Some(n),
            _ => None,
        }
    }

    pub fn s(self) -> Option<Rc<String>> {
        match self {
            Value::S(s) => Some(s),
            _ => None,
        }
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::S(Rc::new(s))
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::S(Rc::new(s.to_owned()))
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::I(v)
    }
}

impl From<u64> for Value {
    fn from(v: u64) -> Self {
        Value::U(v)
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::I(v as i64)
    }
}

impl From<u32> for Value {
    fn from(v: u32) -> Self {
        Value::U(v as u64)
    }
}

impl Object {
    pub fn with_attr<V>(&mut self, name: &'static str, value: V) -> &mut Self
    where
        V: Into<Value>,
    {
        self.inner
            .push((name, bincode::serialize(&value.into()).unwrap()));
        self
    }

    pub fn id(&self) -> Option<u64> {
        self.id
    }

    fn with_id(&mut self, id: u64) -> &mut Self {
        self.id = Some(id);
        self
    }
}

#[macro_export]
macro_rules! object {
    {$( $name:ident: $value:expr ),* $(,)?} => {
        {
            let mut obj = $crate::storage::Object::default();

            $( {obj.with_attr(stringify!($name), $value);} );*

            obj
        }
    };
}

#[test]
fn make_sure_object_works() {
    let o = object!(a: "lol", c: "aaa");

    println!("{:#?}", o);
}
