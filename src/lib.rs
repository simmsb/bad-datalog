#![allow(dead_code)]
#![feature(trait_alias)]

use btreemultimap::BTreeMultiMap;
use byteorder::{BigEndian, ReadBytesExt};
use std::{
    collections::HashSet,
    marker::PhantomData,
    ops::{Bound, RangeInclusive},
};

#[derive(Debug)]
pub struct QueryBuilder {
    var_idx: u64,
    find: Vec<VariableID>,
    variables: HashSet<VariableID>,
    rules: Vec<DataPattern>,
}

impl QueryBuilder {
    pub fn new() -> Self {
        QueryBuilder {
            var_idx: 0,
            find: Vec::new(),
            variables: HashSet::new(),
            rules: Vec::new(),
        }
    }

    pub fn fresh_var(&mut self) -> VariableID {
        let r = self.var_idx;
        self.var_idx += 1;
        VariableID(r)
    }

    pub fn set_results(&mut self, find: &[VariableID]) -> &mut Self {
        self.find = find.to_vec();
        self
    }

    pub fn add_constraint(
        &mut self,
        source: VariableID,
        attribute: &str,
        value: Pattern,
    ) -> &mut Self {
        self.rules.push(DataPattern {
            source,
            attribute: attribute.to_owned(),
            value,
        });
        self
    }
}

enum Value {
    String(String),
    Ref(u64),
}

// pretty much the same as DataFrog
// https://github.com/frankmcsherry/blog/blob/master/posts/2018-05-19.md
struct Tuple<T> {
    lhs: u64,
    rhs: T,
}

// we'll have a sled tree for each relation for each database

struct DBBackedRelation<'db, T> {
    elements: &'db sled::Tree,
    _marker: PhantomData<fn() -> T>,
}

impl<'db, T> DBBackedRelation<'db, T> {
    fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    fn range(&self) -> RangeInclusive<u64> {
        let first = self.elements.first().unwrap();
        let last = self.elements.last().unwrap();

        match (first, last) {
            (Some((first, _)), Some((last, _))) => {
                let first = first.as_ref().read_u64::<BigEndian>().unwrap();
                let last = last.as_ref().read_u64::<BigEndian>().unwrap();
                first..=last
            }
            _ => 1..=0,
        }
    }

    fn next_after(&self, position: u64) -> u64 {
        let (next, _) = self
            .elements
            .get_gt(position.to_be_bytes())
            .unwrap()
            .unwrap();
        next.as_ref().read_u64::<BigEndian>().unwrap()
    }

    fn get(&self, position: u64) -> T
    where
        T: serde::de::DeserializeOwned + Clone,
    {
        let v = self.elements.get(position.to_be_bytes()).unwrap().unwrap();
        bincode::deserialize::<'_, T>(v.as_ref()).unwrap()
    }
}

struct EphemerealRelation<T> {
    elements: BTreeMultiMap<u64, T>,
}

impl<T> EphemerealRelation<T> {
    fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    fn range(&self) -> RangeInclusive<u64> {
        let first = self.elements.iter().next().map(|(k, _)| *k);
        let last = self.elements.iter().next_back().map(|(k, _)| *k);

        match (first, last) {
            (Some(first), Some(last)) => first..=last,
            _ => 1..=0,
        }
    }

    fn next_after(&self, position: u64) -> u64 {
        self.elements
            .range((Bound::Excluded(position), Bound::Unbounded))
            .next()
            .map(|(k, _)| *k)
            .unwrap()
    }

    fn get<'a>(&'a self, position: u64) -> &'a [T]
    where
        T: Clone,
    {
        &self.elements.get_vec(&position).unwrap()
    }
}

enum OnceOrMany<'a, T> {
    Once(T),
    Many(&'a [T]),
}

impl<'a, 'b: 'a, T> IntoIterator for &'b OnceOrMany<'a, T> {
    type Item = &'a T;

    type IntoIter = std::slice::Iter<'b, T>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            OnceOrMany::Once(o) => std::slice::from_ref(o).into_iter(),
            OnceOrMany::Many(m) => m.into_iter(),
        }
    }
}

/// basically a list of:
///   id :attribute value
///
/// either in memory, or backed by a sled database
enum Relation<'db, T> {
    DBBacked(DBBackedRelation<'db, T>),
    Ephemereal(EphemerealRelation<T>),
}

impl<'db, T> Relation<'db, T> {
    fn is_empty(&self) -> bool {
        match self {
            Self::DBBacked(x) => x.is_empty(),
            Self::Ephemereal(x) => x.is_empty(),
        }
    }

    fn range(&self) -> RangeInclusive<u64> {
        match self {
            Relation::DBBacked(x) => x.range(),
            Relation::Ephemereal(x) => x.range(),
        }
    }

    fn next_after(&self, position: u64) -> u64 {
        match self {
            Relation::DBBacked(x) => x.next_after(position),
            Relation::Ephemereal(x) => x.next_after(position),
        }
    }

    fn get<'a>(&'a self, position: u64) -> OnceOrMany<'a, T>
    where
        T: serde::de::DeserializeOwned + Clone,
    {
        match self {
            Relation::DBBacked(x) => OnceOrMany::Once(x.get(position)),
            Relation::Ephemereal(x) => OnceOrMany::Many(x.get(position)),
        }
    }

    fn from_iter<I>(it: I) -> Self
    where
        I: IntoIterator<Item = (u64, T)>,
    {
        Relation::Ephemereal(EphemerealRelation {
            elements: it.into_iter().collect::<BTreeMultiMap<u64, T>>(),
        })
    }
}

struct Variable<'db, T> {
    stable: Vec<Relation<'db, T>>,
    recent: Relation<'db, T>,
    to_add: Vec<Relation<'db, T>>,
}

impl<'db, T> Variable<'db, T> {
    fn insert(&mut self, relation: Relation<'db, T>) {
        if !relation.is_empty() {
            self.to_add.push(relation)
        }
    }
}

fn join_helper<'db, T, U, F>(lhs: &Relation<'db, T>, rhs: &Relation<'db, U>, mut result: F)
where
    F: FnMut(u64, &T, &U),
    T: std::clone::Clone + for<'de> serde::Deserialize<'de>,
    U: std::clone::Clone + for<'de> serde::Deserialize<'de>,
{
    let (mut lhs_pos, lhs_end) = lhs.range().into_inner();
    let (mut rhs_pos, rhs_end) = rhs.range().into_inner();

    while (lhs_pos <= lhs_end) && (rhs_pos <= rhs_end) {
        match lhs_pos.cmp(&rhs_pos) {
            std::cmp::Ordering::Less => lhs_pos = lhs.next_after(rhs_pos - 1),
            std::cmp::Ordering::Equal => {
                for lhs_val in &lhs.get(lhs_pos) {
                    for rhs_val in &rhs.get(rhs_pos) {
                        result(lhs_pos, lhs_val, rhs_val);
                    }
                }

                lhs_pos = lhs.next_after(lhs_pos);
                rhs_pos = rhs.next_after(rhs_pos);
            }
            std::cmp::Ordering::Greater => rhs_pos = rhs.next_after(lhs_pos - 1),
        }
    }
}

fn join_into<'db, T, U, V>(
    lhs: &Variable<'db, T>,
    rhs: &Variable<'db, U>,
    out: &mut Variable<'_, V>,
    mut logic: impl FnMut(u64, &T, &U) -> (u64, V),
) where
    T: Ord + Clone + for<'de> serde::Deserialize<'de>,
    U: Ord + Clone + for<'de> serde::Deserialize<'de>,
    V: Ord,
{
    let mut results = Vec::new();

    for rhs_batch in &rhs.stable {
        join_helper(&lhs.recent, rhs_batch, |k, l, r| {
            results.push(logic(k, l, r));
        });
    }

    for lhs_batch in &lhs.stable {
        join_helper(lhs_batch, &rhs.recent, |k, l, r| {
            results.push(logic(k, l, r));
        });
    }

    join_helper(&lhs.recent, &rhs.recent, |k, l, r| {
        results.push(logic(k, l, r));
    });

    out.insert(Relation::from_iter(results));
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct VariableID(u64);

#[derive(Debug, Clone)]
pub enum Pattern {
    Constant(String),
    Variable(VariableID),
}

#[derive(Debug, Clone)]
struct DataPattern {
    source: VariableID,
    attribute: String,
    value: Pattern,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
