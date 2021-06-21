#![allow(dead_code)]
#![feature(trait_alias)]

use btreemultimap::BTreeMultiMap;
use byteorder::{BigEndian, ReadBytesExt};
use itertools::Itertools;
use std::{
    borrow::Cow,
    cell::{Ref, RefCell},
    marker::PhantomData,
    ops::{Bound, Deref, RangeInclusive},
    rc::Rc,
};

enum Value {
    String(String),
    Ref(u64),
}

// pretty much the same as DataFrog
// https://github.com/frankmcsherry/blog/blob/master/posts/2018-05-19.md
// we'll have a sled tree for each relation for each database

pub struct DBBackedRelation<'db, T> {
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

    // doing stuff this way probably makes joining accidentally quadratic, oh well
    fn next_after(&self, idx: u64) -> Option<u64> {
        let (next, _) = self.elements.get_gt(idx.to_be_bytes()).unwrap()?;
        Some(next.as_ref().read_u64::<BigEndian>().unwrap())
    }

    fn get(&self, idx: u64) -> Option<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let v = self.elements.get(idx.to_be_bytes()).unwrap()?;
        Some(bincode::deserialize::<'_, T>(v.as_ref()).unwrap())
    }
}

pub struct EphemerealRelation<T> {
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

    // doing stuff this way probably makes joining accidentally quadratic, oh well
    fn next_after(&self, idx: u64) -> Option<u64> {
        self.elements
            .range((Bound::Excluded(idx), Bound::Unbounded))
            .next()
            .map(|(k, _)| *k)
    }

    fn get<'a>(&'a self, idx: u64) -> Option<&'a [T]> {
        self.elements.get_vec(&idx).map(|x| x.deref())
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
pub enum Relation<'db, T> {
    DBBacked(DBBackedRelation<'db, T>),
    Ephemereal(EphemerealRelation<T>),
}

// at what point do I trait this?
impl<'db, T> Relation<'db, T> {
    fn is_empty(&self) -> bool {
        match self {
            Self::DBBacked(x) => x.is_empty(),
            Self::Ephemereal(x) => x.is_empty(),
        }
    }

    fn iter(&self) -> RelationIter<T>
    where
        T: Clone + for<'de> serde::Deserialize<'de>,
    {
        self.into_iter()
    }

    fn range(&self) -> RangeInclusive<u64> {
        match self {
            Relation::DBBacked(x) => x.range(),
            Relation::Ephemereal(x) => x.range(),
        }
    }

    fn next_after(&self, idx: u64) -> Option<u64> {
        match self {
            Relation::DBBacked(x) => x.next_after(idx),
            Relation::Ephemereal(x) => x.next_after(idx),
        }
    }

    fn get<'a>(&'a self, idx: u64) -> OnceOrMany<'a, T>
    where
        T: serde::de::DeserializeOwned,
    {
        match self {
            Relation::DBBacked(x) => x
                .get(idx)
                .map_or_else(|| OnceOrMany::Many(&[]), OnceOrMany::Once),
            Relation::Ephemereal(x) => x
                .get(idx)
                .map_or_else(|| OnceOrMany::Many(&[]), OnceOrMany::Many),
        }
    }

    fn len(&self) -> usize {
        match self {
            Relation::DBBacked(x) => x.elements.len(),
            Relation::Ephemereal(x) => x.elements.len(),
        }
    }

    fn empty<'a>() -> Relation<'a, T> {
        Relation::from_iter(std::iter::empty())
    }

    fn merge<'a>(self, other: Self) -> Relation<'a, T>
    where
        T: serde::de::DeserializeOwned + PartialOrd,
    {
        Relation::from_iter(itertools::kmerge([self.into_iter(), other.into_iter()]))
    }

    pub fn from_iter<I>(it: I) -> Self
    where
        I: IntoIterator<Item = (u64, T)>,
    {
        Relation::Ephemereal(EphemerealRelation {
            elements: it.into_iter().collect::<BTreeMultiMap<u64, T>>(),
        })
    }
}

pub enum RelationIter<'a, T> {
    DBBacked(sled::Iter),
    Ephemereal(btreemultimap::MultiIter<'a, u64, T>),
}

impl<'a, T> Iterator for RelationIter<'a, T>
where
    T: Clone + for<'de> serde::Deserialize<'de>,
{
    type Item = (u64, Cow<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            RelationIter::DBBacked(it) => {
                let (idx, val) = it.next()?.unwrap();
                let idx = idx.as_ref().read_u64::<BigEndian>().unwrap();
                let val = bincode::deserialize::<'_, T>(val.as_ref()).unwrap();
                Some((idx, Cow::Owned(val)))
            }
            RelationIter::Ephemereal(it) => it.next().map(|(&k, v)| (k, Cow::Borrowed(v))),
        }
    }
}

impl<'a, 'db, T> IntoIterator for &'a Relation<'db, T>
where
    T: Clone + for<'de> serde::Deserialize<'de>,
{
    type Item = (u64, Cow<'a, T>);

    type IntoIter = RelationIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Relation::DBBacked(x) => RelationIter::DBBacked(x.elements.iter()),
            Relation::Ephemereal(x) => RelationIter::Ephemereal(x.elements.iter()),
        }
    }
}

pub enum RelationIntoIter<T> {
    DBBacked(sled::Iter),
    Ephemereal(
        std::iter::FlatMap<
            std::collections::btree_map::IntoIter<u64, Vec<T>>,
            std::iter::Zip<std::iter::Repeat<u64>, std::vec::IntoIter<T>>,
            fn((u64, Vec<T>)) -> std::iter::Zip<std::iter::Repeat<u64>, std::vec::IntoIter<T>>,
        >,
    ),
}

impl<T> Iterator for RelationIntoIter<T>
where
    T: for<'de> serde::Deserialize<'de>,
{
    type Item = (u64, T);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            RelationIntoIter::DBBacked(it) => {
                let (idx, val) = it.next()?.unwrap();
                let idx = idx.as_ref().read_u64::<BigEndian>().unwrap();
                let val = bincode::deserialize::<'_, T>(val.as_ref()).unwrap();
                Some((idx, val))
            }
            RelationIntoIter::Ephemereal(it) => it.next(),
        }
    }
}

impl<'db, T> IntoIterator for Relation<'db, T>
where
    T: for<'de> serde::Deserialize<'de>,
{
    type Item = (u64, T);

    type IntoIter = RelationIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        fn outer<T>(
            (idx, vals): (u64, Vec<T>),
        ) -> std::iter::Zip<std::iter::Repeat<u64>, std::vec::IntoIter<T>> {
            std::iter::repeat(idx).zip(vals.into_iter())
        }

        match self {
            Relation::DBBacked(x) => RelationIntoIter::DBBacked(x.elements.iter()),
            Relation::Ephemereal(x) => {
                RelationIntoIter::Ephemereal(x.elements.into_iter().flat_map(outer))
            }
        }
    }
}

#[derive(Clone)]
pub struct Variable<'db, T> {
    stable: Rc<RefCell<Vec<Relation<'db, T>>>>,
    recent: Rc<RefCell<Relation<'db, T>>>,
    to_add: Rc<RefCell<Vec<Relation<'db, T>>>>,
}

impl<'db, T> Variable<'db, T> {
    pub fn new() -> Variable<'db, T> {
        Variable {
            stable: Rc::new(RefCell::new(Vec::new())),
            recent: Rc::new(RefCell::new(Relation::empty())),
            to_add: Rc::new(RefCell::new(Vec::new())),
        }
    }

    pub fn as_dyn(&self) -> &dyn VariableMeta<'db>
    where
        T: Ord + for<'de> serde::Deserialize<'de>,
    {
        self
    }

    pub fn insert(&self, relation: Relation<'db, T>) {
        if !relation.is_empty() {
            self.to_add.borrow_mut().push(relation)
        }
    }

    pub fn insert_data<I>(&self, it: I)
    where
        I: IntoIterator<Item = (u64, T)>,
    {
        self.insert(Relation::from_iter(it))
    }

    pub fn from_join<U, V, R, F>(&self, lhs: &Variable<'db, U>, rhs: R, logic: F)
    where
        T: Ord,
        U: Ord + for<'de> serde::Deserialize<'de>,
        V: Ord + for<'de> serde::Deserialize<'de>,
        R: Joinable<'db, V> + Copy,
        F: FnMut(u64, &U, &V) -> (u64, T),
    {
        join_into(lhs, rhs, self, logic);
    }

    pub fn from_map<U, F>(&self, input: &Variable<'db, U>, mut logic: F)
    where
        T: Ord,
        U: Ord + Clone + for<'de> serde::Deserialize<'de>,
        F: FnMut(u64, &U) -> (u64, T),
    {
        self.insert(Relation::from_iter(
            input
                .recent
                .borrow()
                .iter()
                .map(|(idx, v)| logic(idx, v.as_ref())),
        ));
    }

    pub fn into_relation<'a>(self) -> Relation<'a, T>
    where
        T: PartialOrd + for<'de> serde::Deserialize<'de>,
    {
        assert!(self.recent.borrow().is_empty());
        assert!(self.to_add.borrow().is_empty());

        Relation::from_iter(itertools::kmerge(
            self.stable.borrow_mut().drain(..).map(|r| r.into_iter()),
        ))
    }

    fn changed(&self) -> bool
    where
        T: PartialOrd + for<'de> serde::Deserialize<'de>,
    {
        if !self.recent.borrow().is_empty() {
            let mut recent = std::mem::replace(&mut *self.recent.borrow_mut(), Relation::empty());

            while self
                .stable
                .borrow()
                .last()
                .map(|x| x.len() <= 2 * recent.len())
                == Some(true)
            {
                let last = self.stable.borrow_mut().pop().unwrap();

                recent = recent.merge(last);
            }

            self.stable.borrow_mut().push(recent);
        }

        if !self.to_add.borrow().is_empty() {
            let mut to_add =
                itertools::kmerge(self.to_add.borrow_mut().drain(..).map(|r| r.into_iter()))
                    .collect::<Vec<_>>();

            for batch in &*self.stable.borrow() {
                to_add.retain(|(idx, val)| !batch.get(*idx).into_iter().contains(val))
            }

            *self.recent.borrow_mut() = Relation::from_iter(to_add);
        }

        !self.recent.borrow().is_empty()
    }
}

macro_rules! break_on_none {
    ($e:expr) => {
        match $e {
            Some(x) => x,
            None => break,
        }
    };
}

fn join_helper<'db, T, U, F>(lhs: &Relation<'db, T>, rhs: &Relation<'db, U>, mut result: F)
where
    F: FnMut(u64, &T, &U),
    T: for<'de> serde::Deserialize<'de>,
    U: for<'de> serde::Deserialize<'de>,
{
    let (mut lhs_pos, lhs_end) = lhs.range().into_inner();
    let (mut rhs_pos, rhs_end) = rhs.range().into_inner();

    while (lhs_pos <= lhs_end) && (rhs_pos <= rhs_end) {
        match lhs_pos.cmp(&rhs_pos) {
            std::cmp::Ordering::Less => lhs_pos = break_on_none!(lhs.next_after(rhs_pos - 1)),
            std::cmp::Ordering::Equal => {
                for lhs_val in &lhs.get(lhs_pos) {
                    for rhs_val in &rhs.get(rhs_pos) {
                        result(lhs_pos, lhs_val, rhs_val);
                    }
                }

                lhs_pos = break_on_none!(lhs.next_after(lhs_pos));
                rhs_pos = break_on_none!(rhs.next_after(rhs_pos));
            }
            std::cmp::Ordering::Greater => rhs_pos = break_on_none!(rhs.next_after(lhs_pos - 1)),
        }
    }
}

pub trait Joinable<'db, T> {
    type Recent: Deref<Target = Relation<'db, T>>;
    type Stable: Deref<Target = [Relation<'db, T>]>;

    fn recent(self) -> Option<Self::Recent>;
    fn stable(self) -> Self::Stable;
}

impl<'a, 'db, T> Joinable<'db, T> for &'a Variable<'db, T> {
    type Recent = Ref<'a, Relation<'db, T>>;
    type Stable = Ref<'a, [Relation<'db, T>]>;

    fn recent(self) -> Option<Self::Recent> {
        Some(self.recent.borrow())
    }

    fn stable(self) -> Self::Stable {
        Ref::map(self.stable.borrow(), Vec::as_slice)
    }
}

impl<'a, 'db, T> Joinable<'db, T> for &'a Relation<'db, T> {
    type Recent = &'a Relation<'db, T>;
    type Stable = &'a [Relation<'db, T>];

    fn recent(self) -> Option<Self::Recent> {
        None
    }

    fn stable(self) -> Self::Stable {
        std::slice::from_ref(self)
    }
}

fn join_into<'db, T, U, V, R, F>(
    lhs: &Variable<'db, T>,
    rhs: R,
    out: &Variable<'_, V>,
    mut logic: F,
) where
    T: Ord + for<'de> serde::Deserialize<'de>,
    U: Ord + for<'de> serde::Deserialize<'de>,
    V: Ord,
    R: Joinable<'db, U> + Copy,
    F: FnMut(u64, &T, &U) -> (u64, V),
{
    let mut results = Vec::new();

    if let Some(lhs_recent) = lhs.recent() {
        for rhs_batch in rhs.stable().iter() {
            join_helper(&lhs_recent, &rhs_batch, |k, l, r| {
                results.push(logic(k, l, r));
            });
        }
    }

    if let Some(rhs_recent) = rhs.recent() {
        for lhs_batch in lhs.stable().iter() {
            join_helper(&lhs_batch, &rhs_recent, |k, l, r| {
                results.push(logic(k, l, r));
            });
        }
    }

    if let (Some(lhs_recent), Some(rhs_recent)) = (lhs.recent(), rhs.recent()) {
        join_helper(&lhs_recent, &rhs_recent, |k, l, r| {
            results.push(logic(k, l, r));
        });
    }

    out.insert(Relation::from_iter(results));
}

pub trait VariableMeta<'db> {
    fn changed(&self) -> bool;
}

impl<'db, T> VariableMeta<'db> for Variable<'db, T>
where
    T: Ord + for<'de> serde::Deserialize<'de>,
{
    fn changed(&self) -> bool {
        Variable::changed(self)
    }
}

pub fn any_changed<'a, 'db: 'a, I>(variables: I) -> bool
where
    I: IntoIterator<Item = &'a dyn VariableMeta<'db>>,
{
    let mut result = false;

    for variable in variables {
        result |= variable.changed();
    }

    result
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
