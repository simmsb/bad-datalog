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

// pretty much the same as DataFrog
// https://github.com/frankmcsherry/blog/blob/master/posts/2018-05-19.md
// we'll have a sled tree for each relation for each database

/// basically a list of:
///   id :attribute value
///
/// either in memory, or backed by a sled database
pub trait Relation<T: Clone + 'static> {
    type Iter<'a>: Iterator<Item = (u64, Cow<'a, T>)>;

    fn is_empty(&self) -> bool;
    fn range(&self) -> RangeInclusive<u64>;
    fn next_after(&self, idx: u64) -> Option<u64>;
    fn get_inner(&self, idx: u64) -> Option<OnceOrMany<T>>;

    fn get(&self, idx: u64) -> OnceOrMany<T> {
        if let Some(x) = self.get_inner(idx) {
            x
        } else {
            OnceOrMany::Many(&[])
        }
    }

    fn iter<'a>(&'a self) -> Self::Iter<'a>;
    fn len(&self) -> usize;
}

pub struct DBBackedRelation<T> {
    elements: sled::Tree,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Relation<T> for DBBackedRelation<T>
where
    T: Clone + serde::de::DeserializeOwned + 'static,
{
    type Iter<'a> = DBBackedRelationIter<'a, T>;

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
            _ => RangeInclusive::new(1, 0),
        }
    }

    // doing stuff this way probably makes joining accidentally quadratic, oh well
    fn next_after(&self, idx: u64) -> Option<u64> {
        let (next, _) = self.elements.get_gt(idx.to_be_bytes()).unwrap()?;
        Some(next.as_ref().read_u64::<BigEndian>().unwrap())
    }

    fn get_inner(&self, idx: u64) -> Option<OnceOrMany<T>> {
        let v = self.elements.get(idx.to_be_bytes()).unwrap()?;
        Some(OnceOrMany::Once(
            bincode::deserialize::<'_, T>(v.as_ref()).unwrap(),
        ))
    }

    fn iter(&self) -> Self::Iter<'static> {
        DBBackedRelationIter {
            it: self.elements.iter(),
            _marker: PhantomData,
        }
    }

    fn len(&self) -> usize {
        self.elements.len()
    }
}

impl<T> DBBackedRelation<T> {
    pub(crate) fn from_tree<'a>(tree: sled::Tree) -> Self
    where
        T: serde::de::DeserializeOwned + PartialOrd,
    {
        DBBackedRelation {
            elements: tree,
            _marker: PhantomData,
        }
    }
}

pub struct EphemerealRelation<T> {
    elements: BTreeMultiMap<u64, T>,
}

impl<T: Clone + 'static> Relation<T> for EphemerealRelation<T> {
    type Iter<'a> = std::iter::Map<
        btreemultimap::MultiIter<'a, u64, T>,
        for<'b> fn((&'b u64, &'b T)) -> (u64, Cow<'b, T>),
    >;

    fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    fn range(&self) -> RangeInclusive<u64> {
        let first = self.elements.iter().next().map(|(k, _)| *k);
        let last = self.elements.iter().next_back().map(|(k, _)| *k);

        match (first, last) {
            (Some(first), Some(last)) => first..=last,
            _ => RangeInclusive::new(1, 0),
        }
    }

    // doing stuff this way probably makes joining accidentally quadratic, oh well
    fn next_after(&self, idx: u64) -> Option<u64> {
        self.elements
            .range((Bound::Excluded(idx), Bound::Unbounded))
            .next()
            .map(|(k, _)| *k)
    }

    fn get_inner(&self, idx: u64) -> Option<OnceOrMany<T>> {
        self.elements.get_vec(&idx).map(|x| OnceOrMany::Many(x))
    }

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        fn inner<'a, T: Clone>((k, v): (&'a u64, &'a T)) -> (u64, Cow<'a, T>) {
            (*k, Cow::Borrowed(v))
        }

        self.elements.iter().map(inner)
    }

    fn len(&self) -> usize {
        self.elements.len()
    }
}

impl<T: Clone + 'static> EphemerealRelation<T> {
    fn empty() -> EphemerealRelation<T> {
        std::iter::empty().collect()
    }

    fn merge<R>(self, other: R) -> EphemerealRelation<T>
    where
        R: Relation<T> + IntoIterator<Item = (u64, T)>,
        T: Ord,
    {
        itertools::merge(self.into_iter(), other.into_iter()).collect()
    }
}

impl<T> FromIterator<(u64, T)> for EphemerealRelation<T> {
    fn from_iter<I>(it: I) -> Self
    where
        I: IntoIterator<Item = (u64, T)>,
    {
        EphemerealRelation {
            elements: it.into_iter().collect::<BTreeMultiMap<u64, T>>(),
        }
    }
}

pub enum OnceOrMany<'a, T> {
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

pub struct DBBackedRelationIter<'a, T> {
    it: sled::Iter,
    _marker: PhantomData<fn() -> (&'a (), T)>,
}

impl<'a, T> Iterator for DBBackedRelationIter<'a, T>
where
    T: Clone + serde::de::DeserializeOwned + 'static,
{
    type Item = (u64, Cow<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        let (idx, val) = self.it.next()?.unwrap();
        let idx = idx.as_ref().read_u64::<BigEndian>().unwrap();
        let val = bincode::deserialize::<'_, T>(val.as_ref()).unwrap();
        Some((idx, Cow::Owned(val)))
    }
}

pub struct EphemerealRelationIter<'a, T> {
    it: btreemultimap::MultiIter<'a, u64, T>,
}

impl<'a, T: Clone> Iterator for EphemerealRelationIter<'a, T> {
    type Item = (u64, Cow<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(&k, v)| (k, Cow::Borrowed(v)))
    }
}

pub struct DBBackedRelationIntoIter<T> {
    it: sled::Iter,
    _marker: PhantomData<fn() -> T>,
}

type EphemerealRelationIntoIter<T> = std::iter::FlatMap<
    std::collections::btree_map::IntoIter<u64, Vec<T>>,
    std::iter::Zip<std::iter::Repeat<u64>, std::vec::IntoIter<T>>,
    fn((u64, Vec<T>)) -> std::iter::Zip<std::iter::Repeat<u64>, std::vec::IntoIter<T>>,
>;

impl<T> Iterator for DBBackedRelationIntoIter<T>
where
    T: serde::de::DeserializeOwned,
{
    type Item = (u64, T);

    fn next(&mut self) -> Option<Self::Item> {
        let (idx, val) = self.it.next()?.unwrap();
        let idx = idx.as_ref().read_u64::<BigEndian>().unwrap();
        let val = bincode::deserialize::<'_, T>(val.as_ref()).unwrap();
        Some((idx, val))
    }
}

impl<T> IntoIterator for DBBackedRelation<T>
where
    T: serde::de::DeserializeOwned,
{
    type Item = (u64, T);

    type IntoIter = DBBackedRelationIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        DBBackedRelationIntoIter {
            it: self.elements.iter(),
            _marker: PhantomData,
        }
    }
}

impl<T> IntoIterator for EphemerealRelation<T> {
    type Item = (u64, T);

    type IntoIter = EphemerealRelationIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        fn outer<T>(
            (idx, vals): (u64, Vec<T>),
        ) -> std::iter::Zip<std::iter::Repeat<u64>, std::vec::IntoIter<T>> {
            std::iter::repeat(idx).zip(vals.into_iter())
        }
        self.elements.into_iter().flat_map(outer)
    }
}

#[derive(Clone)]
pub struct Variable<T> {
    stable: Rc<RefCell<Vec<EphemerealRelation<T>>>>,
    recent: Rc<RefCell<EphemerealRelation<T>>>,
    to_add: Rc<RefCell<Vec<EphemerealRelation<T>>>>,
}

impl<T: Clone + 'static> Default for Variable<T> {
    fn default() -> Self {
        Variable {
            stable: Rc::new(RefCell::new(Vec::new())),
            recent: Rc::new(RefCell::new(EphemerealRelation::empty())),
            to_add: Rc::new(RefCell::new(Vec::new())),
        }
    }
}

impl<T: Clone + Ord + 'static> Variable<T> {
    pub fn new() -> Variable<T> {
        Default::default()
    }

    pub fn as_dyn(&self) -> &dyn VariableMeta {
        self
    }

    pub fn insert(&self, relation: EphemerealRelation<T>) {
        if !relation.is_empty() {
            self.to_add.borrow_mut().push(relation)
        }
    }

    pub fn insert_data<I>(&self, it: I)
    where
        I: IntoIterator<Item = (u64, T)>,
    {
        self.insert(EphemerealRelation::from_iter(it))
    }

    pub fn from_join<U, V, L, R, F, LR, RR>(&self, lhs: L, rhs: R, logic: F)
    where
        U: Clone + Ord + 'static,
        V: Clone + 'static,
        L: Joinable<U, LR> + Copy,
        LR: Relation<U>,
        R: Joinable<V, RR> + Copy,
        RR: Relation<V>,
        F: FnMut(u64, &U, &V) -> (u64, T),
    {
        join_into(lhs, rhs, self, logic);
    }

    pub fn from_map<U, F>(&self, input: &Variable<U>, mut logic: F)
    where
        U: Clone + 'static,
        F: FnMut(u64, &U) -> (u64, T),
    {
        self.insert(
            input
                .recent
                .borrow()
                .iter()
                .map(|(idx, v)| logic(idx, &v))
                .collect(),
        );
    }

    pub fn into_relation(self) -> EphemerealRelation<T> {
        assert!(self.recent.borrow().is_empty());
        assert!(self.to_add.borrow().is_empty());

        itertools::kmerge(self.stable.borrow_mut().drain(..).map(|r| r.into_iter())).collect()
    }

    fn changed(&self) -> bool {
        if !self.recent.borrow().is_empty() {
            let mut recent =
                std::mem::replace(&mut *self.recent.borrow_mut(), EphemerealRelation::empty());

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

            *self.recent.borrow_mut() = EphemerealRelation::from_iter(to_add);
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

fn join_helper<T, U, F, L, R>(lhs: &L, rhs: &R, mut result: F)
where
    T: Clone + 'static,
    U: Clone + 'static,
    L: Relation<T>,
    R: Relation<U>,
    F: FnMut(u64, &T, &U),
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

pub trait Joinable<T, R>
where
    T: Clone + 'static,
    R: Relation<T>,
{
    type Recent: Deref<Target = R>;
    type Stable: Deref<Target = [R]>;

    fn recent(self) -> Option<Self::Recent>;
    fn stable(self) -> Self::Stable;
}

impl<'a, T> Joinable<T, EphemerealRelation<T>> for &'a Variable<T>
where
    T: Clone + 'static,
{
    type Recent = Ref<'a, EphemerealRelation<T>>;
    type Stable = Ref<'a, [EphemerealRelation<T>]>;

    fn recent(self) -> Option<Self::Recent> {
        Some(self.recent.borrow())
    }

    fn stable(self) -> Self::Stable {
        Ref::map(self.stable.borrow(), Vec::as_slice)
    }
}

impl<'a, T, R> Joinable<T, R> for &'a R
where
    T: Clone + 'static,
    R: Relation<T>,
{
    type Recent = &'a R;
    type Stable = &'a [R];

    fn recent(self) -> Option<Self::Recent> {
        None
    }

    fn stable(self) -> Self::Stable {
        std::slice::from_ref(self)
    }
}

fn join_into<T, U, V, L, R, F, TR, RR>(
    lhs: L, //&Variable<T>,
    rhs: R,
    out: &Variable<V>,
    mut logic: F,
) where
    T: Clone + Ord + 'static,
    U: Clone + 'static,
    V: Clone + Ord + 'static,
    L: Joinable<T, TR> + Copy,
    TR: Relation<T>,
    R: Joinable<U, RR> + Copy,
    RR: Relation<U>,
    F: FnMut(u64, &T, &U) -> (u64, V),
{
    let mut results = Vec::new();

    if let Some(lhs_recent) = lhs.recent() {
        for rhs_batch in rhs.stable().iter() {
            join_helper(&*lhs_recent, rhs_batch, |k, l, r| {
                results.push(logic(k, l, r));
            });
        }
    }

    if let Some(rhs_recent) = rhs.recent() {
        for lhs_batch in lhs.stable().iter() {
            join_helper(lhs_batch, &*rhs_recent, |k, l, r| {
                results.push(logic(k, l, r));
            });
        }
    }

    if let (Some(lhs_recent), Some(rhs_recent)) = (lhs.recent(), rhs.recent()) {
        join_helper(&*lhs_recent, &*rhs_recent, |k, l, r| {
            results.push(logic(k, l, r));
        });
    }

    out.insert(EphemerealRelation::from_iter(results));
}

pub trait VariableMeta {
    fn changed(&self) -> bool;
}

impl<T> VariableMeta for Variable<T>
where
    T: Clone + Ord + 'static,
{
    fn changed(&self) -> bool {
        Variable::changed(self)
    }
}

pub fn any_changed<'a, I>(variables: I) -> bool
where
    I: IntoIterator<Item = &'a dyn VariableMeta>,
{
    let mut result = false;

    for variable in variables {
        result |= variable.changed();
    }

    result
}
