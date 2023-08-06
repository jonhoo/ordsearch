extern crate criterion;
extern crate ordsearch;
extern crate num_traits;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, BenchmarkGroup, measurement::WallTime};
use ordsearch::OrderedCollection;
use std::{collections::BTreeSet, ops::Rem, convert::TryFrom};

// these benchmarks borrow from https://github.com/rust-lang/rust/pull/45333

enum Cache {
    L1,
    L2,
    L3,
}

impl Cache {
    pub fn size(&self) -> usize {
        match *self {
            Cache::L1 => 1000,      // 8kb
            Cache::L2 => 10_000,    // 80kb
            Cache::L3 => 1_000_000, // 8Mb
        }
    }
}

#[inline]
fn nodup_usize(i: usize) -> usize {
    i * 2
}

#[inline]
fn nodup_u8(i: usize) -> u8 {
    nodup_usize(i) as u8
}

#[inline]
fn nodup_u32(i: usize) -> u32 {
    nodup_usize(i) as u32
}

#[inline]
fn dup_usize(i: usize) -> usize {
    i / 16 * 16
}

#[inline]
fn dup_u8(i: usize) -> u8 {
    dup_usize(i) as u8
}

#[inline]
fn dup_u32(i: usize) -> u32 {
    dup_usize(i) as u32
}

fn make_this<T: Ord>(mut v: Vec<T>) -> OrderedCollection<T> {
    v.sort_unstable();
    OrderedCollection::from_sorted_iter(v.into_iter())
}

fn search_this<T: Ord>(c: &OrderedCollection<T>, x: T) -> Option<&T> {
    c.find_gte(x).map(|v| &*v)
}

fn make_btreeset<T: Ord>(v: Vec<T>) -> BTreeSet<T> {
    use std::iter::FromIterator;
    BTreeSet::from_iter(v.into_iter())
}

fn search_btreeset<T: Ord>(c: &BTreeSet<T>, x: T) -> Option<&T> {
    use std::collections::Bound;
    c.range((Bound::Included(x), Bound::Unbounded))
        .next()
        .map(|v| &*v)
}

fn make_sorted_vec<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort_unstable();
    v
}

fn search_sorted_vec<'a, T: Ord>(c: &'a &[T], x: T) -> Option<&'a T> {
    c.binary_search(&x).ok().map(|i| &c[i])
}

fn criterion_benchmark<T, const MAX: usize>(c: &mut Criterion)
    where
    T: TryFrom<usize> + Ord + std::ops::Rem<Output = T> + num_traits::ops::wrapping::WrappingMul,
    <T as TryFrom<usize>>::Error: core::fmt::Debug,
{
    let groupname = format!("Search {}", std::any::type_name::<T>());
    let mut group = c.benchmark_group(groupname);
    for i in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4069, 8192, 16384, 32768, 65536].iter() {
        // search_bench_case("sorted_vec", make_sorted_vec::<T>, search_sorted_vec, &mut group, i);
        search_bench_case("btreeset", make_btreeset::<T>, search_btreeset, &mut group, i);
        search_bench_case("ordsearch", make_this::<T>, search_this, &mut group, i);
    }
    group.finish();
}

fn search_bench_case<T, Coll>(name: &str, setup_fun: impl Fn(Vec<T>) -> Coll, search_fun: impl Fn(&Coll, T) -> Option<&T>, group: &mut BenchmarkGroup<WallTime>, i: &usize)
    where
    T: TryFrom<usize> + Ord + std::ops::Rem<Output = T> + num_traits::ops::wrapping::WrappingMul,
    <T as TryFrom<usize>>::Error: core::fmt::Debug,
{
    group.bench_with_input(BenchmarkId::new(name, i), i, |b, i| {
        let size = *i;
        let mut v: Vec<T> = (0..*i).map(|int| T::try_from(int).unwrap()).collect();
        let mut r = 0usize;
        let c = setup_fun(v);
        b.iter(|| {
            r = r.wrapping_mul(1664525).wrapping_add(1013904223);
            let x = T::try_from(r % size).unwrap();
            search_fun(&c, x);
        })
    });
}

criterion_group!(benches,
                 // criterion_benchmark::<u8, {u8::MAX as usize}>,
                 // criterion_benchmark::<u16, {u16::MAX as usize}>,
                 criterion_benchmark::<u32, {u32::MAX as usize}>,
                 criterion_benchmark::<u64, {u64::MAX as usize}>,
                 criterion_benchmark::<u128, {u64::MAX as usize}>,
);
criterion_main!(benches);
