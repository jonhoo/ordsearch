extern crate criterion;
extern crate ordsearch;
extern crate num_traits;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
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

fn make_this<T: Ord>(v: &mut Vec<T>) -> OrderedCollection<&T> {
    OrderedCollection::from_slice(v)
}

fn search_this<'a, T: Ord>(c: &OrderedCollection<&'a T>, x: T) -> Option<&'a T> {
    c.find_gte(x).map(|v| &**v)
}

fn make_btreeset<T: Ord>(v: &mut Vec<T>) -> BTreeSet<&T> {
    use std::iter::FromIterator;
    BTreeSet::from_iter(v.iter())
}

fn search_btreeset<'a, T: Ord>(c: &BTreeSet<&'a T>, x: T) -> Option<&'a T> {
    use std::collections::Bound;
    c.range((Bound::Included(x), Bound::Unbounded))
        .next()
        .map(|v| &**v)
}

fn make_sorted_vec<T: Ord>(v: &mut Vec<T>) -> &[T] {
    v.sort_unstable();
    &v[..]
}

fn search_sorted_vec<'a, T: Ord>(c: &'a &[T], x: T) -> Option<&'a T> {
    c.binary_search(&x).ok().map(|i| &c[i])
}

fn bench_vec(cache: Cache, input: usize) {
    let size = cache.size();
    let mut v: Vec<_> = (0..size).collect();
    let mut r = 0usize;
}

fn criterion_benchmark<T>(c: &mut Criterion)
    where
    T: TryFrom<usize> + Ord + std::ops::Rem<Output = T> + num_traits::ops::wrapping::WrappingMul,
    <T as TryFrom<usize>>::Error: core::fmt::Debug,
{
    let mut group = c.benchmark_group("Search");
    for i in [2usize, 16, 128, 1024, 4069].iter() {
        group.bench_with_input(BenchmarkId::new("sorted_vec", i), i, |b, i| {
            let size = *i;
            let mut v: Vec<T> = (0..*i).map(|int| T::try_from(int).unwrap()).collect();
            let mut r = 0usize;
            let c = make_sorted_vec(&mut v);
            b.iter(|| {
                r = r.wrapping_mul(1664525).wrapping_add(1013904223);
                let x = T::try_from(r % size).unwrap();
                search_sorted_vec(&c, x);
            })
        });

        group.bench_with_input(BenchmarkId::new("btreeset", i), i, |b, i| {
            let size = *i;
            let mut v: Vec<T> = (0..*i).map(|int| T::try_from(int).unwrap()).collect();
            let mut r = 0usize;
            let c = make_btreeset(&mut v);
            b.iter(|| {
                r = r.wrapping_mul(1664525).wrapping_add(1013904223);
                let x = T::try_from(r % size).unwrap();
                search_btreeset(&c, x);
            })
        });

        group.bench_with_input(BenchmarkId::new("ordsearch", i), i, |b, i| {
            let size = *i;
            let mut v: Vec<T> = (0..*i).map(|int| T::try_from(int).unwrap()).collect();
            let mut r = 0usize;
            let c = make_this(&mut v);
            b.iter(|| {
                r = r.wrapping_mul(1664525).wrapping_add(1013904223);
                let x = T::try_from(r % size).unwrap();
                search_this(&c, x);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark::<u8>);
criterion_main!(benches);
