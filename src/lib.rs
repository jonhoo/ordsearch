//! This crate provides an efficient data structure for approximate lookups in ordered collections.
//!
//! More conretely, given a set `A` of `n` values, and a query value `x`, this library provides an
//! efficient mechanism for finding the smallest value in `A` that is greater than or equal to `x`.
//! In particular, this library caters to the important case where there are many such queries to
//! the same array, `A`.
//!
//! This library is constructed from the best solution identified in [Array Layouts for
//! Comparison-Based Searching](https://arxiv.org/abs/1509.05053) by Paul-Virak Khuong and Pat
//! Morin. For more information, see the paper, [their
//! website](http://cglab.ca/~morin/misc/arraylayout-v2/), and the [C++ implementation
//! repository](https://github.com/patmorin/arraylayout).
//!
//! # Current implementation
//!
//! At the time of writing, this implementation uses a branch-free search over an
//! Eytzinger-arranged array with masked prefetching based on the [C++
//! implementation](https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L253)
//! written by the authors of the aforementioned paper. This is the recommended algorithm from the
//! paper, and what the authors suggested in
//! https://github.com/patmorin/arraylayout/issues/3#issuecomment-338472755.
//!
//! Note that prefetching is *only* enabled with the (non-default) `nightly` feature due to
//! https://github.com/aweinstock314/prefetch/issues/1. Suggestions for workarounds welcome.
//!
//! # Performance
//!
//! The included benchmarks can be run with
//!
//! ```console,ignore
//! $ cargo +nightly bench --features nightly
//! ```
//!
//! This will benchmark both construction and search with different number of values, and
//! differently sized values -- look for the line that aligns closest with your data. The general
//! trend is that `ordsearch` is faster when `n` is smaller and `T` is larger. You may also want to
//! compare with the pending Rust PR "[Improve SliceExt::binary_search
//! performance](https://github.com/rust-lang/rust/pull/45333)".
//! [Summarized](https://github.com/BurntSushi/cargo-benchcmp) results from my laptop (an X1 Carbon
//! with i7-5600U @ 2.60GHz) are given below.
//!
//! Compared to binary search over a sorted vector:
//!
//! ```text,ignore
//! name                        b::sorted_vec:: ns/iter  b::this:: ns/iter  diff ns/iter   diff %  speedup
//! u8::search_l1               46                       38                           -8  -17.39%   x 1.21
//! u8::search_l1_dup           31                       37                            6   19.35%   x 0.84
//! u8::search_l2               45                       58                           13   28.89%   x 0.78
//! u8::search_l2_dup           30                       56                           26   86.67%   x 0.54
//! u8::search_l3               32                       165                         133  415.62%   x 0.19
//! u8::search_l3_dup           30                       120                          90  300.00%   x 0.25
//! u32::search_l1              62                       38                          -24  -38.71%   x 1.63
//! u32::search_l1_dup          40                       38                           -2   -5.00%   x 1.05
//! u32::search_l2              81                       64                          -17  -20.99%   x 1.27
//! u32::search_l2_dup          59                       64                            5    8.47%   x 0.92
//! u32::search_l3              176                      354                         178  101.14%   x 0.50
//! u32::search_l3_dup          152                      355                         203  133.55%   x 0.43
//! usize::search_l1            63                       37                          -26  -41.27%   x 1.70
//! usize::search_l1_dup        39                       37                           -2   -5.13%   x 1.05
//! usize::search_l2            83                       68                          -15  -18.07%   x 1.22
//! usize::search_l2_dup        58                       69                           11   18.97%   x 0.84
//! usize::search_l3            235                      469                         234   99.57%   x 0.50
//! usize::search_l3_dup        193                      474                         281  145.60%   x 0.41
//! ```
//!
//! Compared to a `BTreeSet`:
//!
//! ```text,ignore
//! name                        b::btreeset:: ns/iter  b::this:: ns/iter  diff ns/iter   diff %  speedup
//! u8::search_l1               67                     38                          -29  -43.28%   x 1.76
//! u8::search_l1_dup           50                     37                          -13  -26.00%   x 1.35
//! u8::search_l2               66                     58                           -8  -12.12%   x 1.14
//! u8::search_l2_dup           49                     56                            7   14.29%   x 0.88
//! u8::search_l3               55                     165                         110  200.00%   x 0.33
//! u8::search_l3_dup           49                     120                          71  144.90%   x 0.41
//! u32::search_l1              92                     38                          -54  -58.70%   x 2.42
//! u32::search_l1_dup          62                     38                          -24  -38.71%   x 1.63
//! u32::search_l2              122                    64                          -58  -47.54%   x 1.91
//! u32::search_l2_dup          84                     64                          -20  -23.81%   x 1.31
//! u32::search_l3              386                    354                         -32   -8.29%   x 1.09
//! u32::search_l3_dup          226                    355                         129   57.08%   x 0.64
//! usize::search_l1            93                     37                          -56  -60.22%   x 2.51
//! usize::search_l1_dup        63                     37                          -26  -41.27%   x 1.70
//! usize::search_l2            127                    68                          -59  -46.46%   x 1.87
//! usize::search_l2_dup        85                     69                          -16  -18.82%   x 1.23
//! usize::search_l3            456                    469                          13    2.85%   x 0.97
//! usize::search_l3_dup        272                    474                         202   74.26%   x 0.57
//! ```
//!
//! # Future work
//!
//!  - [ ] Implement aligned operation: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L204
//!  - [ ] Implement deep prefetching for large `T`: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L128
//!
#![deny(missing_docs)]
#![cfg_attr(feature = "nightly", feature(test))]
#![cfg_attr(feature = "nightly", feature(concat_idents))]
#[cfg(feature = "nightly")]
extern crate prefetch;
#[cfg(feature = "nightly")]
extern crate test;

use std::borrow::Borrow;

/// A collection of ordered items that can efficiently satisfy queries for nearby elements.
///
/// The most interesting method here is `find_gte`.
///
/// # Examples
///
/// ```
/// # use ordsearch::OrderedCollection;
/// let x = OrderedCollection::from(vec![1, 2, 4, 8, 16, 32, 64]);
/// assert_eq!(x.find_gte(0), Some(&1));
/// assert_eq!(x.find_gte(1), Some(&1));
/// assert_eq!(x.find_gte(3), Some(&4));
/// assert_eq!(x.find_gte(6), Some(&8));
/// assert_eq!(x.find_gte(8), Some(&8));
/// assert_eq!(x.find_gte(64), Some(&64));
/// assert_eq!(x.find_gte(65), None);
/// ```
pub struct OrderedCollection<T> {
    items: Vec<T>,

    #[cfg(feature = "nightly")] mask: usize,
}

impl<T: Ord> From<Vec<T>> for OrderedCollection<T> {
    /// Construct a new `OrderedCollection` from a vector of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ordsearch::OrderedCollection;
    /// let a = OrderedCollection::from(vec![42, 89, 7, 12]);
    /// assert_eq!(a.find_gte(50), Some(&89));
    /// ```
    fn from(mut v: Vec<T>) -> OrderedCollection<T> {
        v.sort_unstable();
        Self::from_sorted_iter(v.into_iter())
    }
}

/// Insert items from the sorted iterator `iter` into `v` in complete binary tree order.
///
/// Requires `iter` to be a sorted iterator.
/// Requires v's capacity to be set to the number of elements in `iter`.
/// The length of `v` will not be changed by this function.
fn eytzinger_walk<I, T>(v: &mut Vec<T>, iter: &mut I, i: usize)
where
    I: Iterator<Item = T>,
{
    if i >= v.capacity() {
        return;
    }

    // visit left child
    eytzinger_walk(v, iter, 2 * i + 1);

    // put data at the root
    // we know the get_unchecked_mut and unwrap below are safe because we set the Vec's capacity to
    // the length of the iterator.
    *unsafe { v.get_unchecked_mut(i) } = iter.next().unwrap();

    // visit right child
    eytzinger_walk(v, iter, 2 * i + 2);
}

impl<T: Ord> OrderedCollection<T> {
    /// Construct a new `OrderedCollection` from an iterator over sorted elements.
    ///
    /// Note that if the iterator is *not* sorted, no error will be given, but lookups will give
    /// incorrect results. The given iterator must also implement `ExactSizeIterator` so that we
    /// know the size of the lookup array.
    ///
    /// # Examples
    ///
    /// Using an already-sorted iterator:
    ///
    /// ```
    /// # use std::collections::BTreeSet;
    /// # use ordsearch::OrderedCollection;
    ///
    /// let mut s = BTreeSet::new();
    /// s.insert(42);
    /// s.insert(89);
    /// s.insert(7);
    /// s.insert(12);
    /// let a = OrderedCollection::from_sorted_iter(s);
    /// assert_eq!(a.find_gte(50), Some(&89));
    /// ```
    ///
    /// Sorting a collection and then iterating (in this case, you'd likely use `new` instead):
    ///
    /// ```
    /// # use ordsearch::OrderedCollection;
    /// let mut v = vec![42, 89, 7, 12];
    /// v.sort_unstable();
    /// let a = OrderedCollection::from_sorted_iter(v);
    /// assert_eq!(a.find_gte(50), Some(&89));
    /// ```
    ///
    /// The `OrderedCollection` can also be over references to somewhere else:
    ///
    /// ```
    /// # use std::collections::BTreeSet;
    /// # use ordsearch::OrderedCollection;
    ///
    /// let mut s = BTreeSet::new();
    /// s.insert(42);
    /// s.insert(89);
    /// s.insert(7);
    /// s.insert(12);
    /// let a = OrderedCollection::from_sorted_iter(s.iter());
    /// assert_eq!(a.find_gte(50), Some(&&89));
    /// ```
    ///
    pub fn from_sorted_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        let n = iter.len();
        let mut v = Vec::with_capacity(n);
        eytzinger_walk(&mut v, &mut iter, 0);

        // it's now safe to set the length, since all `n` elements have been inserted.
        unsafe { v.set_len(n) };

        #[cfg(feature = "nightly")]
        {
            let mut mask = 1;
            while mask <= n {
                mask <<= 1;
            }
            mask -= 1;

            OrderedCollection {
                items: v,
                mask: mask,
            }
        }
        #[cfg(not(feature = "nightly"))]
        OrderedCollection { items: v }
    }

    /// Construct a new `OrderedCollection` from a slice of elements.
    ///
    /// Note that the underlying slice will be reordered!
    ///
    /// # Examples
    ///
    /// ```
    /// # use ordsearch::OrderedCollection;
    /// let mut vals = [42, 89, 7, 12];
    /// let a = OrderedCollection::from_slice(&mut vals);
    /// assert_eq!(a.find_gte(50), Some(&&89));
    /// ```
    pub fn from_slice<'a>(v: &'a mut [T]) -> OrderedCollection<&'a T> {
        v.sort_unstable();
        OrderedCollection::from_sorted_iter(v.into_iter().map(|x| &*x))
    }

    /// Find the smallest value `v` such that `v >= x`.
    ///
    /// Returns `None` if there is no such `v`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ordsearch::OrderedCollection;
    /// let x = OrderedCollection::from(vec![1, 2, 4, 8, 16, 32, 64]);
    /// assert_eq!(x.find_gte(0), Some(&1));
    /// assert_eq!(x.find_gte(1), Some(&1));
    /// assert_eq!(x.find_gte(3), Some(&4));
    /// assert_eq!(x.find_gte(6), Some(&8));
    /// assert_eq!(x.find_gte(8), Some(&8));
    /// assert_eq!(x.find_gte(64), Some(&64));
    /// assert_eq!(x.find_gte(65), None);
    /// ```
    pub fn find_gte<'a, X>(&'a self, x: X) -> Option<&'a T>
    where
        T: Borrow<X>,
        X: Ord,
    {
        use std::mem;

        let mut i = 0;
        let multiplier = 64 / mem::size_of::<T>();
        let offset = multiplier + multiplier / 2;
        let _ = offset; // avoid warning about unused w/o nightly

        while i < self.items.len() {
            #[cfg(feature = "nightly")]
            {
                use prefetch::prefetch::*;
                // unsafe is safe because pointer is never dereferenced
                prefetch::<Read, High, Data, _>(unsafe {
                    self.items
                        .get_unchecked((multiplier * i + offset) & self.mask)
                } as *const _);
            }

            // safe because i < self.items.len()
            i = if x.borrow() <= unsafe { self.items.get_unchecked(i) }.borrow() {
                2 * i + 1
            } else {
                2 * i + 2
            };
        }

        // we want ffs(~(i + 1))
        // since ctz(x) = ffs(x) - 1
        // we use ctz(~(i + 1)) + 1
        let j = (i + 1) >> ((!(i + 1)).trailing_zeros() + 1);
        if j == 0 {
            None
        } else {
            Some(unsafe { self.items.get_unchecked(j - 1) })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::OrderedCollection;

    #[test]
    fn complete_exact() {
        let x = OrderedCollection::from(vec![1, 2, 4, 8, 16, 32, 64]);
        assert_eq!(x.find_gte(1), Some(&1));
        assert_eq!(x.find_gte(2), Some(&2));
        assert_eq!(x.find_gte(4), Some(&4));
        assert_eq!(x.find_gte(8), Some(&8));
        assert_eq!(x.find_gte(16), Some(&16));
        assert_eq!(x.find_gte(32), Some(&32));
        assert_eq!(x.find_gte(64), Some(&64));
    }

    #[test]
    fn complete_approximate() {
        let x = OrderedCollection::from(vec![1, 2, 4, 8, 16, 32, 64]);
        assert_eq!(x.find_gte(0), Some(&1));
        assert_eq!(x.find_gte(3), Some(&4));
        assert_eq!(x.find_gte(5), Some(&8));
        assert_eq!(x.find_gte(6), Some(&8));
        assert_eq!(x.find_gte(7), Some(&8));
        for i in 9..16 {
            assert_eq!(x.find_gte(i), Some(&16));
        }
        for i in 17..32 {
            assert_eq!(x.find_gte(i), Some(&32));
        }
        for i in 33..64 {
            assert_eq!(x.find_gte(i), Some(&64));
        }
        assert_eq!(x.find_gte(65), None);
    }

    #[test]
    fn unbalanced_exact() {
        let x = OrderedCollection::from(vec![1, 2, 4, 8, 16, 32, 64, 128, 256]);
        assert_eq!(x.find_gte(1), Some(&1));
        assert_eq!(x.find_gte(2), Some(&2));
        assert_eq!(x.find_gte(4), Some(&4));
        assert_eq!(x.find_gte(8), Some(&8));
        assert_eq!(x.find_gte(16), Some(&16));
        assert_eq!(x.find_gte(32), Some(&32));
        assert_eq!(x.find_gte(64), Some(&64));
        assert_eq!(x.find_gte(128), Some(&128));
        assert_eq!(x.find_gte(256), Some(&256));
    }

    #[test]
    fn unbalanced_approximate() {
        let x = OrderedCollection::from(vec![1, 2, 4, 8, 16, 32, 64, 128, 256]);
        assert_eq!(x.find_gte(0), Some(&1));
        assert_eq!(x.find_gte(3), Some(&4));
        assert_eq!(x.find_gte(5), Some(&8));
        assert_eq!(x.find_gte(6), Some(&8));
        assert_eq!(x.find_gte(7), Some(&8));
        for i in 9..16 {
            assert_eq!(x.find_gte(i), Some(&16));
        }
        for i in 17..32 {
            assert_eq!(x.find_gte(i), Some(&32));
        }
        for i in 33..64 {
            assert_eq!(x.find_gte(i), Some(&64));
        }
        for i in 65..128 {
            assert_eq!(x.find_gte(i), Some(&128));
        }
        for i in 129..256 {
            assert_eq!(x.find_gte(i), Some(&256));
        }
        assert_eq!(x.find_gte(257), None);
    }
}

#[cfg(feature = "nightly")]
mod b {
    use super::OrderedCollection;
    use test::Bencher;
    use test::black_box;
    use std::collections::BTreeSet;

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

    macro_rules! benches {
        ($t:ident, $v:ident) => {
            mod $v {
                use super::*;

                fn construction_nodup(c: Cache, b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    let mapper = concat_idents!(nodup_, $v);
                    bench_construction!(c, mk, mapper, b);
                }

                #[bench]
                fn construction_l1(b: &mut Bencher) {
                    construction_nodup(Cache::L1, b);
                }

                #[bench]
                fn construction_l2(b: &mut Bencher) {
                    construction_nodup(Cache::L2, b);
                }

                fn search_nodup(c: Cache, b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    let s = concat_idents!(search_, $t);
                    let mapper = concat_idents!(nodup_, $v);
                    bench_search!(c, mk, s, mapper, b);
                }

                #[bench]
                fn search_l1(b: &mut Bencher) {
                    search_nodup(Cache::L1, b);
                }

                #[bench]
                fn search_l2(b: &mut Bencher) {
                    search_nodup(Cache::L2, b);
                }

                #[bench]
                fn search_l3(b: &mut Bencher) {
                    search_nodup(Cache::L3, b);
                }

                fn construction_dup(c: Cache, b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    let mapper = concat_idents!(dup_, $v);
                    bench_construction!(c, mk, mapper, b);
                }

                #[bench]
                fn construction_l1_dup(b: &mut Bencher) {
                    construction_dup(Cache::L1, b);
                }

                #[bench]
                fn construction_l2_dup(b: &mut Bencher) {
                    construction_dup(Cache::L2, b);
                }

                fn search_dup(c: Cache, b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    let s = concat_idents!(search_, $t);
                    let mapper = concat_idents!(dup_, $v);
                    bench_search!(c, mk, s, mapper, b);
                }

                #[bench]
                fn search_l1_dup(b: &mut Bencher) {
                    search_dup(Cache::L1, b);
                }

                #[bench]
                fn search_l2_dup(b: &mut Bencher) {
                    search_dup(Cache::L2, b);
                }

                #[bench]
                fn search_l3_dup(b: &mut Bencher) {
                    search_dup(Cache::L3, b);
                }
            }
        }
    }

    macro_rules! bench_construction {
        ($cache:expr, $make:ident, $mapper:ident, $b:ident) => {
            let size = $cache.size();
            let mut v: Vec<_> = (0..size).map(&$mapper).collect();
            let mut r = 0usize;

            $b.iter(|| {
                for e in v.iter_mut() {
                    r = r.wrapping_mul(1664525).wrapping_add(1013904223);
                    *e = $mapper(r % size);
                }
                let x = $make(&mut v);
                drop(x);
            });
        }
    }

    macro_rules! bench_search {
        ($cache:expr, $make:ident, $search:ident, $mapper:ident, $b:ident) => {
            let size = $cache.size();
            let mut v: Vec<_> = (0..size).map(&$mapper).collect();
            let mut r = 0usize;

            let c = $make(&mut v);
            $b.iter(move || {
                // LCG constants from https://en.wikipedia.org/wiki/Numerical_Recipes.
                r = r.wrapping_mul(1664525).wrapping_add(1013904223);
                // Lookup the whole range to get 50% hits and 50% misses.
                let x = $mapper(r % size);

                black_box($search(&c, x).is_some());
            });
        }
    }

    fn make_this<T: Ord>(v: &mut Vec<T>) -> OrderedCollection<&T> {
        OrderedCollection::from_slice(v)
    }

    fn search_this<'a, T: Ord>(c: &OrderedCollection<&'a T>, x: T) -> Option<&'a T> {
        c.find_gte(x).map(|v| &**v)
    }

    mod this {
        use super::*;
        benches!(this, u8);
        benches!(this, u32);
        benches!(this, usize);
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

    mod btreeset {
        use super::*;
        benches!(btreeset, u8);
        benches!(btreeset, u32);
        benches!(btreeset, usize);
    }

    fn make_sorted_vec<T: Ord>(v: &mut Vec<T>) -> &[T] {
        v.sort_unstable();
        &v[..]
    }

    fn search_sorted_vec<'a, T: Ord>(c: &'a &[T], x: T) -> Option<&'a T> {
        c.binary_search(&x).ok().map(|i| &c[i])
    }

    mod sorted_vec {
        use super::*;
        benches!(sorted_vec, u8);
        benches!(sorted_vec, u32);
        benches!(sorted_vec, usize);
    }
}
