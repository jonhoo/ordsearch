//! > NOTE: This crate is generally *slower* than using `Vec::binary_search` over a pre-sorted
//! > vector, contrary to the claims in the referenced paper, and is mainly presented for
//! > curiosity's sake at this point.
//!
//! This crate provides a data structure for approximate lookups in ordered collections.
//!
//! More concretely, given a set `A` of `n` values, and a query value `x`, this library provides an
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
//! trend is that `ordsearch` is faster when `n` is smaller and `T` is larger as long as you
//! compile with
//! [`target-cpu=native`](https://github.com/jonhoo/ordsearch/issues/2#issuecomment-390441137) and
//! [`lto=thin`](https://github.com/jonhoo/ordsearch/issues/2#issuecomment-390446671). The
//! performance gain seems to be best on Intel processors, and is smaller since the (relatively)
//! recent improvement to [SliceExt::binary_search
//! performance](https://github.com/rust-lang/rust/pull/45333).
//!
//! Below are [summarized](https://github.com/BurntSushi/cargo-benchcmp) results from an AMD
//! ThreadRipper 2600X CPU run with:
//!
//! ```console
//! $ rustc +nightly --version
//! rustc 1.28.0-nightly (e3bf634e0 2018-06-28)
//! $ env CARGO_INCREMENTAL=0 RUSTFLAGS='-C target-cpu=native -C lto=thin' cargo +nightly bench --features nightly
//! ```
//!
//! Compared to binary search over a sorted vector:
//!
//! ```diff,ignore
//!  name           sorted_vec ns/iter  this ns/iter  diff ns/iter   diff %  speedup
//! -u32::l1        49                  54                       5   10.20%   x 0.91
//! +u32::l1_dup    40                  35                      -5  -12.50%   x 1.14
//! -u32::l2        63                  72                       9   14.29%   x 0.88
//! +u32::l2_dup    64                  62                      -2   -3.12%   x 1.03
//! -u32::l3        120                 273                    153  127.50%   x 0.44
//! -u32::l3_dup    117                 219                    102   87.18%   x 0.53
//! +u8::l1         42                  37                      -5  -11.90%   x 1.14
//! +u8::l1_dup     29                  28                      -1   -3.45%   x 1.04
//! +u8::l2         43                  49                       6   13.95%   x 0.88
//! -u8::l2_dup     33                  35                       2    6.06%   x 0.94
//! -u8::l3         45                  66                      21   46.67%   x 0.68
//! -u8::l3_dup     35                  51                      16   45.71%   x 0.69
//! -usize::l1      49                  54                       5   10.20%   x 0.91
//! +usize::l1_dup  38                  37                      -1   -2.63%   x 1.03
//! -usize::l2      65                  76                      11   16.92%   x 0.86
//! +usize::l2_dup  65                  64                      -1   -1.54%   x 1.02
//! -usize::l3      141                 303                    162  114.89%   x 0.47
//! -usize::l3_dup  140                 274                    134   95.71%   x 0.51
//! ```
//!
//! Compared to a `BTreeSet`:
//!
//! ```diff,ignore
//!  name           btreeset ns/iter  this ns/iter  diff ns/iter   diff %  speedup
//! +u32::l1        68                54                     -14  -20.59%   x 1.26
//! +u32::l1_dup    45                35                     -10  -22.22%   x 1.29
//! +u32::l2        88                72                     -16  -18.18%   x 1.22
//! -u32::l2_dup    61                62                       1    1.64%   x 0.98
//! +u32::l3        346               273                    -73  -21.10%   x 1.27
//! -u32::l3_dup    136               219                     83   61.03%   x 0.62
//! +u8::l1         45                37                      -8  -17.78%   x 1.22
//! +u8::l1_dup     31                28                      -3   -9.68%   x 1.11
//! -u8::l2         44                49                       5   11.36%   x 0.90
//! -u8::l2_dup     31                35                       4   12.90%   x 0.89
//! -u8::l3         43                66                      23   53.49%   x 0.65
//! -u8::l3_dup     30                51                      21   70.00%   x 0.59
//! +usize::l1      67                54                     -13  -19.40%   x 1.24
//! +usize::l1_dup  44                37                      -7  -15.91%   x 1.19
//! +usize::l2      89                76                     -13  -14.61%   x 1.17
//! -usize::l2_dup  60                64                       4    6.67%   x 0.94
//! +usize::l3      393               303                    -90  -22.90%   x 1.30
//! -usize::l3_dup  163               274                    111   68.10%   x 0.59
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
#![cfg_attr(feature = "nightly", feature(core_intrinsics))]
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

    #[cfg(feature = "nightly")]
    mask: usize,
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

        let x = x.borrow();

        let mut i = 0;

        // this computation is a little finicky, so let's walk through it.
        //
        // we want to prefetch a couple of levels down in the tree from where we are.
        // however, we can only fetch one cacheline at a time (assume a line holds 64b).
        // we therefore need to find at what depth a single prefetch fetches all the descendants.
        // it turns out that, at depth k under some node with index i, the leftmost child is at:
        //
        //   2^k * i + 2^(k-1) + 2^(k-2) + ... + 2^0 = 2^k * i + 2^k - 1
        //
        // this follows from the fact that the leftmost immediate child of node i is at 2i + 1 by
        // recursively expanding i. if you're curious, the rightmost child is at:
        //
        //   2^k * i + 2^k + 2^(k-1) + ... + 2^1 = 2^k * i + 2^(k+1) - 1
        //
        // at depth k, there are 2^k children. we can fit 64/sizeof(T) children in a cacheline, so
        // we want to use the depth k that has 64/sizeof(T) children. so, we want:
        //
        //   2^k = 64/sizeof(T)
        //
        // but, we don't actually *need* k. we only ever use 2^k. so, we can just use 64/sizeof(T)
        // directly! nice. we call this the multiplier (because it's what we'll multiply i by).
        let multiplier = 64 / mem::size_of::<T>();
        // now for those additions we had to do above. well, we know that the offset is really just
        // 2^k - 1, and we know that multiplier == 2^k, so we're done. right?
        //
        // right?
        //
        // well, only sort of. the prefetch instruction fetches the cache-line that *holds* the
        // given memory address. let's denote cache lines with []. what if we have:
        //
        //   [..., 2^k + 2^k-1] [2^k + 2^k, ...]
        //
        // essentially, we got unlucky with the alignment so that the leftmost child is not sharing
        // a cacheline with any of the other items at that level! that's not great. so, instead, we
        // prefetch the address that is half-way through the set of children. that way, we ensure
        // that we prefetch at least half of the items.
        let offset = multiplier + multiplier / 2;
        let _ = offset; // avoid warning about unused w/o nightly

        while i < self.items.len() {
            #[cfg(feature = "nightly")]
            // unsafe is safe because pointer is never dereferenced
            unsafe {
                use std::intrinsics::prefetch_read_data;
                prefetch_read_data(
                    self.items
                        .as_ptr()
                        .offset(((multiplier * i + offset) & self.mask) as isize),
                    3,
                )
            };

            // safe because i < self.items.len()
            i = if x <= unsafe { self.items.get_unchecked(i) }.borrow() {
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

#[cfg(all(feature = "nightly", test))]
mod b {
    use super::OrderedCollection;
    use std::collections::BTreeSet;
    use test::black_box;
    use test::Bencher;

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

    macro_rules! construction_benches {
        ($t:ident, $v:ident) => {
            mod $v {
                use super::*;
                fn nodup(c: Cache, b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    let mapper = concat_idents!(nodup_, $v);
                    bench_construction!(c, mk, mapper, b);
                }

                #[bench]
                fn l1(b: &mut Bencher) {
                    nodup(Cache::L1, b);
                }

                #[bench]
                fn l2(b: &mut Bencher) {
                    nodup(Cache::L2, b);
                }

                fn dup(c: Cache, b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    let mapper = concat_idents!(dup_, $v);
                    bench_construction!(c, mk, mapper, b);
                }

                #[bench]
                fn l1_dup(b: &mut Bencher) {
                    dup(Cache::L1, b);
                }

                #[bench]
                fn l2_dup(b: &mut Bencher) {
                    dup(Cache::L2, b);
                }
            }
        };
    }

    macro_rules! search_benches {
        ($t:ident, $v:ident) => {
            mod $v {
                use super::*;
                fn nodup(c: Cache, b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    let s = concat_idents!(search_, $t);
                    let mapper = concat_idents!(nodup_, $v);
                    bench_search!(c, mk, s, mapper, b);
                }

                #[bench]
                fn l1(b: &mut Bencher) {
                    nodup(Cache::L1, b);
                }

                #[bench]
                fn l2(b: &mut Bencher) {
                    nodup(Cache::L2, b);
                }

                #[bench]
                fn l3(b: &mut Bencher) {
                    nodup(Cache::L3, b);
                }

                fn dup(c: Cache, b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    let s = concat_idents!(search_, $t);
                    let mapper = concat_idents!(dup_, $v);
                    bench_search!(c, mk, s, mapper, b);
                }

                #[bench]
                fn l1_dup(b: &mut Bencher) {
                    dup(Cache::L1, b);
                }

                #[bench]
                fn l2_dup(b: &mut Bencher) {
                    dup(Cache::L2, b);
                }

                #[bench]
                fn l3_dup(b: &mut Bencher) {
                    dup(Cache::L3, b);
                }
            }
        };
    }

    macro_rules! benches {
        ($t:ident) => {
            mod $t {
                pub use super::*;
                mod construction {
                    pub use super::*;
                    construction_benches!($t, u8);
                    construction_benches!($t, u32);
                    construction_benches!($t, usize);
                }
                mod search {
                    pub use super::*;
                    search_benches!($t, u8);
                    search_benches!($t, u32);
                    search_benches!($t, usize);
                }
            }
        };
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
                black_box($make(&mut v));
            });
        };
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
        };
    }

    fn make_this<T: Ord>(v: &mut Vec<T>) -> OrderedCollection<&T> {
        OrderedCollection::from_slice(v)
    }

    fn search_this<'a, T: Ord>(c: &OrderedCollection<&'a T>, x: T) -> Option<&'a T> {
        c.find_gte(x).map(|v| &**v)
    }

    benches!(this);

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

    benches!(btreeset);

    fn make_sorted_vec<T: Ord>(v: &mut Vec<T>) -> &[T] {
        v.sort_unstable();
        &v[..]
    }

    fn search_sorted_vec<'a, T: Ord>(c: &'a &[T], x: T) -> Option<&'a T> {
        c.binary_search(&x).ok().map(|i| &c[i])
    }

    benches!(sorted_vec);
}
