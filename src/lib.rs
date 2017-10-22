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
//! These will benchmark both construction and search with different number of values, and
//! differently sized values. `search_common` is likely the metric you want to pay the most
//! attention to: it is the cost of looking up among 1024 `usize` values.
//! [Summarized](https://github.com/BurntSushi/cargo-benchcmp) results from my laptop (an X1 Carbon
//! with i7-5600U @ 2.60GHz) are given below. **Note that these numbers include random number
//! generation**, so they provide only a lower bound.
//!
//! Compared to binary search over a sorted vector:
//!
//! ```text,ignore
//! name                   sorted_vec:: ns/iter  this:: ns/iter  diff ns/iter   diff %  speedup
//! search_common          67                    35                       -32  -47.76%   x 1.91
//! search_few_u32         42                    21                       -21  -50.00%   x 2.00
//! search_few_u8          42                    21                       -21  -50.00%   x 2.00
//! search_many_u32        115                   98                       -17  -14.78%   x 1.17
//! search_many_u8         52                    70                        18   34.62%   x 0.74
//! construction_few_u32   3,365                 3,924                    559   16.61%   x 0.86
//! construction_few_u8    3,483                 4,125                    642   18.43%   x 0.84
//! construction_many_u32  2,338,019             2,619,590            281,571   12.04%   x 0.89
//! construction_many_u8   1,174,587             1,481,139            306,552   26.10%   x 0.79
//! ```
//!
//! Compared to a `BTreeSet`:
//!
//! ```text,ignore
//! name                   btreeset:: ns/iter  this:: ns/iter  diff ns/iter   diff %  speedup
//! search_common          94                  35                       -59  -62.77%   x 2.69
//! search_few_u32         70                  21                       -49  -70.00%   x 3.33
//! search_few_u8          64                  21                       -43  -67.19%   x 3.05
//! search_many_u32        182                 98                       -84  -46.15%   x 1.86
//! search_many_u8         69                  70                         1    1.45%   x 0.99
//! construction_few_u32   9,598               3,924                 -5,674  -59.12%   x 2.45
//! construction_few_u8    8,410               4,125                 -4,285  -50.95%   x 2.04
//! construction_many_u32  10,416,444          2,619,590         -7,796,854  -74.85%   x 3.98
//! construction_many_u8   3,643,919           1,481,139         -2,162,780  -59.35%   x 2.46
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
extern crate rand;
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
mod benchmarks {
    use super::OrderedCollection;
    use test::Bencher;
    use rand::{thread_rng, Rng};
    use std::collections::BTreeSet;

    macro_rules! benches {
        ($t:ident) => {
            mod $t {
                use super::*;

                #[bench]
                fn construction_few_u8(b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    bench_construction!(128, 0u8, mk, b);
                }

                #[bench]
                fn construction_many_u8(b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    bench_construction!(65535, 0u8, mk, b);
                }

                #[bench]
                fn construction_few_u32(b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    bench_construction!(128, 0u32, mk, b);
                }

                #[bench]
                fn construction_many_u32(b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    bench_construction!(65535, 0u32, mk, b);
                }

                #[bench]
                fn search_few_u8(b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    let s = concat_idents!(search_, $t);
                    bench_search!(128, 0u8, mk, s, b);
                }

                #[bench]
                fn search_many_u8(b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    let s = concat_idents!(search_, $t);
                    bench_search!(65535, 0u8, mk, s, b);
                }

                #[bench]
                fn search_common(b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    let s = concat_idents!(search_, $t);
                    bench_search!(1024, 0usize, mk, s, b);
                }

                #[bench]
                fn search_few_u32(b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    let s = concat_idents!(search_, $t);
                    bench_search!(128, 0u32, mk, s, b);
                }

                #[bench]
                fn search_many_u32(b: &mut Bencher) {
                    let mk = concat_idents!(make_, $t);
                    let s = concat_idents!(search_, $t);
                    bench_search!(65535, 0u32, mk, s, b);
                }
            }
        }
    }

    macro_rules! bench_construction {
        ($n:expr, $zero:expr, $make:ident, $b:ident) => {
            let mut v = vec![];
            v.resize($n, $zero);
            let mut rng = thread_rng();
            $b.iter(|| {
                for e in v.iter_mut() {
                    *e = rng.gen();
                }
                let x = $make(&mut v);
                drop(x);
            });
        }
    }

    macro_rules! bench_search {
        ($n:expr, $zero:expr, $make:ident, $search:ident, $b:ident) => {
            let mut v = vec![];
            v.resize($n, $zero);
            let mut rng = thread_rng();
            for e in v.iter_mut() {
                *e = rng.gen();
            }

            let c = $make(&mut v);
            $b.iter(|| {
                #[allow(unused_assignments)]
                let mut x = $zero;
                x = rng.gen();
                $search(&c, x);
            });
        }
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
