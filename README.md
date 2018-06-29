# ordsearch

[![Crates.io](https://img.shields.io/crates/v/ordsearch.svg)](https://crates.io/crates/ordsearch)
[![Documentation](https://docs.rs/ordsearch/badge.svg)](https://docs.rs/ordsearch/)
[![Build Status](https://travis-ci.org/jonhoo/ordsearch.svg?branch=master)](https://travis-ci.org/jonhoo/ordsearch)

> NOTE: This crate is generally *slower* than using `Vec::binary_search` over a pre-sorted
> vector, contrary to the claims in the referenced paper, and is mainly presented for
> curiosity's sake at this point.

This crate provides a data structure for approximate lookups in ordered collections.

More concretely, given a set `A` of `n` values, and a query value `x`, this library provides an
efficient mechanism for finding the smallest value in `A` that is greater than or equal to `x`.
In particular, this library caters to the important case where there are many such queries to
the same array, `A`.

This library is constructed from the best solution identified in [Array Layouts for
Comparison-Based Searching](https://arxiv.org/abs/1509.05053) by Paul-Virak Khuong and Pat
Morin. For more information, see the paper, [their
website](http://cglab.ca/~morin/misc/arraylayout-v2/), and the [C++ implementation
repository](https://github.com/patmorin/arraylayout).

## Current implementation

At the time of writing, this implementation uses a branch-free search over an
Eytzinger-arranged array with masked prefetching based on the [C++
implementation](https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L253)
written by the authors of the aforementioned paper. This is the recommended algorithm from the
paper, and what the authors suggested in
https://github.com/patmorin/arraylayout/issues/3#issuecomment-338472755.

Note that prefetching is *only* enabled with the (non-default) `nightly` feature due to
https://github.com/aweinstock314/prefetch/issues/1. Suggestions for workarounds welcome.

## Performance

The included benchmarks can be run with

```console,ignore
$ cargo +nightly bench --features nightly
```

This will benchmark both construction and search with different number of values, and
differently sized values -- look for the line that aligns closest with your data. The general
trend is that `ordsearch` is faster when `n` is smaller and `T` is larger as long as you
compile with
[`target-cpu=native`](https://github.com/jonhoo/ordsearch/issues/2#issuecomment-390441137) and
[`lto=thin`](https://github.com/jonhoo/ordsearch/issues/2#issuecomment-390446671). The
performance gain seems to be best on Intel processors, and is smaller since the (relatively)
recent improvement to [SliceExt::binary_search
performance](https://github.com/rust-lang/rust/pull/45333).

Below are [summarized](https://github.com/BurntSushi/cargo-benchcmp) results from an AMD
ThreadRipper 2600X CPU run with:

```console
$ rustc +nightly --version
rustc 1.28.0-nightly (e3bf634e0 2018-06-28)
$ env CARGO_INCREMENTAL=0 RUSTFLAGS='-C target-cpu=native -C lto=thin' cargo +nightly bench --features nightly
```

Compared to binary search over a sorted vector:

```diff,ignore
 name           sorted_vec ns/iter  this ns/iter  diff ns/iter   diff %  speedup
-u32::l1        49                  54                       5   10.20%   x 0.91
+u32::l1_dup    40                  35                      -5  -12.50%   x 1.14
-u32::l2        63                  72                       9   14.29%   x 0.88
+u32::l2_dup    64                  62                      -2   -3.12%   x 1.03
-u32::l3        120                 273                    153  127.50%   x 0.44
-u32::l3_dup    117                 219                    102   87.18%   x 0.53
+u8::l1         42                  37                      -5  -11.90%   x 1.14
+u8::l1_dup     29                  28                      -1   -3.45%   x 1.04
+u8::l2         43                  49                       6   13.95%   x 0.88
-u8::l2_dup     33                  35                       2    6.06%   x 0.94
-u8::l3         45                  66                      21   46.67%   x 0.68
-u8::l3_dup     35                  51                      16   45.71%   x 0.69
-usize::l1      49                  54                       5   10.20%   x 0.91
+usize::l1_dup  38                  37                      -1   -2.63%   x 1.03
-usize::l2      65                  76                      11   16.92%   x 0.86
+usize::l2_dup  65                  64                      -1   -1.54%   x 1.02
-usize::l3      141                 303                    162  114.89%   x 0.47
-usize::l3_dup  140                 274                    134   95.71%   x 0.51
```

Compared to a `BTreeSet`:

```diff,ignore
 name           btreeset ns/iter  this ns/iter  diff ns/iter   diff %  speedup
+u32::l1        68                54                     -14  -20.59%   x 1.26
+u32::l1_dup    45                35                     -10  -22.22%   x 1.29
+u32::l2        88                72                     -16  -18.18%   x 1.22
-u32::l2_dup    61                62                       1    1.64%   x 0.98
+u32::l3        346               273                    -73  -21.10%   x 1.27
-u32::l3_dup    136               219                     83   61.03%   x 0.62
+u8::l1         45                37                      -8  -17.78%   x 1.22
+u8::l1_dup     31                28                      -3   -9.68%   x 1.11
-u8::l2         44                49                       5   11.36%   x 0.90
-u8::l2_dup     31                35                       4   12.90%   x 0.89
-u8::l3         43                66                      23   53.49%   x 0.65
-u8::l3_dup     30                51                      21   70.00%   x 0.59
+usize::l1      67                54                     -13  -19.40%   x 1.24
+usize::l1_dup  44                37                      -7  -15.91%   x 1.19
+usize::l2      89                76                     -13  -14.61%   x 1.17
-usize::l2_dup  60                64                       4    6.67%   x 0.94
+usize::l3      393               303                    -90  -22.90%   x 1.30
-usize::l3_dup  163               274                    111   68.10%   x 0.59
```

## Future work

 - [ ] Implement aligned operation: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L204
 - [ ] Implement deep prefetching for large `T`: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L128

