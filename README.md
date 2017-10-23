# ordsearch

[![Crates.io](https://img.shields.io/crates/v/ordsearch.svg)](https://crates.io/crates/ordsearch)
[![Documentation](https://docs.rs/ordsearch/badge.svg)](https://docs.rs/ordsearch/)
[![Build Status](https://travis-ci.org/jonhoo/ordsearch.svg?branch=master)](https://travis-ci.org/jonhoo/ordsearch)

This crate provides an efficient data structure for approximate lookups in ordered collections.

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
trend is that `ordsearch` is faster when `n` is smaller and `T` is larger. You may also want to
compare with the pending Rust PR "[Improve SliceExt::binary_search
performance](https://github.com/rust-lang/rust/pull/45333)".
[Summarized](https://github.com/BurntSushi/cargo-benchcmp) results from my laptop (an X1 Carbon
with i7-5600U @ 2.60GHz) are given below.

Compared to binary search over a sorted vector:

```diff,ignore
 name           sorted_vec ns/iter  this ns/iter  diff ns/iter   diff %  speedup
+u8::l1         46                  37                      -9  -19.57%   x 1.24
-u8::l1_dup     31                  37                       6   19.35%   x 0.84
-u8::l2         44                  58                      14   31.82%   x 0.76
-u8::l2_dup     31                  56                      25   80.65%   x 0.55
-u8::l3         29                  170                    141  486.21%   x 0.17
-u8::l3_dup     30                  127                     97  323.33%   x 0.24
+u32::l1        66                  37                     -29  -43.94%   x 1.78
+u32::l1_dup    41                  37                      -4   -9.76%   x 1.11
+u32::l2        85                  64                     -21  -24.71%   x 1.33
-u32::l2_dup    62                  64                       2    3.23%   x 0.97
-u32::l3        180                 380                    200  111.11%   x 0.47
-u32::l3_dup    156                 381                    225  144.23%   x 0.41
+usize::l1      66                  37                     -29  -43.94%   x 1.78
+usize::l1_dup  41                  37                      -4   -9.76%   x 1.11
+usize::l2      87                  67                     -20  -22.99%   x 1.30
-usize::l2_dup  62                  77                      15   24.19%   x 0.81
-usize::l3      247                 522                    275  111.34%   x 0.47
-usize::l3_dup  203                 614                    411  202.46%   x 0.33
```

Compared to a `BTreeSet`:

```diff,ignore
 name           btreeset ns/iter  this ns/iter  diff ns/iter   diff %  speedup
+u8::l1         48                37                     -11  -22.92%   x 1.30
-u8::l1_dup     35                37                       2    5.71%   x 0.95
-u8::l2         46                58                      12   26.09%   x 0.79
-u8::l2_dup     35                56                      21   60.00%   x 0.62
-u8::l3         36                170                    134  372.22%   x 0.21
-u8::l3_dup     34                127                     93  273.53%   x 0.27
+u32::l1        66                37                     -29  -43.94%   x 1.78
+u32::l1_dup    42                37                      -5  -11.90%   x 1.14
+u32::l2        91                64                     -27  -29.67%   x 1.42
-u32::l2_dup    60                64                       4    6.67%   x 0.94
-u32::l3        351               380                     29    8.26%   x 0.92
-u32::l3_dup    195               381                    186   95.38%   x 0.51
+usize::l1      66                37                     -29  -43.94%   x 1.78
+usize::l1_dup  42                37                      -5  -11.90%   x 1.14
+usize::l2      96                67                     -29  -30.21%   x 1.43
-usize::l2_dup  61                77                      16   26.23%   x 0.79
-usize::l3      441               522                     81   18.37%   x 0.84
-usize::l3_dup  241               614                    373  154.77%   x 0.39
```

## Future work

 - [ ] Implement aligned operation: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L204
 - [ ] Implement deep prefetching for large `T`: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L128

