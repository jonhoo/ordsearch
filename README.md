# ordsearch

[![Crates.io](https://img.shields.io/crates/v/ordsearch.svg)](https://crates.io/crates/ordsearch)
[![Documentation](https://docs.rs/ordsearch/badge.svg)](https://docs.rs/ordsearch/)
[![Build Status](https://travis-ci.org/jonhoo/ordsearch.svg?branch=master)](https://travis-ci.org/jonhoo/ordsearch)

> NOTE: This crate is currently *slower* than using `Vec::binary_search` over a pre-sorted
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
trend is that `ordsearch` is faster when `n` is smaller and `T` is larger. You may also want to
compare with the pending Rust PR "[Improve SliceExt::binary_search
performance](https://github.com/rust-lang/rust/pull/45333)".
[Summarized](https://github.com/BurntSushi/cargo-benchcmp) results from my laptop (an X1 Carbon
with i7-5600U @ 2.60GHz) are given below.

Compared to binary search over a sorted vector:

```diff,ignore
 name           sorted_vec ns/iter  this ns/iter  diff ns/iter   diff %  speedup
-u32::l1        51                  103                     52  101.96%   x 0.50
-u32::l1_dup    42                  90                      48  114.29%   x 0.47
-u32::l2        67                  150                     83  123.88%   x 0.45
-u32::l2_dup    66                  146                     80  121.21%   x 0.45
-u32::l3        118                 352                    234  198.31%   x 0.34
-u32::l3_dup    119                 352                    233  195.80%   x 0.34
-u8::l1         47                  97                      50  106.38%   x 0.48
-u8::l1_dup     36                  85                      49  136.11%   x 0.42
-u8::l2         56                  149                     93  166.07%   x 0.38
-u8::l2_dup     45                  141                     96  213.33%   x 0.32
-u8::l3         68                  224                    156  229.41%   x 0.30
-u8::l3_dup     52                  197                    145  278.85%   x 0.26
-usize::l1      51                  105                     54  105.88%   x 0.49
-usize::l1_dup  42                  91                      49  116.67%   x 0.46
-usize::l2      68                  153                     85  125.00%   x 0.44
-usize::l2_dup  67                  148                     81  120.90%   x 0.45
-usize::l3      139                 463                    324  233.09%   x 0.30
-usize::l3_dup  139                 467                    328  235.97%   x 0.30
```

Compared to a `BTreeSet`:

```diff,ignore
 name           btreeset ns/iter  this ns/iter  diff ns/iter   diff %  speedup
+u32::l1        294               103                   -191  -64.97%   x 2.85
+u32::l1_dup    169               90                     -79  -46.75%   x 1.88
+u32::l2        364               150                   -214  -58.79%   x 2.43
+u32::l2_dup    239               146                    -93  -38.91%   x 1.64
+u32::l3        723               352                   -371  -51.31%   x 2.05
+u32::l3_dup    454               352                   -102  -22.47%   x 1.29
+u8::l1         222               97                    -125  -56.31%   x 2.29
+u8::l1_dup     155               85                     -70  -45.16%   x 1.82
+u8::l2         222               149                    -73  -32.88%   x 1.49
+u8::l2_dup     155               141                    -14   -9.03%   x 1.10
 u8::l3         222               224                      2    0.90%   x 0.99
-u8::l3_dup     155               197                     42   27.10%   x 0.79
+usize::l1      298               105                   -193  -64.77%   x 2.84
+usize::l1_dup  168               91                     -77  -45.83%   x 1.85
+usize::l2      368               153                   -215  -58.42%   x 2.41
+usize::l2_dup  242               148                    -94  -38.84%   x 1.64
+usize::l3      780               463                   -317  -40.64%   x 1.68
+usize::l3_dup  495               467                    -28   -5.66%   x 1.06
```

## Future work

 - [ ] Implement aligned operation: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L204
 - [ ] Implement deep prefetching for large `T`: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L128

