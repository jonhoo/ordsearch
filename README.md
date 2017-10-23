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

```text,ignore
name           sorted_vec ns/iter  this ns/iter  diff ns/iter   diff %  speedup
u8::l1         45                  38                      -7  -15.56%   x 1.18
u8::l1_dup     31                  37                       6   19.35%   x 0.84
u8::l2         45                  58                      13   28.89%   x 0.78
u8::l2_dup     31                  58                      27   87.10%   x 0.53
u8::l3         29                  166                    137  472.41%   x 0.17
u8::l3_dup     30                  118                     88  293.33%   x 0.25
u32::l1        65                  38                     -27  -41.54%   x 1.71
u32::l1_dup    39                  39                       0    0.00%   x 1.00
u32::l2        81                  64                     -17  -20.99%   x 1.27
u32::l2_dup    59                  65                       6   10.17%   x 0.91
u32::l3        172                 383                    211  122.67%   x 0.45
u32::l3_dup    150                 384                    234  156.00%   x 0.39
usize::l1      62                  37                     -25  -40.32%   x 1.68
usize::l1_dup  39                  37                      -2   -5.13%   x 1.05
usize::l2      82                  68                     -14  -17.07%   x 1.21
usize::l2_dup  57                  68                      11   19.30%   x 0.84
usize::l3      241                 520                    279  115.77%   x 0.46
usize::l3_dup  198                 518                    320  161.62%   x 0.38
```

Compared to a `BTreeSet`:

```text,ignore
name           btreeset ns/iter  this ns/iter  diff ns/iter   diff %  speedup
u8::l1         65                38                     -27  -41.54%   x 1.71
u8::l1_dup     48                37                     -11  -22.92%   x 1.30
u8::l2         64                58                      -6   -9.38%   x 1.10
u8::l2_dup     48                58                      10   20.83%   x 0.83
u8::l3         53                166                    113  213.21%   x 0.32
u8::l3_dup     49                118                     69  140.82%   x 0.42
u32::l1        90                38                     -52  -57.78%   x 2.37
u32::l1_dup    60                39                     -21  -35.00%   x 1.54
u32::l2        119               64                     -55  -46.22%   x 1.86
u32::l2_dup    85                65                     -20  -23.53%   x 1.31
u32::l3        390               383                     -7   -1.79%   x 1.02
u32::l3_dup    224               384                    160   71.43%   x 0.58
usize::l1      90                37                     -53  -58.89%   x 2.43
usize::l1_dup  60                37                     -23  -38.33%   x 1.62
usize::l2      122               68                     -54  -44.26%   x 1.79
usize::l2_dup  82                68                     -14  -17.07%   x 1.21
usize::l3      455               520                     65   14.29%   x 0.88
usize::l3_dup  264               518                    254   96.21%   x 0.51
```

## Future work

 - [ ] Implement aligned operation: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L204
 - [ ] Implement deep prefetching for large `T`: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L128

