# ordsearch

[![Crates.io](https://img.shields.io/crates/v/ordsearch.svg)](https://crates.io/crates/ordsearch)
[![Documentation](https://docs.rs/ordsearch/badge.svg)](https://docs.rs/ordsearch/)
[![Build Status](https://travis-ci.org/jonhoo/ordsearch.svg?branch=master)](https://travis-ci.org/jonhoo/ordsearch)

This crate provides an efficient data structure for approximate lookups in ordered collections.

More conretely, given a set `A` of `n` values, and a query value `x`, this library provides an
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
name                        b::sorted_vec:: ns/iter  b::this:: ns/iter  diff ns/iter   diff %  speedup
u8::search_l1               46                       38                           -8  -17.39%   x 1.21
u8::search_l1_dup           31                       37                            6   19.35%   x 0.84
u8::search_l2               45                       58                           13   28.89%   x 0.78
u8::search_l2_dup           30                       56                           26   86.67%   x 0.54
u8::search_l3               32                       165                         133  415.62%   x 0.19
u8::search_l3_dup           30                       120                          90  300.00%   x 0.25
u32::search_l1              62                       38                          -24  -38.71%   x 1.63
u32::search_l1_dup          40                       38                           -2   -5.00%   x 1.05
u32::search_l2              81                       64                          -17  -20.99%   x 1.27
u32::search_l2_dup          59                       64                            5    8.47%   x 0.92
u32::search_l3              176                      354                         178  101.14%   x 0.50
u32::search_l3_dup          152                      355                         203  133.55%   x 0.43
usize::search_l1            63                       37                          -26  -41.27%   x 1.70
usize::search_l1_dup        39                       37                           -2   -5.13%   x 1.05
usize::search_l2            83                       68                          -15  -18.07%   x 1.22
usize::search_l2_dup        58                       69                           11   18.97%   x 0.84
usize::search_l3            235                      469                         234   99.57%   x 0.50
usize::search_l3_dup        193                      474                         281  145.60%   x 0.41
```

Compared to a `BTreeSet`:

```text,ignore
name                        b::btreeset:: ns/iter  b::this:: ns/iter  diff ns/iter   diff %  speedup
u8::search_l1               67                     38                          -29  -43.28%   x 1.76
u8::search_l1_dup           50                     37                          -13  -26.00%   x 1.35
u8::search_l2               66                     58                           -8  -12.12%   x 1.14
u8::search_l2_dup           49                     56                            7   14.29%   x 0.88
u8::search_l3               55                     165                         110  200.00%   x 0.33
u8::search_l3_dup           49                     120                          71  144.90%   x 0.41
u32::search_l1              92                     38                          -54  -58.70%   x 2.42
u32::search_l1_dup          62                     38                          -24  -38.71%   x 1.63
u32::search_l2              122                    64                          -58  -47.54%   x 1.91
u32::search_l2_dup          84                     64                          -20  -23.81%   x 1.31
u32::search_l3              386                    354                         -32   -8.29%   x 1.09
u32::search_l3_dup          226                    355                         129   57.08%   x 0.64
usize::search_l1            93                     37                          -56  -60.22%   x 2.51
usize::search_l1_dup        63                     37                          -26  -41.27%   x 1.70
usize::search_l2            127                    68                          -59  -46.46%   x 1.87
usize::search_l2_dup        85                     69                          -16  -18.82%   x 1.23
usize::search_l3            456                    469                          13    2.85%   x 0.97
usize::search_l3_dup        272                    474                         202   74.26%   x 0.57
```

## Future work

 - [ ] Implement aligned operation: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L204
 - [ ] Implement deep prefetching for large `T`: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L128

