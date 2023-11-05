set term svg enhanced size 800,600 lw 1.2 background rgb 'white'
set output 'plot.svg'
set grid
set key bottom right
set multiplot layout 2,2
set style line 1 lc 'dark-green' lw 1.5 pt 4 ps 0.7
set style line 2 lc 'dark-red' lw 1.5 pt 6 ps 0.7
set style line 3 lc 'dark-yellow' lw 1.5 pt 8 ps 0.7
set size 0.5,0.5

set datafile separator ','

set ylabel "Execution time (ns.)"

unset xlabel
set logscale x 2
set xrange [8/2:65536*2]
input_path = "../target"

set origin 0,0.5
set title "{/:Bold u8}"
plot \
    input_path.'/ordsearch.csv' skip 1 using 1:2 with lp title 'ordsearch' ls 1, \
    input_path.'/sorted_vec.csv' skip 1 using 1:2 with lp title 'binary search' ls 2, \
    input_path.'/btreeset.csv' skip 1 using 1:2 with lp title 'BTree' ls 3

unset ylabel
unset key

set origin 0.5,0.5
set title "{/:Bold u16}"
plot \
    input_path.'/ordsearch.csv' skip 1 using 1:3 with lp title 'ordsearch' ls 1, \
    input_path.'/sorted_vec.csv' skip 1 using 1:3 with lp title 'binary search' ls 2, \
    input_path.'/btreeset.csv' skip 1 using 1:3 with lp title 'BTree' ls 3

set xlabel "Array size"
set ylabel "Execution time (ns.)"

set origin 0,0
set title "{/:Bold u32}"
plot \
    input_path.'/ordsearch.csv' skip 1 using 1:4 with lp title 'ordsearch' ls 1, \
    input_path.'/sorted_vec.csv' skip 1 using 1:4 with lp title 'binary search' ls 2, \
    input_path.'/btreeset.csv' skip 1 using 1:4 with lp title 'BTree' ls 3

unset ylabel

set origin 0.5,0
set title "{/:Bold u64}"
plot \
    input_path.'/ordsearch.csv' skip 1 using 1:5 with lp title 'ordsearch' ls 1, \
    input_path.'/sorted_vec.csv' skip 1 using 1:5 with lp title 'binary search' ls 2, \
    input_path.'/btreeset.csv' skip 1 using 1:5 with lp title 'BTree' ls 3