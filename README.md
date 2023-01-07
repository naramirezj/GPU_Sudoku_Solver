# GPU Sudoku Solver
GPU-based Sudoku solver, using C and CUDA.
The code for the solver is in sudoku.cu, a Makefile is provided so the code can be compiled through the terminal with the command:
~~~C
make
~~~
The file util.h contains a function to convert and keep track of time. The input files containing the sudoku boards are labeled as "tiny.csv", "small.csv", and "medium.csv".
The code is ran like:
~~~C
./sudoku tiny.csv
~~~
