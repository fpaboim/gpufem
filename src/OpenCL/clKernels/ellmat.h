// Kernel code
// Matrizes usando row-major order:
// M(row, col) = *(Matriz + row * Matriz + col)
#define fem_float float

struct ellmat {
  fem_float* data;
  int*       colidx;
  int        rowsz;
};
