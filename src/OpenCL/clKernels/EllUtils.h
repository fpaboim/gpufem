// Kernel code
// Matrizes usando row-major order:
// M(row, col) = *(Matriz + row * Matriz + col)
#ifndef ELLUTILS_H
#define ELLUTILS_H

#define fem_float float

typedef struct Ellmat {
  __global float* elldata;
  __global int*   colidx;
  __global int*   rownnz;
  int ellwidth;
  int matdim;
} Ellmat;

inline int EllMakeMat(__global  float* elldata,
                      __global  int*   colidx,
                      __global  int*   rownnz,
                      int ellwidth,
                      int matdim,
                      Ellmat* ellmat) {
  if (ellmat == 0) {
    return 0;
  }
  ellmat->elldata  = elldata;
  ellmat->colidx   = colidx;
  ellmat->rownnz   = rownnz;
  ellmat->ellwidth = ellwidth;
  ellmat->matdim   = matdim;
  return 1;
}

inline float EllGetVal(Ellmat* ellmat, int row, int col) {
  for (int i = 0; i < ellmat->rownnz[i]; i++) {
    int tempcol = ellmat->colidx[i];
    if (tempcol > col) {
      return 0.0f;
    }
    if (tempcol == col) {
      int rowoffset = ellmat->ellwidth * i;
      return ellmat->elldata[rowoffset+i];
    }
  }
  return 0.0f;
}

inline float LocalBufferToEllmat(Ellmat* ellmat, int row, int col) {
  for (int i = 0; i < ellmat->rownnz[i]; i++) {
    int tempcol = ellmat->colidx[i];
    if (tempcol > col) {
      return 0.0f;
    }
    if (tempcol == col) {
      int rowoffset = ellmat->ellwidth * i;
      return ellmat->elldata[rowoffset+i];
    }
  }
  return 0.0f;
}

inline __global float* EllGetRef(Ellmat* ellmat, int row, int col) {
  for (int i = 0; i < ellmat->rownnz[i]; i++) {
    int tempcol = ellmat->colidx[i];
    if (tempcol > col) {
      return 0;
    }
    if (tempcol == col) {
      int rowoffset = ellmat->ellwidth * i;
      return &ellmat->elldata[rowoffset+i];
    }
  }
  return 0;
}


#endif  // ELLUTILS_H
