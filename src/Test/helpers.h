// Copyright 2012, Francisco Aboim (fpaboim@tecgraf.puc-rio.br)
// All Rights Reserved
//
// This Program is licensed under the MIT License as follows:
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#ifndef HELPERS_H
#define HELPERS_H

#include "SPRmatrix/SPRmatrix.h"
#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<float> Eigsparsef;
typedef Eigen::SparseMatrix<double> Eigsparsed;

// header

template<class eigsparse>
eigsparse Calc_M_Mt_plus_S(eigsparse &M, eigsparse &S, int dim) {
  eigsparse Mt = M.transpose();
  eigsparse M_Mt(dim,dim);
  eigsparse res(dim,dim);
  M_Mt = M*Mt;
  res = M_Mt+S;
  return res;
}

inline void genSparsePosDefMatrices(Eigsparsef &sparsematf,
                                    Eigsparsed &sparsematd,
                                    SPRmatrix* mysparse,
                                    int dim) {
  using namespace Eigen;
  using namespace std;
  srand((unsigned int)time(NULL));
  Eigsparsef randmatf(dim,dim);
  Eigsparsed randmatd(dim,dim);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      double randval = (double) (rand() % 6);
      randmatf.coeffRef(i, j) = (float)randval;
      randmatd.coeffRef(i, j) = randval;
    }
  }
  Eigsparsef ScaledIdentf(dim,dim);
  Eigsparsed ScaledIdentd(dim,dim);
  ScaledIdentf.setIdentity();
  ScaledIdentd.setIdentity();
  ScaledIdentf = ScaledIdentf*10;
  ScaledIdentd = ScaledIdentd*10;
//   cout << "S mats(f /d):" << ScaledIdentf << endl << ScaledIdentd << endl;
  sparsematf = Calc_M_Mt_plus_S(randmatf, ScaledIdentf, dim);
  sparsematd = Calc_M_Mt_plus_S(randmatd, ScaledIdentd, dim);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      mysparse->SetElem(i, j, sparsematf.coeffRef(i,j));
    }
  }
}

#endif