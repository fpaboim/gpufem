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

////////////////////////////////////////////////////////////////////////////////
// SprSolver.cc - Basic Linear Algebra Functions
// Author: Francisco Paulo de Aboim (fpaboim@gmail.com)
////////////////////////////////////////////////////////////////////////////////

#include "LAops/SprSolver.h"

#include <math.h>
#include <time.h>
#include <windows.h>
#include <assert.h>
#include <omp.h>
#include <cstdio>

#include "OpenCL/OCLwrapper.h"
#include "Utils/util.h"
#include "SPRmatrix/SPRmatrix.h"

/******************************************************************************/
//                           Direct Solvers                                //
/******************************************************************************/
////////////////////////////////////////////////////////////////////////////////
// Modifies contents of U address to contain cholesky decomposed lower matrix L
////////////////////////////////////////////////////////////////////////////////
int cholesky(SPRmatrix* matrixA, fem_float* vectorX, fem_float* VectorY,
             int dim, bool print) {
  int i, j, k;
  fem_float sum;
  fem_float*  DiagL = allocVector(dim, false);

  for (i = 0; i < dim; i++) {
    DiagL[i] = matrixA->GetElem(i, i);
  }

  // Decomposition
  for (i = 0; i < dim; i++) {
    DiagL[i] = sqrt(DiagL[i]);
    matrixA->SetElem(i, i, DiagL[i]);

    // Divide all below diagonal by diag value
    for (j = i+1; j < dim; j++) {
      fem_float newval = matrixA->GetElem(j, i) / DiagL[i];
      matrixA->SetElem(j, i, newval);
    }

    for (j = i+1; j < dim; j++) {
      DiagL[j] -= matrixA->GetElem(j, i) * matrixA->GetElem(j, i);
      for (k = j+1; k < dim; k++) {
        fem_float newval = matrixA->GetElem(k, j) - (matrixA->GetElem(j, i) *
                           matrixA->GetElem(k, i));
        matrixA->SetElem(k, j, newval);
      }
    }
  }

  if (print) {
    printf("Cholesky Decomposed Matrix (ignore above diag):\n");
    matrixA->PrintMatrix();
  }

  // Solves the system - L*x = y
  for (i = 0; i < dim; ++i) {
    sum = VectorY[i];
    for (k = i-1; k >= 0; --k) {
      sum = sum - (matrixA->GetElem(i, k)*vectorX[k]);
    }
    vectorX[i] = sum / DiagL[i];
  }
  // Solves L^t*x = y
  for (int i = (dim-1); i >= 0; --i) {
    sum = vectorX[i];
    for ( k = i+1; k < dim; k++ ) {
      sum = sum - (matrixA->GetElem(k, i)*vectorX[k]);
    }
    vectorX[i] = sum / DiagL[i];
  }

  free(DiagL);

  return 1;
}
