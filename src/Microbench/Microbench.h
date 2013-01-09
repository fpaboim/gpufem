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
// Microbench - class to aggregate microbenchmarks used for performance testing
// Author: Francisco Aboim
// TecGraf / PUC-RIO
////////////////////////////////////////////////////////////////////////////////
#ifndef MICROBENCH_H_
#define MICROBENCH_H_

#include "SprMatrix/Sprmatrix.h"
#include "FEM/FemData.h"

class Microbench {
public:
  static void BenchSearch();
  static void BenchCG();
  static void BenchCG2();
  static void BenchMV();

private:
  static double BenchAxyGPU(SPRmatrix* dummymatrix,
                            fem_float* xvec,
                            fem_float* yvec,
                            size_t     localsize,
                            int        nloops);
  static double BenchCGGPU(SPRmatrix* dummymatrix,
                           fem_float* xvec,
                           fem_float* yvec,
                           size_t     localsize,
                           int        nloops);
  static double BenchCGGPU(FemData*   femdata,
                           size_t     localsize,
                           int        nloops);
};

#endif // MICROBENCH_H_