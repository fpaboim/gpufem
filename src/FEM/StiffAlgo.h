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
// StiffAlgo.h - Stiffness Algorithm Selection Parent Class (Strategy Pattern)
// Author: Francisco Paulo de Aboim (fpaboim@gmail.com)
////////////////////////////////////////////////////////////////////////////////

#ifndef STIFFALGO_H_
#define STIFFALGO_H_

#include <vector>

#include "SPRmatrix/SPRmatrix.h"
#include "Utils/util.h"

#define ivecvec std::vector<std::vector<int>>

class StiffAlgo {
public:
  StiffAlgo();
  virtual ~StiffAlgo();

  // creates derived class object according to device type and returns pointer
  virtual double CalcGlobalStiffness() = 0;

  void SetParallelColoring(bool usecoloring) {m_usecoloring = usecoloring;};
  bool GetParallelColoring() {return m_usecoloring;};
  void SetMakeAssembly(bool assemble) {m_assemble = assemble;};
  bool GetMakeAssembly() {return m_assemble;};

protected:
  void GetdShapeFunctNatCoord(int m_model_dim, int nnodes, fem_float* rst,
                              fem_float** dShapeMatrix);
  void GetJacobianMatrix(int m_model_dim, int nnodes, fem_float** Jacobian,
                         fem_float** coordsElem, fem_float** dShapeFunc);
  void BuildBMatrix(int modeldim, int nnodes, fem_float** matrixB,
                    fem_float** dNdcart);
  void BuildBMatrix2(int modeldim, int nnodes, fem_float** matrixB,
                     fem_float** dNdcart);

  bool m_assemble;  //Performs Assembly
  bool m_usecoloring;  //Uses coloring for parallel assembly
};

#endif  // STIFFALGO_H_
