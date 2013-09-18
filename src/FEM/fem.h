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
// fem.h - Finite Element Operations Header
// Author: Francisco Paulo de Aboim (fpaboim@gmail.com)
////////////////////////////////////////////////////////////////////////////////

#ifndef FEM_H_
#define FEM_H_

#include <vector>

#include "FEM/FemData.h"
#include "FEM/StiffAlgo.h"
#include "SPRmatrix/SPRmatrix.h"
#include "Utils/util.h"

class FEM {
public:
  FEM();
  ~FEM();

  enum DeviceMode {
    CPU,
    OMP,
    GPUOMP,
    GPU,
  };

  enum ConstraintMode {
    PEN = 0,
    SUB = 1,
  };

  // creates derived class object according to device type and returns pointer
  void       Init(bool assemble, DeviceMode devicemode);
  // Calls strategy to execute current stiffness calculation algorithm
  double     CalcStiffnessMat() {
    return m_stiffnessalgo->CalcGlobalStiffness(m_femdata);
  };
  // Builds Force Vector
  fem_float* BuildForceVec(int nNodes, int nNodalLoads, fem_float** nodalLoad);
  // Applies Constraint to stiffness matrix by type: 1-Penalty, 2-Substitution
  void       ApplyConstraint(ConstraintMode conmode, int num_supports,
                             int** node_support);

  // Member Access Functions
  void       SetDeviceMode(DeviceMode newdevicemode);
  void       SetAssemble(bool assemble) {
    m_stiffnessalgo->SetMakeAssembly(assemble);
  };
  bool       GetAssemble() {return m_stiffnessalgo->GetMakeAssembly();};
  void       SetUseColoring(bool usecolor) {
    m_stiffnessalgo->SetParallelColoring(usecolor);
  };
  bool       GetUseColoring() {return m_stiffnessalgo->GetParallelColoring();};
  FemData*   GetFemData() {return m_femdata;};

protected:
  bool        m_initialized;
  FemData*    m_femdata;
  StiffAlgo*  m_stiffnessalgo; // Strategy pattern encapsulates algorithms for
                               // calculating stiffness matrix
};

#endif  // FEM_H_
