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

#include "FEM/fem.h"
#include "FEM/StiffAlgoCPU.h"
#include "FEM/StiffAlgoGPU.h"
#include "FEM/StiffAlgoGpuOmp.h"

#include <omp.h>
#include <math.h>
#include <time.h>
#include <windows.h>
#include <assert.h>
#include <cstdio>
#include <cstdlib>

#include "LAops/LAops.h"
#include "Utils/util.h"
#include "Utils/fileIO.h"
#include "SPRmatrix/SPRmatrix.h"

/******************************************************************************/
//                            CPU FEM Operations                              //
/******************************************************************************/

// Constructor and destructor
FEM::FEM() {
  m_initialized = false;
  m_femdata = NULL;
  m_stiffnessalgo = NULL;
}

FEM::~FEM() {
  if (m_femdata)
    delete m_femdata; m_femdata = NULL;
  if (m_stiffnessalgo)
    delete m_stiffnessalgo; m_stiffnessalgo = NULL;
}

////////////////////////////////////////////////////////////////////////////////
// Makes New Finite Element Analysis Object
////////////////////////////////////////////////////////////////////////////////
void FEM::Init(bool assemble, DeviceMode devicemode) {
  if (!m_initialized) {
    m_femdata = new FemData;
    m_initialized = true;
    // Selects stiffness calculation algorithm
    switch (devicemode) {
    case CPU:
      m_stiffnessalgo = new StiffAlgoCPU(m_femdata);
      break;
    case GPUOMP:
      m_stiffnessalgo = new StiffAlgoGpuOmp(m_femdata);
      break;
    case GPU:
      m_stiffnessalgo = new StiffAlgoGPU(m_femdata);
      break;
    default:  //default falls back to CPU
      m_stiffnessalgo = new StiffAlgoCPU(m_femdata);
    }
    m_stiffnessalgo->SetMakeAssembly(assemble);
  } else {
    // Hard Fail
    assert(false);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Applies Constraints to Global Stiffness Matrix: 1-Penalty,  2-Substitution
////////////////////////////////////////////////////////////////////////////////
void FEM :: ApplyConstraint(ConstraintMode conmode, int num_supports,
                            int** node_support) {
  SPRmatrix* stiffmat = m_femdata->GetStiffnessMatrix();
  int modeldim = m_femdata->GetModelDim();
  int numdof = m_femdata->GetNumDof();
  fem_float penaltycoef = 10000;
  assert(stiffmat != NULL);
  if (conmode == PEN) {
    // Applies Constraints by Penalty Method
    // Finds largest value of global matrix diagonal
    fem_float maxDiagKcoef = stiffmat->GetElem(0,0);
    for (int i = 0; i < numdof; ++i) {
      fem_float matelem = stiffmat->GetElem(i,i);
      if ( matelem > maxDiagKcoef )
        maxDiagKcoef = matelem;
    }
    // Penalty coefficient is maximum diagonal value * 10^4
    maxDiagKcoef = maxDiagKcoef * penaltycoef;
    printf("-Penalty coefficient: %f\n", maxDiagKcoef);
    for(int i=0; i<num_supports; ++i) {
      int dof=0;
      for(int j=1; j<(1+modeldim); ++j) {
        // If support if fixed
        if (node_support[i][j] == 1) {
          // DOF index in global matrix is ( m_model_dim*(node-1)+(j-1) )
          dof = modeldim*(node_support[i][0]-1)+(j-1);
          // Multiplies DOF of corresponding node by penalty coefficient
          stiffmat->SetElem(dof, dof, (maxDiagKcoef*stiffmat->GetElem(dof,dof)) );
        }
      }
    }
  }
  // Zero Substitution method
  else if (conmode == SUB) {
#pragma omp parallel for
    for(int i=0; i<num_supports; i++ ) {
      int dof=0;
      for(int j=1; j<(1+modeldim); j++) {
        // If support if fixed
        if (node_support[i][j] == 1) {
          // DOF index in global matrix is ( m_model_dim*(node-1)+(j-1) )
          dof = modeldim*(node_support[i][0]-1)+(j-1);
          // Multiplies DOF of corresponding node by penalty coefficient
          for(int k=0; k<numdof; k++){
             stiffmat->SetElem(dof,k,0);
             stiffmat->SetElem(k,dof,0);
          }
          stiffmat->SetElem(dof,dof,1);
        }
      }
    }
  }

}

////////////////////////////////////////////////////////////////////////////////
// Applies Constraints to Global Stiffness Matrix: 1-Penalty,  2-Substitution
////////////////////////////////////////////////////////////////////////////////
void FEM::SetDeviceMode(DeviceMode newdevicemode) {
  if (m_stiffnessalgo != NULL)
    delete m_stiffnessalgo; m_stiffnessalgo = NULL;
  // Selects stiffness calculation algorithm
  switch (newdevicemode) {
  case CPU:
    m_stiffnessalgo = new StiffAlgoCPU(m_femdata); break;
  case GPU:
    m_stiffnessalgo = new StiffAlgoGPU(m_femdata); break;
  case GPUOMP:
    m_stiffnessalgo = new StiffAlgoGpuOmp(m_femdata); break;
  default:
    m_stiffnessalgo = new StiffAlgoCPU(m_femdata); 
  }
}
