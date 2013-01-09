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

#include "FEM/StiffAlgoCPU.h"

#include <omp.h>
#include <math.h>
#include <time.h>
#include <windows.h>
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "FEM/fem.h"
#include "FEM/FemData.h"
#include "LAops/LAops.h"
#include "Utils/util.h"
#include "Utils/fileIO.h"
#include "SPRmatrix/SPRmatrix.h"

////////////////////////////////////////////////////////////////////////////////
// Constructor Sets Basic Information to Make Code Less Error Prone
////////////////////////////////////////////////////////////////////////////////
StiffAlgoCPU::StiffAlgoCPU(FemData* femdata) {
  m_femdata = femdata;
}

/******************************************************************************/
//                            CPU FEM Operations                              //
/******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// Selects Stiffness Calculation Option
////////////////////////////////////////////////////////////////////////////////
double StiffAlgoCPU::CalcGlobalStiffness() {
  if (m_usecoloring)
    return CalcStiffnessColoring();
  else
    return CalcGlobalStiffnessNaive();
}

////////////////////////////////////////////////////////////////////////////////
// Calculates the Global Sparse Stiffness Matrix K
////////////////////////////////////////////////////////////////////////////////
double StiffAlgoCPU::CalcGlobalStiffnessNaive() {
  int modeldim = m_femdata->GetModelDim();
  int numdof = m_femdata->GetNumDof();
  int nelemdof = m_femdata->GetElemDof();
  int numelem = m_femdata->GetNumElem();
  int numelemnodes = m_femdata->GetNumElemNodes();
  int* elemconnect = m_femdata->GetElemConnect();
  fem_float* nodecoords = m_femdata->GetNodeCoords();
  int nloopgpts = m_femdata->GetNumGaussLoopPts();
  fem_float** gausspts_vec = m_femdata->GetGaussPtsVecCPU();
  fem_float* gaussweight_vec = m_femdata->GetGaussWeightVec();
  fem_float** constit_mat = m_femdata->GetConstituitiveMat();
  SPRmatrix* stiffmat = m_femdata->GetStiffnessMatrix();
  SPRmatrix::SPRformat sprseformat = m_femdata->GetSparseFormat();

  int dim2 = (modeldim-1)*3;

  // Checks global stiffness matrix for allocation or clears it
  if (!stiffmat)
    stiffmat = SPRmatrix::CreateMatrix(numdof, sprseformat);
  else
    stiffmat->Clear();

  double start_time = omp_get_wtime();
#pragma omp parallel
  {
    // Allocates memory for variables used in element stiffness calc. loop
    fem_float** k_temp        = allocMatrix(nelemdof, nelemdof, true);
    fem_float   detJ          = NULL;
    fem_float** dShapeMat     = allocMatrix(modeldim, numelemnodes, false);
    fem_float** J             = allocMatrix(modeldim, modeldim, false);
    fem_float** J_inv         = allocMatrix(modeldim, modeldim, false);
    fem_float** dNdCart       = allocMatrix(modeldim, numelemnodes, false);
    fem_float** B             = allocMatrix(dim2, nelemdof, true);
    fem_float** Bt            = allocMatrix(nelemdof, dim2, false);
    fem_float** CxB           = allocMatrix(dim2, nelemdof, false);
    fem_float** k_local       = allocMatrix(nelemdof, nelemdof, true);
    fem_float** m_elem_coords = allocMatrix(modeldim, numelemnodes, false);

#pragma omp for
    for (int elem = 0; elem < numelem; ++elem) {
      // Gets current element coordinates
      for (int j = 0; j < numelemnodes; ++j) {
        for (int i = 0; i < modeldim; ++i) {
          // node in nodeCoords is nodeCoords[node-1][x,y,z]
          int index = (elemconnect[numelemnodes*elem+j]-1)*modeldim+i;
          m_elem_coords[i][j] = nodecoords[index];
        }
      }

      // resets local matrix
      zeroMatrix(k_local, nelemdof, nelemdof);

      for (int gp = 0; gp < nloopgpts; ++gp) {
        // gets natural coordinates form function
        GetdShapeFunctNatCoord(modeldim, numelemnodes, gausspts_vec[gp],
                               dShapeMat);
        // gets Jacobian matrix
        GetJacobianMatrix(modeldim, numelemnodes, J, m_elem_coords, dShapeMat);
        // gets Jacobian determinant
        detJ = det(modeldim, J);
        // gets Jacobian inverse
        matInverse(modeldim, J, detJ, J_inv);
        // gets dNdCart matrix
        matMult(dNdCart, modeldim, modeldim, numelemnodes, J_inv,
          dShapeMat);
        // gets B matrix
        BuildBMatrix(modeldim, numelemnodes, B, dNdCart);
        // gets B transposed matrix
        matTranp(Bt, dim2, nelemdof, B);
        // Calculates elem stiffness matrix k:
        // k = k + B' * C * B * J * t * w(i);
        matMult(CxB, dim2, dim2, nelemdof, constit_mat, B);
        matMult(k_temp, nelemdof, dim2, nelemdof, Bt, CxB);
        matXScalar(nelemdof, nelemdof, k_temp, (detJ*gaussweight_vec[gp]));
        matPMat(nelemdof, nelemdof, k_local, k_temp);
      }
      // fileIO->writeMatrix3(k_local, m_elem_dofs, m_elem_dofs);

      // Assembles global stiffness matrix
      if (m_assemble) {
#pragma omp critical
        {
          AssembleK(modeldim,
                    numelemnodes,
                    elemconnect,
                    elem,
                    stiffmat,
                    k_local);
        }
      }
    }  // End of Elements loop

    // Cleanup
    freeInnerVectorsF(k_temp, nelemdof);
    freeInnerVectorsF(dShapeMat, modeldim);
    freeInnerVectorsF(J, modeldim);
    freeInnerVectorsF(J_inv, modeldim);
    freeInnerVectorsF(dNdCart, modeldim);
    freeInnerVectorsF(B, dim2);
    freeInnerVectorsF(Bt, nelemdof);
    freeInnerVectorsF(CxB, dim2);
    freeInnerVectorsF(k_local, nelemdof);
    freeInnerVectorsF(m_elem_coords, modeldim);
  }  // end of parallel region

  double exec_time = omp_get_wtime();
  printf("------------------------------------------------\n");
  printf("+Total Execution Time:%.3fms\n", exec_time-start_time);

  return (exec_time-start_time);
}


////////////////////////////////////////////////////////////////////////////////
// Calculates the Global Sparse Stiffness Matrix K
////////////////////////////////////////////////////////////////////////////////
double StiffAlgoCPU::CalcStiffnessColoring() {
  int modeldim                     = m_femdata->GetModelDim();
  int numdof                       = m_femdata->GetNumDof();
  int nelemdof                     = m_femdata->GetElemDof();
  int nelemnodes                   = m_femdata->GetNumElemNodes();
  int* elemconnect                 = m_femdata->GetElemConnect();
  fem_float* nodecoords            = m_femdata->GetNodeCoords();
  int nloopgpts                    = m_femdata->GetNumGaussLoopPts();
  fem_float** gausspts_vec         = m_femdata->GetGaussPtsVecCPU();
  fem_float* gaussweight_vec       = m_femdata->GetGaussWeightVec();
  fem_float** constmat             = m_femdata->GetConstituitiveMat();
  SPRmatrix* stiffmat              = m_femdata->GetStiffnessMatrix();
  SPRmatrix::SPRformat sprseformat = m_femdata->GetSparseFormat();
  const ivecvec colorelem          = m_femdata->GetColorVector();

  int dim2 = (modeldim-1)*3;

  // Checks global stiffness matrix for allocation or clears it
  if (!stiffmat)
    stiffmat = SPRmatrix::CreateMatrix(numdof, sprseformat);
  else
    stiffmat->Clear();

  double start_time = omp_get_wtime();

  // Parallel section starts her so that each processor has his own copy of
  // variables used in calculation loop
#pragma omp parallel
  {
    // Allocates memory for variables used in element stiffness calc. loop
    fem_float** k_temp        = allocMatrix(nelemdof, nelemdof, true);
    fem_float   detJ          = NULL;
    fem_float** dShapeMat     = allocMatrix(modeldim, nelemnodes, false);
    fem_float** J             = allocMatrix(modeldim, modeldim, false);
    fem_float** J_inv         = allocMatrix(modeldim, modeldim, false);
    fem_float** dNdCart       = allocMatrix(modeldim, nelemnodes, false);
    fem_float** B             = allocMatrix(dim2, nelemdof,  true);
    fem_float** Bt            = allocMatrix(nelemdof, dim2, false);
    fem_float** CxB           = allocMatrix(dim2, nelemdof, false);
    fem_float** k_local       = allocMatrix(nelemdof, nelemdof, true);
    fem_float** m_elem_coords = allocMatrix(modeldim, nelemnodes, false);

    // Outer loop ensures colors are assembled serially, within color the
    // assembly, or rather the whole stiffness calculation in this case,
    // can be done in parallel without race conditions
    size_t nColors = colorelem.size();
    for (size_t color = 0; color < nColors; ++color) {
      int nColorElems = (int)colorelem[color].size();
      // Loops over current color doing stiffness calculation in parallel
#pragma omp for
      for (int elemPos = 0; elemPos < nColorElems; ++elemPos) {
        int elem = colorelem[color][elemPos];
        // Gets current element coordinates
        for (int j = 0; j < nelemnodes; ++j) {
          for (int i = 0; i < modeldim; ++i) {
            // node in nodeCoords is nodeCoords[node-1][x,y,z]
            int index = (elemconnect[nelemnodes*elem+j]-1)*modeldim+i;
            m_elem_coords[i][j] = nodecoords[index];
          }
        }

        // resets local matrix
        zeroMatrix(k_local, nelemdof, nelemdof);

        for (int gp = 0; gp < nloopgpts; ++gp) {
          // gets natural coordinates form function
          GetdShapeFunctNatCoord(modeldim, nelemnodes, gausspts_vec[gp], dShapeMat);
          // gets Jacobian matrix
          GetJacobianMatrix(modeldim, nelemnodes, J, m_elem_coords, dShapeMat);
          // gets Jacobian determinant
          detJ = det(modeldim, J);
          // gets Jacobian inverse
          matInverse(modeldim, J, detJ, J_inv);
          // gets dNdCart matrix
          matMult(dNdCart, modeldim, modeldim, nelemnodes, J_inv,
            dShapeMat);
          // gets B matrix
          BuildBMatrix(modeldim, nelemnodes, B, dNdCart);
          // gets B transposed matrix
          matTranp(Bt, dim2, nelemdof, B);
          // Calculates elem stiffness matrix k:
          // k = k + B' * C * B * J * t * w(i);
          matMult(CxB, dim2, dim2, nelemdof, constmat, B);
          matMult(k_temp, nelemdof, dim2, nelemdof, Bt, CxB);
          matXScalar(nelemdof, nelemdof, k_temp, (detJ*gaussweight_vec[gp]));
          matPMat(nelemdof, nelemdof, k_local, k_temp);
        }
        // fileIO->writeMatrix3(k_local, m_elem_dofs, m_elem_dofs);

        // Assembles global stiffness matrix
        if (m_assemble) {
          AssembleK(modeldim,
                    nelemnodes,
                    elemconnect,
                    elem,
                    stiffmat,
                    k_local);
        }  // End of Assembly loop
      }  // End of Elements loop
    }

    //Cleanup
    freeInnerVectorsF(k_temp, nelemdof);
    freeInnerVectorsF(dShapeMat, modeldim);
    freeInnerVectorsF(J, modeldim);
    freeInnerVectorsF(J_inv, modeldim);
    freeInnerVectorsF(dNdCart, modeldim);
    freeInnerVectorsF(B, dim2);
    freeInnerVectorsF(Bt, nelemdof);
    freeInnerVectorsF(CxB, dim2);
    freeInnerVectorsF(k_local, nelemdof);
    freeInnerVectorsF(m_elem_coords, modeldim);
  }  // end of parallel region

  double exec_time = omp_get_wtime();
  printf("------------------------------------------------\n");
  printf("+Total Execution Time:%.3fms\n", exec_time-start_time);

  return (exec_time-start_time);
}


////////////////////////////////////////////////////////////////////////////////
// Calculates the Global Sparse Stiffness Matrix K
////////////////////////////////////////////////////////////////////////////////
void StiffAlgoCPU::AssembleK(int modeldim,
                             int numelemnodes,
                             int* elemconnect,
                             int elem,
                             SPRmatrix* stiffmat,
                             fem_float** k_local) {
  if (modeldim == 2) {
    for (int i = 0; i < numelemnodes; ++i) {
      int gblDOFi = (elemconnect[numelemnodes*elem+i]-1)*modeldim;
      for (int j = 0; j < numelemnodes; ++j) {
        int gblDOFj = (elemconnect[numelemnodes*elem+j]-1)*modeldim;
        stiffmat->AddElem(gblDOFi  , gblDOFj  , k_local[(modeldim*i)  ][(modeldim*j)  ]);
        stiffmat->AddElem(gblDOFi+1, gblDOFj  , k_local[(modeldim*i)+1][(modeldim*j)  ]);
        stiffmat->AddElem(gblDOFi  , gblDOFj+1, k_local[(modeldim*i)  ][(modeldim*j)+1]);
        stiffmat->AddElem(gblDOFi+1, gblDOFj+1, k_local[(modeldim*i)+1][(modeldim*j)+1]);
      }
    }
  } else if ( modeldim == 3 ) {
    for (int i = 0; i < numelemnodes; i++) {
      int gblDOFi = (elemconnect[numelemnodes*elem+i]-1)*modeldim;
      for (int j = 0; j < numelemnodes; ++j) {
        int gblDOFj = (elemconnect[numelemnodes*elem+j]-1)*modeldim;
        stiffmat->AddElem(gblDOFi  , gblDOFj  , k_local[(modeldim*i)  ][(modeldim*j)  ]);
        stiffmat->AddElem(gblDOFi+1, gblDOFj  , k_local[(modeldim*i)+1][(modeldim*j)  ]);
        stiffmat->AddElem(gblDOFi+2, gblDOFj  , k_local[(modeldim*i)+2][(modeldim*j)  ]);
        stiffmat->AddElem(gblDOFi  , gblDOFj+1, k_local[(modeldim*i)  ][(modeldim*j)+1]);
        stiffmat->AddElem(gblDOFi+1, gblDOFj+1, k_local[(modeldim*i)+1][(modeldim*j)+1]);
        stiffmat->AddElem(gblDOFi+2, gblDOFj+1, k_local[(modeldim*i)+2][(modeldim*j)+1]);
        stiffmat->AddElem(gblDOFi  , gblDOFj+2, k_local[(modeldim*i)  ][(modeldim*j)+2]);
        stiffmat->AddElem(gblDOFi+1, gblDOFj+2, k_local[(modeldim*i)+1][(modeldim*j)+2]);
        stiffmat->AddElem(gblDOFi+2, gblDOFj+2, k_local[(modeldim*i)+2][(modeldim*j)+2]);
      }
    }
  }
}
