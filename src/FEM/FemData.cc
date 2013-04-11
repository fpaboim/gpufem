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

#include "FEM/FemData.h"

#include <cstdio>
#include <cstdlib>
#include <math.h>

#include "LAops/LAops.h"
#include "Utils/util.h"
#include "Utils/fileIO.h"
#include "SPRmatrix/SPRmatrix.h"

/******************************************************************************/
//                            CPU FEM Operations                              //
/******************************************************************************/

// Constructor and destructor
FemData::FemData() {
  m_model_dim   = NULL;
  m_num_nodes   = NULL;
  m_node_coords = NULL;
  m_num_dof     = NULL;

  m_const_mat = NULL;
  m_E_mod     = NULL;
  m_Nu_coef   = NULL;

  m_elem_nnodes  = NULL;
  m_elem_dofs    = NULL;
  m_elem_num     = NULL;
  m_elem_connect = NULL;
  m_elem_coords  = NULL;

  m_x_gausspts_cpu     = NULL;
  m_x_gausspts_gpu     = NULL;
  m_w_gaussweights_vec = NULL;
  m_num_gpts           = NULL;
  m_num_loop_gpts      = NULL;

  m_k_global = NULL;

  m_num_loads    = NULL;
  m_force_vec    = NULL;
  m_displace_vec = NULL;

  m_initialized = false;
}

FemData :: ~FemData() {
  if (m_k_global)
    delete m_k_global; m_k_global = NULL;
  if (m_elem_coords) {
    freeInnerVectorsF(m_elem_coords, m_model_dim);
    m_elem_coords = NULL;
  }
  // Gauss
  if (m_x_gausspts_cpu) {
    freeInnerVectorsF(m_x_gausspts_cpu, m_num_loop_gpts);
    m_x_gausspts_cpu = NULL;
  }
  if (m_x_gausspts_gpu) {
    free(m_x_gausspts_gpu);
    m_x_gausspts_gpu = NULL;
  }
  if (m_w_gaussweights_vec) {
    free(m_w_gaussweights_vec);
    m_w_gaussweights_vec = NULL;
  }
  if (m_force_vec) {
    free(m_force_vec);
  }
  // Const Mat
  if (m_const_mat) {
    freeInnerVectorsF(m_const_mat, (m_model_dim - 1) * 3);
    m_const_mat = NULL;
  }
  // Displ vec
  if (m_displace_vec) {
    free(m_displace_vec);
    m_displace_vec = NULL;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Makes New Finite Element Analysis Object
////////////////////////////////////////////////////////////////////////////////
void FemData::Init(SPRmatrix::SPRformat sprse_format,
                   int n_gpts,
                   fem_float E,
                   fem_float Nu,
                   FileIO* fileIO) {
  if (m_initialized) {
    return;
  }
  // Gets data from file operations
  const int   modeldim     = fileIO->getModelDim();
  const char* analysis     = fileIO->getAnalysis();
  const int   nnodes       = fileIO->getNumNodes();
  const int   ndof         = modeldim*fileIO->getNumNodes();
  const int   nelem        = fileIO->getNumElements();
  const int   nsupports    = fileIO->getNumSupports();
  const int   nnodalloads  = fileIO->getNumNodalLoads();
  fem_float*  nodecoords   = fileIO->getNodeCoords();
  int         nelemnodes   = fileIO->getNumElemNodes();
  int*        elemconnect  = fileIO->getElemConnect();
  int**       nodesupports = fileIO->getNodeSupports();
  fem_float** nodalloads   = fileIO->getNodalLoads();

  std::cout << std::endl;
  std::cout << "** STATS **" << std::endl;
  std::cout << "Elements: "<< nelem << std::endl;
//  std::cout << "Nodes: "<< nnodes << std::endl;
  std::cout << "DOF: "<< ndof << std::endl;
//  std::cout << "Nodal Loads: "<< nnodalloads << std::endl;
//  std::cout << "Element Nodes: "<< nelemnodes << std::endl;
  std::cout << "Gauss Points: "<< n_gpts << std::endl;
  std::cout << "Sprseformat: "<< sprse_format << std::endl;

  // Initializes member variables
  m_sparse_format = sprse_format;
  m_model_dim     = modeldim;
  m_num_gpts      = n_gpts;
  m_analysis      = analysis;
  m_E_mod         = E;
  m_Nu_coef       = Nu;
  m_elem_nnodes   = nelemnodes;
  m_elem_num      = nelem;
  m_elem_connect  = elemconnect;
  m_num_nodes     = nnodes;
  m_node_coords   = nodecoords;
  m_num_loads     = nnodalloads;
  m_elem_dofs     = m_model_dim * nelemnodes;
  m_num_dof       = m_model_dim * nnodes;
  m_elem_coords   = allocMatrix(m_model_dim, nelemnodes, false);

  if (m_k_global) {
    delete(m_k_global);
  }
  m_k_global = SPRmatrix::CreateMatrix(m_num_dof, m_sparse_format);

  if (m_displace_vec) {
    free(m_displace_vec);
    m_displace_vec = NULL;
  }
  m_displace_vec = allocVector(m_num_dof, true);

  SetGaussPoints(n_gpts);
  CalcConstituitiveData();

  if (m_num_loads > 0) {
    CalcForceVector(nodalloads);
  } else {
    printf("\n**ERROR: UNABLE TO FIND LOADS IN NEUTRAL FILE**\n");
  }

  m_initialized = true;
}

////////////////////////////////////////////////////////////////////////////////
// Sets number of gauss points used for integrating Gaussian quadrature
////////////////////////////////////////////////////////////////////////////////
void FemData::SetGaussPoints(int n_gpts) {
  // Checks for previous allocation
  if (m_x_gausspts_cpu)
    freeInnerVectorsF(m_x_gausspts_cpu, m_num_loop_gpts);
  if (m_x_gausspts_gpu)
    free(m_x_gausspts_gpu);
  if (m_w_gaussweights_vec)
    free(m_w_gaussweights_vec);
  // Sets new gauss points and loop points
  m_num_gpts = n_gpts;
  if (m_model_dim == 2)
    m_num_loop_gpts = n_gpts*n_gpts;
  else
    m_num_loop_gpts = n_gpts*n_gpts*n_gpts;
  // First index corresponds to distinct combinations of gauss points where the
  // increment order is from last coordinate to first (n^modelDim combinations)
  m_x_gausspts_cpu = allocMatrix(m_num_loop_gpts, m_model_dim, false);
  m_x_gausspts_gpu = allocVector(m_num_loop_gpts * m_model_dim, false);
  m_w_gaussweights_vec = allocVector(m_num_loop_gpts, false);

  // Builds the gauss point vectors
  CalcGaussVectors();
}

////////////////////////////////////////////////////////////////////////////////
// Gets Gaussian quadrature integration points
////////////////////////////////////////////////////////////////////////////////
void FemData::CalcGaussVectors() {
  fem_float  x[3];
  fem_float  w[3];

  if (m_num_gpts != 2 && m_num_gpts != 3)
    return;

  // For 2 gauss points:
  if (m_num_gpts == 2) {
    x[0] = -1/sqrt((fem_float)3);
    x[1] =  -x[0];
    w[0] = 1;
    w[1] = 1;
  } else if (m_num_gpts == 3) {  // For 3 gauss points
    x[0] = -sqrt((fem_float)3/5);
    x[1] = 0;
    x[2] =  -x[0];
    w[0] = (fem_float)5/9;
    w[1] = (fem_float)8/9;
    w[2] = (fem_float)w[0];
  }

  // 3D Model
  if (m_model_dim == 3) {
    // For 2 gauss points:
    if (m_num_gpts == 2) {
      int s = 0;
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
          for (int k = 0; k < 2; ++k) {
            m_x_gausspts_cpu[s][0] = x[i];
            m_x_gausspts_cpu[s][1] = x[j];
            m_x_gausspts_cpu[s][2] = x[k];

            m_x_gausspts_gpu[3*s+0] = x[i];
            m_x_gausspts_gpu[3*s+1] = x[j];
            m_x_gausspts_gpu[3*s+2] = x[k];

            // W[s] = w[i]*w[j]*w[k] = 1;
            m_w_gaussweights_vec[s] = 1;
            s++;
          }
    } else if (m_num_gpts == 3) {  // For 3 gauss points
      int s = 0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          for (int k = 0; k < 3; ++k) {
            m_x_gausspts_cpu[s][0] = x[i];
            m_x_gausspts_cpu[s][1] = x[j];
            m_x_gausspts_cpu[s][2] = x[k];

            m_x_gausspts_gpu[3*s+0] = x[i];
            m_x_gausspts_gpu[3*s+1] = x[j];
            m_x_gausspts_gpu[3*s+2] = x[k];

            m_w_gaussweights_vec[s] = w[i] * w[j] * w[k];
            s++;
          }
    }
  } else {  // 2D Model
    // For 2 gauss points:
    if (m_num_gpts == 2) {
      int s = 0;
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            m_x_gausspts_cpu[s][0] = x[i];
            m_x_gausspts_cpu[s][1] = x[j];

            m_x_gausspts_gpu[2*s+0] = x[i];
            m_x_gausspts_gpu[2*s+1] = x[j];

            // W[s] = w[i]*w[j]*w[k] = 1;
            m_w_gaussweights_vec[s] = 1;
            s++;
          }
    } else if (m_num_gpts == 3) {  // For 3 gauss points
      int s = 0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            m_x_gausspts_cpu[s][0] = x[i];
            m_x_gausspts_cpu[s][1] = x[j];

            m_x_gausspts_gpu[2*s+0] = x[i];
            m_x_gausspts_gpu[2*s+1] = x[j];

            m_w_gaussweights_vec[s] = w[i] * w[j];
            s++;
          }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
// Makes and stores Constitutive Matrix
////////////////////////////////////////////////////////////////////////////////
void FemData::CalcConstituitiveData() {
  // Allocates matrix
  m_const_mat = allocMatrix((m_model_dim - 1) * 3, (m_model_dim - 1) * 3, true);

  // 2D Model case
  if (m_model_dim == 2) {
    // First sets parameters according to analysis type
    fem_float c1, c2, c3, auxcoef;
    if (strcmp(m_analysis, "plane_stress")) {
      auxcoef = m_E_mod / (1 - (m_Nu_coef * m_Nu_coef));
      c1 = auxcoef;
      c2 = auxcoef * ((1 - m_Nu_coef) / 2);
      c3 = auxcoef * m_Nu_coef;
    } else if (strcmp(m_analysis, "plane_strain")) {
      auxcoef = m_E_mod / ((1 - 2 * m_Nu_coef) * (1 + m_Nu_coef));
      c1 = auxcoef * (1 - m_Nu_coef);
      c2 = auxcoef * ((1 - 2 * m_Nu_coef) / 2);
      c3 = auxcoef * m_Nu_coef;
    }
    m_const_mat[0][0] = c1;
    m_const_mat[1][1] = c1;
    m_const_mat[2][2] = c2;
    m_const_mat[0][1] = c3;
    m_const_mat[1][0] = c3;
  } else {  // 3D Model case
    fem_float c1 = m_E_mod * (1.0f - m_Nu_coef) /
      ((1.0f + m_Nu_coef) * (1.0f - 2.0f * m_Nu_coef));
    fem_float c2 = m_Nu_coef / (1.0f-m_Nu_coef);
    fem_float c3 = (1.0f-2.0f*m_Nu_coef) / (2.0f*(1.0f-m_Nu_coef));

    m_const_mat[0][0] = c1;
    m_const_mat[0][1] = c1 * c2;
    m_const_mat[0][2] = c1 * c2;
    m_const_mat[1][0] = c1 * c2;
    m_const_mat[1][1] = c1;
    m_const_mat[1][2] = c1 * c2;
    m_const_mat[2][0] = c1 * c2;
    m_const_mat[2][1] = c1 * c2;
    m_const_mat[2][2] = c1;
    m_const_mat[3][3] = c1 * c3;
    m_const_mat[4][4] = c1 * c3;
    m_const_mat[5][5] = c1 * c3;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Builds the force vector
////////////////////////////////////////////////////////////////////////////////
void FemData::CalcForceVector(fem_float** nodal_loads_vec) {
  if (m_force_vec) free(m_force_vec);
  m_force_vec = allocVector(m_num_nodes * m_model_dim, true);
  // Adds Forces to Force vector
  for (int i = 0; i < m_num_loads; ++i) {
    int dof = 0;
    for (int j = 1; j < 4; ++j) {
      // If load for DOF exists
      if (nodal_loads_vec[i][j] != 0) {
        // DOF index in global matrix is ( m_model_dim*(node-1)+(j-1) )
        dof = m_model_dim*((int)nodal_loads_vec[i][0] - 1)+(j - 1);
        // Multiplies DOF of corresponding Node by Coef
        m_force_vec[dof] = nodal_loads_vec[i][j];
      }
    }
  }
}
