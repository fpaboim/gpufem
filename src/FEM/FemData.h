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
// femData.h - Finite Element Data Encapsulator
// Author: Francisco Paulo de Aboim (fpaboim@gmail.com)
////////////////////////////////////////////////////////////////////////////////

#ifndef FEM_DATA_H_
#define FEM_DATA_H_

#include <vector>

#include "SPRmatrix/SPRmatrix.h"
#include "Utils/fileIO.h"
#include "Utils/util.h"

class StiffAlgo;

class FemData {
public:
  FemData();
  ~FemData();

  // Initializes all data to prevent errors of unset member variables, variables
  // who can be changed for analytical purposes have their own set methods
  void Init(SPRmatrix::SPRformat sprse_format,
            int n_gpts,
            fem_float E,
            fem_float Nu,
            FileIO* fileIO);

  // Builds gauss points position matrix and weight vector
  void SetGaussPoints(int n_gpts);
  // Changes elastic modulus and recalculates constitutive matrix
  void SetElasticModulus(fem_float E_modulus) {
    m_E_mod = E_modulus;
    CalcConstituitiveData();
  };
  // Changes Poisson coefficient and recalculates constitutive matrix
  void SetPoissonCoefficient(fem_float Nu_coefficient) {
    m_Nu_coef = Nu_coefficient;
    CalcConstituitiveData();
  };
  // Changes analysis type and recalculates constitutive matrix
  void SetAnalysisType(const char* analysistype) {
    m_analysis = analysistype;
    CalcConstituitiveData();
  };
  // sets sparse matrix format
  void SetMatrixFormat(SPRmatrix::SPRformat sprse_format) {
    m_sparse_format = sprse_format;
  };
  // sets sparse matrix format
  void SetColorVector(std::vector<std::vector<int>> colorElem) {
    m_colorElem = colorElem;
  };

  int         GetModelDim() {return m_model_dim;};
  int         GetNumGaussPts() {return m_num_gpts;};
  fem_float** GetGaussPtsVecCPU() {return m_x_gausspts_cpu;};
  fem_float*  GetGaussPtsVecGPU() {return m_x_gausspts_gpu;};
  fem_float*  GetGaussWeightVec() {return m_w_gaussweights_vec;};
  int         GetNumGaussLoopPts() {return m_num_loop_gpts;};
  const char* GetAnalysisType() {return m_analysis;};
  fem_float   GetElasticModulus() {return m_E_mod;};
  fem_float   GetPoissonCoef() {return m_Nu_coef;};
  fem_float** GetConstituitiveMat() {return m_const_mat;};
  int         GetNumElemNodes() {return m_elem_nnodes;};
  int         GetNumElem() {return m_elem_num;};
  int*        GetElemConnect() {return m_elem_connect;};
  int         GetElemDof() {return m_elem_dofs;};
  fem_float** GetElemCoord() {return m_elem_coords;};
  int         GetNumNodes() {return m_num_nodes;};
  fem_float*  GetNodeCoords() {return m_node_coords;};
  int         GetNodeNumConstraints() {return m_node_nconstr;};
  int**       GetNodeConstraints() {return m_node_constr;};
  int         GetNumLoads() {return m_num_loads;};
  int         GetNumDof() {return m_num_dof;};
  SPRmatrix*  GetStiffnessMatrix() {return m_k_global;};
  SPRmatrix::SPRformat GetSparseFormat() {return m_sparse_format;};
  fem_float*  GetForceVector() {return m_force_vec;};
  fem_float*  GetDisplVector() {return m_displace_vec;};
  std::vector<std::vector<int>>
              GetColorVector() {return m_colorElem;};

protected:
  // Builds Gauss Vectors
  void CalcGaussVectors();
  // Builds constitutive matrix
  void CalcConstituitiveData();
  // Builds Force Vector
  void CalcForceVector(fem_float** nodal_loads_vec);

protected:
  int         m_model_dim; // Model dimensions in use
  int         m_num_nodes; // Total number of nodes in model
  fem_float*  m_node_coords; // Node coordinates vector
  int         m_node_nconstr; // Node coordinates vector
  int**       m_node_constr; // Node coordinates vector
  int         m_num_dof; // Total number of nodes in model

  fem_float** m_const_mat; // Constitutive Matrix
  fem_float   m_E_mod; // Elastic Modulus (Young) of Material
  fem_float   m_Nu_coef;  //Poisson's Coefficient of Material
  const char* m_analysis;  //Analysis Type

  int         m_elem_nnodes; // Number of Nodes in Element
  int         m_elem_dofs; // Number of Degrees of Freedom of Element
  int         m_elem_num; // Total Number of Elements in Model
  int*        m_elem_connect; // Element conn. in vec. format with elements
                              // sequentially one after the other (in order)
  fem_float** m_elem_coords; // Element Coordinate Matrix

  fem_float** m_x_gausspts_cpu; // "x" coordinate of gauss points (matrix)
  fem_float*  m_x_gausspts_gpu; // "x" coordinate of gauss points (vector)
  fem_float*  m_w_gaussweights_vec; // Weight associated with gauss point
  int         m_num_gpts; // total number of gauss points in element
  int         m_num_loop_gpts; // total number of gauss points in element

  SPRmatrix::SPRformat m_sparse_format;
  SPRmatrix*           m_k_global;

  int         m_num_loads;
  fem_float*  m_force_vec;
  fem_float*  m_displace_vec;

  std::vector<std::vector<int>> m_colorElem; // Vector for parallel assembly by
                                             // coloring

  bool m_initialized;
};

#endif  // FEM_DATA_H_
