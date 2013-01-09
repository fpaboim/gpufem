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

/////////////////////////////////////////////////////////////////////
// Finite Element Coloring Header
/////////////////////////////////////////////////////////////////////
#ifndef FEMCOLOR_H_
#define FEMCOLOR_H_

#include <iostream>
#include <vector>
#include <list>

#include "metis.h"

#include "FEM/FemData.h"

class femColor {
 public:
  femColor();
  ~femColor();

  void makeMetisGraph(FemData* femdata, bool makenodal);
  int  MakeGreedyColoring(FemData* femdata);
  void CalcNNZ(FemData* femdata, int &nnz, int &band);
  int  GetNumColors() {return m_nColors;};

private:
  int* neighbor_elements_q4_mesh(int nelem, int* elemconn);
  int  adj_size_q4_mesh(int nnodes, int nelem, int* elemconn, int* elemNeighbor,
                        int* adjrow);
  int  getAdjacentDOFs(int elem1, int elem2,  FemData* femdata);
  void printCSRVector(int* index_vec, int n_indices, int* values_vec);
  void printNodeConnect();
  void printElemConnect();
  //-------------------------------------------
  // member variables
  //-------------------------------------------
  idx_t *m_ElemAdjIndex, *m_ElemAdjVector;// METIS CSR adjacency vector pointers
                                          // allocated by METIS needs cleanup(!)
  idx_t *m_NodeAdjIndex, *m_NodeAdjVector;// METIS CSR adjacency vector pointers
                                          // allocated by METIS needs cleanup(!)
  std::vector<std::vector<int>> m_colorElem; // Index is color, value is vector
                                             // with elements belonging to that
                                             // color
  int   m_nColors;   // Number of colors used
  int*  m_elemColor; // Index is element - value is color (holds color of elem)
  int   m_nNodes;    // Number of nodes
  int   m_nElem;     // Number of elements
};

#endif  // FEMCOLOR_H_
