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
// Finite Element Graph Coloring for Parallel Assembly
// Author: Francisco Aboim
// TecGraf / PUC-RIO
////////////////////////////////////////////////////////////////////////////////

// Headers
#include "assert.h"
#include <vector>
#include <list>

#include "FEM/femColor.h"
#include "utils/util.h"
#include "metis.h"

// number of colors available for greedy coloring
#define MAXNEIGHBORS 64

//-----------------------------------------------------------------------
femColor::femColor() {
  m_ElemAdjIndex  = NULL;
  m_ElemAdjVector = NULL;
  m_NodeAdjIndex  = NULL;
  m_NodeAdjVector = NULL;
  m_elemColor     = NULL;
  m_nNodes        = 0;
  m_nElem         = 0;
  m_nColors       = 0;
}

//-----------------------------------------------------------------------
femColor::~femColor() {
  if (m_ElemAdjVector) {
    METIS_Free(m_ElemAdjVector);
    m_ElemAdjVector = NULL;
  }
  if (m_ElemAdjIndex) {
    METIS_Free(m_ElemAdjIndex);
    m_ElemAdjIndex = NULL;
  }
  if (m_NodeAdjVector) {
    METIS_Free(m_NodeAdjVector);
    m_NodeAdjVector = NULL;
  }
  if (m_NodeAdjIndex) {
    METIS_Free(m_NodeAdjIndex);
    m_NodeAdjIndex = NULL;
  }
}
//-----------------------------------------------------------------------


////////////////////////////////////////////////////////////////////////////////
// Builds Metis Adjacency CSR Vector
////////////////////////////////////////////////////////////////////////////////
void femColor::makeMetisGraph(FemData* femdata, bool makenodal) {
  int isok;
  m_nElem  = femdata->GetNumElem();
  m_nNodes = femdata->GetNumNodes();
  int  nelemnodes  = femdata->GetNumElemNodes();
  int* elemconnect = femdata->GetElemConnect();

  // makes and index pointer for metis to know where to find the connectivity of
  // the i-th element
  int* eptr = (int*)malloc((m_nElem + 1) * sizeof(int));
  for (int i = 0; i < m_nElem + 1; ++i) {
    eptr[i] = i * nelemnodes;
  }

  // checks allocation to prevent memory leaks (both allocated by metis)
  if (m_ElemAdjVector) {
    METIS_Free(m_ElemAdjVector);
    m_ElemAdjVector = NULL;
  }
  if (m_ElemAdjIndex) {
    METIS_Free(m_ElemAdjIndex);
    m_ElemAdjIndex = NULL;
  }

  int ncommon = 1; // Metis Options
  int cstyle = 0;
  isok = METIS_MeshToDual(&m_nElem, &m_nNodes, eptr, elemconnect,
                          &ncommon, &cstyle, &m_ElemAdjIndex, &m_ElemAdjVector);
  assert(isok == METIS_OK);

  if (makenodal) {
    // checks allocation to prevent memory leaks (both allocated by metis)
    if (m_NodeAdjVector) {
      METIS_Free(m_NodeAdjVector);
      m_NodeAdjVector = NULL;
    }
    if (m_NodeAdjIndex) {
      METIS_Free(m_NodeAdjIndex);
      m_NodeAdjIndex = NULL;
    }
    isok = METIS_MeshToNodal(&m_nElem, &m_nNodes, eptr, elemconnect,
                             &cstyle, &m_NodeAdjIndex, &m_NodeAdjVector);
    assert(isok == METIS_OK);
  }

  //printCSRVector(m_NodeAdjIndex, m_nNodes, m_NodeAdjVector);

  free(eptr);
}

////////////////////////////////////////////////////////////////////////////////
// Makes Serial Graph Coloring
////////////////////////////////////////////////////////////////////////////////
int femColor::MakeGreedyColoring(FemData* femdata) {
  if (m_ElemAdjVector == NULL) {
    return 0;
  }
  // checks for allocation to prevent memory leaks by losing pointer
  if (m_elemColor) {
    free(m_elemColor);
  }
  // uses calloc to initialize vector contents as 0
  m_elemColor = (int*) calloc(m_nElem, sizeof(int));
  // creates auxiliary vector to use for bookeeping on neighbors colors
  int* neighborColors = (int*) malloc(MAXNEIGHBORS*sizeof(int));

  // does greedy coloring
  for (int elem = 0; elem < m_nElem; ++elem) {
    int currIndex  = m_ElemAdjIndex[elem];
    int nNeighbors = m_ElemAdjIndex[elem+1] - m_ElemAdjIndex[elem];
    for (int j = 0; j < nNeighbors; ++j) {
      neighborColors[j] = m_elemColor[m_ElemAdjVector[currIndex+j]];
    } // gets colors of neighbors in static vector

    int color = 1;
    bool found = false;
    while (!found) {
      // iterates over neighbors and assigns next available color starting w/ 1
      for (int j = 0; j < nNeighbors; ++j) {
        if (neighborColors[j] == color) {
          color++;
          j = -1;
          continue;
        }
      }
      found = true;
      m_elemColor[elem] = color;
      if (color > m_nColors) {
        m_nColors++;
      }
    }
  }

  // builds color -> elem vector
  m_colorElem = femdata->GetColorVector();
  if (m_colorElem.size()!= 0) {
    m_colorElem.clear();
  }
  m_colorElem.resize(m_nColors);
  for (int elem = 0; elem < m_nElem; ++elem) {
    m_colorElem[m_elemColor[elem]-1].push_back(elem);
  }
  femdata->SetColorVector(m_colorElem);

//   printVectori(m_elemColor, m_nElem);
//   printMatrixSTL(m_colorElem);

  //cleanup
  free(m_elemColor); m_elemColor = NULL;
  free(neighborColors);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////
// Get NNZ by Nodal Adjacency, returns nnz and maximum band(width) of matrix
////////////////////////////////////////////////////////////////////////////////
void femColor::CalcNNZ(FemData* femdata, int &nnz, int &band) {
  if (m_NodeAdjVector == NULL || m_NodeAdjIndex == NULL)
    return;

  int adjdof = 0;
  int numnodes = femdata->GetNumNodes();
  int modeldim = femdata->GetModelDim();
  int maxadjecency = 0;

  for (int node = 1; node < m_nNodes; ++node) {
    int adjdoftemp = (m_NodeAdjIndex[node + 1] - m_NodeAdjIndex[node]);
    // Fixes metis not allocating self node in adjacency list
    for (int i = 0; i < adjdoftemp; ++i) {
      if (m_NodeAdjVector[m_NodeAdjIndex[node] + i] == node) {
        adjdoftemp--;
      }
    }
    if (adjdoftemp > maxadjecency) {
      maxadjecency = adjdoftemp;
    }
    adjdof += adjdoftemp;
  }
  adjdof += numnodes;

  // adjacent dof have to be divided by or they will be counted twice
  nnz = adjdof * modeldim * 2;
  band = (maxadjecency + 1) * 2;
}

////////////////////////////////////////////////////////////////////////////////
// Prints Metis Format CSR Vector
////////////////////////////////////////////////////////////////////////////////
void femColor::printCSRVector(int* index_vec, int n_indices, int* values_vec) {
  printf("\n++++++++ Printing CSR Vector... ++++++++\n");
  for (int i = 0; i < (n_indices); ++i) {
    printf("index:%i\n",(i));
    for (int j = index_vec[i]; j < index_vec[i+1]; ++j) {
      printf("    value:%i\n", (values_vec[j]));
    }
  }
}
