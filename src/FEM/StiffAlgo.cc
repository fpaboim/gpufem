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

#include "FEM/StiffAlgo.h"
#include "LAops/LAops.h"

#include <omp.h>

/******************************************************************************/
//                            CPU FEM Operations                              //
/******************************************************************************/

// Constructor and destructor
StiffAlgo::StiffAlgo() {
  m_assemble    = true;
  m_usecoloring = false;
}

StiffAlgo :: ~StiffAlgo() {
}

////////////////////////////////////////////////////////////////////////////////
// GetdShapeFunctNatCoord
// Gets the Derivative of the Shape Matrix in relation to natural coordinates
// Formulas obtained by deriving in Matlab and using simplify()
////////////////////////////////////////////////////////////////////////////////
void
StiffAlgo::GetdShapeFunctNatCoord(int modeldim, int nnodes, fem_float* rst,
                                  fem_float** dShapeMatrix ) {
  // 3D
  if (modeldim == 2) {
    if (nnodes == 4) {
      fem_float r = rst[0];
      fem_float s = rst[1];

      //// dShapeMatrix[0][i] = d/dr[Ni]
      dShapeMatrix[0][0] =  0.25f * (s-1);
      dShapeMatrix[0][1] = -0.25f * (s-1);
      dShapeMatrix[0][2] =  0.25f * (s+1);
      dShapeMatrix[0][3] = -0.25f * (s+1);
      //// dShapeMatrix[1][i] = d/ds[Ni]
      dShapeMatrix[1][0] =  0.25f * (r-1);
      dShapeMatrix[1][1] = -0.25f * (r+1);
      dShapeMatrix[1][2] =  0.25f * (r+1);
      dShapeMatrix[1][3] = -0.25f * (r-1);
    }
    if(nnodes == 8) {
      fem_float r = rst[0];
      fem_float s = rst[1];

      //// dShapeMatrix[0][i] = d/dr[Ni]
      dShapeMatrix[0][0] = -((2*r + s)*(s - 1))/4;
      dShapeMatrix[0][1] = r*(s - 1);
      dShapeMatrix[0][2] = -((2*r - s)*(s - 1))/4;
      dShapeMatrix[0][3] = 0.5f * (1 - (s*s));
      dShapeMatrix[0][4] = ((2*r + s)*(s + 1))/4;
      dShapeMatrix[0][5] = -r*(s + 1);
      dShapeMatrix[0][6] = ((2*r - s)*(s + 1))/4;
      dShapeMatrix[0][7] = 0.5f * ((s*s) - 1);
      //// dShapeMatrix[1][i] = d/ds[Ni]
      dShapeMatrix[1][0] = -((r + 2*s)*(r - 1))/4;
      dShapeMatrix[1][1] = 0.5f * ((r*r) - 1);
      dShapeMatrix[1][2] = -((r - 2*s)*(r + 1))/4;
      dShapeMatrix[1][3] = -s*(r + 1);
      dShapeMatrix[1][4] = ((r + 2*s)*(r + 1))/4;
      dShapeMatrix[1][5] = 0.5f * (1 - (r*r));
      dShapeMatrix[1][6] = ((r - 2*s)*(r - 1))/4;
      dShapeMatrix[1][7] = s*(r - 1);
    }
  } else { // 3D
    if (nnodes == 20) { // BRICK20
      fem_float r = rst[0];
      fem_float s = rst[1];
      fem_float t = rst[2];

      //// dShapeMatrix[0][i] = d/dr[Ni]
      // Corner Nodes
      dShapeMatrix[0][0 ] = 0.125f * (1-s)*(1-t)*(2*r-s-t-1);
      dShapeMatrix[0][2 ] = 0.125f * (1+s)*(1-t)*(2*r+s-t-1);
      dShapeMatrix[0][4 ] = 0.125f * (1+s)*(1+t)*(2*r+s+t-1);
      dShapeMatrix[0][6 ] = 0.125f * (1-s)*(1+t)*(2*r-s+t-1);
      dShapeMatrix[0][12] = 0.125f * (1-s)*(1-t)*(2*r+s+t+1);
      dShapeMatrix[0][14] = 0.125f * (1+s)*(1-t)*(2*r-s+t+1);
      dShapeMatrix[0][16] = 0.125f * (1+s)*(1+t)*(2*r-s-t+1);
      dShapeMatrix[0][18] = 0.125f * (1-s)*(1+t)*(2*r+s-t+1);
      // Mid-Point Nodes
      dShapeMatrix[0][1 ] = 0.25f * 1     *(1-s*s)*(1-t);
      dShapeMatrix[0][3 ] = 0.25f * 1     *(1+s)  *(1-t*t);
      dShapeMatrix[0][5 ] = 0.25f * 1     *(1-s*s)*(1+t);
      dShapeMatrix[0][7 ] = 0.25f * 1     *(1-s)  *(1-t*t);
      dShapeMatrix[0][8 ] = 0.25f * (-2*r)*(1-s)  *(1-t);
      dShapeMatrix[0][9 ] = 0.25f * (-2*r)*(1+s)  *(1-t);
      dShapeMatrix[0][10] = 0.25f * (-2*r)*(1+s)  *(1+t);
      dShapeMatrix[0][11] = 0.25f * (-2*r)*(1-s)  *(1+t);
      dShapeMatrix[0][13] = 0.25f * -1    *(1-s*s)*(1-t);
      dShapeMatrix[0][15] = 0.25f * -1    *(1+s)  *(1-t*t);
      dShapeMatrix[0][17] = 0.25f * -1    *(1-s*s)*(1+t);
      dShapeMatrix[0][19] = 0.25f * -1    *(1-s)  *(1-t*t);

      //// dShapeMatrix[1][i] = d/ds[Ni]
      // Corner Nodes
      dShapeMatrix[1][0 ] = 0.125f * (1+r)*(1-t)*(-r+2*s+t+1);
      dShapeMatrix[1][2 ] = 0.125f * (1+r)*(1-t)*( r+2*s-t-1);
      dShapeMatrix[1][4 ] = 0.125f * (1+r)*(1+t)*( r+2*s+t-1);
      dShapeMatrix[1][6 ] = 0.125f * (1+r)*(1+t)*(-r+2*s-t+1);
      dShapeMatrix[1][12] = 0.125f * (1-r)*(1-t)*( r+2*s+t+1);
      dShapeMatrix[1][14] = 0.125f * (1-r)*(1-t)*(-r+2*s-t-1);
      dShapeMatrix[1][16] = 0.125f * (1-r)*(1+t)*(-r+2*s+t-1);
      dShapeMatrix[1][18] = 0.125f * (1-r)*(1+t)*( r+2*s-t+1);
      // Mid-Point Nodes
      dShapeMatrix[1][1 ] = 0.25f * (1+r)  *(-2*s)*(1-t);
      dShapeMatrix[1][3 ] = 0.25f * (1+r)  * 1    *(1-t*t);
      dShapeMatrix[1][5 ] = 0.25f * (1+r)  *(-2*s)*(1+t);
      dShapeMatrix[1][7 ] = 0.25f * (1+r)  *-1    *(1-t*t);
      dShapeMatrix[1][8 ] = 0.25f * (1-r*r)*-1    *(1-t);
      dShapeMatrix[1][9 ] = 0.25f * (1-r*r)* 1    *(1-t);
      dShapeMatrix[1][10] = 0.25f * (1-r*r)* 1    *(1+t);
      dShapeMatrix[1][11] = 0.25f * (1-r*r)*-1    *(1+t);
      dShapeMatrix[1][13] = 0.25f * (1-r)  *(-2*s)*(1-t);
      dShapeMatrix[1][15] = 0.25f * (1-r)  * 1    *(1-t*t);
      dShapeMatrix[1][17] = 0.25f * (1-r)  *(-2*s)*(1+t);
      dShapeMatrix[1][19] = 0.25f * (1-r)  *-1    *(1-t*t);

      //// dShapeMatrix[2][i] = d/dt[Ni]
      // Corner Nodes
      dShapeMatrix[2][0 ] = 0.125f * (1+r)*(1-s)*(-r+s+2*t+1);
      dShapeMatrix[2][2 ] = 0.125f * (1+r)*(1+s)*(-r-s+2*t+1);
      dShapeMatrix[2][4 ] = 0.125f * (1+r)*(1+s)*( r+s+2*t-1);
      dShapeMatrix[2][6 ] = 0.125f * (1+r)*(1-s)*( r-s+2*t-1);
      dShapeMatrix[2][12] = 0.125f * (1-r)*(1-s)*( r+s+2*t+1);
      dShapeMatrix[2][14] = 0.125f * (1-r)*(1+s)*( r-s+2*t+1);
      dShapeMatrix[2][16] = 0.125f * (1-r)*(1+s)*(-r+s+2*t-1);
      dShapeMatrix[2][18] = 0.125f * (1-r)*(1-s)*(-r-s+2*t-1);
      // Mid-Point Nodes
      dShapeMatrix[2][1 ] = 0.25f * (1+r)  *(1-s*s)*-1;
      dShapeMatrix[2][3 ] = 0.25f * (1+r)  *(1+s)  *(-2*t);
      dShapeMatrix[2][5 ] = 0.25f * (1+r)  *(1-s*s)* 1;
      dShapeMatrix[2][7 ] = 0.25f * (1+r)  *(1-s)  *(-2*t);
      dShapeMatrix[2][8 ] = 0.25f * (1-r*r)*(1-s)  *-1;
      dShapeMatrix[2][9 ] = 0.25f * (1-r*r)*(1+s)  *-1;
      dShapeMatrix[2][10] = 0.25f * (1-r*r)*(1+s)  * 1;
      dShapeMatrix[2][11] = 0.25f * (1-r*r)*(1-s)  * 1;
      dShapeMatrix[2][13] = 0.25f * (1-r)  *(1-s*s)*-1;
      dShapeMatrix[2][15] = 0.25f * (1-r)  *(1+s)  *(-2*t);
      dShapeMatrix[2][17] = 0.25f * (1-r)  *(1-s*s)* 1;
      dShapeMatrix[2][19] = 0.25f * (1-r)  *(1-s)  *(-2*t);
    } else if (nnodes == 8) { //BRICK8
      fem_float r = rst[0];
      fem_float s = rst[1];
      fem_float t = rst[2];

      //// dShapeMatrix[0][i] = d/dr[Ni]
      dShapeMatrix[0][0] =  0.125f * (s-1)*(t-1);
      dShapeMatrix[0][1] = -0.125f * (s+1)*(t-1);
      dShapeMatrix[0][2] =  0.125f * (s+1)*(t+1);
      dShapeMatrix[0][3] = -0.125f * (s-1)*(t+1);
      dShapeMatrix[0][4] = -0.125f * (s-1)*(t-1);
      dShapeMatrix[0][5] =  0.125f * (s+1)*(t-1);
      dShapeMatrix[0][6] = -0.125f * (s+1)*(t+1);
      dShapeMatrix[0][7] =  0.125f * (s-1)*(t+1);
      //// dShapeMatrix[1][i] = d/ds[Ni]
      dShapeMatrix[1][0] =  0.125f * (r+1)*(t-1);
      dShapeMatrix[1][1] = -0.125f * (r+1)*(t-1);
      dShapeMatrix[1][2] =  0.125f * (r+1)*(t+1);
      dShapeMatrix[1][3] = -0.125f * (r+1)*(t+1);
      dShapeMatrix[1][4] = -0.125f * (r-1)*(t-1);
      dShapeMatrix[1][5] =  0.125f * (r-1)*(t-1);
      dShapeMatrix[1][6] = -0.125f * (r-1)*(t+1);
      dShapeMatrix[1][7] =  0.125f * (r-1)*(t+1);
      //// dShapeMatrix[2][i] = d/dt[Ni]
      dShapeMatrix[2][0] =  0.125f * (r+1)*(s-1);
      dShapeMatrix[2][1] = -0.125f * (r+1)*(s+1);
      dShapeMatrix[2][2] =  0.125f * (r+1)*(s+1);
      dShapeMatrix[2][3] = -0.125f * (r+1)*(s-1);
      dShapeMatrix[2][4] = -0.125f * (r-1)*(s-1);
      dShapeMatrix[2][5] =  0.125f * (r-1)*(s+1);
      dShapeMatrix[2][6] = -0.125f * (r-1)*(s+1);
      dShapeMatrix[2][7] =  0.125f * (r-1)*(s-1);
    }
  }

}

////////////////////////////////////////////////////////////////////////////////
// Makes and Return Jacobian matrix
////////////////////////////////////////////////////////////////////////////////
void StiffAlgo::GetJacobianMatrix(int modeldim,
                                  int nnodes,
                                  fem_float** Jacobian,
                                  fem_float** coordsElem,
                                  fem_float** dShapeFunc) {
  if (modeldim == 2) {
    fem_float dxdr = dotProduct(nnodes, coordsElem[0], dShapeFunc[0]);
    fem_float dydr = dotProduct(nnodes, coordsElem[1], dShapeFunc[0]);
    fem_float dxds = dotProduct(nnodes, coordsElem[0], dShapeFunc[1]);
    fem_float dyds = dotProduct(nnodes, coordsElem[1], dShapeFunc[1]);

    Jacobian[0][0] = dxdr;
    Jacobian[0][1] = dydr;
    Jacobian[1][0] = dxds;
    Jacobian[1][1] = dyds;
  } else {
    fem_float dxdr = dotProduct(nnodes, coordsElem[0], dShapeFunc[0]);
    fem_float dydr = dotProduct(nnodes, coordsElem[1], dShapeFunc[0]);
    fem_float dzdr = dotProduct(nnodes, coordsElem[2], dShapeFunc[0]);
    fem_float dxds = dotProduct(nnodes, coordsElem[0], dShapeFunc[1]);
    fem_float dyds = dotProduct(nnodes, coordsElem[1], dShapeFunc[1]);
    fem_float dzds = dotProduct(nnodes, coordsElem[2], dShapeFunc[1]);
    fem_float dxdt = dotProduct(nnodes, coordsElem[0], dShapeFunc[2]);
    fem_float dydt = dotProduct(nnodes, coordsElem[1], dShapeFunc[2]);
    fem_float dzdt = dotProduct(nnodes, coordsElem[2], dShapeFunc[2]);

    Jacobian[0][0] = dxdr;
    Jacobian[0][1] = dydr;
    Jacobian[0][2] = dzdr;
    Jacobian[1][0] = dxds;
    Jacobian[1][1] = dyds;
    Jacobian[1][2] = dzds;
    Jacobian[2][0] = dxdt;
    Jacobian[2][1] = dydt;
    Jacobian[2][2] = dzdt;
  }

}

////////////////////////////////////////////////////////////////////////////////
// Builds the B matrix from the derivatives of Shape Functions in Cart. Coords.
////////////////////////////////////////////////////////////////////////////////
void StiffAlgo::BuildBMatrix(int modeldim,
                             int nnodes,
                             fem_float** matrixB,
                             fem_float** dNdcart) {
  int i;

  if(modeldim == 2) {
    // builds B matrix
#pragma omp parallel for private(i)
    for(i=0; i<nnodes; ++i) {
      matrixB[0][(2*i)  ] = dNdcart[0][i];
      matrixB[1][(2*i)+1] = dNdcart[1][i];
      matrixB[2][(2*i)  ] = dNdcart[1][i];
      matrixB[2][(2*i)+1] = dNdcart[0][i];
    }
  } else {
    // builds B matrix
#pragma omp parallel for private(i)
    for(i=0; i<nnodes; ++i) {
      matrixB[0][ 3*i   ] = dNdcart[0][i];
      matrixB[1][(3*i)+1] = dNdcart[1][i];
      matrixB[2][(3*i)+2] = dNdcart[2][i];

      matrixB[3][ 3*i   ] = dNdcart[1][i]; matrixB[3][(3*i)+1] = dNdcart[0][i];
      matrixB[4][(3*i)+1] = dNdcart[2][i]; matrixB[4][(3*i)+2] = dNdcart[1][i];
      matrixB[5][ 3*i   ] = dNdcart[2][i]; matrixB[5][(3*i)+2] = dNdcart[0][i];
    }
  }

}

////////////////////////////////////////////////////////////////////////////////
// Builds the B matrix from the derivatives of Shape Functions in Cart. Coords.
////////////////////////////////////////////////////////////////////////////////
void StiffAlgo::BuildBMatrix2(int modeldim,
                              int nnodes,
                              fem_float** matrixB,
                              fem_float** dNdcart ) {
  int i;

  // builds B matrix
  for(i=0; i<nnodes; i++) {
    matrixB[0][ 3*i   ] = dNdcart[0][i];
    matrixB[1][(3*i)+1] = dNdcart[1][i];
    matrixB[2][(3*i)+2] = dNdcart[2][i];

    matrixB[3][ 3*i   ] = dNdcart[1][i]; matrixB[3][(3*i)+1] = dNdcart[0][i];
    matrixB[4][(3*i)+1] = dNdcart[2][i]; matrixB[4][(3*i)+2] = dNdcart[1][i];
    matrixB[5][ 3*i   ] = dNdcart[2][i]; matrixB[5][(3*i)+2] = dNdcart[0][i];
  }
}


// Calculates the Global Sparse Stiffness Matrix K
////////////////////////////////////////////////////////////////////////////////
void StiffAlgo::AssembleK(int modeldim,
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

// Calculates the Global Sparse Stiffness Matrix K
////////////////////////////////////////////////////////////////////////////////
void StiffAlgo::AssembleKcol(int modeldim,
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
        if (stiffmat->GetAllocTrigger()) {
          #pragma omp critical (memalloc)
          {
            stiffmat->AddElem(gblDOFi  , gblDOFj  , k_local[(modeldim*i)  ][(modeldim*j)  ]);
            stiffmat->AddElem(gblDOFi+1, gblDOFj  , k_local[(modeldim*i)+1][(modeldim*j)  ]);
            stiffmat->AddElem(gblDOFi  , gblDOFj+1, k_local[(modeldim*i)  ][(modeldim*j)+1]);
            stiffmat->AddElem(gblDOFi+1, gblDOFj+1, k_local[(modeldim*i)+1][(modeldim*j)+1]);
          }
        } else {
          stiffmat->AddElem(gblDOFi  , gblDOFj  , k_local[(modeldim*i)  ][(modeldim*j)  ]);
          stiffmat->AddElem(gblDOFi+1, gblDOFj  , k_local[(modeldim*i)+1][(modeldim*j)  ]);
          stiffmat->AddElem(gblDOFi  , gblDOFj+1, k_local[(modeldim*i)  ][(modeldim*j)+1]);
          stiffmat->AddElem(gblDOFi+1, gblDOFj+1, k_local[(modeldim*i)+1][(modeldim*j)+1]);
        }
      }
    }
  } else if ( modeldim == 3 ) {
    for (int i = 0; i < numelemnodes; i++) {
      int gblDOFi = (elemconnect[numelemnodes*elem+i]-1)*modeldim;
      for (int j = 0; j < numelemnodes; ++j) {
        int gblDOFj = (elemconnect[numelemnodes*elem+j]-1)*modeldim;
        if (stiffmat->GetAllocTrigger()) {
          #pragma omp critical (memalloc)
          {
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
        } else {
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
}
