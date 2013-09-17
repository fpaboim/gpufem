// Matrices using row-major order order:
// M(row, col) = *(Matriz + row * Matriz + col)
#include <gpuFEM_Q4.h>
#include <ellmat.h>

////////////////////////////////////////////////////////////////////////////////
// OpenCL kernel for calculating all local stiffness matrices which are stored
// sequentially in a vector and then read by CPU for assembly
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(4,1,1)))
void getStiffnessQ4(__private  fem_float  E,
                    __private  fem_float  Nu,
                    __private  int        gpts,
                    __constant fem_float* X_gausspts,
                    __constant fem_float* W_gaussweights,
                    __global   fem_float* nodeCoords,
                    __global   int*       elemConnect,
                    __global   fem_float* global_Kaux) { //output
  uint i, j;

  // Allocates memory for variables used in element stiffness calc. loop
  // Makes constitutive matrix
  local fem_float elemCoords[2][4];
  local fem_float dShapeMat[2][4];
  local fem_float dNdCart[2][4];
  local fem_float B[3][8];
  local fem_float CxB[3][8];
  local fem_float k_local[8][8];
  local fem_float aux[4];

  fem_float detJ;
  fem_float J[2][2];
  fem_float J_inv[2][2];

  // Gets Compute Unit Index Values
  uint lidx = get_local_id(0);
  uint elem = get_group_id(0);

  //Gets current element coordinates
  elemCoords[0][lidx] = nodeCoords[(elemConnect[4*elem+lidx]-1)*2  ];
  elemCoords[1][lidx] = nodeCoords[(elemConnect[4*elem+lidx]-1)*2+1];

  fem_float C[3][3];
  {
    for( i=0; i<3; i++){
      for( j=0; j<3; j++){
        C[i][j] = 0;
      }
    }
    fem_float auxcoef = E / (1.0f-(Nu*Nu));
    fem_float c1 = auxcoef;
    fem_float c2 = auxcoef*((1.0f-Nu)/2.0f);
    fem_float c3 = auxcoef*Nu;
    C[0][0] = c1;
    C[1][1] = c1;
    C[2][2] = c2;
    C[0][1] = c3;
    C[1][0] = c3;
  }
  for(uint i = 0; i < 2; i++) {
    for(uint j = 0; j < 8; j+=4) {
      k_local[j  ][lidx+(i*4)] = 0;
      k_local[j+1][lidx+(i*4)] = 0;
      k_local[j+2][lidx+(i*4)] = 0;
      k_local[j+3][lidx+(i*4)] = 0;
    }
  }

  // Calculates element stiffness matrix
  //Loops over gauss points
  for (uint gp = 0; gp < (gpts*gpts); gp++) {
    getdShapeMat(gp, X_gausspts, dShapeMat);
    local_getJacobianMatrix(J, elemCoords, dShapeMat, aux);
    detJ = det2x2(J);
    inverse2x2(J_inv, J, detJ);
    local_matMult_2_2_4(lidx, dNdCart, J_inv, dShapeMat);
    local_buildBMatrix(lidx, B, dNdCart);
    local_matMult_3_3_8(lidx, CxB, C, B);
    local_matAddMultTranspScal_8_3_8( lidx, k_local, B, CxB, (detJ*W_gaussweights[gp]) );
  }

  writeToGlobalKaux(lidx, elem, k_local, global_Kaux);
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL kernel for calculating stiffness matrix and assembling on the GPU
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(4,1,1)))
void assembleStiffGPU(__private  fem_float  E,
                      __private  fem_float  Nu,
                      __private  int        gpts,
                      __constant fem_float* X_gausspts,
                      __constant fem_float* W_gaussweights,
                      __global   fem_float* nodeCoords,
                      __global   int*       elemConnect,
                      __global   fem_float* glblKdata,
                      __global   int*       glblKcolidx,
                      __private  int        glblKrowlen) { //output
  uint i, j;

  // Allocates memory for variables used in element stiffness calc. loop
  // Makes constitutive matrix
  local fem_float elemCoords[2][4];
  local fem_float dShapeMat[2][4];
  local fem_float dNdCart[2][4];
  local fem_float B[3][8];
  local fem_float CxB[3][8];
  local fem_float k_local[8][8];
  local fem_float aux[4];

  fem_float detJ;
  fem_float J[2][2];
  fem_float J_inv[2][2];

  // Gets Compute Unit Index Values
  uint lidx = get_local_id(0);
  uint elem = get_group_id(0);

  //Gets current element coordinates
  elemCoords[0][lidx] = nodeCoords[(elemConnect[4*elem+lidx]-1)*2  ];
  elemCoords[1][lidx] = nodeCoords[(elemConnect[4*elem+lidx]-1)*2+1];

  fem_float C[3][3];
  {
    for( i=0; i<3; i++){
      for( j=0; j<3; j++){
        C[i][j] = 0;
      }
    }
    fem_float auxcoef = E / (1.0f-(Nu*Nu));
    fem_float c1 = auxcoef;
    fem_float c2 = auxcoef*((1.0f-Nu)/2.0f);
    fem_float c3 = auxcoef*Nu;
    C[0][0] = c1;
    C[1][1] = c1;
    C[2][2] = c2;
    C[0][1] = c3;
    C[1][0] = c3;
  }

  for(uint i = 0; i < 2; i++) {
    for(uint j = 0; j < 8; j+=4) {
      k_local[j  ][lidx+(i*4)] = 0;
      k_local[j+1][lidx+(i*4)] = 0;
      k_local[j+2][lidx+(i*4)] = 0;
      k_local[j+3][lidx+(i*4)] = 0;
    }
  }

  // Calculates element stiffness matrix
  //Loops over gauss points
  for (uint gp = 0; gp < (gpts*gpts); gp++) {
    getdShapeMat(gp, X_gausspts, dShapeMat);
    local_getJacobianMatrix( J, elemCoords, dShapeMat, aux);
    detJ = det2x2( J );
    inverse2x2( J_inv, J, detJ);
    local_matMult_2_2_4( lidx, dNdCart, J_inv, dShapeMat);
    local_buildBMatrix( lidx, B, dNdCart );
    local_matMult_3_3_8( lidx, CxB, C, B );
    local_matAddMultTranspScal_8_3_8( lidx, k_local, B, CxB, (detJ*W_gaussweights[gp]) );
  }

  //writeToGlobalKaux(lidx, elem, k_local, global_Kaux);
}


// Gets the Derivative of the Shape Matrix in relation to natural coordinates
////////////////////////////////////////////////////////////////////////////////
void getdShapeMat(uint gp,
                  __constant fem_float* rst,
                  __local fem_float dShapeMatrix[2][4]) {
  // Shape func for midpoint nodes (e.g. for r=0):
  //                                Ni = 1/4(1-r^2 )(1+s*si)(1+t*ti)
  // Shape func for corner nodes:   Ni = 1/8(1+r*ri)(1+s*si)(1+t*ti)
    if( get_local_id(0) == 0 )
    {
      fem_float r = rst[2*gp];
      fem_float s = rst[2*gp+1];

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
    barrier(CLK_LOCAL_MEM_FENCE);
}

// Gets the Jacobian Matrix
////////////////////////////////////////////////////////////////////////////////
void local_getJacobianMatrix(fem_float Jacobian[2][2],
                             __local fem_float coordsElem[2][4],
                             __local fem_float dShapeFunc[2][4],
                             __local fem_float* aux ) {

  Jacobian[0][0] = localDotProd( coordsElem[0], dShapeFunc[0], aux );
  Jacobian[0][1] = localDotProd( coordsElem[1], dShapeFunc[0], aux );
  Jacobian[1][0] = localDotProd( coordsElem[0], dShapeFunc[1], aux );
  Jacobian[1][1] = localDotProd( coordsElem[1], dShapeFunc[1], aux );
}

// Gets the Jacobian Matrix
////////////////////////////////////////////////////////////////////////////////
fem_float localDotProd(__local fem_float* v1,
                       __local fem_float* v2,
                       __local fem_float* aux) {
  uint lidx = get_local_id(0);

  aux[lidx] = v1[lidx] * v2[lidx];

  fem_float sum1 = aux[0];
  fem_float sum2 = aux[1];
  fem_float sum3 = aux[2];
  fem_float sum4 = aux[3];

  fem_float sum = sum1+sum2+sum3+sum4;

  return sum;
}

// Computes the determinant of a 3x3 matrix
////////////////////////////////////////////////////////////////////////////////
fem_float det2x2(fem_float Matrix[2][2]) {
  fem_float det;
  det = Matrix[0][0]*Matrix[1][1]-Matrix[0][1]*Matrix[1][0];

  return det;
}

// Computes the determinant of a 2x2 matrix
////////////////////////////////////////////////////////////////////////////////
void inverse2x2(fem_float inverse[2][2], fem_float matrix[2][2], fem_float det)
{
  inverse[0][0] =   matrix[1][1]/det;
  inverse[0][1] =  -matrix[0][1]/det;
  inverse[1][0] =  -matrix[1][0]/det;
  inverse[1][1] =   matrix[0][0]/det;
}

// Computes Matrix Multipication for 2x2 * 2x4 matrices
////////////////////////////////////////////////////////////////////////////////
void local_matMult_2_2_4( uint lidx, __local fem_float matrixC[2][4], fem_float matrixA[2][2], __local fem_float matrixB[2][4] )
{
  barrier(CLK_LOCAL_MEM_FENCE);
  matrixC[0][lidx] = (matrixA[0][0]*matrixB[0][lidx]) + (matrixA[0][1]*matrixB[1][lidx]);
  matrixC[1][lidx] = (matrixA[1][0]*matrixB[0][lidx]) + (matrixA[1][1]*matrixB[1][lidx]);
}

// Computes Matrix Multipication for 3x3 * 3x8 matrices
////////////////////////////////////////////////////////////////////////////////
void local_matMult_3_3_8( uint lidx, __local fem_float matrixC[3][8], fem_float matrixA[3][3], __local fem_float matrixB[3][8] )
{
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int i = 0; i<2; i++)
  {
    fem_float matC0 = (matrixA[0][0]*matrixB[0][lidx]) + (matrixA[0][1]*matrixB[1][lidx]) + (matrixA[0][2]*matrixB[2][lidx]);
    fem_float matC1 = (matrixA[1][0]*matrixB[0][lidx]) + (matrixA[1][1]*matrixB[1][lidx]) + (matrixA[1][2]*matrixB[2][lidx]);
    fem_float matC2 = (matrixA[2][0]*matrixB[0][lidx]) + (matrixA[2][1]*matrixB[1][lidx]) + (matrixA[2][2]*matrixB[2][lidx]);

    matrixC[0][lidx] = matC0;
    matrixC[1][lidx] = matC1;
    matrixC[2][lidx] = matC2;

    lidx += 4;
  }
}

// Computes Matrix Multipication for 2x2 * 2x4 matrices
////////////////////////////////////////////////////////////////////////////////
void local_matAddMultTranspScal_8_3_8( uint lidx, __local fem_float matrixC[8][8], __local fem_float matrixA[3][8],
                                       __local fem_float matrixB[3][8], fem_float scal )
{
  barrier(CLK_LOCAL_MEM_FENCE);
  for(uint i = 0; i<2; i++)
  {
    for(uint j = 0; j<8; j+=4)
    {
      fem_float tempC0 = ( (matrixA[0][j  ]*matrixB[0][lidx]) + (matrixA[1][j  ]*matrixB[1][lidx]) + (matrixA[2][j  ]*matrixB[2][lidx]) ) * scal ;
      fem_float tempC1 = ( (matrixA[0][j+1]*matrixB[0][lidx]) + (matrixA[1][j+1]*matrixB[1][lidx]) + (matrixA[2][j+1]*matrixB[2][lidx]) ) * scal ;
      fem_float tempC2 = ( (matrixA[0][j+2]*matrixB[0][lidx]) + (matrixA[1][j+2]*matrixB[1][lidx]) + (matrixA[2][j+2]*matrixB[2][lidx]) ) * scal ;
      fem_float tempC3 = ( (matrixA[0][j+3]*matrixB[0][lidx]) + (matrixA[1][j+3]*matrixB[1][lidx]) + (matrixA[2][j+3]*matrixB[2][lidx]) ) * scal ;

      matrixC[j  ][lidx] += tempC0;
      matrixC[j+1][lidx] += tempC1;
      matrixC[j+2][lidx] += tempC2;
      matrixC[j+3][lidx] += tempC3;
    }
    lidx += 4;
  }
}

// Builds B Matrix
////////////////////////////////////////////////////////////////////////////////
void local_buildBMatrix( uint lidx, __local fem_float matrixB[3][8], __local fem_float dNdcart[2][4]  )
{
  barrier(CLK_LOCAL_MEM_FENCE);
  // Resets B Matrix
  matrixB[0][lidx  ] = 0;
  matrixB[1][lidx  ] = 0;
  matrixB[2][lidx  ] = 0;
  matrixB[0][lidx+4] = 0;
  matrixB[1][lidx+4] = 0;
  matrixB[2][lidx+4] = 0;

  matrixB[0][(2*lidx)  ] = dNdcart[0][lidx];
  matrixB[1][(2*lidx)+1] = dNdcart[1][lidx];
  matrixB[2][(2*lidx)  ] = dNdcart[1][lidx];
  matrixB[2][(2*lidx)+1] = dNdcart[0][lidx];
}

// Builds B Matrix
////////////////////////////////////////////////////////////////////////////////
void writeToGlobalKaux( uint lidx, uint elem, __local fem_float k_local[8][8], __global float* global_Kaux)
{
  uint elemStride = elem*8;
  uint rowStride  = 8*get_num_groups(0);

  barrier(CLK_LOCAL_MEM_FENCE);
  for(uint i = 0; i<2; i++)
  {
    for(uint j = 0; j<8; j+=4)
    {
      fem_float Kauxtemp0 = k_local[j  ][lidx];
      fem_float Kauxtemp1 = k_local[j+1][lidx];
      fem_float Kauxtemp2 = k_local[j+2][lidx];
      fem_float Kauxtemp3 = k_local[j+3][lidx];

      global_Kaux[elemStride+((j  )*rowStride)+lidx] = Kauxtemp0;
      global_Kaux[elemStride+((j+1)*rowStride)+lidx] = Kauxtemp1;
      global_Kaux[elemStride+((j+2)*rowStride)+lidx] = Kauxtemp2;
      global_Kaux[elemStride+((j+3)*rowStride)+lidx] = Kauxtemp3;
      //printf("elemStride+((j  )*rowStride)+lidx: %i\n", (elemStride+((j  )*rowStride)+lidx));
    }
    lidx += 4;
  }
}



void printMatrix(__local float mat[8][8])
{
  barrier(CLK_LOCAL_MEM_FENCE);

  if( get_local_id(0) == 0 )
  {
    printf("[ ");
    for(uint i=0; i<8; i++)
    {
      for(uint j=0; j<8; j++)
        printf("%4.3f ", mat[i][j]);
        printf("\n ");
    }
  }
}
