// Kernel code
// Matrizes usando row-major order:
// M(row, col) = *(Matriz + row * Matriz + col)

#define fem_float float

// Headers
void      getdShapeMat( __local fem_float dShapeMatrix[3][20], uint gp, __constant fem_float* rst );
void      local_getJacobianMatrix( fem_float Jacobian[3][3], __local fem_float coordsElem[3][20], __local fem_float dShapeFunc[3][20], __local fem_float aux[20] );
fem_float localDotProd( __local fem_float* v1, __local fem_float* v2, __local fem_float* aux );
fem_float localDotProd2( __local fem_float* v1, __local fem_float* v2, __local fem_float* aux );   
fem_float localDotProd3( __local fem_float* v1, __local fem_float* v2, __local fem_float* aux );
fem_float det3x3(fem_float Matrix[3][3]);
void      inverse3x3( fem_float inverse[3][3], fem_float matrix[3][3], fem_float det);
void      local_matMult_3_3_20( uint lidx, __local fem_float matrixC[3][20], fem_float matrixA[3][3], __local fem_float matrixB[3][20] );
void      local_matMult_6_6_60( uint lidx, __local fem_float matrixC[6][60], fem_float matrixA[6][6], __local fem_float matrixB[6][60] );
void      local_matAddMultTranspScal_60_6_60( uint lidx, __local fem_float matrixC[60][60], __local fem_float matrixA[6][60], 
                                              __local fem_float matrixB[6][60], fem_float scal );
void      local_buildBMatrix( uint lidx, __local fem_float matrixB[6][60], __local fem_float dNdcart[3][20]  );
void      writeToGlobalKaux( uint lidx, uint elem, __local fem_float k_local[60][60], __global float* global_Kaux);
void      printMatrix(float mat[3][3]);
void      printLocalMatrix(__local float mat[6][60]);



#pragma OPENCL EXTENSION cl_amd_printf : enable  
/////////////////////////////////////////////////////////////////////////////
//          Matrix multiplication kernel called by MatMulHost()            //
/////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(20,1,1)))
void getStiffnessUR20(__private  fem_float  E,   
                      __private  fem_float  Nu,  
                      __private  int        gpts,
                      __constant fem_float* X_gausspts,
                      __constant fem_float* W_gaussweights,
                      __global   fem_float* nodeCoords,
                      __global   int*       elemConnect,
                      __global   fem_float* global_Kaux)//output
{
  uint i, j;

  // Allocates memory for variables used in element stiffness calc. loop
  // Makes constitutive matrix
  local fem_float elemCoords[3][20];
  local fem_float dShapeMat[3][20];
  local fem_float dNdCart[3][20];
  local fem_float B[6][60];
  local fem_float CxB[6][60];
  local fem_float k_local[60][60];
  local fem_float aux[20];
  
  fem_float detJ;
  fem_float J[3][3];
  fem_float J_inv[3][3];
  
  // Gets Compute Unit Index Values
  uint lidx = get_local_id(0);
  uint elem = get_group_id(0); 
  
  //Gets current element coordinates
  elemCoords[0][lidx] = nodeCoords[(elemConnect[20*elem+lidx]-1)*3  ];
  elemCoords[1][lidx] = nodeCoords[(elemConnect[20*elem+lidx]-1)*3+1];
  elemCoords[2][lidx] = nodeCoords[(elemConnect[20*elem+lidx]-1)*3+2];
  
  fem_float C[6][6];
  {
    for( i=0; i<6; i++){
      for( j=0; j<6; j++){
        C[i][j] = 0;
      }
    }
    fem_float c1 = E*(1.0f-Nu) / ((1.0f+Nu)*(1.0f-2.0f*Nu));
    fem_float c2 = Nu / (1.0f-Nu);
    fem_float c3 = (1.0f-2.0f*Nu) / (2.0f*(1.0f-Nu));
    C[0][0] = c1;
    C[0][1] = c1 * c2;
    C[0][2] = c1 * c2;
    C[1][0] = c1 * c2;
    C[1][1] = c1;
    C[1][2] = c1 * c2;
    C[2][0] = c1 * c2;
    C[2][1] = c1 * c2;
    C[2][2] = c1;
    C[3][3] = c1 * c3;
    C[4][4] = c1 * c3;
    C[5][5] = c1 * c3;
  }
  
  for(uint i = 0; i<3; i++)
  {
    for(uint j = 0; j<60; j+=4)
    {
      k_local[j  ][lidx+(i*20)] = 0;
      k_local[j+1][lidx+(i*20)] = 0;
      k_local[j+2][lidx+(i*20)] = 0;
      k_local[j+3][lidx+(i*20)] = 0;
    }
  }
  
  // Calculates element stiffness matrix
  //Loops over gauss points
  for( uint gp=0; gp<(gpts*gpts*gpts); gp++)
  {
    getdShapeMat( dShapeMat, gp, X_gausspts);
    local_getJacobianMatrix( J, elemCoords, dShapeMat, aux);
    detJ = det3x3( J );
    inverse3x3( J_inv, J, detJ);
    local_matMult_3_3_20( lidx, dNdCart, J_inv, dShapeMat);
    local_buildBMatrix( lidx, B, dNdCart );
    local_matMult_6_6_60( lidx, CxB, C, B );
    local_matAddMultTranspScal_60_6_60( lidx, k_local, B, CxB, (detJ*W_gaussweights[gp]) );
  }
  writeToGlobalKaux(lidx, elem, k_local, global_Kaux);
}


// Gets the Derivative of the Shape Matrix in relation to natural coordinates
////////////////////////////////////////////////////////////////////////////////
void getdShapeMat( __local fem_float dShapeMatrix[3][20], uint gp, __constant fem_float* rst )
{
  // Shape func for midpoint nodes (e.g. for r=0): 
  //                                Ni = 1/4(1-r^2 )(1+s*si)(1+t*ti)
  // Shape func for corner nodes:   Ni = 1/8(1+r*ri)(1+s*si)(1+t*ti)
    if( get_local_id(0) == 0 )
    {
      fem_float r = rst[3*gp+0];
      fem_float s = rst[3*gp+1];
      fem_float t = rst[3*gp+2];
      
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
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

// Gets the Jacobian Matrix
////////////////////////////////////////////////////////////////////////////////
void local_getJacobianMatrix( fem_float Jacobian[3][3], __local fem_float coordsElem[3][20], __local fem_float dShapeFunc[3][20], __local fem_float* aux )
{
  Jacobian[0][0] = localDotProd3( coordsElem[0], dShapeFunc[0], aux );
  Jacobian[0][1] = localDotProd3( coordsElem[1], dShapeFunc[0], aux );
  Jacobian[0][2] = localDotProd3( coordsElem[2], dShapeFunc[0], aux );
  Jacobian[1][0] = localDotProd3( coordsElem[0], dShapeFunc[1], aux );
  Jacobian[1][1] = localDotProd3( coordsElem[1], dShapeFunc[1], aux );
  Jacobian[1][2] = localDotProd3( coordsElem[2], dShapeFunc[1], aux );
  Jacobian[2][0] = localDotProd3( coordsElem[0], dShapeFunc[2], aux );
  Jacobian[2][1] = localDotProd3( coordsElem[1], dShapeFunc[2], aux );
  Jacobian[2][2] = localDotProd3( coordsElem[2], dShapeFunc[2], aux );
}

// Gets the Jacobian Matrix
////////////////////////////////////////////////////////////////////////////////
fem_float localDotProd( __local fem_float* v1, __local fem_float* v2, __local fem_float* aux )
{
  uint lidx = get_local_id(0);
  
  aux[lidx] = v1[lidx] * v2[lidx];
 
  // Reduction
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lidx < 5)
  {
    fem_float sum0 = aux[lidx] + aux[lidx+5] + aux[lidx+10] + aux[lidx+15];
    aux[lidx] = sum0;
  }
  
  fem_float sum = aux[0]+aux[1]+aux[2]+aux[3]+aux[4];
  
  barrier(CLK_LOCAL_MEM_FENCE);
  return sum;
}

// Computes the determinant of a 3x3 matrix
////////////////////////////////////////////////////////////////////////////////
fem_float localDotProd2( __local fem_float* v1, __local fem_float* v2, __local fem_float* aux )
{
  uint lidx = get_local_id(0);
  fem_float sum = 0.0f;
 
  // Reduction
  for( uint i=0; i<20; i+=4)
  {
    fem_float sum1 = v1[i  ] * v2[i  ];
    fem_float sum2 = v1[i+1] * v2[i+1];
    fem_float sum3 = v1[i+2] * v2[i+2];
    fem_float sum4 = v1[i+3] * v2[i+3];
    
    sum += sum1+sum2+sum3+sum4;
  }
  
  return sum;
}

// Gets the Jacobian Matrix
////////////////////////////////////////////////////////////////////////////////
fem_float localDotProd3( __local fem_float* v1, __local fem_float* v2, __local fem_float* aux )
{
  uint lidx = get_local_id(0);
  fem_float sum = 0;
  aux[lidx] = v1[lidx] * v2[lidx];
  
  // Reduction
  for( uint i=0; i<20; i+=4)
  {
    fem_float sum1 = aux[i  ];
    fem_float sum2 = aux[i+1];
    fem_float sum3 = aux[i+2];
    fem_float sum4 = aux[i+3];
    
    sum += sum1+sum2+sum3+sum4;
  }
  
  return sum;
}

// Computes the determinant of a 3x3 matrix
////////////////////////////////////////////////////////////////////////////////
fem_float det3x3(fem_float Matrix[3][3])
{
  fem_float det;
  det = Matrix[0][0]*Matrix[1][1]*Matrix[2][2]+
        Matrix[1][0]*Matrix[2][1]*Matrix[0][2]+
        Matrix[2][0]*Matrix[0][1]*Matrix[1][2]-
        Matrix[0][0]*Matrix[2][1]*Matrix[1][2]-
        Matrix[1][0]*Matrix[0][1]*Matrix[2][2]-
        Matrix[2][0]*Matrix[1][1]*Matrix[0][2];
        
  return det;
}

// Computes the determinant of a 3x3 matrix
////////////////////////////////////////////////////////////////////////////////
void inverse3x3( fem_float inverse[3][3], fem_float matrix[3][3], fem_float det)
{
  inverse[0][0] =  (matrix[2][2]*matrix[1][1]-matrix[2][1]*matrix[1][2])/det;
  inverse[0][1] = -(matrix[2][2]*matrix[0][1]-matrix[2][1]*matrix[0][2])/det;
  inverse[0][2] =  (matrix[1][2]*matrix[0][1]-matrix[1][1]*matrix[0][2])/det;
  inverse[1][0] = -(matrix[2][2]*matrix[1][0]-matrix[2][0]*matrix[1][2])/det;
  inverse[1][1] =  (matrix[2][2]*matrix[0][0]-matrix[2][0]*matrix[0][2])/det;
  inverse[1][2] = -(matrix[1][2]*matrix[0][0]-matrix[1][0]*matrix[0][2])/det;
  inverse[2][0] =  (matrix[2][1]*matrix[1][0]-matrix[2][0]*matrix[1][1])/det;
  inverse[2][1] = -(matrix[2][1]*matrix[0][0]-matrix[2][0]*matrix[0][1])/det;
  inverse[2][2] =  (matrix[1][1]*matrix[0][0]-matrix[1][0]*matrix[0][1])/det;
}

// Computes Matrix Multipication for 3x3 * 3x20 matrices
////////////////////////////////////////////////////////////////////////////////
void local_matMult_3_3_20( uint lidx, __local fem_float matrixC[3][20], fem_float matrixA[3][3], __local fem_float matrixB[3][20] )
{
  barrier(CLK_LOCAL_MEM_FENCE);
  matrixC[0][lidx] = (matrixA[0][0]*matrixB[0][lidx]) + (matrixA[0][1]*matrixB[1][lidx]) + (matrixA[0][2]*matrixB[2][lidx]);
  matrixC[1][lidx] = (matrixA[1][0]*matrixB[0][lidx]) + (matrixA[1][1]*matrixB[1][lidx]) + (matrixA[1][2]*matrixB[2][lidx]);
  matrixC[2][lidx] = (matrixA[2][0]*matrixB[0][lidx]) + (matrixA[2][1]*matrixB[1][lidx]) + (matrixA[2][2]*matrixB[2][lidx]);
}

// Computes Matrix Multipication for 3x3 * 3x20 matrices
////////////////////////////////////////////////////////////////////////////////
void local_matMult_6_6_60( uint lidx, __local fem_float matrixC[6][60], fem_float matrixA[6][6], __local fem_float matrixB[6][60] )
{
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int i = 0; i<3; i++)
  {
    fem_float matC0 = (matrixA[0][0]*matrixB[0][lidx]) + (matrixA[0][1]*matrixB[1][lidx]) + (matrixA[0][2]*matrixB[2][lidx])+
                      (matrixA[0][3]*matrixB[3][lidx]) + (matrixA[0][4]*matrixB[4][lidx]) + (matrixA[0][5]*matrixB[5][lidx]);
    fem_float matC1 = (matrixA[1][0]*matrixB[0][lidx]) + (matrixA[1][1]*matrixB[1][lidx]) + (matrixA[1][2]*matrixB[2][lidx])+
                      (matrixA[1][3]*matrixB[3][lidx]) + (matrixA[1][4]*matrixB[4][lidx]) + (matrixA[1][5]*matrixB[5][lidx]);
    fem_float matC2 = (matrixA[2][0]*matrixB[0][lidx]) + (matrixA[2][1]*matrixB[1][lidx]) + (matrixA[2][2]*matrixB[2][lidx])+
                      (matrixA[2][3]*matrixB[3][lidx]) + (matrixA[2][4]*matrixB[4][lidx]) + (matrixA[2][5]*matrixB[5][lidx]);
    fem_float matC3 = (matrixA[3][0]*matrixB[0][lidx]) + (matrixA[3][1]*matrixB[1][lidx]) + (matrixA[3][2]*matrixB[2][lidx])+
                      (matrixA[3][3]*matrixB[3][lidx]) + (matrixA[3][4]*matrixB[4][lidx]) + (matrixA[3][5]*matrixB[5][lidx]);
    fem_float matC4 = (matrixA[4][0]*matrixB[0][lidx]) + (matrixA[4][1]*matrixB[1][lidx]) + (matrixA[4][2]*matrixB[2][lidx])+
                      (matrixA[4][3]*matrixB[3][lidx]) + (matrixA[4][4]*matrixB[4][lidx]) + (matrixA[4][5]*matrixB[5][lidx]);
    fem_float matC5 = (matrixA[5][0]*matrixB[0][lidx]) + (matrixA[5][1]*matrixB[1][lidx]) + (matrixA[5][2]*matrixB[2][lidx])+
                      (matrixA[5][3]*matrixB[3][lidx]) + (matrixA[5][4]*matrixB[4][lidx]) + (matrixA[5][5]*matrixB[5][lidx]);    
    
    matrixC[0][lidx] = matC0;
    matrixC[1][lidx] = matC1;
    matrixC[2][lidx] = matC2;
    matrixC[3][lidx] = matC3;
    matrixC[4][lidx] = matC4;
    matrixC[5][lidx] = matC5;
    
    lidx += 20;
  }
}

// Computes Matrix Multipication for 3x3 * 3x20 matrices
////////////////////////////////////////////////////////////////////////////////
void local_matAddMultTranspScal_60_6_60( uint lidx, __local fem_float matrixC[60][60], __local fem_float matrixA[6][60], 
                                         __local fem_float matrixB[6][60], fem_float scal )
{
  barrier(CLK_LOCAL_MEM_FENCE);
  for(uint i = 0; i<3; i++)
  {
    for(uint j = 0; j<60; j+=4)
    {
      fem_float tempC0 = ( (matrixA[0][j  ]*matrixB[0][lidx]) + (matrixA[1][j  ]*matrixB[1][lidx]) + (matrixA[2][j  ]*matrixB[2][lidx])+
                           (matrixA[3][j  ]*matrixB[3][lidx]) + (matrixA[4][j  ]*matrixB[4][lidx]) + (matrixA[5][j  ]*matrixB[5][lidx]) ) * scal ;
      fem_float tempC1 = ( (matrixA[0][j+1]*matrixB[0][lidx]) + (matrixA[1][j+1]*matrixB[1][lidx]) + (matrixA[2][j+1]*matrixB[2][lidx])+
                           (matrixA[3][j+1]*matrixB[3][lidx]) + (matrixA[4][j+1]*matrixB[4][lidx]) + (matrixA[5][j+1]*matrixB[5][lidx]) ) * scal ;
      fem_float tempC2 = ( (matrixA[0][j+2]*matrixB[0][lidx]) + (matrixA[1][j+2]*matrixB[1][lidx]) + (matrixA[2][j+2]*matrixB[2][lidx])+
                           (matrixA[3][j+2]*matrixB[3][lidx]) + (matrixA[4][j+2]*matrixB[4][lidx]) + (matrixA[5][j+2]*matrixB[5][lidx]) ) * scal ;
      fem_float tempC3 = ( (matrixA[0][j+3]*matrixB[0][lidx]) + (matrixA[1][j+3]*matrixB[1][lidx]) + (matrixA[2][j+3]*matrixB[2][lidx])+
                           (matrixA[3][j+3]*matrixB[3][lidx]) + (matrixA[4][j+3]*matrixB[4][lidx]) + (matrixA[5][j+3]*matrixB[5][lidx]) ) * scal ;
      
      matrixC[j  ][lidx] += tempC0;
      matrixC[j+1][lidx] += tempC1;
      matrixC[j+2][lidx] += tempC2;
      matrixC[j+3][lidx] += tempC3;
    }
    lidx += 20;
  }
}

// Builds B Matrix
////////////////////////////////////////////////////////////////////////////////
void local_buildBMatrix( uint lidx, __local fem_float matrixB[6][60], __local fem_float dNdcart[3][20]  )
{
  barrier(CLK_LOCAL_MEM_FENCE);
  // Resets B Matrix
  for(uint i = 0; i<3; i++)
  {
    matrixB[0][lidx+(20*i)] = 0;
    matrixB[1][lidx+(20*i)] = 0;
    matrixB[2][lidx+(20*i)] = 0;
    matrixB[3][lidx+(20*i)] = 0;
    matrixB[4][lidx+(20*i)] = 0;
    matrixB[5][lidx+(20*i)] = 0;
  }
  
  // builds B matrix
  matrixB[0][ 3*lidx   ] = dNdcart[0][lidx];
  matrixB[1][(3*lidx)+1] = dNdcart[1][lidx];
  matrixB[2][(3*lidx)+2] = dNdcart[2][lidx];
  
  matrixB[3][ 3*lidx   ] = dNdcart[1][lidx]; matrixB[3][(3*lidx)+1] = dNdcart[0][lidx];
  matrixB[4][(3*lidx)+1] = dNdcart[2][lidx]; matrixB[4][(3*lidx)+2] = dNdcart[1][lidx];
  matrixB[5][ 3*lidx   ] = dNdcart[2][lidx]; matrixB[5][(3*lidx)+2] = dNdcart[0][lidx];
}

// Builds B Matrix
////////////////////////////////////////////////////////////////////////////////
void writeToGlobalKaux( uint lidx, uint elem, __local fem_float k_local[60][60], __global float* global_Kaux)
{
  uint elemStride = elem*60;
  uint rowStride  = 60*get_num_groups(0);
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  for(uint i = 0; i<3; i++)
  {
    for(uint j = 0; j<60; j+=4)
    {
      fem_float Kauxtemp0 = k_local[j  ][lidx];
      fem_float Kauxtemp1 = k_local[j+1][lidx];
      fem_float Kauxtemp2 = k_local[j+2][lidx];
      fem_float Kauxtemp3 = k_local[j+3][lidx];
      
      global_Kaux[elemStride+((j  )*rowStride)+lidx] = Kauxtemp0;
      global_Kaux[elemStride+((j+1)*rowStride)+lidx] = Kauxtemp1;
      global_Kaux[elemStride+((j+2)*rowStride)+lidx] = Kauxtemp2;
      global_Kaux[elemStride+((j+3)*rowStride)+lidx] = Kauxtemp3;
    }
    lidx += 20;
  }
}

// printMatrix Utlity Function
////////////////////////////////////////////////////////////////////////////////
void printMatrix(float mat[3][3])
{
  if( get_global_id(0) == 0 )
  {
    printf("[ ");
    for(uint i=0; i<3; i++)
    {
      for(uint j=0; j<3; j++)
        printf("%4.3f ", mat[i][j]);
        printf("\n ");  
    }
  }
}

// printMatrix Utlity Function
////////////////////////////////////////////////////////////////////////////////
void printLocalMatrix(__local float mat[6][60])
{
  barrier(CLK_LOCAL_MEM_FENCE);
  if( get_global_id(0) == 0 )
  {
    printf("[ ");
    for(uint i=0; i<3; i++)
    {
      for(uint j=0; j<3; j++)
        printf("%4.3f ", mat[i][j]);
        printf("\n ");  
    }
  }
}
