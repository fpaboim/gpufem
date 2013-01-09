// Kernel code
// Matrizes usando row-major order:
// M(row, col) = *(Matriz + row * Matriz + col)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#define fem_float float

// Headers
void getdShapeMat(uint gp, __constant fem_float* rst,
                  __local fem_float dShapeMatrix[2][4]);
void local_getJacobianMatrix(fem_float Jacobian[2][2],
                             __local fem_float coordsElem[2][4],
                             __local fem_float dShapeFunc[2][4],
                             __local fem_float aux[4]);
fem_float localDotProd(__local fem_float* v1, __local fem_float* v2,
                       __local fem_float* aux );
fem_float det2x2(fem_float Matrix[2][2]);
void inverse2x2(fem_float inverse[2][2], fem_float matrix[2][2], fem_float det);
void local_matMult_2_2_4(uint lidx,
                         __local fem_float matrixC[2][4],
                         fem_float matrixA[2][2],
                         __local fem_float matrixB[2][4]);
void local_matMult_3_3_8(uint lidx,
                         __local fem_float matrixC[3][8],
                         fem_float matrixA[3][3],
                         __local fem_float matrixB[3][8] );
void local_matAddMultTranspScal_8_3_8(uint lidx,
                                      __local fem_float matrixC[8][8],
                                      __local fem_float matrixA[3][8],
                                      __local fem_float matrixB[3][8],
                                      fem_float scal);
void local_buildBMatrix(uint lidx,
                        __local fem_float matrixB[3][8],
                        __local fem_float dNdcart[2][4]);
void writeToGlobalKaux(uint lidx, uint elem,
                       __local fem_float k_local[8][8],
                       __global float* global_Kaux);
void printMatrix(__local float mat[38][8]);
