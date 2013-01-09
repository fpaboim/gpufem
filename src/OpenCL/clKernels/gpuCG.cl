#define TILE_SIZEx 4
#define TILE_SIZEy 16

#pragma OPENCL EXTENSION cl_amd_printf : enable
/////////////////////////////////////////////////////////////////////////////
//          Matrix multiplication kernel called by MatMulHost()            //
/////////////////////////////////////////////////////////////////////////////
__kernel void MatMulKernelGlobal(__private  uint   width,
                                 __constant float* matA,
                                 __constant float* vecX,
                                 __global   float* vecY,
                                 __local    float* auxShared) {
  uint localid = get_local_id(0);
  //vecY[get_global_id(0)]=0;
  for (uint i = width/get_local_size(0) ; i > 0; i-- ) {
    auxShared[localid] = matA[get_group_id(0)*get_local_size(0)+localid] * vecX[get_global_id(0)];
    // bitwise >>= 1 is faster
    for (uint stride = get_local_size(0)>>1; stride > 0; stride >>= 1 ) {
      // Synchronize to make sure each work-item is done updating
      barrier(CLK_LOCAL_MEM_FENCE);
      // Only the first work-items in the work-group add elements together
      if (localid < stride) {
        // Add two elements from the "auxShared" array
        auxShared[localid] += auxShared[localid + stride];
      }
    }
    // Writes to vector
    if (get_local_id(0)==0)
      vecY[0] += auxShared[0];
  }
}

/////////////////////////////////////////////////////////////////////////////
//          Matrix multiplication kernel called by MatrixMul()             //
/////////////////////////////////////////////////////////////////////////////
__kernel void MatMulKernelTiled(__private uint   width,
                                __global  float* matA,
                                __global  float* vecX,
                                __global  float* vecY) {
  local float submatA[TILE_SIZEx][TILE_SIZEy];

  uint lidx = get_local_id(0);
  uint lidy = get_local_id(1);
  uint gidx = get_global_id(0);
  uint gidy = get_global_id(1);

  //grouprow = get_group_id(1) * TILE_SIZE * width;
  //groupcol = get_group_id(0) * TILE_SIZE;

  // Loops over xid of group to find final vect
  for ( uint i=0; i<(width/TILE_SIZEx); i++ ) {
    // Colaborative Loading of Submatrix
    submatA[lidx][lidy] = matA[ gidy*width + gidx ];
    submatA[lidx][lidy] *= vecX[gidx];
    for (uint stride = TILE_SIZEx / 2; stride > 0; stride >>= 1 ) {
      // Synchronize to make sure each work-item is done updating
      // shared memory; this is necessary because work-items read
      // results that have been written by other work-items
      barrier(CLK_LOCAL_MEM_FENCE);
      // Only the first work-items in the work-group add elements together
      if (get_local_id(0) < stride) {
        // Add two elements from the "auxShared" array
        // and store the result in auxShared[index]
        submatA[get_local_id(0)][get_local_id(1)] += submatA[get_local_id(0) + stride][get_local_id(1)];
      }
    }
    if (get_local_id(0) == 0)
      vecY[get_group_id(1)*TILE_SIZEy+get_local_id(1)] += submatA[0][get_local_id(1)];
  }
}



/////////////////////////////////////////////////////////////////////////////
//          Matrix multiplication kernel called by MatrixMul()
/////////////////////////////////////////////////////////////////////////////
__kernel void MatMulKernelCoalesced(__private uint   width,
                                    __global  float* M,
                                    __global  float* V,
                                    __global  float* W,
                                    __local   float* auxShared) {
  // Each work-group handles as many matrix rows as necessary
  for (uint i = get_group_id(0); i < width; i += get_num_groups(0)) {
    // Row pointer
    __global float* row = M + i*width;
    // Each work-item accumulates as many products as necessary
    // into local variable "sum"
    float sum = 0;
    for (uint j = get_local_id(0); j < width; j += get_local_size(0))
      sum += row[j] * V[j];
    // Each partial dot product is stored in shared memory
    auxShared[get_local_id(0)] = sum;
    // Perform parallel reduction to add each work-item's
    // partial dot product together
    for (uint stride = get_local_size(0) / 2; stride > 0; stride >>= 1 ) {
      // Synchronize to make sure each work-item is done updating
      // shared memory; this is necessary because work-items read
      // results that have been written by other work-items
      barrier(CLK_LOCAL_MEM_FENCE);
      // Only the first work-items in the work-group add elements together
      if (get_local_id(0) < stride) {
        // Add two elements from the "auxShared" array
        // and store the result in auxShared[index]
        auxShared[get_local_id(0)] += auxShared[get_local_id(0) + stride];
      }
    }
    // Write the result of the reduction to global memory
    if (get_local_id(0) == 0)
      W[i] = auxShared[0];
    // Synchronize to make sure the first work-item is done with
    // reading auxShared
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}


/////////////////////////////////////////////////////////////////////////////
//          Matrix multiplication kernel called by MatMulHost()            //
/////////////////////////////////////////////////////////////////////////////
void EllSpmv(__private int    matDim,
             __private int    ELLwidth,
             __global  float* matA,
             __global  int*   colIdx,
             __global  int*   rowNnz,
             __global  float* vecX,
             __global  float* vecY) {
  
}

/////////////////////////////////////////////////////////////////////////////
//               Scalar Alpha X Plus Y -> y = alpha*x + y
/////////////////////////////////////////////////////////////////////////////
void addScaledVectToSelf(__global float* self_vect,
                         __global float* add_vect,
                         const float alpha,
                         const int n) {
  // get index into global data array
  uint tid = get_global_id(0);
  if (tid < n)
    self_vect[tid] += (alpha * add_vect[tid]);
}

/////////////////////////////////////////////////////////////////////////////
//             Scalar Alpha X Plus Y -> y = x + beta*y
/////////////////////////////////////////////////////////////////////////////
void addSelfScaledToVect(__global float* self_vect,
                         __global const float* add_vect,
                         const float beta,
                         int n) {
  // get index into global data array
  uint tid = get_global_id(0);
  if (tid < n)
    self_vect[tid] = (beta * self_vect[tid]) + add_vect[tid];
}

/////////////////////////////////////////////////////////////////////////////
//      Computes a vector - vector dot product x * y = alpha
/////////////////////////////////////////////////////////////////////////////
float DotProdCoalesced(__private uint   width,
                       __global  float* M,
                       __global  float* V,
                       __local   float* auxShared) {
  // Each work-item accumulates as many products as necessary
  // into local variable "sum"
  float sum = 0;
  for (uint x = get_local_id(0); x < width; x += get_local_size(0))
      sum += M[x] * V[x];
  // Each partial dot product is stored in shared memory
  auxShared[get_local_id(0)] = sum;
  // Perform parallel reduction to add each work-item's
  // partial dot product together
  for (uint stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
    // Synchronize to make sure each work-item is done updating
    // shared memory; this is necessary because work-items read
    // results that have been written by other work-items
    barrier(CLK_LOCAL_MEM_FENCE);
    // Only the first work-items in the work-group add elements together
    if (get_local_id(0) < stride) {
      // Add two elements from the "auxShared" array
      // and store the result in auxShared[index]
      auxShared[get_local_id(0)] += auxShared[get_local_id(0) + stride];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // Write the result of the reduction to global memory
  return auxShared[0];
}

/////////////////////////////////////////////////////////////////////////////
//      Computes a vector - vector dot product x * y = alpha
/////////////////////////////////////////////////////////////////////////////
float DotProdCoalesced2(__private uint   width,
                        __global  float* vec1,
                        __global  float* vec2,
                        __local   float* auxShared) {
  float accumulator = 0;
  for (uint i = 0; i < width; ++i) {
    float x1 = vec1[i];
    float x2 = vec2[i];
    accumulator += x1 * x2;
  }
  printf("gid:%i - wid:%i - vec1[2]:%f - accumulator: %f\n", 
         get_global_id(0), width, vec1[0], accumulator); 

  return accumulator;
}

/////////////////////////////////////////////////////////////////////////////
//          Matrix multiplication kernel called by MatrixMul()
/////////////////////////////////////////////////////////////////////////////
__kernel void CGSparseELL(__global  float* matData,
                          __global  int*   colIdx,
                          __global  int*   rowNnz,
                          __private int    ELLwidth,
                          __private int    matDim,
                          __private float  epsilon,
                          __private int    nIter,
                          __private float  deltaInit,
                          __global  float* vector_X,
                          __global  float* vector_d,
                          __global  float* vector_q,
                          __global  float* vector_r,
                          __local   float* auxShared) {
  // Direction Vector
  float delta_new = deltaInit;
  float delta_old = 0.0f;
  float err_bound = 0.0f;
  float alpha     = 0.0f;
  float beta      = 0.0f;
  
  uint gid = get_global_id(0);
  printf("gid:%i - nIter:%i - vecR[2]:%f - epsilon: %f\n", 
         gid, nIter, vector_r[2], epsilon);

  // zero initial value for x
  // r = b - Ax   ;   d = r
  //delta_new = DotProdCoalesced2(matDim, vector_d, vector_d, auxShared);
  err_bound = epsilon * epsilon * delta_new;
  
  for (uint i = 0; (i < nIter) && (delta_new > err_bound); i++) {
    // q = A * d
    EllSpmv(matDim, ELLwidth, matData, colIdx, rowNnz, vector_d, vector_q);
    
    //alpha = rDotrNew / (d dot q)
    alpha = delta_new / DotProdCoalesced(matDim, vector_d, vector_q, auxShared);
   
    printf("alpha[%i]: %.2f\n", i, alpha);
    //x = x + alpha * d
    addScaledVectToSelf(vector_X, vector_d, alpha, matDim);
    
    //r = r - alpha * q
    addScaledVectToSelf(vector_r, vector_q, -alpha, matDim);
    
    //rDotrOld = rDotrNew
    delta_old = delta_new;
    
    //rDotrNew = r dot r
    delta_new = DotProdCoalesced(matDim, vector_r, vector_r, auxShared);

    //beta = rDotrNew / rDotrOld
    //beta = delta_new / delta_old;

    //d = r + beta * d
    //addSelfScaledToVect(vector_d, vector_r, beta, matDim);
  }
}
