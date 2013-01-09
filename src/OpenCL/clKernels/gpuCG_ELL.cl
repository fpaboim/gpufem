// Kernel code
// Matrizes usando row-major order:
// M(row, col) = *(Matriz + row * Matriz + col)

#define TILE_SIZEx 4
#define TILE_SIZEy 16


/////////////////////////////////////////////////////////////////////////////
//          Matrix multiplication kernel called by MatMulHost()            //
/////////////////////////////////////////////////////////////////////////////
__kernel void MatMulKernelGlobal(__private  uint   width,
                                 __constant float* matA,
                                 __constant float* vecX,
                                 __global   float* vecY,
                                 __local    float* partialDotProduct) {
  uint localid = get_local_id(0);
  //vecY[get_global_id(0)]=0;
  for (uint i = width/get_local_size(0) ; i > 0; i-- ) {
    partialDotProduct[localid] = matA[get_group_id(0)*get_local_size(0)+localid] * vecX[get_global_id(0)];
    // bitwise >>= 1 is faster
    for (uint stride = get_local_size(0)>>1; stride > 0; stride >>= 1 ) {
      // Synchronize to make sure each work-item is done updating
      barrier(CLK_LOCAL_MEM_FENCE);
      // Only the first work-items in the work-group add elements together
      if (localid < stride) {
        // Add two elements from the "partialDotProduct" array
        partialDotProduct[localid] += partialDotProduct[localid + stride];
      }
    }
    // Writes to vector
    if (get_local_id(0)==0)
      vecY[0] += partialDotProduct[0];
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
        // Add two elements from the "partialDotProduct" array
        // and store the result in partialDotProduct[index]
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
__kernel void MatMulKernelCoalesced(uint width,
                                    __global  float* M,
                                    __global  float* V,
                                    __global float* W,
                                    __local float* partialDotProduct) {
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
    partialDotProduct[get_local_id(0)] = sum;
    // Perform parallel reduction to add each work-item's
    // partial dot product together
    for (uint stride = get_local_size(0) / 2; stride > 0; stride >>= 1 ) {
      // Synchronize to make sure each work-item is done updating
      // shared memory; this is necessary because work-items read
      // results that have been written by other work-items
      barrier(CLK_LOCAL_MEM_FENCE);
      // Only the first work-items in the work-group add elements together
      if (get_local_id(0) < stride) {
        // Add two elements from the "partialDotProduct" array
        // and store the result in partialDotProduct[index]
        partialDotProduct[get_local_id(0)] += partialDotProduct[get_local_id(0) + stride];
      }
    }
    // Write the result of the reduction to global memory
    if (get_local_id(0) == 0)
      W[i] = partialDotProduct[0];
    // Synchronize to make sure the first work-item is done with
    // reading partialDotProduct
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

/////////////////////////////////////////////////////////////////////////////
//               Scalar Alpha X Plus Y -> y = alpha*x + y
/////////////////////////////////////////////////////////////////////////////
void addScaledVectToSelf( __global float* self_vect,
                          __global const float* add_vect,
                          const float alpha,
                          int n) {
  // get index into global data array
  uint tid = get_global_id(0);
  // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
  if (tid >= n)
    return;
  self_vect[tid] = self_vect[tid] + (alpha*add_vect[tid]);
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
  // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
  if (tid >= n)
    return;
  self_vect[tid] = (beta*self_vect[tid]) + add_vect[tid];
}

/////////////////////////////////////////////////////////////////////////////
//      Computes a vector - vector dot product x * y = alpha
/////////////////////////////////////////////////////////////////////////////
float DotProdCoalesced(uint width,
                       __global const float* M,
                       __global const float* V,
                       __local float* partialDotProduct) {
  // Each work-item accumulates as many products as necessary
  // into local variable "sum"
  float sum = 0;
  for (uint x = get_local_id(0); x < width; x += get_local_size(0))
      sum += M[x] * V[x];
  // Each partial dot product is stored in shared memory
  partialDotProduct[get_local_id(0)] = sum;
  // Perform parallel reduction to add each work-item's
  // partial dot product together
  for (uint stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
    // Synchronize to make sure each work-item is done updating
    // shared memory; this is necessary because work-items read
    // results that have been written by other work-items
    barrier(CLK_LOCAL_MEM_FENCE);
    // Only the first work-items in the work-group add elements together
    if (get_local_id(0) < stride) {
      // Add two elements from the "partialDotProduct" array
      // and store the result in partialDotProduct[index]
      partialDotProduct[get_local_id(0)] += partialDotProduct[get_local_id(0) + stride];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // Write the result of the reduction to global memory
  return partialDotProduct[0];
}

/////////////////////////////////////////////////////////////////////////////
//          Matrix multiplication kernel called by MatrixMul()
/////////////////////////////////////////////////////////////////////////////
__kernel void CGMethod_CSR(uint width,
                          __global  float* matrix_A,
                          __global  float* vector_X,
                          __global  float* vector_d,
                          __global  float* vector_q,
                          __global  float* vector_r,
                          __private float* epsilon,
                          __local   float* partialDotProduct) {
  uint i;
  float epsilon = 0.0001f;
  uint localid = get_local_id(0);
  // Direction Vector
  float delta_new = 0.0f;
  float delta_old = 0.0f;
  float delta_ini = 0.0f;
  float err_bound = 0.0f;
  float alpha     = 0.0f;
  float beta      = 0.0f;

  // zero initial value for x
  // r = b - Ax
  // d = r
  delta_new = DotProdCoalesced(width, vector_r, vector_r, partialDotProduct);
  delta_ini = delta_new;
  err_bound = epsilon*epsilon*delta_ini;

  for (i = 0; (i < width) && (delta_new>err_bound); i++) {
    //MatMulKernelCoalesced(width, matrix_A, vector_d, vector_q, partialDotProduct);
    //alpha = rDotrNew / (d dot q)
    alpha = delta_new / DotProdCoalesced(width, vector_d, vector_q, partialDotProduct);
    //x = x + alpha * d
    addScaledVectToSelf( vector_X, vector_d, alpha, width);
    //r = r - alpha * q
    addScaledVectToSelf( vector_r, vector_q, -alpha, width);
    //rDotrOld = rDotrNew
    delta_old = delta_new;
    //rDotrNew = r dot r
    delta_new = DotProdCoalesced( width, vector_r, vector_r, partialDotProduct);
    //beta = rDotrNew / rDotrOld
    beta = delta_new / delta_old;
    //d = r + beta * d
    addSelfScaledToVect( vector_d, vector_r, beta, width);
  }
}