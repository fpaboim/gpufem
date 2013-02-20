#define TILE_SIZEx 4
#define TILE_SIZEy 16

////////////////////////////////////////////////////////////////////////////////
//  SparseMV: Ellpack version has matrix data structure with matData matrix
//  arrays scanned column-wise and colIdx vector with corresponding cols for
//  each data entry. ELLwidth is number of columns saved (usually more than
//  used to mainain 2^n scalability. vector_d is input vector and vector_q
//  is CG output vector, result of MV operation. auxShare is a local vector
//  of the same size as ELLwidth used for parallel optimizations
////////////////////////////////////////////////////////////////////////////////

// SpMVCoal: Coalesced version of SpMV kernel (using blocking)
////////////////////////////////////////////////////////////////////////////////
__kernel void SpMVCoal(__global  float* matData,     // INPUT MATRIX DATA
                       __global  int*   colIdx,
                       __global  int*   rowNnz,
                       __private int    ELLwidth,
                       __private int    matDim,
                       __global  float* vector_x,    // INPUT
                       __global  float* vector_y,    // OUTPUT
                       __local   float* auxShared) { // LOCAL SHARED BUFFER
  //uint grpidy  = get_group_id(1);
  uint gloidy  = get_global_id(1);
  uint locidx  = get_local_id(0);
  uint locidy  = get_local_id(1);
  uint loclenx = get_local_size(0);

  // Zero out local memory
  auxShared[loclenx * locidy + locidx] = 0;

  uint slabsize   = matDim * loclenx;
  uint NumBlocksX = ELLwidth / loclenx;
  float sum = 0;
  // Loops the vertical "slab" over ELLwidth
  for (int i = 0; i < NumBlocksX; ++i) {
    //         baseaddress  +  locate group + locate in group
    int index  = i * slabsize + gloidy + (matDim * locidx);
    int col    = colIdx[index];
    float aval = matData[index];
    float xval = vector_x[col];
    sum  += aval * xval;
  }
  auxShared[locidy * loclenx + locidx] = sum;
  // Only one thread per row reduces
  if (locidx == 0) {
    barrier(CLK_LOCAL_MEM_FENCE);
    sum = 0;
    uint localbase = locidy * loclenx;
    for (int i = 0; i < loclenx; ++i) {
      sum += auxShared[localbase + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    vector_y[gloidy] = sum;
  }
}

// SpMVCoal: Coalesced version of SpMV kernel (using blocking)
////////////////////////////////////////////////////////////////////////////////
__kernel void SpMVCoalUR(__global  float* matData,     // INPUT MATRIX DATA
                         __global  int*   colIdx,
                         __global  int*   rowNnz,
                         __private int    ELLwidth,
                         __private int    matDim,
                         __global  float* vector_x,    // INPUT
                         __global  float* vector_y,    // OUTPUT
                         __local   float* auxShared) { // LOCAL SHARED BUFFER
  uint gloidy  = get_global_id(1);
  uint locidx  = get_local_id(0);
  uint locidy  = get_local_id(1);
  uint loclenx = get_local_size(0);
  uint locleny = get_local_size(1);

  // Zero out local memory
  auxShared[loclenx * locidy + locidx] = 0;

  uint slabsize   = matDim * loclenx;
  uint NumBlocksX = ELLwidth / loclenx;

  // Loops the vertical "slab" over ELLwidth
  float4 sum4 = 0;
  for (int i = 0; i < NumBlocksX; i += 4) {
    int4 index, col;
    float4 aval, xval;

    index.x = i * slabsize + gloidy + (matDim * locidx);
    index.y = (i + 1) * slabsize + gloidy + (matDim * locidx);
    index.z = (i + 2) * slabsize + gloidy + (matDim * locidx);
    index.w = (i + 3) * slabsize + gloidy + (matDim * locidx);
    col.x   = colIdx[index.x];
    col.y   = colIdx[index.y];
    col.z   = colIdx[index.z];
    col.w   = colIdx[index.w];
    aval.x  = matData[index.x];
    aval.y  = matData[index.y];
    aval.z  = matData[index.z];
    aval.w  = matData[index.w];
    xval.x  = vector_x[col.x];
    xval.y  = vector_x[col.y];
    xval.z  = vector_x[col.z];
    xval.w  = vector_x[col.w];
    sum4    += aval * xval;
  }
  auxShared[locidy * loclenx + locidx] = sum4.x + sum4.y + sum4.z + sum4.w;

  // Only one thread per row reduces
  if (locidx == 0) {
    barrier(CLK_LOCAL_MEM_FENCE);
    sum4.x = 0; sum4.y = 0; sum4.z = 0; sum4.w = 0;
    uint localbase = locidy * loclenx;
    for (int i = 0; i < loclenx; i += 4) {
      sum4.x += auxShared[localbase + i];
      sum4.y += auxShared[localbase + i + 1];
      sum4.z += auxShared[localbase + i + 2];
      sum4.w += auxShared[localbase + i + 3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    vector_y[gloidy] = sum4.x + sum4.y + sum4.z + sum4.w;
  }
}

// SpMVCoal: Coalesced version of SpMV kernel (using blocking)
////////////////////////////////////////////////////////////////////////////////
__kernel void SpMVCoalUR2(__global  float* matData,     // INPUT MATRIX DATA
                          __global  int*   colIdx,
                          __global  int*   rowNnz,
                          __private int    ELLwidth,
                          __private int    matDim,
                          __global  float* vector_x,    // INPUT
                          __global  float* vector_y,    // OUTPUT
                          __local   float* auxShared) { // LOCAL SHARED BUFFER
  //uint grpidy  = get_group_id(1);
  uint gloidy  = get_global_id(1);
  uint locidx  = get_local_id(0);
  uint locidy  = get_local_id(1);
  uint loclenx = get_local_size(0);
  uint locleny = get_local_size(1);

  // Zero out local memory
  auxShared[loclenx * locidy + locidx] = 0;

  uint localbase = (locidy * loclenx) + locidx;
  uint slabsize   = matDim * loclenx;
  uint NumBlocksX = ELLwidth / loclenx;

  // Loops the vertical "slab" over ELLwidth
  float4 sum4 = 0;
  for (int i = 0; i < NumBlocksX; i += 4) {
    int4 index, col;
    float4 aval, xval;

    index.x = i * slabsize + gloidy + (matDim * locidx);
    index.y = (i + 1) * slabsize + gloidy + (matDim * locidx);
    index.z = (i + 2) * slabsize + gloidy + (matDim * locidx);
    index.w = (i + 3) * slabsize + gloidy + (matDim * locidx);
    col.x = colIdx[index.x];
    col.y = colIdx[index.y];
    col.z = colIdx[index.z];
    col.w = colIdx[index.w];
    aval.x = matData[index.x];
    aval.y = matData[index.y];
    aval.z = matData[index.z];
    aval.w = matData[index.w];
    xval.x = vector_x[col.x];
    xval.y = vector_x[col.y];
    xval.z = vector_x[col.z];
    xval.w = vector_x[col.w];
    sum4  += aval * xval;
  }
  auxShared[localbase] = sum4.x + sum4.y + sum4.z + sum4.w;

  // Only one thread per row reduces
  if (locidx < 4) {
    barrier(CLK_LOCAL_MEM_FENCE);
    sum4.x = 0; sum4.y = 0; sum4.z = 0; sum4.w = 0;
    for (int i = 0; i < loclenx; i += 4) {
      sum4.x += auxShared[localbase + i];
      sum4.y += auxShared[localbase + i + 1];
      sum4.z += auxShared[localbase + i + 2];
      sum4.w += auxShared[localbase + i + 3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (locidx == 0)
      vector_y[gloidy] = sum4.x + sum4.y + sum4.z + sum4.w;
  }
}

// SpMVStag: matrix vector product computed with staggered offsets
////////////////////////////////////////////////////////////////////////////////
__kernel void SpMVStag2(__global  float* matData,     // INPUT MATRIX DATA
                       __global  int*   colIdx,
                       __global  int*   rowNnz,
                       __private int    ELLwidth,
                       __private int    matDim,
                       __global  float* vector_x,    // INPUT
                       __global  float* vector_y,    // OUTPUT
                       __local   float* auxShared) { // LOCAL SHARED BUFFER
  uint grpid  = get_group_id(0);
  uint locid  = get_local_id(0);
  uint loclen = get_local_size(0);
  if (grpid < matDim) {
    uint  nnz = rowNnz[grpid];
    uint  tid = grpid  + (locid * matDim);
    auxShared[locid] = 0;
    for (int i = 0; i < ELLwidth; i += loclen) {
      int index  = (i * matDim) + tid;
      int col    = colIdx[index];
      float aval = matData[index];
      float xval = vector_x[col];
      float val  = aval * xval;
      auxShared[locid] += val;
    }

    // parallel reduction
    barrier(CLK_LOCAL_MEM_FENCE);
    {
      for (int stride = loclen/2; stride > 0; stride /= 2) {
        if (locid < stride) {
          auxShared[locid] += auxShared[stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if (locid == 0)
        vector_y[grpid] = auxShared[0];
    }
  }
}

// SpMVStag: matrix vector product computed with staggered offsets
////////////////////////////////////////////////////////////////////////////////
__kernel void SpMVStag(__global  float* matData,     // INPUT MATRIX DATA
                       __global  int*   colIdx,
                       __global  int*   rowNnz,
                       __private int    ELLwidth,
                       __private int    matDim,
                       __global  float* vector_x,    // INPUT
                       __global  float* vector_y,    // OUTPUT
                       __local   float* auxShared) { // LOCAL SHARED BUFFER
  uint grpid       = get_group_id(0);
  uint locid       = get_local_id(0);
  uint loclen      = get_local_size(0);
  uint numblocks   = ELLwidth / loclen;
  uint blockstride = loclen * matDim;
  uint slaboffset  = grpid + (locid * matDim);

  if (grpid < matDim) {
    auxShared[locid] = 0;
    for (int i = 0; i < numblocks; i++) {
      //             vertical slab   +  row  + x_pos
      int index  = (i * blockstride) + slaboffset;
      int col    = colIdx[index];
      float aval = matData[index];
      float xval = vector_x[col];
      float val  = aval * xval;
      auxShared[locid] += val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
      float sum = 0;
      for (int i = 0; i < loclen; i++) {
        sum += auxShared[i];
      }
      vector_y[grpid] = sum;
    }
  }
}

// SpMVNaive: naive implementation of sparse matrix vector product
////////////////////////////////////////////////////////////////////////////////
__kernel void SpMVNaive(__global  float* matData,     // INPUT MATRIX DATA
                        __global  int*   colIdx,
                        __global  int*   rowNnz,
                        __private int    ELLwidth,
                        __private int    matDim,
                        __global  float* vector_x,    // INPUT
                        __global  float* vector_y,    // OUTPUT
                        __local   float* auxShared) { // LOCAL SHARED BUFFER
  uint gid = get_global_id(0);
  if (gid < matDim) {
    uint nnz    = rowNnz[gid];
    float sum = 0;
    for (int i = 0; i < nnz; i++) {
      int index   = i * matDim + gid;
      int col     = colIdx[index];
      float aval  = matData[index];
      float xval  = vector_x[col];
      sum  += aval * xval;
    }
    vector_y[gid] = sum;
  }
}

// SpMVNaive: naive implementation of sparse matrix vector product
////////////////////////////////////////////////////////////////////////////////
__kernel void SpMVNaiveUR(__global  float* matData,     // INPUT MATRIX DATA
                          __global  int*   colIdx,
                          __global  int*   rowNnz,
                          __private int    ELLwidth,
                          __private int    matDim,
                          __global  float* vector_x,    // INPUT
                          __global  float* vector_y,    // OUTPUT
                          __local   float* auxShared) { // LOCAL SHARED BUFFER
  uint gid = get_global_id(0);
  if (gid < matDim) {
    uint nnz    = rowNnz[gid];
    float4 sum = 0;
    for (int i = 0; i < nnz; i += 4) {
      int4 index, col;
      float4 aval, xval;
      index.s0 = i * matDim + gid;
      index.s1 = (i + 1) * matDim + gid;
      index.s2 = (i + 2) * matDim + gid;
      index.s3 = (i + 3) * matDim + gid;
      col.s0   = colIdx[index.s0];
      col.s1   = colIdx[index.s1];
      col.s2   = colIdx[index.s2];
      col.s3   = colIdx[index.s3];
      aval.s0  = matData[index.s0];
      aval.s1  = matData[index.s1];
      aval.s2  = matData[index.s2];
      aval.s3  = matData[index.s3];
      xval.s0  = vector_x[col.s0];
      xval.s1  = vector_x[col.s1];
      xval.s2  = vector_x[col.s2];
      xval.s3  = vector_x[col.s3];
      sum.s0  += aval.s0 * xval.s0;
      sum.s1  += aval.s1 * xval.s1;
      sum.s2  += aval.s2 * xval.s2;
      sum.s3  += aval.s3 * xval.s3;
    }
    vector_y[gid] = sum.s0 + sum.s1 + sum.s2 + sum.s3;
  }
}
