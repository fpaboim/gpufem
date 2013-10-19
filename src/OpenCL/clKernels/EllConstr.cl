// Finite Element Constraint Application for Ellpack Matrix Format
#define TILE_SIZEx 4
#define TILE_SIZEy 16

__kernel void ApplyContrPen(__global  float* matData,     // INPUT MATRIX DATA
                            __global  int*   colIdx,
                            __global  int*   rowNnz,
                            __private int    ELLwidth,
                            __private int    matDim,
                            __local   float* auxShared) { // LOCAL SHARED BUFFER

}

// Based on AMD reduce tutorial by Bryan Catanzaro ("Optimization Case Study:
// Simple Reductions") parallel reduction with bitwise & local bounds checking
__kernel
void reduce(__global float* buffer,
            __local float*  auxShared,
            __const int     length,
            __global float* result) {
  int gloid = get_global_id(0);
  int locid = get_local_id(0);
  // Load data into local memory
  if (gloid < length) {
    ausShared[locid] = buffer[gloid];
  } else {
    // Infinity is the identity element for the min operation
    ausShared[locid] = INFINITY;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // each iteration counts from n = 1 to n^2 til localsz(0)
  for(int offset = 1; offset < get_local_size(0); offset <<= 1) {
    int mask = (offset << 1) - 1;
    if ((locid & mask) == 0) { // only up til mask(offset^2-1)
      float other = ausShared[locid + offset];
      float mine = ausShared[locid];
      ausShared[locid] = (mine < other) ? mine : other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (locid == 0) {
    result[get_group_id(0)] = ausShared[0];
  }
}

  // Zero out local memory
}
