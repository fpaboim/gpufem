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
// Sparsely Sparse Matrix Library - Main Matrix Creation Interface
// Author: Francisco Aboim
// TecGraf / PUC-RIO
////////////////////////////////////////////////////////////////////////////////

// Headers
#include "SPRmatrix.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DENmatrix.h"
#include "DIAmatrix.h"
#include "CSRmatrix.h"
#include "ELLmatrix.h"
#include "ELLmatrix2.h"
#include "EIGmatrix.h"

// Constructor
////////////////////////////////////////////////////////////////////////////////
SPRmatrix::SPRmatrix() {
  m_matdim            = 0;
  m_matformat         = DEN;
  m_optimizationstrat = UNDEF;
  m_verboseerrors     = true;
}

// Destructor
////////////////////////////////////////////////////////////////////////////////
SPRmatrix::~SPRmatrix() {
}

// CreateMatrix: creates a new sparse matrix of given format and dimensions
////////////////////////////////////////////////////////////////////////////////
SPRmatrix*
SPRmatrix::CreateMatrix(const int dim, SPRformat matformat) {

  if (matformat == DEN)
    return new DENmatrix(dim);
  else if (matformat == DIA)
    return new DIAmatrix(dim);
  else if (matformat == CSR)
    return new CSRmatrix(dim);
  else if (matformat == ELL)
    return new ELLmatrix(dim);
  else if (matformat == EL2)
    return new ELLmatrix2(dim);
  else if (matformat == EIG)
    return new EIGmatrix(dim);
  else
    assert(false); // Error!

  return NULL;
}


////////////////////////////////////////////////////////////////////////////////
// Matrix Manipulation Functions
////////////////////////////////////////////////////////////////////////////////

// InputIsOK: Checks if input is within matrix bounds and not adding a zero
////////////////////////////////////////////////////////////////////////////////
int SPRmatrix::InputIsOK(fem_float val, int row, int col) {
  if (val == 0)
    return 0;

  return BoundsOK(row, col);
}

// BoundsOK: Checks if row and column are inside matrix boundaries
////////////////////////////////////////////////////////////////////////////////
int SPRmatrix::BoundsOK(int row, int col) {
  if (row < 0 || row >= m_matdim || col < 0 || col >= m_matdim) {
    if (m_verboseerrors)
      printf("**WARNING** - Out of bounds matrix access: (dim:%i row:%i col:%i)\n",
             m_matdim, row, col);
    return 0;
  }
  return 1;
}

// BinSearchInt: preforms binary search in ordered int array for val and length,
//               returns the position
////////////////////////////////////////////////////////////////////////////////
int SPRmatrix::BinSearchInt(const int* intvector, const int val,
                            const int length) const {
  if(length == 0)
    return 0;
  if(length == 1)
    return (val < intvector[0] ? 0 : 1);

  int end   = length-1;
  int start = 0;
  int mid;

  while (start <= end) {
    mid = (end+start) >> 1; // mid is midpoint of array
    if (intvector[mid] < val)
      start = mid + 1;
    else if (intvector[mid] > val)
      end = mid - 1;
    else
      return mid; // found element, return position
  }

  return start; // not found, returns insertion position
}

// BinSearchInt: preforms binary search in ordered int array for val and length,
//               returns the position
////////////////////////////////////////////////////////////////////////////////
int SPRmatrix::BinSearchInt2(const int* intvector, const int val,
                             const int len) const {
  int min = 0, max = len;

  while (min < max) {
    int middle = (min + max) >> 1;
    if (val > intvector[middle])
      min = middle + 1;
    else
      max = middle;
  }
  return min;
}

// Same as above but with SSE instructions
////////////////////////////////////////////////////////////////////////////////
int SPRmatrix::BinSearchIntCmov(const int* intvector, const int val,
                                const int len) const {
  int min = 0, max = len;
  while (min < max) {
    int middle = (min + max) >> 1;
    if (val > intvector[middle])
      min = middle + 1;
    else
      max = middle;
  }
//  while (min < max) {
//    int middle = (min + max) >> 1;
//    __asm {
//      "cmpl %3, %2"
//      "cmovg %4, %0"
//      "cmovle %5, %1"
//      :"+r" (min), "+r" (max)
//      : "r" (val), "g" (intvector[middle]), "g" (middle + 1), "g" (middle)
//    };
//    __asm {
//      mov EAX, [val]
//      mov EBX, [intvector+middle]
//      cmp EAX, EBX
//      mov EAX, middle
//      add EAX, 1
//      mov EBX, middle
//      cmovg [min], EAX
//      cmovle [max], EBX
//    }
//  }

  return min;
}


// BinSmart: less conditionals per loop
////////////////////////////////////////////////////////////////////////////////
int SPRmatrix::BinSmart(const int *intvector, const int val, const int len)
                        const {
  int idx;
  //idx = 2^(largest i| 2^i < n):
  for (idx = 1; idx < len; idx <<= 1);
    if (intvector[idx] > val) {
      return BinSmart(intvector, idx, val);
    } else {
      //search_func_t binary_cmov_unrolled = get_binary_cmov_unrolled(n);
      int newlen = len - idx;
      return BinSearchInt2(intvector + idx, val, newlen);
    }
}


// LinGetPosI: preforms linear search in ordered int array for val and length,
// position sorted number should be inserted in.
////////////////////////////////////////////////////////////////////////////////
int SPRmatrix::LinSearchI(const int* intvector, const int val,
                          const int length) const {
  int i = 0;
  for(i = 0; i < length; i++) {
    if (intvector[i] >= val)
      break;
  }

  return i;
}

// LinSearchISSE: Returns index of found value or sorted insert point if not
// found; Input vector must be 16 bit aligned, multiple of 16 and padded with
// INT_MAX on first of undefined values
////////////////////////////////////////////////////////////////////////////////
int SPRmatrix::LinSearchISSE(const int* intvector, const int val,
                             const int len) const {

  if ((len == 0) || (val < intvector[0]))
    return 0;

  __declspec(align(16)) int v4[4] = {val, val, val, val};
  const __m128i* v4ptr = (__m128i*)v4;
  __m128i key4 = _mm_load_si128(v4ptr);

  int i = 0;
  __declspec(align(16)) unsigned short res;
  for (i = 0; i <= len; i += 16) {
    const __m128i* in0ptr = (__m128i*)&intvector[i    ];
    const __m128i* in1ptr = (__m128i*)&intvector[i + 4];
    const __m128i* in2ptr = (__m128i*)&intvector[i + 8];
    const __m128i* in3ptr = (__m128i*)&intvector[i + 12];
    __m128i in0 = _mm_load_si128(in0ptr);
    __m128i in1 = _mm_load_si128(in1ptr);
    __m128i in2 = _mm_load_si128(in2ptr);
    __m128i in3 = _mm_load_si128(in3ptr);
    __m128i cmp0 = _mm_cmpgt_epi32(key4, in0);
    __m128i cmp1 = _mm_cmpgt_epi32(key4, in1);
    __m128i cmp2 = _mm_cmpgt_epi32(key4, in2);
    __m128i cmp3 = _mm_cmpgt_epi32(key4, in3);
//     __m128i cmp0 = _mm_cmpgt_epi32(key4, *in0ptr);
//     __m128i cmp1 = _mm_cmpgt_epi32(key4, *in1ptr);
//     __m128i cmp2 = _mm_cmpgt_epi32(key4, *in2ptr);
//     __m128i cmp3 = _mm_cmpgt_epi32(key4, *in3ptr);
    __m128i pack01 = _mm_packs_epi32(cmp0, cmp1);
    __m128i pack23 = _mm_packs_epi32(cmp2, cmp3);
    __m128i pack0123 = _mm_packs_epi16(pack01, pack23);
    res = _mm_movemask_epi8(pack0123);
    if (res != 0xffff)
      break;
  }

  int count = 0;
  if (res) {
    unsigned long rb = 0;
    _BitScanForward(&rb, (unsigned long)~res);
    count = rb;
  }

  return i + count;
}


// LinSearchISSE2: Returns index of found value or sorted insert point if not
// found; Input vector must be 16 bit aligned, multiple of 16 and padded with
// INT_MAX on first of undefined values
////////////////////////////////////////////////////////////////////////////////
int SPRmatrix::LinSearchISSE2(const int* intvector, const int val,
                              const int len) const {

    if ((len == 0) || (val < intvector[0]))
      return 0;

    __declspec(align(16)) int v4[4] = {val, val, val, val};
    __m128i key4 = _mm_load_si128((__m128i*)v4);

    int i = 0;
    __declspec(align(16)) unsigned short res;
    for (i = 0; i <= len; i += 4) {
      const __m128i* in0ptr = (__m128i*)&intvector[i];
      __m128i cmp0 = _mm_cmpgt_epi32(key4, *in0ptr);
      res = _mm_movemask_epi8(cmp0);
      if (res != 0xffff)
        break;
    }

    int count = 0;
    if (res) {
      unsigned long rb = 0;
      _BitScanForward(&rb, (unsigned long)~res);
      count = rb>>2;
    }

    return i + count;
}

// PrintMatrix: prints the sparse matrix to console
////////////////////////////////////////////////////////////////////////////////
void
SPRmatrix::PrintMatrix() {
  int i, j;

  printf("[ ");
  for (i = 0; i < m_matdim; i++) {
    for (j = 0; j < m_matdim; j++) {
      fem_float value = GetElem(i, j);
      if (value < 0)
        printf("%4.1f ", value);
      else
        printf("%4.2f ", value);
    }
    printf("%s", (i != (m_matdim-1)) ? "\n  " : "]\n");
  }
}
