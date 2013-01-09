// Headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DIAmatrix.hpp"


//-----------------------------------------------------------------------
DIAmatrix::DIAmatrix(int matdim)
{
  m_matdim     = matdim;
  m_ndiag      = 0;
  m_maxentries = 128;
  m_DIAdata    = (fem_float**)calloc(m_maxentries*m_matdim,sizeof(fem_float*));
  m_posvect    = (int*)       malloc(m_maxentries*sizeof(int));
}

//-----------------------------------------------------------------------
DIAmatrix::~DIAmatrix()
{
}
//-----------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
// Matrix Manipulation Functions
////////////////////////////////////////////////////////////////////////////////

void
DIAmatrix::setElem(int row, int col, fem_float val)
{
  int diag = col - row;
  
  // finds matrix position
  int i=0;
  bool found = false;
  while(m_posvect[i]<=diag && m_posvect[i] != NULL )
  {
    if(m_posvect[i]==diag)
    {
      found = true;
      break;
    }
    i++;
  }
  // if diagonal of element not found allocates memory
  if(!found)
  {
    insertDiag(i);
    m_posvect[i] = diag;
  }
  m_DIAdata[i][row] = val;
}

//-----------------------------------------------------------------------

void
DIAmatrix::addElem(const int row, const int col, const fem_float val)
{
  const int diag = col - row;
  if(diag<0)
    return;

  // finds matrix position
  int pos = binSearchInt(m_posvect, diag, m_ndiag);

  // if diagonal of element not found allocates memory
  if(m_posvect[pos]!=diag)
  {
    insertDiag(pos);
    m_posvect[pos] = diag;
  }
  m_DIAdata[pos][row] += val;
}

//-----------------------------------------------------------------------

fem_float
DIAmatrix::getElem(int row, int col)
{
  int diag = col - row;
  
  int i=0;
  while(m_posvect[i]<=diag)
  {
    if(m_posvect[i]==diag)
      return m_DIAdata[i][row];
    i++;
  }

  return 0;
}

//-----------------------------------------------------------------------

size_t
DIAmatrix::getMatSize()
{
  return (m_maxentries*m_matdim*sizeof(fem_float));
}

//-----------------------------------------------------------------------

inline void
DIAmatrix::insertDiag(const int pos)
{
  m_ndiag++;
  if(m_ndiag>m_maxentries)
  {
    m_maxentries += 4;
    m_DIAdata = (fem_float**)realloc(m_DIAdata, (m_maxentries * sizeof(fem_float*)) );
    m_posvect = (int*)       realloc(m_posvect, (m_maxentries * sizeof(int)) );
  }

  // Moves Memory: (to , from , blocksize)
  memmove(&m_posvect[pos+1], &m_posvect[pos], (m_ndiag-(pos+1))*sizeof(int) );
  memmove(&m_DIAdata[pos+1], &m_DIAdata[pos], (m_ndiag-(pos+1))*sizeof(fem_float*) );
  m_DIAdata[pos] = (fem_float*)calloc(m_matdim, sizeof(fem_float));
}

//-----------------------------------------------------------------------

inline int
DIAmatrix::binSearchInt( const int* intvector, const int val, const int length) const
{
  int end   = length-1;
  int start = 0;

  while(end>start)
  {
    size_t mid = (end+start)>>1;
    if (intvector[mid]<val)
      start = mid+1;
    else
      end = mid;
  }
  return end;
}