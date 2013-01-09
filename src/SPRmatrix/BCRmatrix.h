/////////////////////////////////////////////////////////////////////
// Sparsely Sparse Matrix Library
// Author: Francisco Aboim
// TecGraf / PUC-RIO
/////////////////////////////////////////////////////////////////////
#ifndef DIAMATRIX_H
#define DIAMATRIX_H

#define fem_float float

class DIAmatrix
{
public:
  DIAmatrix(int matdim);
  ~DIAmatrix();

  void         setElem(int row, int col, fem_float val);
  void         addElem(const int row, const int col, const fem_float val);
  fem_float    getElem(int row, int col);
  size_t       getMatSize();
  inline void  insertDiag(const int pos);
  inline int   binSearchInt( const int* intvector, const int val, const int length) const;


private:
  //-------------------------------------------
  // member variables
  //-------------------------------------------

  // DIA Matrix Format Data Structure
  int         m_ndiag;    /* number of diagonals     */
  int         m_matdim;   /* square matrix dimension */
  int         m_maxentries;   /* square matrix dimension */
  fem_float** m_DIAdata;  /* matrix data vector      */
  int*        m_posvect;  /* positional data vector  */

};

#endif