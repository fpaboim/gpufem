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

/////////////////////////////////////////////////////////////////////
// fileIO.h- File I/O Operations Header
/////////////////////////////////////////////////////////////////////
#ifndef FILEIO_H_
#define FILEIO_H_

#include <iostream>
#include <string>

#include "SPRmatrix/SPRmatrix.h"
#include "util.h"


// Class Definition
class FileIO {
public:
  enum ElemType {
    UNDEF   = 0,
    BRICK20 = 1,
    BRICK8  = 2,
    Q4      = 3,
    Q8      = 4
  };

  FileIO();
  ~FileIO();

  int ReadNF(std::string neutralfile_name);
  // Output handling
  int OpenOutputFile(std::string out_filename);
  int CloseOutputFile();
  int writeOutputHeader();
  int writeNFInfo();
  int writeString(const char* str);
  int writeString(std::string str);
  int writeStringTab(const char* str);
  int writeStringTab(std::string str);
  int writeNumTab(fem_float num);
  int writeNumTab(double num);
  int writeNumTab(int num);
  int writeNewLine();
  int writeMatrix(SPRmatrix* matrix, int m, int n);
  int writeMatrixNonzeros(SPRmatrix* matrix, int m, int n);
  int writeMatrix2(fem_float* matrix, int m, int n);
  int writeMatrix3(fem_float** matrix, int m, int n);
  int writeVector(fem_float*  vector, int n);
  int writeXYZVector(fem_float* u_vector, int n);

  // Data Structure Query Functions
  /////////////////////////////////////////////////
  int         getModelDim() const { return m_modeldim; }
  const char* getAnalysis() const { return m_analysis; }
  int         getNumNodes() const { return m_numnodes; }
  fem_float*  getNodeCoords() const { return m_nodecoord; }
  int         getNumSupports() const { return m_numsupports; }
  int**       getNodeSupports() const { return m_nodesupport; }
  int         getNumNodalLoads() const { return m_numnodalloads; }
  fem_float** getNodalLoads() const { return m_nodalloads; }
  int         getNumElements() const { return m_numelem; }
  int         getNumElemNodes() const { return m_elemnodes; }
  int*        getElemConnect() const { return m_elemconnect; }

  // Verbosity
  void SetVerbose(bool verbose) { m_verbose = verbose; }

private:
  int extractHeaderAnalysis(std::ifstream &inputStream);
  int extractNodes(std::ifstream &inputStream);
  int extractElementsAndModeldim(std::ifstream &inputStream);
  int extractNodalLoads(std::ifstream &inputStream);
  int extractNodeSupports(std::ifstream &inputStream);
  // Output file error control
  int outIsOK();
  // Given an "ifstream" go the the next line until find the string "param"
  // in the text
  int findParameter(std::ifstream& inputStream, std::string param);
  // Finds string with element type in input stream and returns the string
  std::string findElemString(std::ifstream* inputStream);

  // Frees internal memory allocation
  void FreeFileMembers();
  //-------------------------------------------
  // member variables
  //-------------------------------------------
  // FEM Member Variables
  int         m_modeldim;
  const char* m_analysis;
  int         m_numnodes;
  fem_float*  m_nodecoord;
  int         m_numsupports;
  int**       m_nodesupport;
  int         m_numnodalloads;
  fem_float** m_nodalloads;
  ElemType    m_elemtype;
  int         m_elemnodes;
  int         m_numelem;
  int*        m_elemconnect;

  bool        m_verbose;

  // File Member Vars - filenames do not include directory
  std::string m_inputfilename;
  std::string m_outputfilename;
  FILE*  m_outputfile;
};

#endif  // FILEIO_H_
