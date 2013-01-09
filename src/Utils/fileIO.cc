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

// Headers
#include <ctime>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <errno.h>

#include "fileIO.h"
#include "SPRmatrix/SPRmatrix.h"
#include "Utils/util.h"

using namespace std;

// Constructor
////////////////////////////////////////////////////////////////////////////////
FileIO::FileIO() {
  m_modeldim       = NULL;
  m_analysis       = NULL;
  m_numnodes       = NULL;
  m_nodecoord      = NULL;
  m_numsupports    = NULL;
  m_nodesupport    = NULL;
  m_numnodalloads  = NULL;
  m_nodalloads     = NULL;
  m_elemtype       = UNDEF;
  m_elemnodes      = NULL;
  m_numelem        = NULL;
  m_elemconnect    = NULL;
  m_inputfilename  = "";
  m_outputfilename = "";
  m_outputfile     = NULL;
  m_verbose        = false;
}

// Destructor
////////////////////////////////////////////////////////////////////////////////
FileIO::~FileIO() {
  FreeFileMembers();
  CloseOutputFile();
}

// Extracts information from the neutral file
////////////////////////////////////////////////////////////////////////////////
int FileIO::ReadNF(string neutralfile_name) {
  string auxstring;

  // checks file extension
  if (neutralfile_name.find(".nf", neutralfile_name.size() - 3) == string::npos)
    return 0;

  if (m_inputfilename != "") {
    FreeFileMembers();
    CloseOutputFile();
  }

  cout<< "\n**Reading Neutral File:" << neutralfile_name << endl;
  m_inputfilename = neutralfile_name;

  // opens the input file "filename.dat"
  ifstream inputStream;
  inputStream.open(m_inputfilename.c_str());
  // checks if it opened correctly
  if (!inputStream.is_open()) {
    cerr << "\n*************** ERROR ***************"  << endl;
    cerr << "-> Error opening file " << neutralfile_name << endl;
    system("pause");
    return 0;
  }

  extractElementsAndModeldim(inputStream);
  extractHeaderAnalysis(inputStream);
  extractNodes(inputStream);
  extractNodeSupports(inputStream);
  extractNodalLoads(inputStream);

  cout << "..DONE READING!" << endl;
  if (inputStream)
    inputStream.close();

  if(outIsOK()) {
    writeNFInfo();
  }

  return 1;
}

// Extracts element information
////////////////////////////////////////////////////////////////////////////////
int FileIO::extractElementsAndModeldim(ifstream &inputStream) {
  string elem = findElemString(&inputStream);
  if (elem == "") {
    return 0;
  }
  string elem_brick20_str = "%ELEMENT.BRICK20";
  string elem_brick8_str  = "%ELEMENT.BRICK8";
  string elem_Q4_str      = "%ELEMENT.Q4";
  string elem_Q8_str      = "%ELEMENT.Q8";
  string auxstring;

  // finds %NODE.COORD string
  if (elem == elem_brick20_str) {
    m_elemtype  = BRICK20;
    m_elemnodes = 20;
    m_modeldim  = 3;
  } else if (elem == elem_brick8_str) {
    m_elemtype  = BRICK8;
    m_elemnodes = 8;
    m_modeldim  = 3;
  } else if (elem == elem_Q4_str) {
    m_elemtype  = Q4;
    m_elemnodes = 4;
    m_modeldim  = 2;
  } else if (elem == elem_Q8_str) {
    m_elemtype  = Q8;
    m_elemnodes = 8;
    m_modeldim  = 2;
  } else {
    cerr << "\n*************** ERROR ***************"  << endl;
    cerr << "-> Element Type Not Found " << endl;
    return 0;
  }

  // extracts string with number of elements
  getline(inputStream, auxstring);
  istringstream(auxstring) >> m_numelem;

  // Alloc Node Coordinate Vector
  if (!m_elemconnect)
    m_elemconnect = (int*)malloc(m_elemnodes * m_numelem * sizeof(int));

  if (m_verbose)
    printf("\nNumber of Elements: %i\n", m_numelem);

  for (int i = 0; i < m_numelem; ++i) {
    int* elemAux = (int*)malloc(m_elemnodes * sizeof(int));
    // goes to the next line and creates ISS
    getline(inputStream, auxstring);
    istringstream iss(auxstring);

    // discards first three values
    iss >> elemAux[0];
    iss >> elemAux[0];
    iss >> elemAux[0];
    if ((m_elemtype == Q8) || (m_elemtype == Q4))
      iss >> elemAux[0];
    for (int j = 0 ; j < m_elemnodes; ++j)
      iss >> elemAux[j];

    if (m_verbose) {
      printf("Node ID: %i\n", i+1);
      for (int j = 0; j < m_elemnodes; ++j)
        printf("  Connectivity[%i]: %i\n", j, elemAux[j]);
    }

    // Corrects neutral file numbering
    if (m_elemtype == BRICK20) {
      m_elemconnect[m_elemnodes*i+0 ] = elemAux[0];
      m_elemconnect[m_elemnodes*i+1 ] = elemAux[7];
      m_elemconnect[m_elemnodes*i+2 ] = elemAux[6];
      m_elemconnect[m_elemnodes*i+3 ] = elemAux[11];
      m_elemconnect[m_elemnodes*i+4 ] = elemAux[18];
      m_elemconnect[m_elemnodes*i+5 ] = elemAux[19];
      m_elemconnect[m_elemnodes*i+6 ] = elemAux[12];
      m_elemconnect[m_elemnodes*i+7 ] = elemAux[8];
      m_elemconnect[m_elemnodes*i+8 ] = elemAux[1];
      m_elemconnect[m_elemnodes*i+9 ] = elemAux[5];
      m_elemconnect[m_elemnodes*i+10] = elemAux[17];
      m_elemconnect[m_elemnodes*i+11] = elemAux[13];
      m_elemconnect[m_elemnodes*i+12] = elemAux[2];
      m_elemconnect[m_elemnodes*i+13] = elemAux[3];
      m_elemconnect[m_elemnodes*i+14] = elemAux[4];
      m_elemconnect[m_elemnodes*i+15] = elemAux[10];
      m_elemconnect[m_elemnodes*i+16] = elemAux[16];
      m_elemconnect[m_elemnodes*i+17] = elemAux[15];
      m_elemconnect[m_elemnodes*i+18] = elemAux[14];
      m_elemconnect[m_elemnodes*i+19] = elemAux[9];
    } else if ((m_elemtype == BRICK8) || (m_elemtype == Q8)) {
      m_elemconnect[m_elemnodes*i+0 ] = elemAux[0];
      m_elemconnect[m_elemnodes*i+1 ] = elemAux[1];
      m_elemconnect[m_elemnodes*i+2 ] = elemAux[2];
      m_elemconnect[m_elemnodes*i+3 ] = elemAux[3];
      m_elemconnect[m_elemnodes*i+4 ] = elemAux[4];
      m_elemconnect[m_elemnodes*i+5 ] = elemAux[5];
      m_elemconnect[m_elemnodes*i+6 ] = elemAux[6];
      m_elemconnect[m_elemnodes*i+7 ] = elemAux[7];
    } else if (m_elemtype == Q4) {
      m_elemconnect[m_elemnodes*i+0 ] = elemAux[0];
      m_elemconnect[m_elemnodes*i+1 ] = elemAux[1];
      m_elemconnect[m_elemnodes*i+2 ] = elemAux[2];
      m_elemconnect[m_elemnodes*i+3 ] = elemAux[3];
    } else if (m_elemtype == Q8) {
      m_elemconnect[m_elemnodes*i+0 ] = elemAux[0];
      m_elemconnect[m_elemnodes*i+1 ] = elemAux[1];
      m_elemconnect[m_elemnodes*i+2 ] = elemAux[2];
      m_elemconnect[m_elemnodes*i+3 ] = elemAux[3];
      m_elemconnect[m_elemnodes*i+4 ] = elemAux[4];
      m_elemconnect[m_elemnodes*i+5 ] = elemAux[5];
      m_elemconnect[m_elemnodes*i+6 ] = elemAux[6];
      m_elemconnect[m_elemnodes*i+7 ] = elemAux[7];
    }

    if (m_verbose) {
      printf("Element ID: %i\n", i + 1);
      for (int j = 0 ; j < m_elemnodes; ++j)
        printf("  Connectivity[%i]: %i\n", j, m_elemconnect[m_elemnodes*i+j]);
    }
    free(elemAux);
  }

  return 1;
}

// Extracts element information
////////////////////////////////////////////////////////////////////////////////
int FileIO::extractHeaderAnalysis(ifstream &inputStream) {
  string auxstring;
  if (m_modeldim == 2) {
    findParameter(inputStream, "%HEADER.ANALYSIS");
    getline(inputStream, auxstring);
    string str;
    istringstream(auxstring) >> str;
    m_analysis = str.c_str();
  }

  return 1;
}

// Extracts element information
////////////////////////////////////////////////////////////////////////////////
int FileIO::extractNodes(ifstream &inputStream) {
  string node_coord_str = "%NODE.COORD";
  string s;

  // finds %NODE.COORD string
  findParameter(inputStream, node_coord_str);

  // Gets Number of Nodes
  getline(inputStream, s);
  istringstream(s) >> m_numnodes;

  // Alloc Node Coordinate Vector
  if (!m_nodecoord)
    m_nodecoord = (fem_float*)malloc(3 * m_numnodes * sizeof(fem_float));

  if (m_verbose)
    printf("\nNumber of nodes: %i\n", m_numnodes);

  // Nodes are read (and stored) in order - e.g. m_nodeCoord[4] is a vector
  // with 3 components of the 5th node (0 is 1st, 1 is 2nd, etc...)
  for (int i = 0; i < m_numnodes; ++i) {
    int nodeId = 0;
    // goes to the next line and creates ISS
    getline(inputStream, s);
    istringstream iss(s);

    iss >> nodeId;
    iss >> m_nodecoord[m_modeldim*i];
    iss >> m_nodecoord[m_modeldim*i+1];
    if ( m_modeldim == 3 )
      iss >> m_nodecoord[m_modeldim*i+2];

    if (m_verbose) {
      printf("Node ID: %i\n", nodeId);
      printf("  Coord x: %.4f\n", m_nodecoord[3*i  ]);
      printf("  Coord y: %.4f\n", m_nodecoord[3*i+1]);
      printf("  Coord z: %.4f\n", m_nodecoord[3*i+2]);
    }
  }

  return 1;
}

// Extracts nodal loads(forces)
////////////////////////////////////////////////////////////////////////////////
int FileIO::extractNodalLoads(ifstream &inputStream) {
  string nodal_force_case_str = "%LOAD.CASE.NODAL.FORCE";
  string s;

  if (m_nodalloads) {
    freeInnerVectorsF(m_nodalloads, m_numnodalloads);
  }
  // finds %NODE.COORD string
  findParameter(inputStream, nodal_force_case_str);

  // extracts string with number of support nodes
  getline(inputStream, s);
  istringstream(s) >> m_numnodalloads;

  // Alloc NodeId + Loads Vector
  m_nodalloads = (fem_float**)malloc(m_numnodalloads * sizeof(fem_float*));
  for (int i = 0; i < m_numnodalloads; i++) {
    m_nodalloads[i] = NULL;
  }

  if (m_numnodalloads == 0) {
    printf("\n** FileIO: NO FORCES APPLIED TO MODEL **\n");
  }

  if (m_verbose)
    printf("\nNumber of Nodes w/ Nodal Loads: %i\n", m_numnodalloads);

  // Gets nodal load info w/ first element node then x, y, z global components
  for (int i = 0; i < m_numnodalloads; ++i) {
    fem_float* nodalLoads = (fem_float*)malloc((m_modeldim + 1) *
      sizeof(fem_float));

    // extracts nodal loads from iss string stream
    getline(inputStream, s);
    istringstream iss(s);
    for (int j = 0 ; j < 1 + m_modeldim; ++j) {
      float node_and_loadsxyz;
      iss >> node_and_loadsxyz;
      nodalLoads[j] = node_and_loadsxyz;
    }

    // If in verbose mode prints data
    if (m_verbose) {
      printf("Nodal Load (Node) ID: %i\n", (int)nodalLoads[0]);
      printf("  POS: x=%f, y=%f, z=%f\n", nodalLoads[1],
        nodalLoads[2], nodalLoads[3]);
    }

    m_nodalloads[i] = nodalLoads;
  }

  return 1;
}

// Extracts nodal supports
////////////////////////////////////////////////////////////////////////////////
int FileIO::extractNodeSupports(ifstream &inputStream) {
  string node_support_str = "%NODE.SUPPORT";
  string s;

  // finds %NODE.COORD string
  findParameter(inputStream, node_support_str);

  // extracts string with number of support nodes
  getline(inputStream, s);
  istringstream(s) >> m_numsupports;

  // Alloc NodeId + Supports Vector
  if (!m_nodesupport)
    m_nodesupport = (int**)malloc(m_numsupports * sizeof(int*));
  for (int i = 0; i < m_numsupports; i++) {
    m_nodesupport[i] = NULL;
  }

  if (m_verbose)
    printf("\nNumber of Nodes w/ Support: %i\n", m_numsupports);

  // Gets support info w/ first element node, rest is bool support condition
  for (int i = 0; i < m_numsupports; ++i) {
    int* nodeSupport = (int*)malloc((m_modeldim + 1) * sizeof(int));

    // goes to the next line and creates ISS
    getline(inputStream, s);
    istringstream iss(s);

    for (int j = 0 ; j < (m_modeldim + 1); ++j)
      iss >> nodeSupport[j];

    if (m_verbose) {
      printf("Support Node ID: %i\n", nodeSupport[0]);
      printf("  Bool Support: x=%i, y=%i, z=%i\n", nodeSupport[1],
        nodeSupport[2], nodeSupport[3]);
    }

    m_nodesupport[i] = nodeSupport;
  }

  return 1;
}

// Closes Output File
////////////////////////////////////////////////////////////////////////////////
int FileIO::OpenOutputFile(std::string out_filename) {
  m_outputfilename = out_filename;

  if (m_outputfile)
    CloseOutputFile();

  int err = fopen_s(&m_outputfile, m_outputfilename.c_str() , "w");
  if (err != 0)
    return 0;

  // Write creation time
  time_t t = time(0);
  struct tm* datestruct = localtime(&t);
  writeString("FEMGPU Output File\n");
  writeString("Date: ");
  writeString(asctime(datestruct));
  writeNewLine();

  return 1;
}

// Closes Output File
////////////////////////////////////////////////////////////////////////////////
int FileIO::CloseOutputFile() {
  if (m_outputfile) {
    fclose(m_outputfile);
    m_outputfile = NULL;
  }
  return 1;
}

// Error control for output file writing - checks if output exists
////////////////////////////////////////////////////////////////////////////////
int FileIO::outIsOK() {
  if (m_outputfile == NULL)
    return 0;

  return 1;
}

// Writes tab delimited information to be exported
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeOutputHeader() {
  writeStringTab("DOF");
  writeStringTab("TIME");
  writeStringTab("ELEM");
  writeStringTab("NODES");
  writeStringTab("SIZE");
  writeNewLine();

  return 1;
}

// Writes neutral file data to output file
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeNFInfo() {
  return 1;
}

// Writes a string to output file and goes to new line
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeString(const char* str) {
  if (!outIsOK())
    return 0;
  fprintf(m_outputfile, "%s", str);

  return 1;
}

// Writes a string to output file and goes to new line
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeString(string str) {
  if (!outIsOK())
    return 0;
  fprintf(m_outputfile, "%s", str);

  return 1;
}

// Writes a string to output file followed by tab
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeStringTab(const char* str) {
  if (!outIsOK())
    return 0;
  fprintf(m_outputfile, "%s\t", str);

  return 1;
}

// Writes a string to output file followed by tab
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeStringTab(string str) {
  if (!outIsOK())
    return 0;
  fprintf(m_outputfile, "%s\t", str);

  return 1;
}

// Writes a number to output file followed by tab
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeNumTab(fem_float num) {
  if (!outIsOK())
    return 0;
  fprintf(m_outputfile, "%.2f\t", num);

  return 1;
}

// Writes a number to output file followed by tab
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeNumTab(double num) {
  if (!outIsOK())
    return 0;
  fprintf(m_outputfile, "%.2f\t", num);

  return 1;
}

// Writes a number to output file followed by tab
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeNumTab(int num) {
  if (!outIsOK())
    return 0;
  fprintf(m_outputfile, "%i\t", num);

  return 1;
}

// Writes a number to output file followed by tab
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeNewLine() {
  if (!outIsOK())
    return 0;
  fprintf(m_outputfile, "\n");

  return 1;
}

// Writes a Matrix to Output File
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeMatrix(SPRmatrix* matrix, int m, int n) {
  if (!outIsOK())
    return 0;

  int i, j;
  // Matrix row numbering
  fprintf(m_outputfile, "\nMatrix(%i,%i) Dump:\n ", m, n);
  for (i = 0; i < n; ++i)
    fprintf(m_outputfile, "--%03i--+", i);
  fprintf(m_outputfile, "\n ");
  for (i = 0; i < n; ++i)
    fprintf(m_outputfile, "=======+");

  // Prints actual matrix
  fprintf(m_outputfile, "\n[ ");
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j)
      fprintf(m_outputfile, "%+7.2f ", matrix->GetElem(i, j));
    fprintf(m_outputfile, "%s", (i != (m-1)) ? "\n  " : "]\n");
  }

  return 1;
}

// Writes a Matrix to Output In Format Nonzero = 1
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeMatrixNonzeros(SPRmatrix* matrix, int m, int n) {
  if (!outIsOK())
    return 0;

  int i, j;
  // Matrix row numbering
  fprintf(m_outputfile, "\nMatrix(%i,%i) Dump:\n ", m, n);
  for (i = 0; i < n; ++i)
    fprintf(m_outputfile, "%2i", i);

  int nnz = 0;
  // Prints actual matrix
  fprintf(m_outputfile, "\n[ ");
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      int val;
      if (matrix->GetElem(i, j) == 0) {
        val = 0;
      } else {
        val = 1;
        nnz++;
      }
      fprintf(m_outputfile, "%i ", val);
    }
    fprintf(m_outputfile, "%s", (i != (m-1)) ? "\n  " : "]\n");
  }
  fprintf(m_outputfile, "Total NNZ: %i\n", nnz);

  return 1;
}

// Writes a Matrix to Output File
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeMatrix2(fem_float* matrix, int m, int n) {
  if (!outIsOK())
    return 0;

  int i, j;
  // Matrix row numbering
  fprintf(m_outputfile, "\nMatrix Dump:\n ");
  for (i = 0; i < n; ++i)
    fprintf(m_outputfile, "-%03i-+", i);
  fprintf(m_outputfile, "\n ");
  for (i = 0; i < n; ++i)
    fprintf(m_outputfile, "=====+");

  // Prints actual matrix
  fprintf(m_outputfile, "\n[ ");
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j)
      fprintf(m_outputfile, "% 5.2f ", matrix[i*n+j]);
    fprintf(m_outputfile, "%s", (i != (m-1)) ? "\n  " : "]\n");
  }

  return 1;
}

// Writes a Matrix to Output File
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeMatrix3(fem_float** matrix, int m, int n) {
  if (!outIsOK())
    return 0;

  int i, j;
  // Matrix row numbering
  fprintf(m_outputfile, "\nMatrix Dump:\n ");
  for (i = 0; i < n; ++i)
    fprintf(m_outputfile, "-%03i-+", i);
  fprintf(m_outputfile, "\n ");
  for (i = 0; i < n; ++i)
    fprintf(m_outputfile, "=====+");

  // Prints actual matrix
  fprintf(m_outputfile, "\n[ ");
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j)
      fprintf(m_outputfile, "% 5.2f ", matrix[i][j]);
    fprintf(m_outputfile, "%s", (i != (m-1)) ? "\n  " : "]\n");
  }

  return 1;
}

// Writes a Vector to Output File
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeVector(fem_float* vector, int n) {
  if (!outIsOK())
    return 0;

  int i;

  fprintf(m_outputfile, "\nVector Dump:\n");
  fprintf(m_outputfile, "DOF -    Value\n");
  fprintf(m_outputfile, "==============\n");
  fprintf(m_outputfile, "  1 -[% 9.4f\n", vector[0]);
  for (i = 1; i < (n-1); ++i) {
    fprintf(m_outputfile, "%3i - % 9.4f\n", (i+1), vector[i]);
  }
  fprintf(m_outputfile, "%3i - % 9.4f ]\n", (i+1), vector[n-1]);

  return 1;
}

// Writes a 3xN matrix to Output File
////////////////////////////////////////////////////////////////////////////////
int FileIO::writeXYZVector(fem_float* u_vector, int n) {
  if (!outIsOK())
    return 0;

  int i, j;
  // Matrix row numbering
  fprintf(m_outputfile, "\nDisplacement Vector Dump:\n ");
  fprintf(m_outputfile, "Node+===x===+===y===+===z===+");
  // Prints actual matrix
  fprintf(m_outputfile, "\n");
  for (i = 0; i < (n/3); ++i) {
    fprintf(m_outputfile, "%3i -", (i+1));
    for (j = 0; j < 3; ++j)
      fprintf(m_outputfile, " %7.2f", u_vector[(3*i)+j]);
    fprintf(m_outputfile, "\n");
  }

  return 1;
}

//---------------------------
// PRIVATE METHODS
//---------------------------

// Searches for a paramater string given the ifstream
////////////////////////////////////////////////////////////////////////////////
int FileIO::findParameter(ifstream& _inputStream, string _param) {
  // cout << " -> searching parameter = " << _param ;
  string s;
  size_t found, foundEND;
  bool secondtry = false;

  while (getline(_inputStream, s)) {
    // Verifies which line that starts with '%' and contains the specified
    // 'param'
    found = s.find(_param);
    if (found != s.npos) {
      return 1;
    }

    foundEND = s.find("%END");
    if (foundEND != s.npos) {
      // Tries to read the file again in case the parameter was in a part of the
      // buffer that was already removed
      if (secondtry == true) {
        return 0;
      }
      if (m_verbose) {
        cout << "  FOUND %END BEFORE PARAMETER: " << _param << endl;
        cout << "  *** Reopening File  *** " << endl;
      }
      secondtry = true;
      _inputStream.close();
      _inputStream.open(m_inputfilename.c_str());
    }
  }

  return 1;
}

// Finds string with element type in input stream and returns the string
////////////////////////////////////////////////////////////////////////////////
string FileIO::findElemString(ifstream* _inputStream) {
  string elem_str = "%ELEMENT.";
  string s = "";
  bool secondtry = false;
  size_t found, foundEND;

  while (getline((*_inputStream), s)) {
    found = s.find(elem_str);
    if (found != s.npos) {
      return s;
    }

    foundEND = s.find("%END");
    if (foundEND != s.npos) {
      // Tries to read the file again in case the parameter was in a part of the
      // buffer that was already removed
      if (secondtry == true) {
        return 0;
      }
      secondtry = true;
      _inputStream->close();
      _inputStream->open(m_inputfilename.c_str());
    }
  }

  return NULL;
}

// Frees memory allocated by fileIO
////////////////////////////////////////////////////////////////////////////////
void FileIO::FreeFileMembers() {
  printf("Freeing File members....\n");
  // Cleanup
  if (m_nodecoord) {
    free(m_nodecoord);
    m_nodecoord = NULL;
  }
  if (m_elemconnect) {
    free(m_elemconnect);
    m_elemconnect = NULL;
  }
  freeInnerVectorsI(m_nodesupport, m_numsupports);
  m_nodesupport = NULL;
  freeInnerVectorsF(m_nodalloads, m_numnodalloads);
  m_nodalloads = NULL;
}
