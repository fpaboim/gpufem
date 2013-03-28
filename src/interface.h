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

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <assert.h>
#include <omp.h>

#include "Lua/LuaWrap.h"
#include "FEM/fem.h"
#include "FEM/FemData.h"
#include "SPRmatrix/SPRmatrix.h"
#include "Utils/fileIO.h"
#include "OpenCL/OCLwrapper.h"

typedef enum BatchMode {
  INTERFACE = 1,
  ASEMBATCH = 2,
  SOLVBATCH = 3,
} BatchMode;

// Analysis Interface Functions
////////////////////////////////////////////////////////////////////////////////
int LuaRegisterCFunctions(lua_State* L);

static int lua_RunAnalysis(lua_State* L);

static int lua_GetMaxThreads(lua_State* L);

void outfileInit(FileIO* Filehandler,
  std::string outfile,
  bool append,
  int gausspts,
  bool iscoloring,
  SPRmatrix::SPRformat sprseformat);

void outfileWriteHeader(FileIO* Filehandler,
  int nthreads,
  bool writesolheader);

void outfileWriteGptsColoring(FileIO* Filehandler, int gausspts,
  bool iscoloring);

void outfileWriteResults(const bool isbatch,
  const bool solve,
  FileIO* Filehandler,
  FemData* femdata,
  float matsizeMB,
  double tstiff,
  double tsolve);

void outfileWriteBatchResults(const bool isbatch,
  const bool solve,
  FileIO* Filehandler,
  FemData* femdata,
  float matsizeMB,
  double tstiffcpu,
  double tstiffgpu,
  double tsolvecpu,
  double tsolvegpu);

void colorMesh(FemData* femdata, bool usennz);

double solveDisplacements(FEM::DeviceMode deviceType, FemData* femdata);

void printAnalysisEnd();

int RunAnalysis(std::vector<std::string> files,
  SPRmatrix::SPRformat sprseformat,
  FEM::DeviceMode deviceType,
  const int numCPUthreads,
  const int gausspts,
  const fem_float Emod,
  const fem_float Nucoef,
  const bool usecolor,
  const bool makenodal,
  const bool printstiff,
  const bool solve,
  const bool view,
  std::string outfile,
  const bool appendmode);

int RunAsmBatchAnalysis(std::vector<std::string> files,
  SPRmatrix::SPRformat sprseformat,
  FEM::DeviceMode deviceType,
  const int numCPUthreads,
  const int gausspts,
  const fem_float Emod,
  const fem_float Nucoef,
  const bool usecolor,
  const bool makenodal,
  const bool printstiff,
  const bool solve,
  const bool view,
  std::string outfile,
  const bool appendmode);

int RunSolBatchAnalysis(std::vector<std::string> files,
  SPRmatrix::SPRformat sprseformat,
  FEM::DeviceMode deviceType,
  const int numCPUthreads,
  int gausspts,
  const fem_float Emod,
  const fem_float Nucoef,
  const bool usecolor,
  const bool makenodal,
  const bool printstiff,
  const bool solve,
  const bool view,
  std::string outfile,
  bool appendmode);

int RunAsmBatches(std::vector<std::string> files,
  SPRmatrix::SPRformat sprseformat,
  FEM::DeviceMode deviceType,
  const int numCPUthreads,
  const int gausspts,
  const fem_float Emod,
  const fem_float Nucoef,
  const bool usecolor,
  const bool makenodal,
  const bool printstiff,
  const bool solve,
  const bool view,
  std::string outfile,
  const bool appendmode);