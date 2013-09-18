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

#include "iup.h"
#include "iuplua.h"
#include "iupcontrols.h"
#include "iupluacontrols.h"

#include "Lua/LuaWrap.h"
#include "Utils/fileIO.h"
#include "FEM/fem.h"
#include "FEM/FemData.h"
#include "FEM/femColor.h"
#include "LAops/SprSolver.h"
#include "SPRmatrix/SPRmatrix.h"
#include "OpenCL/OCLwrapper.h"
#include "Vis/Vis.h"

#include "interface.h"

// LuaRegisterCFunctions: C function to be registered and called by Lua code
////////////////////////////////////////////////////////////////////////////////
int LuaRegisterCFunctions(lua_State* L) {
  static const struct luaL_Reg FEMlib [] = {
    {"GetMaxThreads", lua_GetMaxThreads},
    {"RunAnalysis",   lua_RunAnalysis},
    {NULL,            NULL} //sentinel value
  };
  luaL_register(L, "FEMlib", FEMlib);

  return 1;
}

// lua_GetMaxThreads: Gets OMP maximum number of threads in PC
////////////////////////////////////////////////////////////////////////////////
static int lua_GetMaxThreads(lua_State* L) {
  int maxthreads = omp_get_max_threads();
  lua_pushnumber(L, maxthreads);
  return 1;
}

// lua_RunAnalysis: C function to be registered and called by Lua code
//   Lua Function:
//   FEMlib.RunAnalysis(files, matfmt, ocl, nthreads, gausspts, color,
//                      solve, view, outfile)
////////////////////////////////////////////////////////////////////////////////
static int lua_RunAnalysis(lua_State* L) {
  int narg = lua_gettop(L);
  assert(narg == 11);
  using namespace std;

  fem_float Emod  = 100.0f;
  fem_float Nu    = 0.25f;
  bool makenodal  = true;
  bool printstiff = false;

  int arg = 1;
  // Table with names of files to execute
  if (!lua_istable(L, arg)) {
    lua_pushstring(L, "Error getting file list table");
    lua_error(L);
  }
  vector<string> files;
  int nfiles = lua_objlen(L, arg);
  for (int i = 1; i <= nfiles; ++i) {
    lua_rawgeti(L, arg, i);
    if (!lua_isstring(L, -1)) {
      lua_pushstring(L, "Error getting file string");
      lua_error(L);
    }
    const char* fileptr = lua_tolstring(L, -1, NULL);
    string file(fileptr);
    files.push_back(file);
    lua_pop(L, 1);
  } arg++;
  // Matrix Format (1-Dense, 2-Diagonal, 3- CSR, 4-EllPack, 5- Ellpack2,6-Eigen)
  if (!lua_isnumber(L, arg)) {
    lua_pushstring(L, "Error getting matrix format");
    lua_error(L);
  }
  int matnum = lua_tonumber(L, arg); arg++;
  SPRmatrix::SPRformat matfmt = (SPRmatrix::SPRformat)matnum;
  // Device (0-CPU; 1-GPU)
  if (!lua_isboolean(L, arg)) {
    lua_pushstring(L, "Error getting device type");
    lua_error(L);
  }
  int device = lua_toboolean(L, arg); arg++;
  FEM::DeviceMode devicemode = (FEM::DeviceMode)device;
  // Number of Threads
  if (!lua_isnumber(L, arg)) {
    lua_pushstring(L, "Error getting number of threads");
    lua_error(L);
  }
  int nthreads = lua_tonumber(L, arg); arg++;
  // Gauss Points
  if (!lua_isnumber(L, arg)) {
    lua_pushstring(L, "Error getting gauss points");
    lua_error(L);
  }
  int gausspts = lua_tonumber(L, arg); arg++;
  // Mesh Coloring
  if (!lua_isboolean(L, arg)) {
    lua_pushstring(L, "Error getting use mesh coloring");
    lua_error(L);
  }
  bool color = (lua_toboolean(L, arg) == 1); arg++;
  // Solve for Displacements
  if (!lua_isboolean(L, arg)) {
    lua_pushstring(L, "Error getting solve for displacements");
    lua_error(L);
  }
  bool solve = (lua_toboolean(L, arg) == 1); arg++;
  // View Results with Vtk
  if (!lua_isboolean(L, arg)) {
    lua_pushstring(L, "Error getting view results");
    lua_error(L);
  }
  bool view  = (lua_toboolean(L, arg) == 1); arg++;
  // Set output file
  if (!lua_isstring(L, arg)) {
    lua_pushstring(L, "Error getting output file");
    lua_error(L);
  }
  std::string outfile = lua_tostring(L, arg); arg++;
  // Output file append or overwrite mode
  if (!lua_isboolean(L, arg)) {
    lua_pushstring(L, "Error getting view results");
    lua_error(L);
  }
  bool appendmode = (lua_toboolean(L, arg) == 1); arg++;
  // Batch mode
  if (!lua_isnumber(L, arg)) {
    lua_pushstring(L, "Error getting batch mode");
    lua_error(L);
  }
  int bmode = lua_tonumber(L, arg); arg++;

  BatchMode itfbatchmode = (BatchMode) bmode;
  switch (itfbatchmode) {
    // Analysis using interface values
    case INTERFACE:
      RunAnalysis(files, matfmt, devicemode, nthreads, gausspts, Emod, Nu,
                  color, makenodal, printstiff, solve, view, outfile,
                  appendmode);
      break;
    // Analysis using ASSEMBLY batch benchmarks
    case ASEMBATCH:
      // loops varying numbers of threads (1, 2, 4, 8.. maxthreads)
      RunAsmBatchAnalysis(files, matfmt, devicemode, nthreads, gausspts, Emod,
                          Nu, color, makenodal, printstiff, solve, view,
                          outfile, appendmode);
      break;
    // Analysis using SOLVER batch benchmarks
    case SOLVBATCH:
      RunSolBatchAnalysis(files, matfmt, devicemode, nthreads, gausspts, Emod,
                          Nu, color, makenodal, printstiff, true, view,
                          outfile, appendmode);
      break;
    default:  //default falls back to CPU
      assert(false);
  }

  // function returns to lua 0 results
  return 0;
}

// outfileInit: Initializes output file for dumping output
////////////////////////////////////////////////////////////////////////////////
void outfileInit(FileIO* Filehandler,
                 std::string outfile,
                 bool append,
                 int gausspts,
                 bool iscoloring,
                 SPRmatrix::SPRformat sprseformat) {
  Filehandler->OpenOutputFile(outfile, append);
  if (append == false) {
    outfileWriteGptsColoring(Filehandler, gausspts, iscoloring);
  }
  Filehandler->writeMatFormat(sprseformat);
}

// outfileWriteGptsColoringHeader: writes gausspts and coloring state to header
////////////////////////////////////////////////////////////////////////////////
void outfileWriteGptsColoring(FileIO* Filehandler, int gausspts,
                              bool iscoloring) {
  Filehandler->writeString("Gps:");
  Filehandler->writeNumTab(gausspts);
  Filehandler->writeString("Coloring:");
  Filehandler->writeNumTab((int)iscoloring);
  Filehandler->writeNewLine();
}

// outfileWriteHeader: Writes the header in the output file
////////////////////////////////////////////////////////////////////////////////
void outfileWriteHeader(FileIO* Filehandler,
                        int nthreads,
                        bool writesolheader) {
  Filehandler->writeString("Threads:");
  Filehandler->writeNumTab(nthreads);
  Filehandler->writeNewLine();
  if (writesolheader) {
    Filehandler->writeSolOutputHeader();
  } else {
    Filehandler->writeAsmOutputHeader();
  }
}

// outfileWriteResults: Writes results depending if solving or not
////////////////////////////////////////////////////////////////////////////////
void outfileWriteResults(const bool solve,
                         FileIO* Filehandler,
                         FemData* femdata,
                         float matsizeMB,
                         double tstiff,
                         double tsolve) {
  // Write to output file
  Filehandler->writeNumTab(femdata->GetNumDof());
  Filehandler->writeNumTab(femdata->GetNumNodes());
  Filehandler->writeNumTab(femdata->GetNumElem());
  Filehandler->writeNumTab(matsizeMB);
  Filehandler->writeNumTab(tstiff);
  if (solve) {
    Filehandler->writeNumTab(tsolve);
  }
  Filehandler->writeNewLine();
}

// outfileWriteAsmResults: Writes assembly benchmark results to file
////////////////////////////////////////////////////////////////////////////////
void outfileWriteAsmBatchResults(FileIO* Filehandler,
                                 FemData* femdata,
                                 float matsizeMB,
                                 double tstiffcpu,
                                 double tstiffgpu) {
  // Write to output file
  Filehandler->writeNumTab(femdata->GetNumDof());
  Filehandler->writeNumTab(femdata->GetNumNodes());
  Filehandler->writeNumTab(femdata->GetNumElem());
  Filehandler->writeNumTab(matsizeMB);
  Filehandler->writeNumTab(tstiffcpu);
  Filehandler->writeNumTab(tstiffgpu);
  Filehandler->writeNewLine();
}

// outfileWriteSolResults: Writes solver benchmark results to file
////////////////////////////////////////////////////////////////////////////////
void outfileWriteSolBatchResults(FileIO* Filehandler,
                                 FemData* femdata,
                                 float matsizeMB,
                                 double tstiffcpu,
                                 double tstiffgpu,
                                 double tsolvecpu,
                                 double tsolveomp,
                                 double tsolvegpu_naive,
                                 double tsolvegpu_naiveur,
                                 double tsolvegpu_share,
                                 double tsolvegpu_blk,
                                 double tsolvegpu_blkur) {
  // Write to output file
  Filehandler->writeNumTab(femdata->GetNumDof());
  Filehandler->writeNumTab(femdata->GetNumElem());
  Filehandler->writeNumTab(matsizeMB);
  Filehandler->writeNumTab(tstiffcpu);
  Filehandler->writeNumTab(tstiffgpu);
  Filehandler->writeNumTab(tsolvecpu);
  Filehandler->writeNumTab(tsolvegpu_naive);
  Filehandler->writeNumTab(tsolvegpu_naiveur);
  Filehandler->writeNumTab(tsolvegpu_share);
  Filehandler->writeNumTab(tsolvegpu_blk);
  Filehandler->writeNumTab(tsolvegpu_blkur);
  Filehandler->writeNewLine();
}
// colorMesh: performs mesh coloring and updates femdata with info
////////////////////////////////////////////////////////////////////////////////
void colorMesh( FemData* femdata, bool usennz ) {
  femColor* mshColorObj = new femColor();
  double t1 = omp_get_wtime();
  mshColorObj->makeMetisGraph(femdata, usennz);
  double t2 = omp_get_wtime();
  cout << "Connect Graph Build Time: " << (t2-t1) << endl;
  mshColorObj->MakeGreedyColoring(femdata);
  double t3 = omp_get_wtime();
  int metisnnz, band;
  if (usennz) {
    mshColorObj->CalcNNZ(femdata, metisnnz, band);
    femdata->GetStiffnessMatrix()->SetNNZInfo(metisnnz, band);
    printf("Number of nonzeros: %i\n", metisnnz);
  }
  double t4 = omp_get_wtime();

  if (false) {
    printf("Number of colors: %i\n", mshColorObj->GetNumColors());
    printf("Metis element (+nodal) graph time: %3.4f\n", (t2-t1));
    printf("Greedy coloring time: %3.4f\n", (t3-t2));
    if (usennz) {
      printf("NNZ/Band counting time: %3.4f\n", (t4-t3));
    }
    printf("+ Total coloring time: %3.4f\n", (t4-t1));
  }
  delete(mshColorObj);
}

// solveDisplacements: Solves linear syst. Ku = f for u (displacements)
////////////////////////////////////////////////////////////////////////////////
double solveDisplacements(FEM::DeviceMode deviceType, FemData* femdata) {
  printf("..Solving for displacements:\n");
  double tsolve = omp_get_wtime();
  if (deviceType == FEM::CPU) {
    femdata->GetStiffnessMatrix()->SetDeviceMode(SPRmatrix::DEV_CPU);
    femdata->GetStiffnessMatrix()->CG(
      femdata->GetDisplVector(),
      femdata->GetForceVector(),
      3000,
      0.001f);
    tsolve = omp_get_wtime() - tsolve;
  } else {
    femdata->GetStiffnessMatrix()->SetDeviceMode(SPRmatrix::DEV_GPU);
    femdata->GetStiffnessMatrix()->SetOclLocalSize(8);
    femdata->GetStiffnessMatrix()->SetOclStrategy(SPRmatrix::STRAT_NAIVE);
    femdata->GetStiffnessMatrix()->CG(
      femdata->GetDisplVector(),
      femdata->GetForceVector(),
      3000,
      0.001f);
    tsolve = omp_get_wtime() - tsolve;
  }
  // printVectorf(femdata->GetDisplVector(), ndof);
  return tsolve;
}

// printAnalysisEnd: Prints done message and separator
////////////////////////////////////////////////////////////////////////////////
void printAnalysisEnd() {
  printf("\n***************************************************************\n");
  printf("** DONE! **\n");
  printf("***************************************************************\n");
}

// RunAnalysis: Runs FEM analysis according to options
////////////////////////////////////////////////////////////////////////////////
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
                const bool appendmode) {
  omp_set_num_threads(numCPUthreads);
  omp_set_dynamic(0);
  bool usennz = false;
  FileIO* Filehandler = new FileIO();
  outfileInit(Filehandler, outfile, appendmode, gausspts, usecolor, sprseformat);
  outfileWriteHeader(Filehandler, numCPUthreads, solve);
  printf("* OMP Number of Threads: %i**\n", omp_get_max_threads());
  std::vector<std::string>::iterator vecitr;
  for (vecitr = files.begin(); vecitr != files.end(); ++vecitr) {
    FEM* FEM_test = new FEM();
    FEM_test->Init(true, deviceType);
    int err = Filehandler->ReadNF(*vecitr);
    if (err == FEM_ERR) {
      return 0;
    }
    // Sets-up finite element info
    FemData* femdata = FEM_test->GetFemData();
    femdata->Init(sprseformat,
                  gausspts,
                  Emod,
                  Nucoef,
                  Filehandler);
    // Preprocessing element coloring
    if (usecolor) {
      colorMesh(femdata, usennz);
    }
    // Gets global stiffness matrix using selected device
    FEM_test->SetUseColoring(usecolor);
    FEM_test->SetDeviceMode(deviceType);
    femdata->GetStiffnessMatrix()->Clear();
    double tstiff = FEM_test->CalcStiffnessMat();

    // Print Stiffness Matrix to default output if option selected
    if (printstiff) {
      Filehandler->writeMatrixNonzeros(femdata->GetStiffnessMatrix(),
                                       femdata->GetNumDof(),
                                       femdata->GetNumDof());
    }
    printf("..Applying Constraints:\n");
    FEM_test->ApplyConstraint(FEM::PEN,
                              Filehandler->getNumSupports(),
                              Filehandler->getNodeSupports());
    // Solves linear system for displacements
    double tsolve = 0;
    if (solve) {
      tsolve = solveDisplacements(deviceType, femdata);
    }
    // Calculation info
    float matsizeMB =
      (float) femdata->GetStiffnessMatrix()->GetMatSize() / (1024 * 1024);
    printf("^ Matrix size: %.2fMB\n", matsizeMB);
    // Visualization - setup VTK through Vis class
    if (view) {
      Vis femVis;
      femVis.BuildMesh(femdata);
    }
    outfileWriteResults(solve, Filehandler, femdata, matsizeMB, tstiff, tsolve);
    delete(FEM_test);
  }
  Filehandler->CloseOutputFile();
  delete(Filehandler);
  printAnalysisEnd();

  return 1;
}

// RunAsmBatchAnalysis: Runs FEM assembly batch analysis for benchmarking
////////////////////////////////////////////////////////////////////////////////
int RunAsmBatchAnalysis(std::vector<std::string> files,
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
                        bool appendmode) {
  // 2 Gausspts
  gausspts = 2;
  printf("Running CSR matrix, GPTS:%i/n", gausspts);
  sprseformat = SPRmatrix::CSR;
  RunBatches(files, sprseformat, deviceType, numCPUthreads, gausspts, Emod,
    Nucoef, usecolor, makenodal, printstiff, solve, view, outfile, appendmode);
  appendmode = true; // first overwrite then append to file
  printf("Running ELL matrix, GPTS:%i/n", gausspts);
  sprseformat = SPRmatrix::ELL;
  RunBatches(files, sprseformat, deviceType, numCPUthreads, gausspts, Emod,
    Nucoef, usecolor, makenodal, printstiff, solve, view, outfile, appendmode);
  printf("Running EIG matrix, GPTS:%i/n", gausspts);
  sprseformat = SPRmatrix::EIG;
  RunBatches(files, sprseformat, deviceType, numCPUthreads, gausspts, Emod,
    Nucoef, usecolor, makenodal, printstiff, solve, view, outfile, appendmode);
  // 3 Gausspts
  gausspts = 3;
  FileIO* filehand = new FileIO();
  outfileInit(filehand, outfile, true, gausspts, usecolor, SPRmatrix::NIL);
  outfileWriteGptsColoring(filehand, gausspts, usecolor);
  delete(filehand);
  printf("Running CSR matrix, GPTS:%i/n", gausspts);
  sprseformat = SPRmatrix::CSR;
  RunBatches(files, sprseformat, deviceType, numCPUthreads, gausspts, Emod,
    Nucoef, usecolor, makenodal, printstiff, solve, view, outfile, appendmode);
  printf("Running ELL matrix, GPTS:%i/n", gausspts);
  sprseformat = SPRmatrix::ELL;
  RunBatches(files, sprseformat, deviceType, numCPUthreads, gausspts, Emod,
    Nucoef, usecolor, makenodal, printstiff, solve, view, outfile, appendmode);
  printf("Running EIG matrix, GPTS:%i/n", gausspts);
  sprseformat = SPRmatrix::EIG;
  RunBatches(files, sprseformat, deviceType, numCPUthreads, gausspts, Emod,
    Nucoef, usecolor, makenodal, printstiff, solve, view, outfile, appendmode);

  return 1;
}

// RunAsmBatchAnalysis: Runs FEM assembly batch analysis for benchmarking
////////////////////////////////////////////////////////////////////////////////
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
                        bool appendmode) {
  // 2 Gausspts
  gausspts = 2;
  printf("Running ELL matrix, GPTS:%i/n", gausspts);
  sprseformat = SPRmatrix::ELL;
  RunBatches(files, sprseformat, deviceType, numCPUthreads, gausspts, Emod,
    Nucoef, usecolor, makenodal, printstiff, solve, view, outfile, appendmode);
  appendmode = true; // first overwrite then append to file
  // 3 Gausspts
  gausspts = 3;
  printf("Running ELL matrix, GPTS:%i/n", gausspts);
  sprseformat = SPRmatrix::ELL;
  RunBatches(files, sprseformat, deviceType, numCPUthreads, gausspts, Emod,
    Nucoef, usecolor, makenodal, printstiff, solve, view, outfile, appendmode);

  return 1;
}

// RunAsmBatchAnalysis: Runs FEM assembly batch analysis for benchmarking
////////////////////////////////////////////////////////////////////////////////
int RunBatches(std::vector<std::string> files,
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
                  const bool appendmode) {
  omp_set_dynamic(0);
  bool usennz   = false;
  bool writesol = false;

  FileIO* Filehandler = new FileIO();
  outfileInit(Filehandler, outfile, appendmode, gausspts, usecolor, sprseformat);
  int maxthreads = omp_get_num_procs();
  // Loops over all input files doing FEM calculations
  for (int threads = 1; threads <= maxthreads; threads *= 2) {
    omp_set_num_threads(threads);
    printf("** OMP Number of Threads: %i**\n", omp_get_max_threads());
    outfileWriteHeader(Filehandler, threads, solve);
    std::vector<std::string>::iterator vecitr;
    for (vecitr = files.begin(); vecitr != files.end(); ++vecitr) {
      FEM* FEM_test = new FEM();
      FEM_test->Init(true, deviceType);
      int err = Filehandler->ReadNF(*vecitr);
      if (err == FEM_ERR) {
        return 0;
      }
      // Sets-up finite element info
      FemData* femdata = FEM_test->GetFemData();
      femdata->Init(sprseformat,
                    gausspts,
                    Emod,
                    Nucoef,
                    Filehandler);
      // Preprocessing element coloring
      if (usecolor) {
        colorMesh(femdata, usennz);
      }
      // Gets global stiffness matrix using selected device
      FEM_test->SetUseColoring(usecolor);
      FEM_test->SetDeviceMode(FEM::CPU);
      femdata->GetStiffnessMatrix()->Clear();
      double tstiffcpu = FEM_test->CalcStiffnessMat();
      FEM_test->SetDeviceMode(FEM::GPUOMP);
      femdata->GetStiffnessMatrix()->Clear();
      double tstiffgpu = FEM_test->CalcStiffnessMat();
      // Print Stiffness Matrix to default output
      if (printstiff) {
        Filehandler->writeMatrixNonzeros(femdata->GetStiffnessMatrix(),
                                         femdata->GetNumDof(),
                                         femdata->GetNumDof());
      }
      printf("..Applying Constraints:\n");
      FEM_test->ApplyConstraint(FEM::PEN,
                                Filehandler->getNumSupports(),
                                Filehandler->getNodeSupports());
      // Calculation info
      float matsizeMB =
        (float) femdata->GetStiffnessMatrix()->GetMatSize() / (1024 * 1024);
      printf("^ Matrix size: %.2fMB\n", matsizeMB);
      // Solves linear system for displacements

      if (solve) {
        double tsolvecpu = 0, tsolveomp = 0, tsolvegpu_naive = 0, tsolvegpu_naiveur = 0,
          tsolvegpu_share = 0, tsolvegpu_blk = 0, tsolvegpu_blkur = 0;
        tsolvecpu = solveDisplacements(FEM::CPU, femdata);
        tsolveomp = solveDisplacements(FEM::OMP, femdata);
        tsolvegpu_naive   = GPUSolveTime(femdata, SPRmatrix::STRAT_NAIVE);
        tsolvegpu_naiveur = GPUSolveTime(femdata, SPRmatrix::STRAT_NAIVEUR);
        tsolvegpu_share   = GPUSolveTime(femdata, SPRmatrix::STRAT_SHARE);
        tsolvegpu_blk     = GPUSolveTime(femdata, SPRmatrix::STRAT_BLOCK);
        tsolvegpu_blkur   = GPUSolveTime(femdata, SPRmatrix::STRAT_BLOCKUR);
        outfileWriteSolBatchResults(Filehandler,
          femdata,
          matsizeMB,
          tstiffcpu,
          tstiffgpu,
          tsolvecpu,
          tsolveomp,
          tsolvegpu_naive,
          tsolvegpu_naiveur,
          tsolvegpu_share,
          tsolvegpu_blk,
          tsolvegpu_blkur);
      } else {
        outfileWriteAsmBatchResults(Filehandler,
          femdata,
          matsizeMB,
          tstiffcpu,
          tstiffgpu);
      }
      delete(FEM_test);
    }
    Filehandler->writeNewLine();
  }

  Filehandler->CloseOutputFile();
  delete(Filehandler);
  printAnalysisEnd();

  return 1;
}

////////////////////////////////////////////////////////////////////////////////
double GPUSolveTime(FemData* femdata, SPRmatrix::OclStrategy oclstrategy) {
  femdata->GetStiffnessMatrix()->SetOclStrategy(oclstrategy);
  return solveDisplacements(FEM::GPU, femdata);
}