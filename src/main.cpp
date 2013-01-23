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

#include "lua.hpp"
#include "iup.h"
#include "iuplua.h"
#include "iupcontrols.h"
#include "iupluacontrols.h"

#include "Lua/LuaWrap.h"
#include "Utils/util.h"
#include "Utils/fileIO.h"
#include "FEM/fem.h"
#include "FEM/FemData.h"
#include "FEM/femColor.h"
#include "LAops/LAops.h"
#include "LAops/SprSolver.h"
#include "SPRmatrix/SPRmatrix.h"
#include "OpenCL/OCLwrapper.h"
#include "Microbench/Microbench.h"
#include "Vis/Vis.h"


// Headers
////////////////////////////////////////////////////////////////////////////////
int LuaRegisterCFunctions(lua_State* L);
static int lua_RunAnalysis(lua_State* L);
static int lua_GetMaxThreads(lua_State* L);
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


// main
////////////////////////////////////////////////////////////////////////////////
int main(int ac, char** av) {
  using namespace std;

  cout << endl;
  cout << "********************************************************************" << endl;
  cout << "**    GPU based Finite Element Method Analysis                    **" << endl;
  cout << "**    Authors: Andre Pereira   - UFF/Tecgraf                      **" << endl;
  cout << "**             Francisco Aboim - PUC-RIO/Tecgraf                  **" << endl;
  cout << "**    Date: April/2012                                            **" << endl;
  cout << "********************************************************************" << endl;
  cout << endl;

  //Microbench::BenchSearch();
  //Microbench::BenchCG();
  //Microbench::BenchMV();

  IupOpen(NULL, NULL);

  /* load Lua base libraries */
  WLua.init();
  lua_State* L = WLua.getLuaState();

  /* run the script */
  iuplua_open(L);
  LuaRegisterCFunctions(L);
  IupImageLibOpen();

  // Does Dialog File
  WLua.dofile("./../src/Lua/maindlg.lua");
  IupMainLoop();

  // End Session
  cout << "\n**** EXITING.. ****" << endl;
  IupClose();

  //std::getchar(); // keep console window open until Return keystroke

  return 1;
}

// LuaRegisterCFunctions: C function to be registered and called by Lua code
////////////////////////////////////////////////////////////////////////////////
int LuaRegisterCFunctions(lua_State* L) {
  static const struct luaL_Reg FEMlib [] = {
    {"GetMaxThreads", lua_GetMaxThreads},
    {"RunAnalysis", lua_RunAnalysis},
    {NULL, NULL} //sentinel value
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
  assert(narg == 10);
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
  // Matrix Format (1-Dense, 2-Diagonal, 3- CSR, 4-EllPack, 5-Eigen)
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

  RunAnalysis(files,
              matfmt,
              devicemode,
              nthreads,
              gausspts,
              Emod,
              Nu,
              color,
              makenodal,
              printstiff,
              solve,
              view,
              outfile,
              appendmode);

  // function returns to lua 0 results

  return 0;
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
  bool writesol = false;
  printf("**OMP Number of Threads: %i**\n", omp_get_max_threads());

  FileIO* Filehandler = new FileIO();
  Filehandler->OpenOutputFile(outfile, appendmode);
  Filehandler->writeMatFormat(sprseformat);

  // Loops over all input files doing FEM calculations
  for (int threads = 1; threads <= 8; threads *= 2) {
    Filehandler->writeString("Threads:");
    Filehandler->writeNumTab(threads);
    Filehandler->writeNewLine();
    if (writesol)
      Filehandler->writeSolOutputHeader();
    else
      Filehandler->writeAsmOutputHeader();

    omp_set_num_threads(threads);
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

        printf("Number of colors: %i\n", mshColorObj->GetNumColors());
        printf("Metis element (+nodal) graph time: %3.4f\n", (t2-t1));
        printf("Greedy coloring time: %3.4f\n", (t3-t2));
        if (usennz) {
          printf("NNZ/Band counting time: %3.4f\n", (t4-t3));
        }
        printf("++ Total coloring time: %3.4f\n", (t4-t1));
        delete(mshColorObj);
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

      // Solves linear system for displacements
      //////////////////////////////////////////////////////////////////////////////
      double tsolvecpu = omp_get_wtime();
      double tsolvegpu = 0;
      if (solve) {
        printf("..Solving for displacements:\n");

        CPU_CG(femdata->GetStiffnessMatrix(),
          femdata->GetDisplVector(),
          femdata->GetForceVector(),
          femdata->GetNumDof(),
          3000,
          0.001f,
          false);
        tsolvecpu = omp_get_wtime() - tsolvecpu;
        tsolvegpu = omp_get_wtime();

        femdata->GetStiffnessMatrix()->SolveCgGpu(femdata->GetDisplVector(),
          femdata->GetForceVector(),
          3000,
          0.001f,
          1);

        tsolvegpu = omp_get_wtime() - tsolvegpu;
        // printVectorf(femdata->GetDisplVector(), ndof);
      }

      // Calculation info
      float matsizeMB =
        (float) femdata->GetStiffnessMatrix()->GetMatSize() / (1024 * 1024);
      printf("\n# MATRIX SIZE: %.2fMB\n", matsizeMB);
      //printf("# Execution Time (seconds): %.2f\n", tstiff);

      if (solve) {
        //printf("# Solver Time: %3.4f -> %.2f%% of Stiffness Time\n",
        //  tsolve, (tsolve/tstiff)*100);
      }
      printf("\n********************************************\n");

      // Visualization - setup VTK through Vis class
      if (view) {
        Vis femVis;
        femVis.BuildMesh(femdata);
      }

      if (writesol) {
        // Write to output file
        Filehandler->writeNumTab(femdata->GetNumDof());
        Filehandler->writeNumTab(femdata->GetNumNodes());
        Filehandler->writeNumTab(femdata->GetNumElem());
        Filehandler->writeNumTab(matsizeMB);
        Filehandler->writeNumTab(tstiffcpu);
        Filehandler->writeNumTab(tstiffgpu);
        Filehandler->writeNumTab(tsolvecpu);
  //       Filehandler->writeNumTab(tsolvegpu);
        Filehandler->writeNewLine();
      } else {
        // Write to output file
        Filehandler->writeNumTab(femdata->GetNumDof());
        Filehandler->writeNumTab(femdata->GetNumNodes());
        Filehandler->writeNumTab(femdata->GetNumElem());
        Filehandler->writeNumTab(matsizeMB);
        Filehandler->writeNumTab(tstiffcpu);
        Filehandler->writeNumTab(tstiffgpu);
        Filehandler->writeNumTab(tsolvecpu);
  //       Filehandler->writeNumTab(tsolvegpu);
        Filehandler->writeNewLine();
      }

      delete(FEM_test);
    }
    Filehandler->writeNewLine();
  }

  Filehandler->CloseOutputFile();
  delete(Filehandler);
  printf("** DONE! **\n");

  return 1;
}

