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

#include "interface.h"

// main
////////////////////////////////////////////////////////////////////////////////
int main(int ac, char** av) {
  using namespace std;

  cout << endl;
  cout << "********************************************************************" << endl;
  cout << "**    GPU based Finite Element Method Analysis                    **" << endl;
  cout << "**    Authors: Andre Pereira   - UFF/Tecgraf                      **" << endl;
  cout << "**             Francisco Aboim - PUC-RIO/Tecgraf                  **" << endl;
  cout << "**    Date: March/2013                                            **" << endl;
  cout << "********************************************************************" << endl;
  cout << endl;

  //Microbench::BenchSearch();
  Microbench::BenchMV();
  Microbench::BenchCG();

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
