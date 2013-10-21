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

#include "gtest/gtest.h"
#include "checkcrt.h"

#include "FEM/FemData.h"
#include "FEM/FemData.cc"
#include "SPRmatrix/SPRmatrix.h"

// Creation and teardown does not leak
////////////////////////////////////////////////////////////////////////////////
TEST(FemDataTest, creation_and_teardown_noleak) {
  CheckMemory chk;
  FemData* femdata = new FemData();

  delete(femdata);
}

// Filehandler reading file and save femdata
////////////////////////////////////////////////////////////////////////////////
TEST(FemDataTest, check_init_femdata_with_filehandler_noleak) {
  CheckMemory chk;
  FileIO* Filehandler = new FileIO();
  FemData* Femdata = new FemData();
  delete(Femdata);
  delete(Filehandler);
}

// Filehandler reading file and save femdata
////////////////////////////////////////////////////////////////////////////////
TEST(FemDataTest, check_init_femdata_does_not_leak) {
  CheckMemory chk;
  FileIO* Filehandler = new FileIO();
  FemData* Femdata = new FemData();

  int err = Filehandler->ReadNF("../../test_models/_testmodels/Q4.nf");
  ASSERT_EQ(err, 1);
  Femdata->Init(SPRmatrix::ELL, 2, 100, 0.2, Filehandler);

  delete(Femdata);
  delete(Filehandler);
}
