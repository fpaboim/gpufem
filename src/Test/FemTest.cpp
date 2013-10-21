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

#include <omp.h>

#include "gtest/gtest.h"
#include "checkcrt.h"

#include "FEM/Fem.h"
#include "FEM/Fem.cc"
#include "FEM/StiffAlgo.cc"
#include "FEM/StiffAlgo.h"
#include "FEM/StiffAlgoCPU.cc"
#include "FEM/StiffAlgoCPU.h"
#include "FEM/StiffAlgoGpuOmp.cc"
#include "FEM/StiffAlgoGpuOmp.h"
#include "FEM/StiffAlgoGPU.cc"
#include "FEM/StiffAlgoGPU.h"
#include "SPRmatrix/SPRmatrix.h"

// The fixture for value parameterized testing class AxyGPUTest.
////////////////////////////////////////////////////////////////////////////////
class FemTest : public ::testing::TestWithParam<FEM::DeviceMode> {
 protected:
  FemTest() {
    m_devicemode = FEM::CPU;
  }

  virtual void SetUp() {
    m_devicemode = GetParam();
  }

  virtual void TearDown() {
  }

  FEM::DeviceMode m_devicemode;
};

// Creation and teardown does not leak
////////////////////////////////////////////////////////////////////////////////
TEST_P(FemTest, creation_and_teardown_noleak) {
  CheckMemory chk;
  FEM* FEM_test = new FEM();

  delete(FEM_test);
}

// Initialized creation and teardown does not leak
////////////////////////////////////////////////////////////////////////////////
TEST_P(FemTest, creation_and_teardown_initialized_noleak) {
  CheckMemory chk;
  FEM* FEM_test = new FEM();
  FEM_test->Init(true, m_devicemode);

  delete(FEM_test);
}

// Filehandler with initilized fem object does not leak
////////////////////////////////////////////////////////////////////////////////
TEST_P(FemTest, with_filehandler_initialized_noleak) {
  CheckMemory chk;
  FEM* FEM_test = new FEM();
  FileIO* Filehandler = new FileIO();
  FEM_test->Init(true, m_devicemode);

  delete(Filehandler);
  delete(FEM_test);
}

////////////////////////////////////////////////////////////////////////////////
inline void MakeFEMExample(const char* filename,
                           FEM::DeviceMode devmode,
                           FEM* femtest) {
  FileIO* Filehandler = new FileIO();
  int err = Filehandler->ReadNF(filename);
  ASSERT_EQ(err, 1);
  // Sets-up finite element info
  femtest->Init(true, devmode);
  femtest->GetFemData()->Init(SPRmatrix::ELL, 2, 100, 0.25, Filehandler);
  delete(Filehandler);
}

// Filehandler reading file and save femdata
////////////////////////////////////////////////////////////////////////////////
TEST_P(FemTest, with_filehandler_and_femdata) {
  CheckMemory chk;
  FEM* FEM_test = new FEM();
  const char* filename = "../../test_models/placa2_Q4.nf";
  MakeFEMExample(filename, m_devicemode, FEM_test);
  delete(FEM_test);
}

////////////////////////////////////////////////////////////////////////////////
TEST_P(FemTest, macro_test_stiffness_coloring_noleak) {
  CheckMemory chk;
  FEM* FEM_test = new FEM();
  const char* filename = "../../test_models/placa2_Q4.nf";
  MakeFEMExample(filename, m_devicemode, FEM_test);
  FEM_test->SetUseColoring(true);
  double tstiff = FEM_test->CalcStiffnessMat();
  FEM_test->ApplyConstraint(FEM::PEN);

  delete(FEM_test);
  OCL.teardown();
}

////////////////////////////////////////////////////////////////////////////////
TEST_P(FemTest, macro_test_3dstiffness_diag_nonzero) {
  CheckMemory chk;
  FEM* FEM_test = new FEM();
  const char* filename = "../../test_models/brick_brk20v2.nf";
  MakeFEMExample(filename, m_devicemode, FEM_test);
  SPRmatrix* stiffmat = FEM_test->GetFemData()->GetStiffnessMatrix();
  stiffmat->SetOclLocalSize(4);
  FEM_test->SetUseColoring(true);
  double tstiff = FEM_test->CalcStiffnessMat();
  fem_float diag;
  for (int i = 0; i < stiffmat->GetMatDim(); i++) {
    diag = stiffmat->GetElem(i,i);
    ASSERT_GT(diag, 0.0f);
  }
  delete(FEM_test);
  OCL.teardown();
}

////////////////////////////////////////////////////////////////////////////////
TEST_P(FemTest, macro_test_2dstiffness_diag_nonzero) {
  CheckMemory chk;
  FEM* FEM_test = new FEM();
  const char* filename = "../../test_models/placa2_Q4.nf";
  MakeFEMExample(filename, m_devicemode, FEM_test);
  SPRmatrix* stiffmat = FEM_test->GetFemData()->GetStiffnessMatrix();
  stiffmat->SetOclLocalSize(4);
  FEM_test->SetUseColoring(false);

  double tstiff = FEM_test->CalcStiffnessMat();
  stiffmat->PrintMatrix();
  fem_float elm00 = stiffmat->GetElem(0,0);
  FEM_test->ApplyConstraint(FEM::PEN);
  fem_float elm00_const = stiffmat->GetElem(0,0);
  ASSERT_LE(elm00, elm00_const);

  delete(FEM_test);
  OCL.teardown();
}

// Compare with reference matlab implementation and check constraints
////////////////////////////////////////////////////////////////////////////////
TEST_P(FemTest, macro_test_stiffness_constraint) {
  CheckMemory chk;
  FEM* FEM_test = new FEM();
  const char* filename = "../../test_models/placa2_Q4.nf";
  MakeFEMExample(filename, m_devicemode, FEM_test);
  SPRmatrix* stiffmat = FEM_test->GetFemData()->GetStiffnessMatrix();
  stiffmat->SetOclLocalSize(4);

  FEM_test->SetUseColoring(false);
  double tstiff = FEM_test->CalcStiffnessMat();
  stiffmat->PrintMatrix();
  fem_float elm00 = stiffmat->GetElem(0,0);
  FEM_test->ApplyConstraint(FEM::PEN);
  fem_float elm00_const = stiffmat->GetElem(0,0);
  ASSERT_LE(elm00, elm00_const);

  delete(FEM_test);
  OCL.teardown();
}

using ::testing::Values;
INSTANTIATE_TEST_CASE_P(FemTest_CPU_ALGO,    FemTest, Values(FEM::CPU));
INSTANTIATE_TEST_CASE_P(FemTest_GPU_ALGO,    FemTest, Values(FEM::GPU));
INSTANTIATE_TEST_CASE_P(FemTest_GPUOMP_ALGO, FemTest, Values(FEM::GPUOMP));
