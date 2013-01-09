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

#include "OpenCL/OCLwrapper.h"


// The fixture for testing class SPRmatrixTest.
////////////////////////////////////////////////////////////////////////////////
class OclTest : public ::testing::Test {
 protected:
  OclTest() {
  }
  virtual ~OclTest() {
  }

  virtual void SetUp() {
  }
  virtual void TearDown() {
  }
};

////////////////////////////////////////////////////////////////////////////////
TEST_F(OclTest, normal_load_and_teardown) {
  CheckMemory chk;
  OCL.setDir("C://Users//fpaboim//Desktop//parallel_projects//GPU_FEM//fpaboim_gpufem//src//OpenCL//clKernels//");
  OCL.loadSource("gpuFEM_Q4.cl");
  OCL.loadSource("LAopsEll.cl");
  OCL.loadKernel("SpMVStag");
  OCL.teardown();
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(OclTest, double_source_load_nofail) {
  CheckMemory chk;
  OCL.setDir("C://Users//fpaboim//Desktop//parallel_projects//GPU_FEM//fpaboim_gpufem//src//OpenCL//clKernels//");
  OCL.loadSource("LAopsEll.cl");
  OCL.loadSource("LAopsEll.cl");
  OCL.teardown();
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(OclTest, double_kernel_load_nofail) {
  CheckMemory chk;
  OCL.setDir("C://Users//fpaboim//Desktop//parallel_projects//GPU_FEM//fpaboim_gpufem//src//OpenCL//clKernels//");
  OCL.loadSource("LAopsEll.cl");
  OCL.loadKernel("SpMVStag");
  OCL.loadKernel("SpMVStag");
  OCL.teardown();
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(OclTest, double_teardown_nofail) {
  CheckMemory chk;
  OCL.setDir("C://Users//fpaboim//Desktop//parallel_projects//GPU_FEM//fpaboim_gpufem//src//OpenCL//clKernels//");
  OCL.loadSource("gpuFEM_Q4.cl");
  OCL.loadSource("LAopsEll.cl");
  OCL.teardown();
  OCL.teardown();
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(OclTest, change_dir_nofail) {
  CheckMemory chk;
  OCL.setDir(".\\..\\src\\OpenCL\\clKernels\\");
  OCL.setDir("C://Users//fpaboim//Desktop//parallel_projects//GPU_FEM//fpaboim_gpufem//src//OpenCL//clKernels//");
  OCL.loadSource("LAopsEll.cl");
  OCL.loadSource("gpuFEM_Q4.cl");
  OCL.loadKernel("getStiffnessQ4");
  OCL.teardown();
}
