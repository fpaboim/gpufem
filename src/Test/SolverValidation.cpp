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

#include "SPRmatrix/SPRmatrix.h"
#include "SPRmatrix/CSRmatrix.h"
#include "SPRmatrix/DIAmatrix.h"
#include "SPRmatrix/DENmatrix.h"
#include "SPRmatrix/ELLmatrix.h"
#include "SPRmatrix/EIGmatrix.h"
#include "SPRmatrix/EIGmatrix.h"
#include "OpenCL/OCLwrapper.h"
#include "LAops/SprSolver.h"

#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>

// The fixture for value parameterized testing class AxyGPUTest.
////////////////////////////////////////////////////////////////////////////////
class SolverValidation : public ::testing::TestWithParam<SPRmatrix::OclStrategy> {
};

TEST(SolverValidation, CompareEigenSolverWithSparseImplementation) {
  CheckMemory check;
  int matdim      = 32;
  int localsz     = 4;
  int maxiter     = 100;

  Eigen::Matrix3f A;
  Eigen::Vector3f b;
  A << 1,2,3,  4,5,6,  7,8,10;
  b << 3, 3, 4;
  std::cout << "Here is the matrix A:\n" << A << std::endl;
  std::cout << "Here is the vector b:\n" << b << std::endl;
  Eigen::Vector3f x = A.colPivHouseholderQr().solve(b);
  std::cout << "The solution is:\n" << x << std::endl;
  std::cout << "The inverse of A is:\n" << A.inverse() << std::endl;
}

TEST(SolverValidation, CompareEigenSolverWithSparseImplementation2) {
  using namespace Eigen;
  using namespace std;
  CheckMemory check;
  int matdim      = 32;
  int localsz     = 4;
  int maxiter     = 100;
  int matsz = 10;

  MatrixXd A = MatrixXd::Random(matsz,matsz);
  MatrixXd b = MatrixXd::Random(matsz,1);
  MatrixXd x = A.fullPivLu().solve(b);
  double relative_error = (A*x - b).norm() / b.norm(); // norm() is L2 norm
  cout << "The relative error is:\n" << relative_error << endl;
}

TEST(SolverValidation, CompareEigenSolverWithSparseImplementation3) {
  using namespace Eigen;
  using namespace std;
  CheckMemory check;
  int matsz = 10;

  Matrix3d m = Matrix3d::Random();
  cout << "Here is the matrix m:" << endl << m << endl;
  Matrix3d inverse;
  bool invertible;
  double determinant;
  m.computeInverseWithCheck(inverse,invertible);
  cout << "Its determinant is " << determinant << endl;
  if(invertible) {
    cout << "It is invertible, and its inverse is:" << endl << inverse << endl;
  }
  else {
    cout << "It is not invertible." << endl;
  }

}

// 
// using ::testing::Values;
// INSTANTIATE_TEST_CASE_P(AxyNAIVE,   AxyGPUTest, Values(SPRmatrix::STRAT_NAIVE));
// INSTANTIATE_TEST_CASE_P(AxyNAIVEUR, AxyGPUTest, Values(SPRmatrix::STRAT_NAIVEUR));
// INSTANTIATE_TEST_CASE_P(AxySHARE,   AxyGPUTest, Values(SPRmatrix::STRAT_SHARE));
// INSTANTIATE_TEST_CASE_P(AxyBLOCK,   AxyGPUTest, Values(SPRmatrix::STRAT_BLOCK));
// INSTANTIATE_TEST_CASE_P(AxyBLOCKUR, AxyGPUTest, Values(SPRmatrix::STRAT_BLOCKUR));
