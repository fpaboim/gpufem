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

#include <string>
#include "gtest/gtest.h"
#include "Test/checkcrt.h"

#include "Utils/fileIO.cc"
#include "Utils/fileIO.h"

// Creation and teardown does not leak
////////////////////////////////////////////////////////////////////////////////
TEST(FileIOTest, creation_and_teardown_noleak) {
  CheckMemory chk;
  FileIO* filehandler = new FileIO();
  delete(filehandler);
}

// Reads sample Q4 mesh with two elements and checks memory allocation
////////////////////////////////////////////////////////////////////////////////
TEST(FileIOTest, reading_2Q4_test_NF) {
  CheckMemory chk;
  FileIO* filehandler = new FileIO();
  string filename = "../../test_models/2Q4.nf";
  filehandler->ReadNF(filename);
  delete(filehandler);
}

// Reads sample q4 mesh multiple times with correct object allocation
////////////////////////////////////////////////////////////////////////////////
TEST(FileIOTest, multiple_read_NF) {
  CheckMemory chk;
  string filename = "../../test_models/2Q4.nf";
  for (int i = 0; i < 4; i++) {
    FileIO* filehandler = new FileIO();
    filehandler->ReadNF(filename);
    delete(filehandler);
  }
}

// Reads sample q4 mesh multiple times reusing the same handler
////////////////////////////////////////////////////////////////////////////////
TEST(FileIOTest, multi_reusing_handler_read_NF) {
  CheckMemory chk;
  string filename = "../../test_models/2Q4.nf";
  FileIO* filehandler = new FileIO();
  for (int i = 0; i < 4; i++) {
    filehandler->ReadNF(filename);
  }
  delete(filehandler);
}
