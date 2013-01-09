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


////////////////////////////////////////////////////////////////////////////////
// SPRmatrix.h - Sparse matrix class interface header
// Author: Francisco Paulo de Aboim (fpaboim@gmail.com)
////////////////////////////////////////////////////////////////////////////////

#ifndef VIS_H
#define VIS_H

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>

#include "FEM/FemData.h"

class Vis
{
public:
  Vis();
  ~Vis();

  int BuildMesh(FemData* femdata);
  int BuildQ4Mesh(FemData* femdata);
  int BuildQ8Mesh(FemData* femdata);
  int BuildBrk8Mesh(FemData* femdata);
  int BuildBrk20Mesh(FemData* femdata);

  vtkSmartPointer<vtkUnstructuredGrid> m_unstructgrid;
  vtkSmartPointer<vtkDataSetMapper> m_mapper;
  vtkSmartPointer<vtkRenderer> m_renderer;
  vtkSmartPointer<vtkRenderWindow> m_renderwindow;
  vtkSmartPointer<vtkRenderWindowInteractor> m_renderwindowinteractor;
  vtkSmartPointer<vtkActor> m_actor;

  double m_scalefactor;
};

#endif
