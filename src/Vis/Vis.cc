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
// Sparsely Sparse Matrix Library - Main Matrix Creation Interface
// Author: Francisco Aboim
// TecGraf / PUC-RIO
////////////////////////////////////////////////////////////////////////////////

// Headers
#include "Vis.h"

#include <vtkVersion.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkQuad.h>
#include <vtkQuadraticQuad.h>
#include <vtkHexahedron.h>
#include <vtkQuadraticHexahedron.h>
#include <vtkUnstructuredGrid.h>

#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkGeometryFilter.h>
#include <vtkWarpVector.h>
#include <vtkDoubleArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkDataSetMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>

#include "FEM/FemData.h"

// Constructor
////////////////////////////////////////////////////////////////////////////////
Vis::Vis() {
  m_unstructgrid            = vtkSmartPointer<vtkUnstructuredGrid>::New();
  m_mapper                  = vtkSmartPointer<vtkDataSetMapper>::New();
  m_renderer                = vtkSmartPointer<vtkRenderer>::New();
  m_renderwindow            = vtkSmartPointer<vtkRenderWindow>::New();
  m_renderwindowinteractor  = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  m_actor                   = vtkSmartPointer<vtkActor>::New();

  m_scalefactor = 1;
}

// Destructor
////////////////////////////////////////////////////////////////////////////////
Vis::~Vis() {
}

// buildMesh: Forwards data to appropriate mesh building method
////////////////////////////////////////////////////////////////////////////////
int Vis::BuildMesh(FemData* femdata) {

  if (femdata->GetModelDim() == 2) { // 2D Case
    if (femdata->GetNumElemNodes() == 4) { // Builds Q4 Element Mesh
      BuildQ4Mesh(femdata);
      return 1;
    } else if (femdata->GetNumElemNodes() == 8) { // Builds Q8 Element Mesh
      BuildQ8Mesh(femdata);
      return 1;
    }
  } else if (femdata->GetModelDim() == 3) { // 3D Case
    if (femdata->GetNumElemNodes() == 8) { // Builds brick8 Element Mesh
      BuildBrk8Mesh(femdata);
      return 1;
    } else if (femdata->GetNumElemNodes() == 20) { // Builds brick20 Element Mesh
      BuildBrk20Mesh(femdata);
      return 1;
    }
  }

  return 0;
}

// BuildQ4Mesh: Builds mesh for Q4 element
////////////////////////////////////////////////////////////////////////////////
int Vis::BuildQ4Mesh(FemData* femdata) {
  int*      elemconnect = femdata->GetElemConnect();
  fem_float* nodecoords = femdata->GetNodeCoords();
  fem_float* displvec   = femdata->GetDisplVector();

  int celltype = 0;
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkCellArray> quadarray =
    vtkSmartPointer<vtkCellArray>::New();
  vtkSmartPointer<vtkDoubleArray> warpData =
    vtkSmartPointer<vtkDoubleArray>::New();
  warpData->SetNumberOfComponents(3);
  warpData->SetName("warpData");

  // Loops over elements
  for (int elem = 0; elem < femdata->GetNumElem(); elem++) {

    // Gets current element coordinates
    fem_float** elemcoords = allocMatrix(4, 2, false);
    fem_float** elemdispls = allocMatrix(4, 2, false);
    for (int i = 0; i < 4; ++i) { // number of element nodes
      for (int j = 0; j < 2; ++j) { // model dimension
        // node in nodeCoords is nodeCoords[node-1][x,y,z]
        int index = (elemconnect[4 * elem + i] - 1) * 2 + j;
        elemcoords[i][j] = nodecoords[index];
        elemdispls[i][j] = displvec[index];
      }
    }

    // Setup the coordinates of eight points
    // (the two faces must be in counter clockwise order as viewed from the
    // outside)
    double P0[3] = {elemcoords[0][0], elemcoords[0][1], 0};
    double P1[3] = {elemcoords[1][0], elemcoords[1][1], 0};
    double P2[3] = {elemcoords[2][0], elemcoords[2][1], 0};
    double P3[3] = {elemcoords[3][0], elemcoords[3][1], 0};
    // Create the points
    points->InsertNextPoint(P0);
    points->InsertNextPoint(P1);
    points->InsertNextPoint(P2);
    points->InsertNextPoint(P3);

    // Sets up displacement information for warping display
    double D0[3] = {elemdispls[0][0], elemdispls[0][1], 0};
    double D1[3] = {elemdispls[1][0], elemdispls[1][1], 0};
    double D2[3] = {elemdispls[2][0], elemdispls[2][1], 0};
    double D3[3] = {elemdispls[3][0], elemdispls[3][1], 0};
    // Insert displacement data to displacement vector
    warpData->InsertNextTuple(D0);
    warpData->InsertNextTuple(D1);
    warpData->InsertNextTuple(D2);
    warpData->InsertNextTuple(D3);



    // Create a hexahedron from the points
    vtkSmartPointer<vtkQuad> quad = vtkSmartPointer<vtkQuad>::New();
    quad->GetPointIds()->SetId(0, (4*elem)+0);
    quad->GetPointIds()->SetId(1, (4*elem)+1);
    quad->GetPointIds()->SetId(2, (4*elem)+2);
    quad->GetPointIds()->SetId(3, (4*elem)+3);

    // Add the hexahedron to a cell array
    celltype = quad->GetCellType();
    m_unstructgrid->InsertNextCell(celltype, quad->GetPointIds());
  }
  // Add the points and hexahedron to an unstructured grid
  m_unstructgrid->SetPoints(points);
  m_unstructgrid->GetPointData()->AddArray(warpData);
  m_unstructgrid->GetPointData()->SetActiveVectors(warpData->GetName());

  // Geometry Filter
  vtkSmartPointer<vtkGeometryFilter> geomfilter =
    vtkSmartPointer<vtkGeometryFilter>::New();
  geomfilter->SetInput(m_unstructgrid);
  vtkSmartPointer<vtkWarpVector> dislfilter =
    vtkSmartPointer<vtkWarpVector>::New();
  dislfilter->SetInput(geomfilter->GetOutput());
  dislfilter->SetScaleFactor(m_scalefactor);
  dislfilter->Update();

  // Mappers
  vtkSmartPointer<vtkPolyDataMapper> elementsmapper =
    vtkSmartPointer<vtkPolyDataMapper>::New();
  elementsmapper->SetInput(geomfilter->GetOutput());
  elementsmapper->ScalarVisibilityOff();

  vtkSmartPointer<vtkDataSetMapper> warpmapper =
    vtkSmartPointer<vtkDataSetMapper>::New();
  warpmapper->SetInput(dislfilter->GetOutput());
  warpmapper->ScalarVisibilityOff();

  // Actors
  vtkSmartPointer<vtkActor> elementsactor =
    vtkSmartPointer<vtkActor>::New();
  elementsactor->SetMapper(elementsmapper);
  elementsactor->GetProperty()->SetColor(0.91, 0.87, 0.67);
  elementsactor->GetProperty()->SetDiffuse(0);
  elementsactor->GetProperty()->SetAmbient(1);
  elementsactor->GetProperty()->SetInterpolationToFlat();
  elementsactor->GetProperty()->SetEdgeVisibility(1);
  elementsactor->GetProperty()->SetEdgeColor(0, 0, 0);

  vtkSmartPointer<vtkActor> warpactor =
    vtkSmartPointer<vtkActor>::New();
  warpactor->SetMapper(warpmapper);
  warpactor->GetProperty()->SetColor(0.91, 0.90, 0.10);
  warpactor->GetProperty()->SetDiffuse(0);
  warpactor->GetProperty()->SetAmbient(1);
  warpactor->GetProperty()->SetInterpolationToFlat();
  warpactor->GetProperty()->SetEdgeVisibility(1);
  warpactor->GetProperty()->SetEdgeColor(0.1, 0.1, 0.1);
  warpactor->GetProperty()->SetOpacity(0.5);

  // Set Renderer Actor and Info
  m_renderer->AddActor(warpactor);
  m_renderer->AddActor(elementsactor);
  m_renderer->SetBackground(.2, .3, .4);
  m_renderer->SetBackground2(.9, .95, 1);
  m_renderer->SetGradientBackground(true);

  // Set RenderWindow Info
  m_renderwindow->AddRenderer(m_renderer);
  int winsize[2] = {600,400};
  m_renderwindow->SetSize(winsize);

  // RenderWindowInteractor Settings
  m_renderwindowinteractor->SetRenderWindow(m_renderwindow);

  m_renderwindow->Render();
  m_renderwindowinteractor->Start();

  return 1;
}

// BuildQ4Mesh: Builds mesh for Q8 element
////////////////////////////////////////////////////////////////////////////////
int Vis::BuildQ8Mesh(FemData* femdata) {
  int*      elemconnect = femdata->GetElemConnect();
  fem_float* nodecoords = femdata->GetNodeCoords();
  fem_float* displvec   = femdata->GetDisplVector();

  int celltype = 0;
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkCellArray> quadarray =
    vtkSmartPointer<vtkCellArray>::New();
  vtkSmartPointer<vtkDoubleArray> warpData =
    vtkSmartPointer<vtkDoubleArray>::New();
  warpData->SetNumberOfComponents(3);
  warpData->SetName("warpData");

  // Loops over elements
  for (int elem = 0; elem < femdata->GetNumElem(); elem++) {

    // Gets current element coordinates
    fem_float** elemcoords = allocMatrix(8, 2, false);
    fem_float** elemdispls = allocMatrix(8, 2, false);
    for (int i = 0; i < 8; ++i) { // number of element nodes
      for (int j = 0; j < 2; ++j) { // model dimension
        // node in nodeCoords is nodeCoords[node-1][x,y,z]
        int index = (elemconnect[8 * elem + i] - 1) * 2 + j;
        elemcoords[i][j] = nodecoords[index];
        elemdispls[i][j] = displvec[index];
      }
    }

    // Setup the coordinates of eight points
    // (the two faces must be in counter clockwise order as viewed from the
    // outside)
    double P0[3] = {elemcoords[0][0], elemcoords[0][1], 0};
    double P1[3] = {elemcoords[1][0], elemcoords[1][1], 0};
    double P2[3] = {elemcoords[2][0], elemcoords[2][1], 0};
    double P3[3] = {elemcoords[3][0], elemcoords[3][1], 0};
    double P4[3] = {elemcoords[4][0], elemcoords[4][1], 0};
    double P5[3] = {elemcoords[5][0], elemcoords[5][1], 0};
    double P6[3] = {elemcoords[6][0], elemcoords[6][1], 0};
    double P7[3] = {elemcoords[7][0], elemcoords[7][1], 0};
    // Create the points
    points->InsertNextPoint(P0);
    points->InsertNextPoint(P1);
    points->InsertNextPoint(P2);
    points->InsertNextPoint(P3);
    points->InsertNextPoint(P4);
    points->InsertNextPoint(P5);
    points->InsertNextPoint(P6);
    points->InsertNextPoint(P7);

    // Sets up displacement information for warping display
    double D0[3] = {elemdispls[0][0], elemdispls[0][1], 0};
    double D1[3] = {elemdispls[1][0], elemdispls[1][1], 0};
    double D2[3] = {elemdispls[2][0], elemdispls[2][1], 0};
    double D3[3] = {elemdispls[3][0], elemdispls[3][1], 0};
    double D4[3] = {elemdispls[4][0], elemdispls[4][1], 0};
    double D5[3] = {elemdispls[5][0], elemdispls[5][1], 0};
    double D6[3] = {elemdispls[6][0], elemdispls[6][1], 0};
    double D7[3] = {elemdispls[7][0], elemdispls[7][1], 0};
    // Insert displacement data to displacement vector
    warpData->InsertNextTuple(D0);
    warpData->InsertNextTuple(D1);
    warpData->InsertNextTuple(D2);
    warpData->InsertNextTuple(D3);
    warpData->InsertNextTuple(D4);
    warpData->InsertNextTuple(D5);
    warpData->InsertNextTuple(D6);
    warpData->InsertNextTuple(D7);


    // Create a hexahedron from the points
    vtkSmartPointer<vtkQuad> quad = vtkSmartPointer<vtkQuad>::New();
    quad->GetPointIds()->SetId(0, (8*elem)+0);
    quad->GetPointIds()->SetId(1, (8*elem)+1);
    quad->GetPointIds()->SetId(2, (8*elem)+2);
    quad->GetPointIds()->SetId(3, (8*elem)+3);
    quad->GetPointIds()->SetId(4, (8*elem)+4);
    quad->GetPointIds()->SetId(5, (8*elem)+5);
    quad->GetPointIds()->SetId(6, (8*elem)+6);
    quad->GetPointIds()->SetId(7, (8*elem)+7);

    // Add the hexahedron to a cell array
    celltype = quad->GetCellType();
    m_unstructgrid->InsertNextCell(celltype, quad->GetPointIds());
  }
  // Add the points and hexahedron to an unstructured grid
  m_unstructgrid->SetPoints(points);
  m_unstructgrid->GetPointData()->AddArray(warpData);
  m_unstructgrid->GetPointData()->SetActiveVectors(warpData->GetName());

  // Geometry Filter
  vtkSmartPointer<vtkGeometryFilter> geomfilter =
    vtkSmartPointer<vtkGeometryFilter>::New();
  geomfilter->SetInput(m_unstructgrid);
  vtkSmartPointer<vtkWarpVector> dislfilter =
    vtkSmartPointer<vtkWarpVector>::New();
  dislfilter->SetInput(geomfilter->GetOutput());
  dislfilter->SetScaleFactor(m_scalefactor);
  dislfilter->Update();

  // Mappers
  vtkSmartPointer<vtkPolyDataMapper> elementsmapper =
    vtkSmartPointer<vtkPolyDataMapper>::New();
  elementsmapper->SetInput(geomfilter->GetOutput());
  elementsmapper->ScalarVisibilityOff();

  vtkSmartPointer<vtkDataSetMapper> warpmapper =
    vtkSmartPointer<vtkDataSetMapper>::New();
  warpmapper->SetInput(dislfilter->GetOutput());
  warpmapper->ScalarVisibilityOff();

  // Actors
  vtkSmartPointer<vtkActor> elementsactor =
    vtkSmartPointer<vtkActor>::New();
  elementsactor->SetMapper(elementsmapper);
  elementsactor->GetProperty()->SetColor(0.91, 0.87, 0.67);
  elementsactor->GetProperty()->SetDiffuse(0);
  elementsactor->GetProperty()->SetAmbient(1);
  elementsactor->GetProperty()->SetInterpolationToFlat();
  elementsactor->GetProperty()->SetEdgeVisibility(1);
  elementsactor->GetProperty()->SetEdgeColor(0, 0, 0);

  vtkSmartPointer<vtkActor> warpactor =
    vtkSmartPointer<vtkActor>::New();
  warpactor->SetMapper(warpmapper);
  warpactor->GetProperty()->SetColor(0.91, 0.90, 0.10);
  warpactor->GetProperty()->SetDiffuse(0);
  warpactor->GetProperty()->SetAmbient(1);
  warpactor->GetProperty()->SetInterpolationToFlat();
  warpactor->GetProperty()->SetEdgeVisibility(1);
  warpactor->GetProperty()->SetEdgeColor(0.1, 0.1, 0.1);
  warpactor->GetProperty()->SetOpacity(0.5);

  // Set Renderer Actor and Info
  m_renderer->AddActor(warpactor);
  m_renderer->AddActor(elementsactor);
  m_renderer->SetBackground(.2, .3, .4);
  m_renderer->SetBackground2(.9, .95, 1);
  m_renderer->SetGradientBackground(true);

  // Set RenderWindow Info
  m_renderwindow->AddRenderer(m_renderer);
  int winsize[2] = {600,400};
  m_renderwindow->SetSize(winsize);

  // RenderWindowInteractor Settings
  m_renderwindowinteractor->SetRenderWindow(m_renderwindow);

  m_renderwindow->Render();
  m_renderwindowinteractor->Start();

  return 1;
}

// BuildQ4Mesh: Builds mesh for Q4 element
////////////////////////////////////////////////////////////////////////////////
int Vis::BuildBrk8Mesh(FemData* femdata) {
  return 1;
}

// BuildQ4Mesh: Builds mesh for Q4 element
////////////////////////////////////////////////////////////////////////////////
int Vis::BuildBrk20Mesh(FemData* femdata) {
  return 1;
}
