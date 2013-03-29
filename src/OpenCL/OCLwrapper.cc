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

// Headers
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <assert.h>
#include <sys/stat.h>

#include "CL/opencl.h"

#include "OCLwrapper.h"

// initializations
OCLwrapper* OCLwrapper::m_oclwrap = NULL;

// Constructor defaults to GPU and sets up environment
////////////////////////////////////////////////////////////////////////////////
OCLwrapper::OCLwrapper() {
  m_verbose          = false;
  m_usecpu           = false;
  m_clcurrentdevice  = NULL;
  m_cldeviceCPU      = NULL;
  m_cldeviceGPU      = NULL;
  m_clerr            = 0;
  m_computeunits     = 0;
  m_maxworkdim       = 0;
  m_maxworkgroupsize = 0;
  m_localmemsize     = 0;
  m_globalmemsize    = 0;
  m_globalmaxmemsize = 0;
  m_exectime         = 0;
  m_clprogram[0]     = NULL;
  for (int i=0; i < 3; ++i) {
    m_maxworkitemsizes[i] = 0;
    m_localworksize[i]    = 0;
    m_globalworksize[i]   = 0;
  }
  m_clsrcfilehandle = NULL;

  // Gets Device Information
  m_clerr = clGetPlatformIDs(1, &m_clplatformid, NULL);
  assert(m_clerr == CL_SUCCESS);
  // Find the CPU CL device
  m_clerr = clGetDeviceIDs(m_clplatformid, CL_DEVICE_TYPE_CPU, 1,
    &m_cldeviceCPU, NULL);
  assert(m_clerr == CL_SUCCESS);
  // Find the GPU CL device, if not possible fall back to CPU
  m_clerr = clGetDeviceIDs(m_clplatformid, CL_DEVICE_TYPE_GPU, 1,
    &m_cldeviceGPU, NULL);
  // Selects device for computation - GPU is default if present
  if (m_clerr != CL_SUCCESS) {
    setDevice(CPU);
  } else {
    setDevice(GPU);
  }
}

OCLwrapper::~OCLwrapper() {
}

// gets OCLwrapper single instance
////////////////////////////////////////////////////////////////////////////////
OCLwrapper& OCLwrapper::instance() {
  return m_oclwrap ? *m_oclwrap : *(m_oclwrap = new OCLwrapper());
}

// Changes OpenCL device in use
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::getDeviceInfo() {
  cl_char vendor_name[1024] = {0};
  cl_char device_name[1024] = {0};
  m_clerr = clGetDeviceInfo(m_clcurrentdevice,
                            CL_DEVICE_VENDOR,
                            sizeof(vendor_name),
                            vendor_name,
                            NULL);
  m_clerr |= clGetDeviceInfo(m_clcurrentdevice,
                             CL_DEVICE_NAME,
                             sizeof(device_name),
                             device_name,
                             NULL);
  m_clerr |= clGetDeviceInfo(m_clcurrentdevice,
                             CL_DEVICE_MAX_COMPUTE_UNITS,
                             sizeof(cl_uint),
                             &m_computeunits,
                             NULL);
  m_clerr |= clGetDeviceInfo(m_clcurrentdevice,
                             CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                             sizeof(cl_uint),
                             &m_maxworkdim,
                             NULL);
  m_clerr |= clGetDeviceInfo(m_clcurrentdevice,
                             CL_DEVICE_MAX_WORK_ITEM_SIZES,
                             sizeof(m_maxworkitemsizes),
                             &m_maxworkitemsizes,
                             NULL);
  m_clerr |= clGetDeviceInfo(m_clcurrentdevice,
                             CL_DEVICE_MAX_WORK_GROUP_SIZE,
                             sizeof(size_t),
                             &m_maxworkgroupsize,
                             NULL);
  m_clerr |= clGetDeviceInfo(m_clcurrentdevice,
                             CL_DEVICE_LOCAL_MEM_SIZE,
                             sizeof(cl_ulong),
                             &m_localmemsize,
                             NULL);
  m_clerr |= clGetDeviceInfo(m_clcurrentdevice,
                             CL_DEVICE_LOCAL_MEM_SIZE,
                             sizeof(cl_ulong),
                             &m_localmemsize,
                             NULL);
  m_clerr |= clGetDeviceInfo(m_clcurrentdevice,
                             CL_DEVICE_GLOBAL_MEM_SIZE,
                             sizeof(cl_ulong),
                             &m_globalmemsize,
                             NULL);
  m_clerr |= clGetDeviceInfo(m_clcurrentdevice,
                             CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                             sizeof(cl_ulong),
                             &m_globalmaxmemsize,
                             NULL);
  assert(m_clerr == CL_SUCCESS);
  printf("Connecting to %s:%s...\n", vendor_name, device_name);
  if (m_verbose) {
    printf("->  Compute Units: %u\n", m_computeunits);
    printf("->  Local Memsize: %lu Kbytes\n", m_localmemsize/1024);
    printf("-> Global Memsize: %lu Mbytes\n", m_globalmemsize/(1024*1024));
    printf("->  MaxGlbl Memsz: %lu Mbytes\n", m_globalmaxmemsize/(1024*1024));
    printf("->    Max WI Dims: %u\n", m_maxworkdim);
    printf("->    Max WG Size: %u\n", m_maxworkgroupsize);
    printf("->   Max WI Sizes: %u\n", m_maxworkitemsizes[0]);
    printf("->                 %u\n", m_maxworkitemsizes[1]);
    printf("->                 %u\n", m_maxworkitemsizes[2]);
  }
}

// Changes OpenCL device in use
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::setDevice(Device devicetype) {
  if (devicetype == CPU)
    m_clcurrentdevice = m_cldeviceCPU;
  else
    m_clcurrentdevice = m_cldeviceGPU;
  getDeviceInfo();
  m_clcontext = clCreateContext(NULL,
                                1,
                                &m_clcurrentdevice,
                                NULL,
                                NULL,
                                &m_clerr);
  assert(m_clerr == CL_SUCCESS);
  cl_command_queue_properties prop = 0;
  prop |= CL_QUEUE_PROFILING_ENABLE;
  m_clcmdqueue = clCreateCommandQueue(m_clcontext,
                                      m_clcurrentdevice,
                                      prop,
                                      NULL);
}

// Changes OpenCL device in use
////////////////////////////////////////////////////////////////////////////////
OCLwrapper::Device OCLwrapper::getDevice() {
  OCLwrapper::Device devicetype = NONE;
  if (m_clcurrentdevice == m_cldeviceCPU) {
    devicetype = CPU;
  } else if (m_clcurrentdevice == m_cldeviceGPU) {
    devicetype = GPU;
  }
  return devicetype;
}

// Loads The OpenCL Source File
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::setDir(std::string dir) {
  // Checks last line for "//" - later string has filename appended
  if ((dir.find("//", dir.size()-2) || dir.find("\\", dir.size()-2))
      == std::string::npos)
    dir.append("//");
  m_dir = dir;
  std::string includedir("-I ");
  includedir.append(dir);
  setBuildOptions(includedir);
}

// Loads The OpenCL Source File
////////////////////////////////////////////////////////////////////////////////
int OCLwrapper::loadSource(std::string sourcefilename) {
  if (m_dir.empty()) {
    return 0;
  }
  if (isProgramLoaded(sourcefilename)) {
    m_clprogram[0] = m_loadedprograms[sourcefilename];
    return 1;
  }
  // Gets source from file
  struct stat statbuf;
  std::string filename = m_dir;
  filename.append(sourcefilename);
  if (m_clsrcfilehandle) {
    fclose(m_clsrcfilehandle);
    m_clsrcfilehandle = NULL;
  }
  m_clsrcfilehandle = fopen(filename.c_str(), "r");
  assert(m_clsrcfilehandle != 0);
  stat(filename.c_str(), &statbuf);
  char* program_source = (char*) malloc(statbuf.st_size + 1);
  fread(program_source, statbuf.st_size, 1, m_clsrcfilehandle);
  program_source[statbuf.st_size] = '\0';
  const char* finalsource = program_source;

  // Create OpenCL program from source
  m_clprogram[0] = clCreateProgramWithSource(m_clcontext,
                                             1,
                                             &finalsource,
                                             NULL,
                                             &m_clerr);
  assert(m_clerr == CL_SUCCESS);
  // Build OpenCL program
  m_clerr = clBuildProgram(m_clprogram[0], 0, NULL, m_buildoptions.c_str(),
                           NULL, NULL);
  if (m_clerr != CL_SUCCESS)
    getBuildLog(m_clerr, m_clprogram, m_clcurrentdevice);
  assert(m_clerr == CL_SUCCESS);

  m_loadedprograms[sourcefilename] = m_clprogram[0];
  free(program_source);

  return 1;
}

// Creates the Kernel From Loaded Source
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::loadKernel(const char* kernelname) {
  if (isKernelLoaded(kernelname)) {
    m_clkernel[0] = m_loadedkernels[kernelname];
    return;
  }
  assert(m_clprogram[0] != NULL);
  // Now create the kernel "objects" that we want to use in the example file
  m_clkernel[0] = clCreateKernel(m_clprogram[0], kernelname, &m_clerr);
  assert(m_clerr == CL_SUCCESS);
  m_loadedkernels[kernelname] = m_clkernel[0];
}

// Gets OpenCL build log
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::getBuildLog(cl_int err,
                             cl_program* program,
                             cl_device_id device ) {
  cl_int logStatus;
  char *buildLog = NULL;
  size_t buildLogSize = 0;
  logStatus = clGetProgramBuildInfo(program[0],
    device,
    CL_PROGRAM_BUILD_LOG,
    buildLogSize,
    buildLog,
    &buildLogSize);

  buildLog = (char*)malloc(buildLogSize);

  memset(buildLog, 0, buildLogSize);

  logStatus = clGetProgramBuildInfo(program[0],
                                    device,
                                    CL_PROGRAM_BUILD_LOG,
                                    buildLogSize,
                                    buildLog,
                                    NULL);

  std::cout << " \n\t\t\tBUILD LOG\n";
  std::cout << " ************************************************\n";
  std::cout << buildLog << std::endl;
  std::cout << " ************************************************\n";
  system("pause");
  free(buildLog);
}

// Creates Memory Buffer
////////////////////////////////////////////////////////////////////////////////
cl_mem OCLwrapper::createBuffer(size_t memsize, cl_mem_flags flag) {
  cl_mem buffer;
  buffer = clCreateBuffer(m_clcontext, flag, memsize, NULL, &m_clerr);
  assert(m_clerr == CL_SUCCESS);

  return buffer;
}

// Enqueues Memory Buffer for Writing
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::enqueueWriteBuffer(cl_mem buffer,
                                    size_t memsize,
                                    void* hostData,
                                    cl_bool blocking) {
  m_clerr = clEnqueueWriteBuffer(m_clcmdqueue,
                                 buffer,
                                 blocking,
                                 0,
                                 memsize,
                                 hostData,
                                 0,
                                 NULL,
                                 NULL);
  assert(m_clerr == CL_SUCCESS);
}

// Enqueues Memory Buffer for Reading
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::enqueueReadBuffer(cl_mem buffer,
                                   size_t memsize,
                                   void* hostData,
                                   cl_bool blocking)
{
  m_clerr = clEnqueueReadBuffer(m_clcmdqueue,
                                buffer,
                                blocking,
                                0,
                                memsize,
                                hostData,
                                0,
                                NULL,
                                NULL);
  assert(m_clerr == CL_SUCCESS);
}

// Sets Kernel Arguments
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::setKernelArg(int argnum,
                              size_t argSpecifierSize,
                              const void* argAddress) {
  m_clerr  = clSetKernelArg(m_clkernel[0],
                            argnum,
                            argSpecifierSize,
                            argAddress);
  assert(m_clerr == CL_SUCCESS);
}

// Sets Global Dimensions
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::setGlobalWorksize(cl_uint dim, cl_uint value) {
  if (dim < 0 || dim > m_maxworkdim) {
    printf("\n*** ERROR - INVALID DIMENSION SET: %u ***\n", dim);
    return;
  }
  if (value > m_maxworkgroupsize || value < 0) {
    printf("\n*** ERROR - INVALID WORKGROUPSIZE[%u] SET: %u ***\n", dim, value);
  }
  m_globalworksize[dim] = value;
}

// Sets Local Dimensions
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::setLocalWorksize(cl_uint dim, cl_uint value) {
  if (dim < 0 || dim > m_maxworkdim) {
    printf("\n*** ERROR - INVALID DIMENSION SET: %i ***\n", dim);
    return;
  }
  if (value > m_maxworkitemsizes[dim] || value < 0) {
    printf("\n*** ERROR - INVALID WORKSIZE[%u] SET: %u ***\n", dim, value);
  }
  m_localworksize[dim] = value;
}

// Sets Local Dimensions
////////////////////////////////////////////////////////////////////////////////
bool OCLwrapper::localSizeIsOK(size_t memsize) {
  if (memsize <= m_localmemsize) {
    return true;
  }

  return false;
}

// Enqueues NDRange Kernel for Execution
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::enquequeNDRangeKernel(cl_uint ndrange, bool getKernelTime) {
  if (getKernelTime) {
    m_clerr = clEnqueueNDRangeKernel(m_clcmdqueue,
                                     m_clkernel[0],
                                     ndrange,
                                     NULL,
                                     m_globalworksize,
                                     m_localworksize,
                                     0,
                                     NULL,
                                     &m_clevents[0]);
    assert(m_clerr == CL_SUCCESS);

    m_clerr = clWaitForEvents(1, &m_clevents[0]);
    assert(m_clerr == CL_SUCCESS);

    /* Calculate performance */
    cl_ulong startTime;
    cl_ulong endTime;

    m_clerr = clGetEventProfilingInfo(m_clevents[0],
      CL_PROFILING_COMMAND_START,
      sizeof(cl_ulong),
      &startTime,
      0);
    assert(m_clerr == CL_SUCCESS);
    m_clerr = clGetEventProfilingInfo(m_clevents[0],
      CL_PROFILING_COMMAND_END,
      sizeof(cl_ulong),
      &endTime,
      0);
    assert(m_clerr == CL_SUCCESS);
    // convert from ns to ms
    m_exectime = 1e-6 * (endTime - startTime);
    printf("@OpenCL Kernel Exec Time: %.4f ms\n", m_exectime);
  } else {
    m_clerr = clEnqueueNDRangeKernel(m_clcmdqueue,
                                     m_clkernel[0],
                                     ndrange,
                                     NULL,
                                     m_globalworksize,
                                     m_localworksize,
                                     0,
                                     NULL,
                                     NULL);
    checkErr(m_clerr);
  }
}

// Calls ClFinish for Current Command Queue
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::finish() {
  clFinish(m_clcmdqueue);
}

// Cleans Up OpenCL Memory Object
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::releaseMem(cl_mem memobj) {
  clReleaseMemObject(memobj);
}

// Frees all class memory
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::reset() {
  clearMem();
  Device currentdevice = getDevice();
  setDevice(currentdevice);
}

// Frees all class memory, including created singleton
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::teardown() {
  clearMem();
  delete(m_oclwrap);
  m_oclwrap = NULL;
}

// Clears all allocated memory except for singleton instance
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::clearMem() {
  std::map<std::string, cl_kernel>::iterator kernelitr;
  for (kernelitr = m_loadedkernels.begin();
    kernelitr != m_loadedkernels.end();
    ++kernelitr) {
      cl_kernel kernel = kernelitr->second;
      clReleaseKernel(kernel);
  }
  std::map<std::string, cl_program>::iterator progitr;
  for (progitr = m_loadedprograms.begin();
    progitr != m_loadedprograms.end();
    ++progitr) {
      cl_program prog = progitr->second;
      clReleaseProgram(prog);
  }

  clReleaseContext(m_clcontext);
  clReleaseCommandQueue(m_clcmdqueue);
}

// isKernelLoaded: check is kernel with kernelname was already built
////////////////////////////////////////////////////////////////////////////////
bool OCLwrapper::isKernelLoaded(std::string kernelname) {
  if (m_loadedkernels.find(kernelname) == m_loadedkernels.end())
    return false;
  return true;
}

// isProgramLoaded: checks if program with name sourcefilename is already loaded
////////////////////////////////////////////////////////////////////////////////
bool OCLwrapper::isProgramLoaded(std::string sourcefilename) {
  if (m_loadedprograms.find(sourcefilename) == m_loadedprograms.end())
    return false;
  return true;
}

// checkErr: checks for error
////////////////////////////////////////////////////////////////////////////////
void OCLwrapper::checkErr(cl_int errorcode) {
  if (errorcode == CL_SUCCESS) {
    return;
  }
  else {
    printf("OPENCL ERROR: %i\n", errorcode);
  }
}
