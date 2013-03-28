////////////////////////////////////////////////////////////////////////////////
// OpenCL Wrapper - warning: does work in the constructor and is a singleton!
////////////////////////////////////////////////////////////////////////////////
#ifndef OCLWRAPPER_H
#define OCLWRAPPER_H

#include <string>
#include <map>

#include "CL/opencl.h"

// macro for easy singleton access
#define OCL OCLwrapper::instance()

class OCLwrapper
{
public:
  static OCLwrapper& instance();

protected:
  OCLwrapper();
  virtual ~OCLwrapper();

public:
  typedef enum Device {
    CPU,
    GPU,
    NONE
  } Device;

  void   setDevice(Device device);
  Device getDevice();
  void   setDir(std::string dir);
  int    loadSource(std::string sourcefile);
  void   loadKernel(const char* kernelname);
  void   getBuildLog(cl_int err, cl_program* program, cl_device_id device);
  cl_mem createBuffer(size_t memsize, cl_mem_flags flag);
  void   enqueueWriteBuffer(cl_mem buffer, size_t memsize, void* hostData,
           cl_bool blocking);
  void   enqueueReadBuffer(cl_mem buffer, size_t memsize, void* hostData,
           cl_bool blocking);
  void   setKernelArg(int argnum, size_t argSpecifierSize,
           const void* argAddress);
  void   setGlobalWorksize(cl_uint dim, cl_uint value);
  void   setLocalWorksize(cl_uint dim, cl_uint value);
  bool   localSizeIsOK(size_t memsize);
  void   enquequeNDRangeKernel(cl_uint ndrange, bool getKernelTime);
  void   finish();
  void   releaseMem(cl_mem memobj);
  void   teardown();
  void   reset();
  double getKernelExecTime() const {return m_exectime;}
  void   setBuildOptions(std::string optionsStr) {
    m_buildoptions = optionsStr;
  }

private:
  void   getDeviceInfo();
  void   clearMem();
  bool   isKernelLoaded(std::string kernelname);
  bool   isProgramLoaded(std::string sourcefilename);
  void   checkErr(cl_int errorcode);

  //-------------------------------------------
  // member variables
  //-------------------------------------------
  // pointer to the singleton object
  static OCLwrapper* m_oclwrap;
  // OpenCL vars
  cl_program       m_clprogram[1];
  cl_kernel        m_clkernel[2];
  cl_command_queue m_clcmdqueue;
  cl_context       m_clcontext;
  cl_platform_id   m_clplatformid;
  cl_device_id     m_clcurrentdevice;
  cl_device_id     m_cldeviceCPU;
  cl_device_id     m_cldeviceGPU;
  cl_int           m_clerr;
  cl_event         m_clevents[2];
  cl_uint          m_computeunits;
  cl_uint          m_maxworkdim;
  size_t           m_maxworkgroupsize;
  size_t           m_maxworkitemsizes[3];
  cl_ulong         m_localmemsize;
  cl_ulong         m_globalmemsize;
  cl_ulong         m_globalmaxmemsize;
  // Aux vars
  bool             m_verbose;
  FILE*            m_clsrcfilehandle;
  std::string      m_buildoptions;
  bool             m_usecpu;
  double           m_exectime;
  size_t           m_globalworksize[3];
  size_t           m_localworksize[3];
  std::string      m_dir;
  std::string      m_filename;
  std::map<std::string, cl_kernel>  m_loadedkernels;
  std::map<std::string, cl_program> m_loadedprograms;
};

#endif
