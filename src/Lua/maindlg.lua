-- file: maindlg.lua
-- version: 1.0
-- date: 18/06/2012
require "FEMlib"

function BuildMaindlg()
  -- commonly used vars
  bgcolor   = "224 223 227"
  white     = "255 255 255"
  black     = "0   0   0  "


  --********************************************************************--
  ---               Variable Declaration and Definitions               ---
  --********************************************************************--
  -- Options Frame Data
  ------------------------------------------------------------------------
  local gaussptslbl  = iup.label{TITLE = "Gauss Points:"}
  local gausspts2tgl = iup.toggle{TITLE = "2"}
  local gausspts3tgl = iup.toggle{TITLE = "3"}
  local matformatlbl = iup.label{TITLE = "Matrix Format:"}
  local matformatlst = iup.list {
                         "Dense", "Diagonal", "CSR", "ELLPack", "ELLPack2",
                         "Eigen";
                         DROPDOWN = "YES",
                         BGCOLOR = white,
                         VALUE = "1",
                         VISIBLE_ITEMS = "7"
                       }
  local openmptgl  = iup.toggle{TITLE = "OpenMP"}
  local threadslbl = iup.label{TITLE = "  Number of Threads:", ACTIVE = "NO"}
  local threadslst = iup.list{DROPDOWN = "YES", BGCOLOR = white, ACTIVE = "NO",
                              SCROLLBAR = "NO"}
  local opencltgl  = iup.toggle{TITLE = "OpenCL (GPU)"}
  local colortgl   = iup.toggle{TITLE = "Mesh Coloring"}
  local solvetgl   = iup.toggle{TITLE = "Solve"}
  local viewtgl    = iup.toggle{TITLE = "View"}
  local asmbatchbtn= iup.button{TITLE = "Run Asm Batch"}
  local solbatchbtn= iup.button{TITLE = "Run Solver Batch"}

  -- File Frame Data
  ------------------------------------------------------------------------
  local filelbl   = iup.label {TITLE = " Input File:"; ALIGNMENT = "ABOTTOM"}
  local filebtn   = iup.button{IMAGE = "IUP_FileOpen"}
  local addbtn    = iup.button{IMAGE = "IUP_ActionOk"}
  local delbtn    = iup.button{IMAGE = "IUP_ActionCancel"}
  local fileindlg = iup.filedlg {
                      ALLOWNEW = "NO",
                      DIALOGTYPE = "Open",
                      MULTIPLEFILES = "YES",
                      EXTFILTER = "Neutral File| *.nf"
                    }
  local fileinlst = iup.list {
                      EDITBOX = "YES",
                      SCROLLBAR = "YES",
                      VISIBLELINES = "6",
                      BGCOLOR = white,
                      EXPAND = "YES",
                      SIZE = "450x"
                    };
  local fileoutlbl = iup.label{TITLE = "Output File:"}
  local fileouttxt = iup.text {
                       TITLE = "",
                       BGCOLOR = white,
                       SIZE = fileinlst.SIZE
                     }
  local fileoutbtn = iup.button{IMAGE = "IUP_FileOpen"}
  local fileoutdlg = iup.filedlg {
                       ALLOWNEW = "YES",
                       DIALOGTYPE = "Open",
                       EXTFILTER = "Text file| *.txt"
                     }
  local outappendlbl    = iup.label{TITLE = "Write mode:"}
  local outappendyestgl = iup.toggle{TITLE = "Append"}
  local outappendnotgl  = iup.toggle{TITLE = "Overwrite"}
  local outappendrdo = iup.radio {
    iup.hbox {outappendyestgl, outappendnotgl};
    VALUE = outappendyestgl
  }

  -- Bottom buttons frame data
  ------------------------------------------------------------------------
  local runbtn    = iup.button{TITLE = "  Run  ", SIZE = "80x14"}
  local quitbtn   = iup.button{TITLE = " Quit " , SIZE = "80x14"}

  --********************************************************************--
  ---                   Main Dialog GUI Construction                   ---
  --********************************************************************--
  local separator = iup.label{TITLE="-------------------------"}
  -- Options frame
  ------------------------------------------------------------------------
  local optionsfrm = iup.frame {
    iup.vbox {
      iup.hbox{matformatlbl, matformatlst},
      opencltgl,
      openmptgl,
      iup.hbox{threadslbl, threadslst},
      iup.hbox {
        gaussptslbl,
        iup.radio {
          iup.vbox {
            gausspts2tgl,
            gausspts3tgl
          }
        };
        ALIGNMENT = "ATOP"
      },
      colortgl,
      solvetgl,
      viewtgl,
      iup.hbox{ iup.fill{}, separator, iup.fill{} },
      asmbatchbtn,
      solbatchbtn
    };
    TITLE  = "Options",
    MARGIN = "4x4"
  }

  -- File input/output frame
  ------------------------------------------------------------------------
  local filesfrm = iup.frame {
    iup.vbox {
      iup.hbox{filelbl, fileinlst, addbtn, delbtn, filebtn},
      iup.hbox{fileoutlbl, fileouttxt, fileoutbtn},
      iup.hbox {
        outappendlbl, outappendrdo
      },
    };
    TITLE  = "I/O",
    MARGIN = "4x4"
  }
  -- Bottom buttons frame
  ------------------------------------------------------------------------
  local buttonfrm = iup.frame {
    iup.hbox{iup.fill{}, runbtn, quitbtn, iup.fill{};};
    MARGIN = "4x4"
  }

  -- Box for holding all components
  ------------------------------------------------------------------------
  local mainbox = iup.vbox {
    iup.hbox{optionsfrm, filesfrm},
    iup.hbox{buttonfrm};
    NMARGIN = "4x4"
  }

  -- main dialog construction
  local maindlg   = iup.dialog {
    mainbox;
    FONT         = "Consolas::9",
    NMARGIN      = "3x3",
    GAP          = "2x2",
    BGCOLOR      = "224 223 227",
    FGCOLOR      = "0 0 0",
    MAXBOX       = "YES",
    MINBOX       = "YES",
    RESIZE       = "NO",
    TITLE        = "FEM GPU",
    defaultenter =  runbtn
  }


  --********************************************************************--
  ---                       Main Dialog Callbacks                      ---
  --********************************************************************--
  -- OpenMP toggle callback
  ------------------------------------------------------------------------
  function openmptgl:action(state)
    if (state == 1) then
      threadslst.ACTIVE = "YES"
      threadslbl.ACTIVE = "YES"
    else
      threadslst.ACTIVE = "NO"
      threadslbl.ACTIVE = "NO"
    end
  end

  -- Threads list map callback
  ------------------------------------------------------------------------
  function threadslst:map_cb()
    local maxthreads = FEMlib.GetMaxThreads()
    for i = 1, maxthreads do
      threadslst["INSERTITEM"..tostring(i)] = tostring(i)
    end
    threadslst.VALUE = tostring(maxthreads)
    threadslst.VISIBLE_ITEMS = tostring(maxthreads + 1)
  end

  -- Open file button callback
  ------------------------------------------------------------------------
  function filebtn:action()
    fileindlg:popup()
    if (fileindlg.STATUS == "-1") then return end
    files = GetFileNames(fileindlg.value)
    -- If files are not already in list appends them to list
    for i = 1, table.getn(files) do
      local fileexists = false
      for j = 1, tonumber(fileinlst.COUNT) do
        if (fileinlst[tostring(j)] == files[i]) then
          fileexists = true
        end
      end
      if (fileexists == false) then fileinlst.APPENDITEM = files[i] end
    end
  end

  -- Add button callback
  ------------------------------------------------------------------------
  function addbtn:action()
    for i = 1, tonumber(fileinlst.COUNT) do
      if (fileinlst[i] == fileinlst.value) then
        return
      end
    end
    fileinlst.APPENDITEM = fileinlst.value
  end

  -- Delete button callback
  ------------------------------------------------------------------------
  function delbtn:action()
    for i = 1, tonumber(fileinlst.COUNT) do
      if (fileinlst[i] == fileinlst.value) then
        fileinlst.REMOVEITEM = tostring(i)
        break
      end
    end
  end

  -- Select output file button callback
  ------------------------------------------------------------------------
  function fileoutbtn:action()
    fileoutdlg:popup()
    if (fileoutdlg.STATUS == "-1") then return end
    if (fileoutdlg.STATUS ==  "0") then
      local outfilemode = iup.Alarm("File already exists!",
                                    "Append to existing file?",
                                    "Append",
                                    "Overwrite",
                                    "Cancel")
      -- 2 is option to cancel overwriting
      if (outfilemode == 1) then
        outappendrdo.value = outappendyestgl
      elseif (outfilemode == 2) then
        outappendrdo.value = outappendnotgl
      elseif (outfilemode == 3) then
        return
      end
    end

    fileouttxt.value = fileoutdlg.value
  end

  -- Assembly Batch Button Callback
  ------------------------------------------------------------------------
  function asmbatchbtn:action()
    RunAnalysis(2)
  end

  -- Solver Batch Button Callback
  ------------------------------------------------------------------------
  function solbatchbtn:action()
    RunAnalysis(3)
  end

  -- Run Button Callback
  ------------------------------------------------------------------------
  function runbtn:action()
    RunAnalysis(1)
  end

  -- RunAnalysis - batchmode(1-Interface, 2-Asm batch, 3- Solver batch)
  ------------------------------------------------------------------------
  function RunAnalysis(batchmode)
    local files = {}
    for i = 1, tonumber(fileinlst.COUNT) do
      files[i] = fileinlst[tostring(i)]
    end
    local matfmt   = tonumber(matformatlst.value)
    local ocl      = GetToggleVal(opencltgl.value)
    local nthreads = tonumber(threadslst.value)
    if (openmptgl.value == "OFF") then
      nthreads = 1
    end
    local gausspts = nil
    if (gausspts2tgl.value == "ON") then
      gausspts = 2
    else
      gausspts = 3
    end
    local omp        = GetToggleVal(openmptgl.value)
    local color      = GetToggleVal(colortgl.value)
    local solve      = GetToggleVal(solvetgl.value)
    local view       = GetToggleVal(viewtgl.value)
    local outfile    = fileouttxt.value
    local appendmode = GetToggleVal(outappendyestgl.value)

    FEMlib.RunAnalysis(files,
                       matfmt,
                       ocl,
                       nthreads,
                       gausspts,
                       color,
                       solve,
                       view,
                       outfile,
                       appendmode,
                       batchmode)
  end

  --Quit Button Callback
  ------------------------------------------------------------------------
  function quitbtn:action()
    maindlg:destroy()
  end

  return maindlg
end

--********************************************************************--
---                       Auxiliary Functions                        ---
--********************************************************************--
-- GetFileNames: Input is value from iupfiledlg, outputs table with each value
-- is a name of the selected files
------------------------------------------------------------------------
function GetFileNames(filedlgval)
  files = {}
  local idx = string.find(filedlgval, "|", 0, true)
  if (idx == nil) then
    files[1] = filedlgval
  else
    dir = string.sub(filedlgval, 0, (idx - 1)).."\\"
    filenames = string.sub(filedlgval, (idx + 1))
    idx = string.find(filenames, "|", 0, true)
    local i = 1
    while (idx ~= nil) do
      files[i] = dir..string.sub(filenames, 0, (idx - 1))
      filenames = string.sub(filenames, (idx + 1))
      idx = string.find(filenames, "|", 0, true)
      i = i + 1
    end
  end

  return files
end

-- GetToggleVal: Converts toggle value from string to boolean value
------------------------------------------------------------------------
function GetToggleVal(togglevalue)
  if (togglevalue == "ON") then
    return true;
  elseif (togglevalue == "OFF") then
    return false;
  else
    return nil;
  end
end

maindlg = BuildMaindlg()

-- Shows dialog
maindlg:show()
