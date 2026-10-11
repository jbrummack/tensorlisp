@echo off
rem Compiles ggml-cuda sources (vendor/ggml/src/ggml-cuda) to PTX so the instruction mix of the
rem industry's kernels can be read: scripts\ggml-cuda-ptx.bat mmvq softmax norm template-instances/mmq-instance-q8_0
rem Output: target\ggml-ptx\<name>.ptx. Needs the CUDA toolkit (CUDA_PATH) and Visual Studio's MSVC.
setlocal
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VS=%%i
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
set ROOT=%~dp0..
set SRC=%ROOT%\vendor\ggml\src\ggml-cuda
set OUT=%ROOT%\target\ggml-ptx
if not exist "%OUT%" mkdir "%OUT%"
set INC=-I%ROOT%\vendor\ggml\include -I%ROOT%\vendor\ggml\src -I%SRC%
for %%F in (%*) do (
  echo compiling %%F
  "%CUDA_PATH%\bin\nvcc.exe" --ptx -arch=sm_86 -O3 -std=c++17 -DGGML_CUDA_NO_VMM %INC% "%SRC%\%%F.cu" -o "%OUT%\%%~nF.ptx" 2>&1 | findstr /i "error"
)
