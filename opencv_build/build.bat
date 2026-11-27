@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
rem VENV: numpy가 설치된 Python 3.12 가상환경 위치로 바꿀 것
set VENV=D:\opencv-cuda\venv
set PATH=%VENV%\Scripts;%PATH%
cd /d D:\opencv-cuda
cmake --build build --target install -j 16
