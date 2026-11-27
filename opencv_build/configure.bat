@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
rem VENV: numpy가 설치된 Python 3.12 가상환경 위치로 바꿀 것
set VENV=D:\opencv-cuda\venv
set PATH=%VENV%\Scripts;%PATH%
cd /d D:\opencv-cuda
cmake -S opencv -B build -G Ninja ^
 -DCMAKE_BUILD_TYPE=Release ^
 -DCMAKE_INSTALL_PREFIX=D:/opencv-cuda/install ^
 -DOPENCV_EXTRA_MODULES_PATH=D:/opencv-cuda/opencv_contrib/modules ^
 -DBUILD_LIST=core,imgproc,imgcodecs,videoio,video,cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping,cudaoptflow,python3 ^
 -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DWITH_NVCUVID=OFF -DWITH_NVCUVENC=OFF -DWITH_CUFFT=ON -DWITH_CUBLAS=ON ^
 -DENABLE_CUDA_FIRST_CLASS_LANGUAGE=ON ^
 -DCUDA_ARCH_BIN=8.6 -DCUDA_ARCH_PTX= ^
 -DWITH_FFMPEG=ON -DWITH_OPENCL=OFF -DWITH_IPP=ON ^
 -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF ^
 -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=ON ^
 -DPYTHON3_EXECUTABLE=%VENV%\Scripts\python.exe ^
 -DOPENCV_PYTHON3_INSTALL_PATH=D:/opencv-cuda/install/python ^
 -DOPENCV_SKIP_PYTHON_LOADER=ON
