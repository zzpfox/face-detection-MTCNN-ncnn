set OpenCV_DIR=C:/library/opencv-3.3.1/opencv/build
set "VSCMD_START_DIR=%CD%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64



IF NOT EXIST build-vs2017-x64 mkdir build-vs2017-x64
cd build-vs2017-x64

cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..
nmake

xcopy /s/i/y/q ..\model\* .