A quick python & C++ project to use a GPU to perform a large number of fits to data in parallel. The python script, python_wrapper.py, generates a large number (currently 1 million) individual gaussian distributions each of 100 data points.

The data is generated as a matrix of numpy arrays. Then, using the python ctypes module, the numpy arrays are passed into a compiled dynamic link library (.dll) function. This library is first created by compiling my "GPU_Fit.cpp" script. 

The GPU_Fit script makes a call to the third-party, open source, gpufit library package which is described here: gpufit.readthedocs.io/en/latest/introduction.html
The third-party gpufit.h and Gpufit.dll are included as these are dependencies of the .dll library created as part of the C++ compilation.

The C++-side of the project was compiled using Microsoft Visual Studio 2015.
