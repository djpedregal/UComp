
------Instructions for building the MEX file from source code included on different platforms.

1. Download Armadillo: http://arma.sourceforge.net/download.html(See installation notes to get information about external dependencies)

2. If not installed, download a compatible C++ compiler with MATLAB: https://es.mathworks.com/support/requirements/supported-compilers.html

3. Go to folder where UComp is copied/installed.

4. In the command window, type: mex -setup cpp. With this command you should be able to choose a compiler.

5. Build the MEX file linking the source file with Armadillo and LAPACK/BLAS/OpenBLAS libraries. To do that, use mexUComp.m installer supplied with UComp. In most systems mexUComp only needs the first input that tells the folder where Armadillo library lives. If necessary, a second input tells where libraries Lapack and Blas or substitutes live. Folder cpp includes some pre-compiled versions of these libraries for Windows systems.

Pre-compiled versions of Lapack and Blas libraries for different platforms are available at:
https://icl.cs.utk.edu/lapack-for-windows/lapack/

------It is recommended to compile on your own PC the source file UCompC.cpp.

Further information about MEX files:

https://es.mathworks.com/help/matlab/matlab_external/build-c-mex-programs.html?lang=en

https://es.mathworks.com/help/matlab/call-mex-file-functions.html?lang=en

https://es.mathworks.com/help/matlab/matlab_external/before-you-run-a-mex-file.html?lang=en

------Using C++ files directly or linking to other environments
UCompC in cpp folder shows an example of how a UComp model may be run in pure C++ code. This can be useful to illustrate how to integrate the C++ code directly in your own C++ code.

------Integrating C++ files in a Python or other environments
Integration of UComp C++ files into other environments (say Python) is more complex, and basically consists on writing wrappers in such environment (Python) similar to functions UCompC (this is the most complex, since it involves integrating Python with C++ using the Armadillo library), UC, UCmodel, UCsetup, UCfilter, UCsmooth, UCdisturb, UCcomponents, UCestim and UChp.

