
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

