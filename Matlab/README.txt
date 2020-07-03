This folder includes a MEX file ready to use.
It has been built on Windows using MinGW-W64 compiler and MATLAB R2019a.
In addition, the MEX file has been linked to LAPACK/BLAS libraries included in this folder.

IMPORTANT: It is not possible to run a MEX file from a different platform.

Visit this website before you run the MEX file included:
https://es.mathworks.com/help/matlab/matlab_external/before-you-run-a-mex-file.html?lang=en

------

Instructions for building the MEX file from source code included on different platforms.

1. Download Armadillo: http://arma.sourceforge.net/download.html
(See intallation notes to get information about external dependencies)

2. Download a compatible compiler with MATLAB: https://es.mathworks.com/support/requirements/supported-compilers.html

3. Add UCompC.cpp to the current path.

4. In the command window, type: mex -setup cpp. With this command you should be able to choose a compiler.

5. Build the MEX file linking the source file with Armadillo and LAPACK/BLAS/OpenBLAS libraries. 
To do that, type in the command window: mex -Ipath\to\armadillo\include -Lpath\to\libraries -llapack -lblas UCompC.cpp
e.g: mex -IC:\armadillo-9.850.1\include -LC:\..\UComp\libs -llapack -lblas UCompC.cpp


6. To run the MEX function, add to your path the BLAS and LAPACK libraries you have linked to.

You can find more information about how to build MEX files in: https://es.mathworks.com/help/matlab/matlab_external/build-c-mex-programs.html?lang=en

------

It is recommended to compile on your own PC the source file UCompC.cpp.

MATLAB documentation about MEX files: https://es.mathworks.com/help/matlab/call-mex-file-functions.html?lang=en