This folder includes a MEX file ready to use.
It has been built on Windows 10 using MinGW-W64 compiler and Octave 5.2.

In addition, the MEX file has been linked to LAPACK/BLAS libraries included in this folder.
You can find this libraries in Octave installation folder Octave/mingw64 and build the source file at your own.



IMPORTANT: It is not possible to run a MEX file from a different platform.



Octave documentation about MEX files:

https://octave.org/doc/v5.2.0/Mex_002dFiles.html#Mex_002dFiles



------



Instructions for building the MEX file from source code included on different platforms.



1. Download Armadillo: http://arma.sourceforge.net/download.html

(See intallation notes to get information about external dependencies)



2. Add UCompC.cpp to the current path.



3. Build the MEX file linking the source file with Armadillo and LAPACK/BLAS/OpenBLAS libraries. 

To do that, type in the command window: mex -Ipath\to\armadillo\include (-Lpath\to\lapack-blas\libraries) -llapack -lblas UCompC.cpp

e.g: mex -IC:\armadillo-9.850.1\include -llapack -lblas UCompC.cpp



4. To run the MEX function, add to your path the BLAS and LAPACK libraries you have linked in case
you have used different ones than those provided by Octave.

