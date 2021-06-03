function mexUComp(armadilloPath, lapackPath)
%
% mexUComp(armadilloPath, lapackPath)
%
% Function to compile source UComp code
%
% armadilloPath: string with the path to armadillo library
% lapackPath:    string with the path to Lapack and Blas libraries
if nargin < 1
    error('Please supply Armadillo path!!');
end
if nargin < 2
    lapackPath = '';
end
% Get current path
initialPath = pwd;
currentPath = which('mexUComp.m');
ind = strfind(currentPath, 'mexUComp.m') - 2;
currentPath = currentPath(1 : ind);
cd(currentPath);
if exist('OCTAVE_VERSION', 'builtin')
  copyfile('cpp/UCompCOctave.cpp', 'UCompC.cpp');
else
  copyfile('cpp/UCompCMatlab.cpp', 'UCompC.cpp');
end
% filename = [currentPath '/' platform '/UCompC.cpp'];
if ispc
% lapackPath = [matlabroot '\extern\lib\win64\microsoft']
% mex('UCompC.cpp', ['-I' armadilloPath '/include'], ['-L' lapackPath], '-lmwlapack', '-lmwblas');
% Mex pc
    if isempty(lapackPath)
        mex('UCompC.cpp', ['-I' armadilloPath '/include'], '-llapack', '-lblas');
    else
        mex('UCompC.cpp', ['-I' armadilloPath '/include'], ['-L' lapackPath], '-llapack', '-lblas');
    end
else
% Mex mac or Linux
    mex('UCompC.cpp', ['-I' armadilloPath '/include'], '-llapack', '-lblas');
end
delete('UCompC.cpp');
cd(initialPath);
end