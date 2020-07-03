function mexUComp(armadilloPath, lapackPath)
%
% mexUComp(armadilloPath, lapackPath)
%
% Function to compile source UComp code
%
% armadilloPath: string with the path to armadillo library
% lapackPath:    string with the path to Lapack and Blas libraries
if nargin < 1
    error('Please supply Armaillo path!!');
end
if nargin < 2
    lapackPath = '';
end
% Get current path
initialPath = pwd;
currentPath = which('mexUComp.m');
ind = findstr(currentPath, 'mexUComp.m') - 2;
currentPath = currentPath(1 : ind);
cd(currentPath);
if ispc
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
cd(initialPath);
end