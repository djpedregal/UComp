function mexUComp(armadilloPath, lapackPath)
if nargin < 1
    error('Please supply Armaillo path!!');
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
if ismac
% Mex mac
    mex('UCompC.cpp', ['-I' armadilloPath '/include'], '-llapack', '-lblas');
elseif ispc
% Mex pc
    if isempty(lapackPath)
        mex('UCompC.cpp', ['-I' armadilloPath '/include'], '-llapack', '-lblas');
    else
        mex('UCompC.cpp', ['-I' armadilloPath '/include'], ['-L' lapackPath], '-llapack', '-lblas');
    end
else
% Mex linux
end
cd(initialPath);


% mex -I"/Users/diegopedregal/Google Drive/MATLAB/UCompMATLAB/armadillo/include" -llapack -lblas UCompC.cpp
% 
% mex -I"/Users/diegopedregal/Google Drive/MATLAB/UCompMATLAB/armadillo/include" -L"/Users/diegopedregal/Google Drive/MATLAB/UCompMATLAB/armadillo/examples/lib_win64" -llapack -lblas UCompC.cpp

end