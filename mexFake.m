clc
armadilloPath = 'D:\OneDrive - Universidad de Castilla-La Mancha\cursoCorriente\C++\armadillo-12.8.4';
lapackPath = 'D:\OneDrive - Universidad de Castilla-La Mancha\cursoCorriente\C++\armadillo-12.8.4\examples\lib_win64';
mexUComp(armadilloPath, lapackPath);
% mexUComp('/Users/diego.pedregal/Desktop/armadillo-12.8.4');
load airpas
y = log(airpas);
m = ARIMAmodel(y, 12, 'verbose', true);

y(y > 6) = 6;
m = TETS(y, 12, 'Ymax', 6);



% eval(['mex mymex.cpp -I' armadilloPath '/include -L' ...
%     armadilloPath '/lib -larmadillo']);
% 
% mex('ETSc.cpp', ['-I' armadilloPath '/include'], ['-L' lapackPath], '-lopenblas');
% 
% mex('mymex.cpp', ['-I' armadilloPath '/include'], ['-L' armadilloPath '/lib'], '-larmadillo');
% 
% 
% mex('mymex.cpp', ['-I' armadilloPath '/include'], ['-L' lapackPath], '-lopenblas');
% 
