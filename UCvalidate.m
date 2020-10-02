function sys = UCvalidate(sys, printScreen)
% UCvalidate  - Shows a table of estimation and diagnostics results for UC
% models
%
%   sys = UCvalidate(sys)
%
%   Input:
%       sys: structure of type UComp created with UCmodel
%
%   Output:
%       The same input structure with the appropiate fields filled in, in particular:
%           table: Estimation and validation table
%           v: Estimated innovations (white noise correctly specified models)
%
%   Authors: Diego J. Pedregal, Nerea Urbina
%
%   Example:
%       load data/airpas
%       m = UCmodel(log(y), 12)
%       m = UCvalidate(m)
%
%   See also UC, UCcomponents, UCdisturb, UCestim, UCfilter, UCmodel, UCsetup, UCsmooth
    if nargin < 2
      printScreen = 1;
    end
    
    y = sys.y;
    u = sys.u;

    [v,table] = UCompC('validate',y,u,sys.model,sys.h,sys.comp,sys.compV,sys.p,...
        sys.v,sys.yFit,sys.yFor,sys.yFitV,sys.yForV,sys.a,sys.P,sys.eta,sys.eps,...
        sys.table,sys.outlier,sys.tTest,sys.criterion,sys.periods,sys.rhos,...
        sys.verbose,sys.stepwise,sys.p0,sys.cLlik,sys.criteria,sys.arma,sys.hidden);

    sys.table = table;
    sys.v = v;
    if printScreen
        for i = 1 : length(sys.table)
          disp(sys.table{i}(1 : end - 1))
        end
    end
      
end