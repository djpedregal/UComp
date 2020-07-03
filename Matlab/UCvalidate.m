function sys = UCvalidate(sys)
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
%       load 'airpassengers' - contains 2 variables: y, frequency
%       m = UCmodel(log(y),frequency)
%       m = UCvalidate(m)
%
%   See also UC, UCcomponents, UCdisturb, UCestim, UCfilter, UCmodel, UCsetup, UCsmooth

    y = sys.y;
    u = sys.u;

    if(istable(sys.criteria))
        criteria = table2array(sys.criteria)';
    else
        criteria = sys.criteria;
    end

    [v,table] = UCompC('validate',y,u,sys.model,sys.h,sys.comp,sys.compV,sys.p,...
        sys.v,sys.yFit,sys.yFor,sys.yFitV,sys.yForV,sys.a,sys.P,sys.eta,sys.eps,...
        sys.table,sys.outlier,sys.tTest,sys.criterion,sys.periods,sys.rhos,...
        sys.verbose,sys.stepwise,sys.p0,sys.cLlik,criteria,sys.arma,sys.hidden);

    sys.table = table;
    sys.v = v;

end