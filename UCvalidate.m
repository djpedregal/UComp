function sys = UCvalidate(sys, printScreen)
% UCvalidate  - Shows a table of estimation and diagnostics results for UC
% models
%
%   sys = UCvalidate(sys)
%
%   sys: structure of type UComp created with UCmodel
%
%   Shows a table of estimation and diagnostics results for UC models.
%   The table shows information in four sections:
%   Firstly, information about the model estimated, the relevant 
%   periods of the seasonal component included, and further information aboutconvergence.
%   Secondly, parameters with their names are provided, the asymptotic standard errors, 
%   the ratio of the two, and the gradient at the optimum. One asterisk indicates 
%   concentrated-out parameters and two asterisks signals parameters constrained during estimation.
%   Thirdly, information criteria and the value of the log-likelihood.
%   Finally, diagnostic statistics about innovations, namely, the Ljung-Box Q test of absense
%   of autocorrelation statistic for several lags, the Jarque-Bera gaussianity test, and a
%   standard ratio of variances test.%   Input:
%
%   Output:
%       The same input structure with the appropiate fields filled in, in particular:
%           table: Estimation and validation table
%           v:     Estimated innovations (white noise correctly specified models)
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

    [v,table,coef] = UCompC('validate',sys.y,u,sys.model,sys.h,sys.comp,sys.compV,sys.v,sys.yFit,sys.yFor,sys.yFitV,...
        sys.yForV,sys.a,sys.P,sys.eta,sys.eps,sys.table,sys.outlier,sys.tTest,sys.criterion,...
        sys.periods,sys.rhos,sys.verbose,sys.stepwise,sys.p0,sys.criteria,sys.arma,sys.grad,...
        sys.covp,sys.p,sys.hidden);


    sys.table = table;
    sys.v = v;
    sys.p = coef;
    if printScreen
        for i = 1 : length(sys.table)
          disp(sys.table{i}(1 : end - 1))
        end
    end
      
end