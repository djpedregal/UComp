function sys = UCfilter(sys)
% UCfilter - Runs the Kalman Filter for UC models
%    
%   sys = UCfilter(sys)   
%
%   Input:
%       sys: structure of type UComp created with UCmodel
%    
%   Output:
%       The same input struct with the appropiate fields filled in, in particular:
%           yFit: Fitted values of output
%           yFitV: Variance of fitted values of output
%           a: State estimates
%           P: Variance of state estimates
%    
%   Authors: Diego J. Pedregal, Nerea Urbina
%    
%   Examples:
%       load 'airpassengers' - contains 2 variables: y, frequency
%       m = UCmodel(log(y),frequency)
%       m = UCfilter(m)
%        
%   See also UC, UCcomponents, UCdisturb, UCestim, UCmodel, UCsetup, UCsmooth, UCvalidate

    sys = filter_(sys,'filter');
    
end