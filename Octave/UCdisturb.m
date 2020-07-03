function sys = UCdisturb(sys)
% UCdisturb - Runs the Disturbance Smoother for UC models
%    
%   sys = UCdisturb(sys)   
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
%           eta: State perturbations estimates
%           eps: Observed perturbations estimates        
%
%   Authors: Diego J. Pedregal, Nerea Urbina
%    
%   Examples:
%       load 'airpassengers' - contains 2 variables: y, frequency
%       m = UCmodel(log(y),frequency)
%       m = UCdisturb(m)
%        
%   See also UC, UCcomponents, UCestim, UCfilter, UCmodel, UCsetup, UCsmooth, UCvalidate

    sys = filter_(sys,'disturb');

end