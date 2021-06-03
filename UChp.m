function cycle = UChp(y, frequency, lambda)
% UChp - Hodrick-Prescott filter estimation
%    
%   UChp returns the cycle extracted by the HP filter
%
%   cycle = UChp(y, frequency, lambda)   
%
%   Inputs:
%       y: a time series.
%       frequency: fundamental period, number of observations per year.
%       lambda: smoothing constant for estimation (1600 by default)
%   Output:
%       cycle: HP estimated cycle
%    
%   Author: Diego J. Pedregal
%    
  if (nargin < 3)
    lambda = 1600;
  end
  m = UCsetup(y, frequency, 'model', 'irw/none/arma(0,0)');
  m.p = [log(1 / lambda) / 2; 0];
  m = UCcomponents(m);
  if exist('OCTAVE_VERSION', 'builtin')
    cycle = y - m.comp(1 : length(y), 1);
  else
    cycle = y - table2array(m.comp(1 : length(y), 1));
  end
end
