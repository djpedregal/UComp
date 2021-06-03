function p0 = getp0(y, freq, model, periods)
% getp0 - Provides initial parameters of a given model for the time series.
%    They may be changed arbitrarily by the user to include as an input p0 to
%    UC or UCmodel functions (see example below).
%    There is no guarantee that the model will converge and selecting initial conditions
%    should be used with care.
%
%  p0 = getp0(y, frequency, model, periods)
%
%  Inputs:
%       y: a time series to model.
%       frequency: frequency of time series.
%       model: UComp model in a string.
%       periods: (opt) vector of fundamental period and harmonics. If not entered as input, 
%           it will be calculated from frequency.
%
%   Examples:
%       load data/airpas
%       p0 = getp0(log(airpas), 12);
%       p0 = getp0(log(airpas), 12, 'llt/different/arma(1,1)');
    if nargin < 3
        model = 'llt/none/equal/arma(0,0)';
    end
    if nargin < 4
        periods = freq ./ (1 : floor(freq / 2));
    end
    
    if containsO(model, '?')
        error("UComp ERROR: Model should not contain any \'?\'!!!")
    end
    sys = UCsetup(y, freq, 'model', model, 'periods', periods, 'verbose', false);
    sys = UCestim(sys);
    p1 = coef(sys);
    if ~exist ('OCTAVE_VERSION', 'builtin')   % Matlab
      p0 = array2table(sys.p0,'Rownames',p1.Properties.RowNames,'VariableNames', {'Param'});
    else
      p0 = sys.p0;
    end
end