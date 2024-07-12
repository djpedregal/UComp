function m1 = TETS(y, s, varargin)
% system = TETS(y, period, 'inp1', inp1, 'inp2', inp2, ...)
%
% Runs a TOBIT ETS general univariate models on a time series
% 
% See help of TETSmodel
%    
% Author: Diego J. Pedregal
% 
% See also: TETSmodel, TETSvalidate, TETScomponents, TETSestim
% 
% Examples:
%    m = TETS(y, 12);
%    m = TETS(y, 12, 'model', '???');
%    m = TETS(y, 12, 'model', '?AA');
    m1 = TETSsetup(y, s, varargin{:});
    if (min(m1.Ymax, [], 'omitnan') > max(y, [], 'omitnan') && ...
            max(m1.Ymin, [], 'omitnan') < min(y, [], 'omitnan'))
        m1 = ETSvalidate(m1);
    else
        m1 = TETSvalidate(m1);
    end
end
