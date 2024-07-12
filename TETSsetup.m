function out = TETSsetup(y, s, varargin)
% system = TETSsetup(y, period, 'inp1', inp1, 'inp2', inp2, ...)
%
% Sets up TOBIT TETS general univariate models
% 
% See help of TETSmodel
%    
% Author: Diego J. Pedregal
% 
% See also: TETS, TETSmodel, TETSvalidate, TETScomponents, TETSestim
% 
% Examples:
%    m = TETSsetup(y, 12);
%    m = TETSsetup(y, 12, 'model', '???');
%    m = TETSsetup(y, 12, 'model', '?AA');
    Ymaxi = find(strcmpi('Ymax', varargin));
    if isempty(Ymaxi)
        Ymax = Inf;
    else
        Ymax = varargin{Ymaxi + 1};
        varargin([Ymaxi Ymaxi + 1]) = [];
    end
    Ymini = find(strcmpi('Ymin', varargin));
    if isempty(Ymini)
        Ymin = -Inf;
    else
        Ymin = varargin{Ymini + 1};
        varargin([Ymini Ymini + 1]) = [];
    end
    out = ETSsetup(y, s, varargin{:});
    out.Ymin = Ymin;
    out.Ymax = Ymax;
end
