function m = UCsetup(y,frequency,varargin)
% UCsetup - Sets up UC general univariate models 
%   
%   UCsetup sets up the model with a number of control variables that govern the
%   way the rest of functions in the package will work.
%
%   m = UCsetup(y,frequency)
%   m = UCsetup(y,frequency,'optionalvar1',optvar1,...,'optionalvarN',optvarN)
%
%   Inputs:
%       y: a time series to forecast.
%       frequency: fundamental period, number of observations per year.
%       periods: (opt) vector of fundamental period and harmonics. If not entered as input, 
%           it will be calculated from frequency.
%       u: (opt) a matrix of external regressors included only in the observation equation. 
%          If the output wanted to be forecast, matrix \code{u} should contain future values 
%          for inputs. Default: []
%       model: (opt) the model to estimate. It is a single string indicating the
%           type of model for each component. It allows two formats
%           'trend/seasonal/irregular' or 'trend/cycle/seasonal/irregular'. The
%           possibilities available for each component are:
%           - Trend: ? / none / rw / irw / llt / dt   
%           - Seasonal: ? / none / equal / different 
%           - Irregular: ? / none / arma(0,0) / arma(p,q) - with p and q
%               integer positive orders
%           - Cycles: ? / none / combination of positive or negative numbers.
%           Positive numbers fix the period of the cycle while negative
%           values estimate the period taking as initial condition the
%           absolute value of the period supplied.
%           Several cycles with positive or negative values are possible
%           and if a question mark is included, the model test for the
%           existence of the cycles specified (check the examples below). 
%           The following are valid cyckle models examples with different 
%           meanings: 48, 48?, -48, -48?, 48+60, -48+60, -48-60, 48-60, 
%                     48+60?, -48+60?, -48-60?, 48-60?.
%           Default: '?/none/?/?'
%       outlier: (opt) critical level of outlier tests. If NaN it does not
%           carry out any outlier detection (default). A negative value
%           indicates critical minimum t test for one run of outlier detection after
%           identification. A positive value indicates the critical
%           minimum t test for outlier detection in any model during identification.
%           Default: NaN
%       stepwise: (opt) stepwise identification procedure (true,false). 
%           Default: false.
%       tTest: (opt) augmented Dickey Fuller test for unit roots (true/false).
%           The number of models to search for is reduced, depending on the
%           result of this test. Default: false
%       p0: (opt) initial condition for parameter estimates. Default: NaN
%       h: (opt) forecast horizon. If the model includes inputs h is not used,
%           the length of u is used instead. Default: NaN
%       criterion: (opt) information criterion for identification ('aic','bic' or
%           'aicc'). Default: 'aic'
%       verbose: (opt) intermediate results shown about progress of estimation
%           (true/false). Default: false.
%       arma: (opt) check for arma models for irregular components (true/false).
%           Default: true
%       cLlik: (opt) reserved input
%
%   Output:
%       An object of class UComp. It is a structure with fields including all
%       the inputs and the fields listed below as outputs. All the
%       functions in this package fill in part of the fields of any UComp
%       object as specified in what follows (function UC fills in all of
%       them at once):
%           After running UCmodel or UCestim:
%               p: Estimated parameters
%               v: Estimated innovations (white noise correctly specified
%                   models)
%               yFor: Forecasted values of output
%               yForV: Variance of forecasted values of output
%               criteria: Value of criteria for estimated model
%               iter: Number of iterations in estimation
%               grad: Gradient at estimated parameters
%               covp: Covariance matrix of parameters
% 
%           After running UCvalidate:
%               table: Estimation and validation table
%               v: Estimated innovations (white noise correctly specified models)
%
%           After running UCcomponents:
%               comp: Estimated components in table form
%               compV: Estimated components variance in table form
%             
%           After running UCfilter, UCsmooth or UCdisturb:
%               yFit: Fitted values of output
%               yFitV: Variance of fitted values of output
%               a: State estimates
%               P: Variance of state estimates
%              
%           After running UCdisturb:
%               eta: State perturbations estimates
%               eps: Observed perturbations estimates
%
%   Authors: Diego J. Pedregal, Nerea Urbina
%
%   Examples:
%       load data/airpas
%       m = UCsetup(log(y),frequency)
%       m = UCsetup(log(y),frequency,'model','llt/equal/arma(0,0)')
%       m = UCsetup(log(y),frequency,'outlier',4)
%
%   See also UC, UCcomponents, UCdisturb, UCestim, UCfilter, UCmodel,
%   UCsmooth, UCvalidate

    %Set default values
    p = inputParser;
    addRequired(p,'y',@isfloat);
    addRequired(p,'frequency',@isfloat);
    defaultU = []; addParameter(p,'u',defaultU,@isfloat);
    defaultPeriods = NaN; addParameter(p,'periods',defaultPeriods,@isfloat);
    defaultModel = '?/none/?/?'; addParameter(p,'model',defaultModel,@ischar);
    defaultH = 9999; addParameter(p,'h',defaultH,@isfloat);
    defaultOutlier = 9999; addParameter(p,'outlier',defaultOutlier,@isfloat);
    defaultTTest = false; addParameter(p,'tTest',defaultTTest,@islogical);
    defaultCriterion = 'aic'; addParameter(p,'criterion',defaultCriterion,@ischar);
    defaultVerbose = false; addParameter(p,'verbose',defaultVerbose,@islogical);
    defaultStepwise = false; addParameter(p,'stepwise',defaultStepwise,@islogical);
    defaultP0 = -9999.9; addParameter(p,'p0',defaultP0,@isfloat);
    %defaultCLlik = true; addParameter(p,'cLlik',defaultCLlik,@islogical);
    defaultArma = false; addParameter(p,'armaIdent',defaultArma,@islogical);
    defaultLambda = 1.0; addParameter(p,'lambda',defaultLambda,@isfloat);
    defaultTVP = []; addParameter(p,'TVP',defaultTVP,@isfloat);
    defaultTrendOptions = 'none/rw/llt/dt'; addParameter(p,'trendOptions',defaultTrendOptions,@ischar);
    defaultSeasonalOptions = 'none/equal/different'; addParameter(p,'seasonalOptions',defaultSeasonalOptions,@ischar);
    defaultIrregularOptions = 'none/arma(0,0)'; addParameter(p,'irregularOptions',defaultIrregularOptions,@ischar);
    defaultMSOE = false; addParameter(p,'MSOE',defaultVerbose,@islogical);
    defaultPTSnames = false; addParameter(p,'PTSnames',defaultVerbose,@islogical);

    parse(p,y,frequency,varargin{:});
    
%     h = p.Results.h;
%     if(~isnan(h) && floor(h) ~= h)
%         h = floor(h);
%     end

    model = p.Results.model;
    outlier = p.Results.outlier;
    tTest = p.Results.tTest;
    criterion = p.Results.criterion;
    verbose = p.Results.verbose;
    stepwise = p.Results.stepwise;
    p0 = p.Results.p0(:);
    %cLlik = p.Results.cLlik;
    arma = p.Results.armaIdent;
    h = p.Results.h;
    lambda = p.Results.lambda;
    TVP = p.Results.TVP;
    MSOE = p.Results.MSOE;
    PTSnames = p.Results.PTSnames;

%     rhos = NaN;
%     p = NaN;
    % Checking u
    u = p.Results.u;
    if size(u, 1) > size(u, 2)
        u = u';
    end
    if isempty(u)
        u = [0 0];
    end
    if any(size(y) == 2)
        error('ERROR: output varialbe should be a scalar time series!!');
    end
    % Checking periods
    periods = p.Results.periods;
    if(frequency>1 && isnan(periods(1)))
        periods = frequency./(1:floor(frequency/2));
    elseif(isnan(periods(1)) && frequency<=1)
        periods=1;
    end
    periods = periods(:);
    if isnan(lambda)
        lambda = 9999.99;
    end
%     periods=periods';    periods = p.Results.periods;
%     if (is.ts(y) && is.na(periods) && frequency(y) > 1){
%         periods = frequency(y) / (1 : floor(frequency(y) / 2))
%     } else if (is.ts(y) && is.na(periods)){
%         periods = 1
%     } else if (!is.ts(y) && is.na(periods)){
%         stop("Input \"periods\" should be supplied!!")
%     }
    rhos = ones(length(periods), 1);
    if isempty(TVP) && sum(u) ~= 0
        TVP = zeros(1, size(u, 1));
    end
    if isempty(TVP)
        TVP = -9999.99;
    end

    %Output:
    hidden = struct('d_t',NaN,'estimOk','Not estimated','objFunValue',0,...
        'innVariance',1,'nonStationaryTerms',NaN,'ns',NaN,'nPar',NaN,'harmonics',NaN,...
        'constPar',NaN,'typePar',NaN,'cycleLimits',NaN,'typeOutliers',-ones(1,2), ...
        'beta',NaN,'betaV',NaN,'truePar',NaN,'seas',frequency,'iter',0,'MSOE',MSOE,...
        'PTSnames',PTSnames);
    m = struct('y',y,'u',u,'model',model,'h',h,'comp',NaN,'compV',NaN,'v',NaN, ...
        'yFit',NaN,'yFor',NaN,'yFitV',NaN,'yForV',NaN,'a',NaN,'P',NaN,'eta',NaN,'eps',NaN,...
        'table','','outlier',-abs(outlier),'tTest',tTest,'criterion',criterion,...
        'periods',periods,'rhos',rhos,'verbose',verbose,'stepwise',stepwise,'p0',p0,...
        'criteria',NaN,'arma',arma,'grad',NaN,'covp',NaN,'p',NaN, 'lambda', lambda, 'TVP', TVP, ...
        'trendOptions', p.Results.trendOptions, 'seasonalOptions', p.Results.seasonalOptions, ...
        'irregularOptions', p.Results.irregularOptions, 'hidden',hidden);

end