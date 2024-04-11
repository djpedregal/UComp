function sys = UCestim(sys)
% UCestim - Estimates and forecasts UC models
%    
%   UCestim estimates and forecasts a time series using an UC model
%
%   sys = UCestim(sys)   
%
%   The optimization method is a BFGS quasi-Newton algorithm with a 
%   backtracking line search using Armijo conditions.
%   Parameter names in output table are the following:
%        Damping:   Damping factor for DT trend.}
%        Level:     Variance of level disturbance.}
%        Slope:     Variance of slope disturbance.}
%        Rho(#):    Damping factor of cycle #.}
%        Period(#): EStimated period of cycle #.}
%        Var(#):    Variance of cycle #.}
%        Seas(#)    Seasonal harmonic with period #.}
%        Irregular: Variance of irregular component.}
%        AR(#):     AR parameter of lag #.}
%        MA(#):     MA parameter of lag #.}
%        AO#:       Additive outlier in observation #.}
%        LS#:       Level shift outlier in observation #.}
%        SC#:       Slope change outlier in observation #.}
%        Beta(#):   Beta parameter of input #.}
%
%   Input:
%       sys: structure of type UComp created with UCmodel or UCsetup
%    
%   Output:
%       The same input structure with the appropiate fields filled in, in particular:
%           p:        Estimated parameters
%           v:        Estimated innovations (white noise in correctly specified models)
%           yFor:     Forecasted values of output
%           yForV:    Variance of forecasted values of output
%           criteria: Value of criteria for estimated model
%           covp:     Covariance matrix of estimated transformed parameters
%           grad:     Gradient of log-likelihood at the optimum
%           iter:     Estimation iterations
%    
%   Authors: Diego J. Pedregal, Nerea Urbina
%    
%   Examples:
%       load data/airpas
%       m = UCsetup(log(y), 12)
%       m = UCestim(m)
%        
%   See also UC, UCcomponents, UCdisturb, UCsmooth, UCfilter, UCmodel,
%   UCsetup, UCvalidate

    %Clear  variables to make new estimation
    sys.table = '';
    sys.hidden.constPar = NaN;

    %Estimation
    u = sys.u;
    nu = size(u,2);
    kInitial = size(u,1);
    if(nu == 2)
        nu = length(sys.y)+sys.h;
        kInitial = 0;
    end

    [p,p0,model,yFor,periods,rhos,yForV,estimOk,harmonics,...
        cycleLimits,nonStationaryTerms,beta,betaV,u,typeOutliers,...
        criteria,d_t,innVariance,objFunValue,grad,constPar,typePar,...
        ns,nPar,h,outlier,Iter,lambda] = UCompC('estimate',...
        sys.y,u,sys.model,sys.h,sys.comp,sys.compV,sys.v,sys.yFit,sys.yFor,sys.yFitV,...
        sys.yForV,sys.a,sys.P,sys.eta,sys.eps,sys.table,sys.outlier,sys.tTest,sys.criterion,...
        sys.periods,sys.rhos,sys.verbose,sys.stepwise,sys.p0,sys.criteria,sys.arma,sys.grad,...
        sys.covp,sys.p,sys.lambda,sys.TVP,sys.trendOptions, sys.seasonalOptions, ...
        sys.irregularOptions,sys.hidden);

    if (model == "error")
        sys.model = "error";
        return;
    end
    if(~isempty(yFor))
        sys.yFor = yFor;
        sys.yForV = yForV;
    end
%     sys.p = p;
    sys.hidden.truePar = p;
    sys.p0 = p0;
    if(regexp(sys.model,'\?'))
        sys.model = model;
    end
    %sys.hidden.grad = grad;
    sys.grad = grad;
    sys.hidden.constPar = constPar;
    sys.hidden.typePar = typePar;
    if(size(cycleLimits) == 1)
            sys.hidden.cycleLimits = cycleLimits*ones(round(numel(cycleLimits)/2),2);
    else
            sys.hidden.cycleLimits = reshape(cycleLimits,round(numel(cycleLimits)/2),2);
    end
    sys.hidden.d_t = d_t;
    sys.hidden.innVariance = innVariance;
    sys.hidden.objFunValue = objFunValue;
    sys.hidden.beta = beta;
    sys.hidden.betaV = betaV;
    sys.periods = periods;
    sys.h = h;
    sys.outlier = outlier;
    sys.rhos = rhos;
    if isempty(sys.rhos)
        sys.rhos = ones(length(sys.periods), 1);
    end
    sys.hidden.estimOk = estimOk;
    sys.hidden.nonStationaryTerms = nonStationaryTerms;
    sys.hidden.ns = ns;
    sys.hidden.nPar = nPar;
    sys.hidden.harmonics = harmonics;
    sys.hidden.iter = Iter;
    sys.hidden.lambda = lambda;
    if isempty(sys.hidden.harmonics)
        sys.hidden.harmonics = nan(length(sys.periods), 1);
    end
    if(size(criteria) == 1)
            sys.criteria = criteria*ones(4, 1);
    else
            sys.criteria = reshape(criteria, 4, 1);
    end
    if(~isnan(sys.outlier) && ~isempty(u))
        nu = length(sys.y) + sys.h;
        k = numel(u)/nu;
        nOut = k-kInitial;
        if(nOut > 0)
            sys.u = reshape(u, k, nu);
            sys.hidden.typeOutliers = typeOutliers;
%             if(size(u) == 1)
%                 sys.u = u*ones(k,nu);
%             else
%                 sys.u = reshape(u,k,nu);
%             end
%             sys.hidden.typeOutliers = NaN(nOut,2);
%             sys.hidden.typeOutliers(:,1) = typeOutliers;
%             for i=1:nOut
%                 sys.hidden.typeOutliers(i,2) = find(sys.u(kInitial+i,:)==1,1); 
%             end
        end
    end
end

