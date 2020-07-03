function sys = UCestim(sys)
% UCestim - Estimates and forecasts UC models
%    
%   UCestim estimates and forecasts a time series using an UC model
%
%   sys = UCestim(sys)   
%
%   Input:
%       sys: structure of type UComp created with UCsetup or UCmodel
%    
%   Output:
%       The same input struct with the appropiate fields filled in, in particular:
%           p: Estimated parameters
%           v: Estimated innovations (white noise in correctly specified models)
%           yFor: Forecasted values of output
%           yForV: Variance of forecasted values of output
%           criteria: Value of criteria for estimated model
%    
%   Authors: Diego J. Pedregal, Nerea Urbina
%    
%   Examples:
%       load 'airpassengers' - contains 2 variables: y, frequency
%       m = UCsetup(log(y),frequency)
%       m = UCestim(m)
%        
%   See also UC, UCcomponents, UCdisturb, UCfilter, UCmodel, UCsetup, UCsmooth, UCvalidate

    %Clear  variables to make new estimation
    sys.table = '';
    sys.hidden.constPar = NaN;
    if(isstruct(sys.criteria))
    	criteria=cell2mat(struct2cell(sys.criteria));
    else
    	criteria=sys.criteria;
    end

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
        criteria,d_t,innVariance,objFunValue,grad,constPar,typePar,ns,nPar] = UCompC('estimate',...
        sys.y,u,sys.model,sys.h,sys.comp,sys.compV,sys.p,sys.v,sys.yFit,sys.yFor,sys.yFitV,...
        sys.yForV,sys.a,sys.P,sys.eta,sys.eps,sys.table,sys.outlier,sys.tTest,sys.criterion,...
        sys.periods,sys.rhos,sys.verbose,sys.stepwise,sys.p0,sys.cLlik,criteria,sys.arma,sys.hidden);

    if(~isempty(yFor))
        sys.yFor = yFor;
        sys.yForV = yForV;
    end
    sys.p = p;
    sys.p0 = p0;
    if(strchr(sys.model,'?'))
        sys.model = model;
    end
    sys.hidden.grad = grad;
    sys.hidden.constPar = constPar;
    sys.hidden.typePar = typePar;
    if(size(cycleLimits) == 1)
            sys.hidden.cycleLimits = cycleLimits*ones(round(length(cycleLimits)/2),2);
    else
            sys.hidden.cycleLimits = reshape(cycleLimits,round(length(cycleLimits)/2),2);
    end
    sys.hidden.d_t = d_t;
    sys.hidden.innVariance = innVariance;
    sys.hidden.objFunValue = objFunValue;
    sys.hidden.beta = beta;
    sys.hidden.betaV = betaV;
    sys.periods = periods;
    sys.rhos = rhos;
    sys.hidden.estimOk = estimOk;
    sys.hidden.nonStationaryTerms = nonStationaryTerms;
    sys.hidden.ns = ns;
    sys.hidden.nPar = nPar;
    sys.hidden.harmonics = harmonics;
    if(size(criteria) == 1)
            sys.criteria = criteria*ones(1,4);
    else
            sys.criteria = reshape(criteria,1,4);
    end
    criteriaAux=struct("LLIK",sys.criteria(1,1),"AIC",sys.criteria(1,2),"BIC",sys.criteria(1,3),"AICc",sys.criteria(1,4));
    sys.criteria=criteriaAux;
    if(~isnan(sys.outlier))
        k = numel(u)/nu;
        nOut = k-kInitial;
        if(nOut > 0)
            if(size(u) == 1)
                sys.u = u*ones(k,nu);
            else
                sys.u = reshape(u,k,nu);
            end
            sys.hidden.typeOutliers = NaN(nOut,2);
            sys.hidden.typeOutliers(:,1) = typeOutliers;
            for i=1:nOut
                sys.hidden.typeOutliers(i,2) = find(sys.u(kInitial+i,:)==1,1); 
            end
        end
    end

end

