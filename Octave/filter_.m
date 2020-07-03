function sys = filter_(sys,command)
% filter_ - Auxiliar function of UComp package
% 
%   sys = filter_(sys,command)
% 
%   Inputs:
%       sys: reserved input
%       command: reserved input
%   
%   Author: Diego J. Pedregal

    y = sys.y;
    u = sys.u;
    if(isstruct(sys.criteria))
    	criteria=cell2mat(struct2cell(sys.criteria));
    else
    	criteria=sys.criteria;
    end

    [a,P,v,yFitV,yFit,eps,eta] = UCompC(command,y,u,sys.model,sys.h,sys.comp,sys.compV,...
        sys.p,sys.v,sys.yFit,sys.yFor,sys.yFitV,sys.yForV,sys.a,sys.P,sys.eta,sys.eps,...
        sys.table,sys.outlier,sys.tTest,sys.criterion,sys.periods,sys.rhos,sys.verbose,...
        sys.stepwise,sys.p0,sys.cLlik,criteria,sys.arma,sys.hidden);

    %Re-building to their original sizes
    n = length(sys.y) + sys.h;
    m = numel(a)/n;

    if(strcmp(command,'disturb'))
        n = length(sys.y);
        mEta = numel(eta)/n;
        if(size(eta) == 1)
             sys.eta = eta*ones(mEta,n)';
        else
            sys.eta = reshape(eta,mEta,n)';
        end
        if(size(eps) == 1)
             sys.eps = eps*ones(1,n)';
        else
            sys.eps = reshape(eps,1,n)';
        end
    else
        if(size(a) == 1)
             sys.a = a*ones(m,n)';
        else
            sys.a = reshape(a,m,n)';
        end
        if(size(P) == 1)
             sys.P = P*ones(m,n)';
        else
            sys.P = reshape(P,m,n)';
        end
        sys.yFit = yFit;
        sys.yFitV = yFitV;
        sys.v = v;
    end

end