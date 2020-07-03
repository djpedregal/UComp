function sys = UCcomponents(sys)
% UCcomponents - Estimates components of UC models
% 
%   sys = UCcomponents(sys)
%
%   Input:
%       sys: structure of type UComp created with UCmodel
%    
%   Output:
%       The same input structure with the appropiate fields filled in, in particular:
%           comp: Estimated components in table form
%           compV: Estimated components variance in table form      
%
%   Authors: Diego J. Pedregal, Nerea Urbina
%    
%   Examples:
%       load 'airpassengers' - contains 2 variables: y, frequency
%       m = UC(log(y),frequency)
%       m = UCcomponents(m)
%        
%   See also UC, UCsmooth, UCdisturb, UCestim, UCfilter, UCmodel,
%   UCsetup, UCvalidate

    y = sys.y;
    u = sys.u;
    if(istable(sys.criteria))
        criteria = table2array(sys.criteria)';
    else
        criteria = sys.criteria;
    end

    [comp,compV,m] = UCompC('components',y,u,sys.model,sys.h,sys.comp,sys.compV,...
        sys.p,sys.v,sys.yFit,sys.yFor,sys.yFitV,sys.yForV,sys.a,sys.P,sys.eta,...
        sys.eps,sys.table,sys.outlier,sys.tTest,sys.criterion,sys.periods,sys.rhos,...
        sys.verbose,sys.stepwise,sys.p0,sys.cLlik,criteria,sys.arma,sys.hidden);

    sys.comp = comp;
    sys.compV = compV;
    if(size(u,1) == 1 && size(u,2) == 2)
        k = 0;
    else
        k = size(u,1);
    end
    nCycles = m-k-4;

    %Re-building matrices to their original sizes
    n = numel(sys.comp)/m;

    if(size(sys.comp)==1)
        sys.comp = sys.comp*ones(rows,cols);
    else
        sys.comp = reshape(sys.comp,m,n);
    end
    sys.comp = sys.comp';
    if(size(sys.compV) == 1)
        sys.compV = sys.compV*ones(m,m);
    else
        sys.compV = reshape(sys.compV,m,n);
    end
    sys.compV = sys.compV';
    namesComp = ["Trend","Slope","Seasonal","Irregular"];
    if(nCycles > 0)
        for i=1:nCycles
            namesComp = [namesComp,strcat("Cycle",string(i))];
        end
    end

    %Input names
    if(k > 0)
        nOut = 0;
        if(sys.hidden.typeOutliers(1,2) ~= -1)
            nOut = size(sys.hidden.typeOutliers,1);
        end
        nU = k-nOut;
        if(nU > 0)
            for i=1:nU
                namesComp = [namesComp,strcat("Exogenous",string(i))];
            end
        end
        if(nOut > 0)
            for i=1:nOut
                namei = "AO";
                if(sys.hidden.typeOutliers(i,1)==1)
                    namei = "LS";
                elseif(sys.hidden.typeOutliers(i,1)==2)
                    namei = "SC";
                end
                namesComp = [namesComp,strcat(namei,string(sys.hidden.typeOutliers(i,2)))];
            end
        end
    end
    sys.comp = array2table(sys.comp,'VariableNames',namesComp);
    sys.compV = array2table(sys.compV,'VariableNames',namesComp);

end