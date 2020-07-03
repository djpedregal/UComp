function sys = UCcomponents(sys)
% UCcomponents - Estimates components of UC models
% 
%   sys = UCcomponents(sys)
%
%   Input:
%       sys: structure of type UComp created with UCmodel
%    
%   Output:
%       The same input struct with the appropiate fields filled in, in particular:
%           comp: Estimated components in struct form
%           compV: Estimated components variance in struct form      
%
%   Authors: Diego J. Pedregal, Nerea Urbina
%    
%   Examples:
%       load 'airpassengers' - contains 2 variables: y, frequency
%       m = UC(log(y),frequency)
%       m = UCcomponents(m)
%        
%   See also UC, UCdisturb, UCestim, UCfilter, UCmodel, UCsetup, UCsmooth, UCvalidate
 
    y = sys.y;
    u = sys.u;
    if(isstruct(sys.criteria))
    	criteria=cell2mat(struct2cell(sys.criteria));
    else
    	criteria=sys.criteria;
    end

    [comp,compV,m] = UCompC('components',y,u,sys.model,sys.h,sys.comp,sys.compV,...
        sys.p,sys.v,sys.yFit,sys.yFor,sys.yFitV,sys.yForV,sys.a,sys.P,sys.eta,...
        sys.eps,sys.table,sys.outlier,sys.tTest,sys.criterion,sys.periods,sys.rhos,...
        sys.verbose,sys.stepwise,sys.p0,sys.cLlik,criteria,sys.arma,sys.hidden);

    if(size(u,1) == 1 && size(u,2) == 2)
        k = 0;
    else
        k = size(u,1);
    end
    nCycles = m-k-4;

    %Re-building matrices to their original sizes
    n = numel(comp)/m;

    if(size(comp)==1)
        comp = comp*ones(rows,cols);
    else
        comp = reshape(comp,m,n);
    end
    comp = comp';
    if(size(compV) == 1)
        compV = compV*ones(m,m);
    else
        compV = reshape(compV,m,n);
    end
    compV = compV';
    compAux=struct("Trend",comp(:,1),"Slope",comp(:,2),"Seasonal",comp(:,3),"Irregular",comp(:,4));
    compVAux=struct("Trend",compV(:,1),"Slope",compV(:,2),"Seasonal",compV(:,3),"Irregular",compV(:,4));
    i=0;
    if(nCycles > 0)
        for i=1:nCycles
            compAux=setfield(compAux,strcat("Cycle",num2str(i)),comp(:,4+i));
            compVAux=setfield(compVAux,strcat("Cycle",num2str(i)),compV(:,4+i));
        end
    end

    %Input names
    if(k > 0)
        nOut = 0;
        if(sys.hidden.typeOutliers(1,2) ~= -1)
            nOut = size(sys.hidden.typeOutliers,1);
        end
        nU = k-nOut;
        j=0;
        if(nU > 0)
            for j=1:nU
                compAux=setfield(compAux,strcat("Exogenous",num2str(j)),comp(:,4+i+j));
                compVAux=setfield(compVAux,strcat("Exogenous",num2str(j)),compV(:,4+i+j));
            end
        end
        k=0;
        if(nOut > 0)
            for k=1:nOut
                namei = "AO";
                if(sys.hidden.typeOutliers(k,1)==1)
                    namei = "LS";
                elseif(sys.hidden.typeOutliers(k,1)==2)
                    namei = "SC";
                end
                compAux=setfield(compAux,strcat(namei,num2str(sys.hidden.typeOutliers(k,2))),comp(:,4+i+j+k));
                compVAux=setfield(compVAux,strcat(namei,num2str(sys.hidden.typeOutliers(k,2))),compV(:,4+i+j+k));
            end
        end
    end
    sys.comp=compAux;
    sys.compV=compVAux;

end