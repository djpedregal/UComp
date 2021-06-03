function p = coef(object)
% coef - Extracts model coefficients of UComp object
%   
%   p = coef(object)
%
%  Inputs:
%       object: a UComp model.
%
%   Examples:
%       load data/airpas
%       m = UC(log(airpas), 12);
%       coef(m)
    if isempty(object.table)
        object = UCvalidate(object, false);
    end
    hyphen = 1;
    names = cell(1, length(object.p));
    j = 1;
    for i = 5 : length(object.table)
        linei = object.table{i};
        if strcmp(linei(1), '-')
            hyphen = hyphen + 1;
        end
        if (hyphen == 3) && ~strcmp(linei(1), '-')
                    names{j} = replace(linei(1 : strfind(linei, ':') - 1), ' ', '');
                    j = j + 1;
        end
    end
    if ~exist ('OCTAVE_VERSION', 'builtin')   % Matlab
      p = array2table(object.p,'Rownames',names,'VariableNames', {'Param'});
    else
      p = names;
    end
end