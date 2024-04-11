function results = Accuracy(py, y, s, collectFun)
% Accuracy - Calcula la precisión para una serie de tiempo y varios métodos de pronóstico
%
%   Calcula diversas métricas de precisión para una serie de tiempo y una matriz de pronósticos.
%   La matriz de pronósticos contiene los pronósticos de varios métodos para diferentes pasos hacia adelante.
%   La función devuelve una tabla con los resultados de precisión.
%
%   Sintaxis:
%   results = Accuracy(py, y, s, collectFun)
%
%   Argumentos de entrada:
%   - py: matriz de pronósticos (h x nMethods x nForecasts).
%   - y: matriz de valores reales (n x nForecasts).
%   - s: periodo estacional, número de observaciones por año.
%   - collectFun: función de agregación (mean, median, etc.).
%
%   Resultado:
%   - results: tabla de resultados.
%
%   Autor: Diego J. Pedregal
%
%   Ejemplos:
%   results = Accuracy(py, y, 12, @mean)
    [h, nMethods, nForecasts] = size(py);
    ny = size(y, 2);
    if nargin < 4
        collectFun = @nanmean;
    end
    if nargin < 3
        s = 1;
    end
    spy = length(size(py));
    if isnumeric(s) && isscalar(s)
        s = repmat(s, nForecasts, 1);
    end
    if isnumeric(y) && isscalar(y)
        y = repmat(y, h, nForecasts);
    end
    if isequal(spy, 2)
        py = reshape(py, h, nMethods, nForecasts, 1);
        y = repmat(y, 1, 1, nForecasts);
    end
    insample = size(y, 1) > h;
    out = NaN(nMethods, 7 + 3 * insample);
    rownames = cell(1, nMethods);
    colnames = {'ME', 'RMSE', 'MAE', 'MPE', 'PRMSE', 'MAPE', 'sMAPE'};
    if insample
        colnames = [colnames, 'MASE', 'RelMAE', 'Theil''s U'];
    end
    for i = 1:nMethods
        e = py(:, i, :) - y(end-h+1:end, :);
        p = 100 * e ./ y(end-h+1:end, :);
        aux = [mean(e, 1, 'omitnan');
                sqrt(mean(e.^2, 1, 'omitnan'));
                mean(abs(e), 1, 'omitnan');
                mean(p, 1, 'omitnan');
                sqrt(mean(p.^2, 1, 'omitnan'));
                mean(abs(p), 1, 'omitnan');
                mean(200 * abs(e) ./ (py(:, i, :) + y(end-h+1:end, :)), 1, 'omitnan')];
        
        if insample
            theil = 100 * (y(end-h+1:end, :) - y(end-h:end-1, :)) ./ y(end-h+1:end, :);
            fRW = y(s+1:end-h, :) - y(1:end-h-s, :);
            aux = [aux;
                    mean(abs(e), 1, 'omitnan') ./ mean(abs(fRW), 1, 'omitnan');
                    sum(abs(e), 1, 'omitnan') ./ sum(abs(fRW), 1, 'omitnan');
                    sqrt(sum(p.^2, 1, 'omitnan') ./ sum(theil.^2, 1, 'omitnan'))];
        end
        if nForecasts == 1
            out(i, :) = aux;
        else
            out(i, :) = collectFun(aux);
        end
        rownames{i} = num2str(i);
    end
    if ~exist ('OCTAVE_VERSION', 'builtin')   % Matlab
        results = array2table(out, 'VariableNames', colnames, 'RowNames', rownames);
    end
end
