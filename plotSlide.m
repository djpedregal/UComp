function outj = plotSlide(py1, y, s, orig, step, errorFun, collectFun, varargin)
% plotSlide - Plots aggregated results from rolling forecasts run with slide
%
%   Syntax:
%   outj = plotSlide(py1, y, orig, step, errorFun, collectFun, ...)
%
%   Input Arguments:
%   - py1: vector or matrix of time series.
%   - y: vector or matrix of time series.
%   - s: seasonal period of time series.
%   - orig: initial forecast origin.
%   - step: observations ahead to move the forecast origin.
%   - errorFun: handle to error calculation function.
%   - collectFun: aggregation function (mean, median, etc.).
%   - ...: Additional parameters for errorFun function.
%
%   Output:
%   - outj: matrix with all the dimensions.
%
%   Author: Diego J. Pedregal
%
%   Examples:
%   outj = plotSlide(airpas, 100, @forecFun)
    if nargin < 7
        collectFun = @nanmean;
    end
    py = py1;
    [h, nOrigs, nMethods, nSeries] = size(py);
%     if isnan(nSeries)
%         nSeries = 1;
%         py = cat(5, py, NaN(size(py1, 1), size(py1, 2), size(py1, 3), 1));
%     end
    metrics = NaN(h, nMethods);
    outj = NaN(h, nOrigs, nMethods, nSeries);
    for i = 1:nOrigs
        actuali = y(1:orig + (i - 1) * step + h, :);
        aux = py(:, i, :, :);
        if nSeries == 1
            for j = 1:nMethods
                outj(:, i, j, 1) = errorFun(aux(:, 1, j, 1), actuali, s);
            end
        else
            for k = 1:nSeries
                for j = 1:nMethods
                    outj(:, i, j, k) = errorFun(aux(:, 1, j, k), actuali(:, k), s);
                end
            end
        end
    end
    for m = 1:nMethods
        for j = 1:h
            metrics(j, m) = collectFun(collectFun(outj(j, :, m, :)));
        end
    end
    plot(metrics);
    ylabel('');
    if ~exist ('OCTAVE_VERSION', 'builtin')   % Matlab
        if istable(py1)
            legend(py1.Properties.VariableNames, 'Location', 'best');
        else
            names = cell(size(metrics, 2), 1);
            for i = 1 : size(metrics, 2)
                names{i, 1} = num2str(i);
            end
            legend(names)
        end
end
