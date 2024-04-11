function out = slide(y, s, orig, forecFun, varargin)
% slide - Rolling forecasting of a matrix of time series
%
%   Takes a time series and run forecasting methods implemented in function
%   forecFun h steps ahead along the time series y, starting at forecasting
%   origin orig, and moving step observations ahead. Forecasts may be run in parallel
%   by setting parallel to TRUE. A fixed window width may be
%   specified with input window. The output is of dimensions (h, nOrigs, nModels, nSeries)
%
%   Syntax:
%   out = slide(y, orig, forecFun, h, step, output, window, parallel, ...)
%
%   Input arguments:
%   - y: a vector or matrix of time series.
%   - orig: starting forecasting origin.
%   - forecFun: user function that implements forecasting methods.
%   - h: forecasting horizon (default: 12).
%   - step: observations ahead to move the forecasting origin (default: 1).
%   - output: logical value indicating whether to include output (default: true).
%   - window: fixed window width in number of observations (NaN for non-fixed) (default: NaN).
%   - parallel: logical value indicating whether to run forecasts in parallel (default: false).
%
%   Output:
%   - out: a vector with all the dimensions.
%
%   Author: Diego J. Pedregal
%
%   Example:
%   slide(AirPassengers, 100, forecFun)
    menu = inputParser;
    addRequired(menu, 'y', @isfloat);
    addRequired(menu, 's', @isfloat);
    addRequired(menu, 'orig', @isfloat);
    addRequired(menu, 'forecFun');
    addParameter(menu, 'h', 24, @isfloat);
    addParameter(menu, 'step', 1, @isfloat);
    addParameter(menu, 'output', true, @islogical);
    addParameter(menu, 'window', NaN, @isfloat);
    addParameter(menu, 'parallel', false, @islogical);
    parse(menu, y, s, orig, forecFun, varargin{:});
    y = menu.Results.y;
    s = menu.Results.s;
    orig = menu.Results.orig;
    forecFun = menu.Results.forecFun;
    if ~isa(forecFun, 'function_handle')
        error('Value for forecFun must be a handle to a function!!');
    end
    h = menu.Results.h;
    step = menu.Results.step;
    output = menu.Results.output;
    window = menu.Results.window;
    parallel = menu.Results.parallel;
    if size(y, 2) == 1
        nSeries = 1;
        n = length(y);
    else
        n = size(y, 1);
        nSeries = size(y, 2);
    end
    dataList = cell(1, nSeries);
    for j = 1 : nSeries
        dataList{j} = y(:, j);
    end
    nOr = length(orig : step : (n - h));
    listOut = cell(1, nSeries);
    if ~parallel || (nSeries == 1 && nOr == 1)
        for j = 1:nSeries
            listOut{j} = slideAux(dataList{j}, s, orig, forecFun, h, step, output, false, window, false);
        end
    else
%         if nSeries > 1
%             listOut = cell(1, nSeries);
            parfor j = 1:nSeries
                listOut{j} = slideAux(dataList{j}, s, orig, forecFun, h, step, output, false, window, true);
            end
%             listOut = listOut.Value;
%         elseif nOr > 1
%             listOut = cell(1, nSeries);
%             parfor j = 1:nSeries
%                 listOut{j} = slideAux(dataList{j}, orig, forecFun, h, step, output, false, window, true, varargin{:});
%             end
%         end
    end
    out = NaN(size(listOut{1}, 1), size(listOut{1}, 2), size(listOut{1}, 3), nSeries);
    for j = 1:nSeries
        out(:, :, :, j) = listOut{j};
    end

    % Set the appropriate dimension names if needed
    % dimnames(out){3} = dimnames(listOut[[1]])[[3]];

end

function out = slideAux(y, s, orig, forecFun, h, step, output, graph, window, parallel)
    % Auxiliary function run from slide
    % Inputs:
    %     y: a vector or matrix of time series
    %     s: seasonal period of time series
    %     orig: starting forecasting origin
    %     forecFun: user function that implements forecasting methods
    %     h: forecasting horizon
    %     step: observations ahead to move the forecasting origin
    %     output: output TRUE/FALSE
    %     graph: graphical output TRUE/FALSE
    %     window: fixed window width in number of observations (None for non-fixed)
    %     parallel: run forecasts in parallel
    % Returns:
    %     Next time stamp
    %
    % Author: Diego J. Pedregal

    % Rolling for 1 series
    % out = [h, nOrigs, nModels]
    n = length(y);
    origs = orig : step : n-h;
    nOr = length(origs);

    if isnan(window)
        outi = forecFun(y(1:orig), s, h);
    else
        outi = forecFun(y(orig-window+1:orig), s, h);
    end
    nMethods = size(outi, 2);
    out = nan(h, nOr, nMethods);
    out(:, 1, :) = outi;

    dataList = cell(1, nOr-1);
    if nOr > 1
        for j = 2:nOr
            if isnan(window)
                dataList{j-1} = y(1:orig+j-1);
            else
                dataList{j-1} = y(orig-window+step:orig+j-1);
            end
        end
    end

    if parallel
        pool = gcp();
        listOut = cell(1, nOr-1);
        parfor i = 1:nOr-1
            listOut{i} = forecFun(dataList{i}, s, h);
        end
    else
        listOut = cell(1, nOr-1);
        for i = 1:nOr-1
            listOut{i} = forecFun(dataList{i}, s, h);
        end
    end

    if nOr > 1
        for i = 2:nOr
            if nMethods == 1
                out(:, i, :) = reshape(listOut{i-1}, [], 1);
            else
                out(:, i, :) = listOut{i-1};
            end
        end
    end
end