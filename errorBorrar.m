function rmse = errorBorrar(py, y)
    e = (py - tail(y, length(py)));
    rmse = sqrt(cumsum(e.^2) ./ (1 : length(e))');
end