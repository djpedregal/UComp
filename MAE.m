function salida = MAE(py, y, s)
  h = length(py);
  salida = cumsum(abs(py - y(end - h + 1 : end))) ./ (1 : h)';
end
