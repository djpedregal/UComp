function salida = simpleMethods(y, h)
    if nargin < 2
        h = 4;
    end
    n = length(y);
    predicciones = nan(h, 2);
    residuos = nan(n, 2);
    predicciones(:, 1) = mean(y);
    residuos(:, 1) = y - predicciones(end, 1);
    aux = y(~isnan(y));
    predicciones(:, 2) = y(end);
    residuos(2 : n, 2) = y(2 : n) - y(1 : (n - 1));
    salida = struct('predicciones', predicciones, 'residuos', residuos);
end
