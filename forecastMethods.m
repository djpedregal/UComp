function predicciones = forecastMethods(x, s, h)
    % Función que devuelve predicciones de todos los métodos en columnas
    if nargin < 3
        h = 4;
    end
    nYears = ceil(h / s);
    naive = repmat(x(end), h, 1);
    snaive = repmat(x(end - s + 1 : end), nYears, 1);
    snaive = snaive(1 : h);
    mediaAnual = repmat(mean(x(end - s + 1 : end)), h, 1);
    pETS = ETSmodel(x, s, h=h).yFor;
    pPTS = PTSmodel(x, s, h=h).yFor;
    pred = [snaive mediaAnual pETS pPTS];
    media = mean(pred, 2);
    mediana = median(pred, 2);
    predicciones = [naive snaive mediaAnual pETS pPTS media mediana];
    % predicciones.Properties.VariableNames = {'naive', 'snaive', 'mediaAnual', 'ETS', 'PTS', 'media', 'mediana'};
end