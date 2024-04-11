function [ar, ma] = modelUC2arma(model)
    % ARMA model orders
    ar = 0;
    ma = 0;
    posi = strfind(model, 'arma');
    if isempty(posi)
        ar = 0;
        ma = 0;
        return;
    end
    posi = posi + 5;
    marma = model(posi : end-1);
    coma = strfind(marma, ',');
    ar = str2double(marma(1 : coma-1));
    ma = str2double(marma(coma + 1 : end));
end