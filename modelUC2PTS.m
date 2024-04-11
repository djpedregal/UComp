function model = modelUC2PTS(modelUC)
    % removing cycle from UC model
    % modelUC = strrep(modelUC, '/none/', '/');
    % extracting components
    aux = strsplit(modelUC, '/');
    trend = aux{1};
    seasonal = aux{3};
    noise = aux{4};
    % noise
    model = 'A';
    if strcmp(noise, 'none')
        model = 'N';
    end
    % trend
    if strcmp(trend, 'rw')
        model = [model 'N'];
    elseif strcmp(trend, 'srw')
        model = [model 'Ad'];
    elseif strcmp(trend, 'llt')
        model = [model 'A'];
    elseif strcmp(trend, 'td')
        model = [model 'L'];
    end
    % seasonal
    if strcmp(seasonal, 'none')
        model = [model 'N'];
    elseif strcmp(seasonal, 'equal')
        model = [model 'E'];
    elseif strcmp(seasonal, 'different')
        model = [model 'D'];
    elseif strcmp(seasonal, 'linear')
        model = [model 'L'];
    end
end
