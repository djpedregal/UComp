function modelU = PTS2modelUC(model, armaOrders)
    if nargin < 2
        armaOrders = [0, 0];
    end
    modelU = '';
    n = length(model);
    % noise
    model = lower(model);
    aux = model(1);
    if aux == '?'
        modelU = '/?';
    elseif aux == 'n'
        modelU = '/none';
    elseif aux == 'a'
        modelU = sprintf('/arma(%d,%d)', armaOrders(1), armaOrders(2));
    else
        error('ERROR: incorrect error model!!');
    end
    % seasonal
    aux = model(end);
    if aux == '?'
        modelU = ['/?' modelU];
    elseif aux == 'n'
        modelU = ['/none' modelU];
    elseif aux == 'a'
        modelU = ['/linear' modelU];
    elseif aux == 'e'
        modelU = ['/equal' modelU];
    elseif aux == 'd'
        modelU = ['/different' modelU];
    else
        error('ERROR: incorrect seasonal model!!');
    end
    % trend
    aux = model(2:n-1);
    if aux == '?'
        modelU = ['?/none' modelU];
    elseif aux == 'n'
        modelU = ['rw/none' modelU];
    elseif aux == 'a'
        modelU = ['llt/none' modelU];
    elseif aux == 'ad'
        modelU = ['srw/none' modelU];
    elseif aux == 'l'
        modelU = ['td/none' modelU];
    else
        error('ERROR: incorrect trend model!!');
    end
end
