function mase = MASE(px, actual, s)
    n = length(actual);
    h = length(px);
    tx = actual(n - h + 1 : n);
    error1 = mean(abs(actual(s + 1 : n - h) - actual(1 : n - h - s)));
    mase = cumsum(abs(px - tx)) / error1 ./ (1 : h)';
end
