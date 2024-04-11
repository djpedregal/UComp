function y = box_cox(x, lambda)
    if abs(lambda) < 1e-4
        y = log(x);
    elseif lambda > 0.99 && lambda < 1.01
        y = x;
    else
        y = (x .^ lambda - 1) / lambda;
    end
end

