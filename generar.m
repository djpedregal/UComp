function a = generar(n)
    if nargin < 1
        n = 3000;
    end
    a = randn(n, 1);
    ['Media: ' num2str(mean(a)) '   /   Varianza: ' num2str(var(a))]
    hist(a)
end
