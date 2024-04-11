function out = pruebaBorrar(y, h)
    f = y(end - h + 1 : end);
    out = [f flip(f)];
end