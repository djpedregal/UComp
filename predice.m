function salida = predice(y, s, h)
  m = UC(y, 4, h=8);
  naive = repmat(y(end), 8);
  salida = [naive m.yFor];
end
