% mexUComp('D:\googleDrive\C++\armadillo-12.4.0', 'D:\googleDrive\C++\armadillo-12.4.0/examples/lib_win64');
load airpas
out = slide(airpas, 100, @pruebaBorrar, 12, 1);
plotSlide(out, airpas, 100, 1, @errorBorrar);



return

m = ETS(airpas, 12);
m = ETSvalidate(m);







m = UCmodel(airpas, 12, 'verbose', true, 'lambda', NaN);
m = UCvalidate(m);
m = UCfilter(m);
m = UCsmooth(m);
m = UCdisturb(m);
m = UCcomponents(m);






