%% Automatic identification of structural Unobserved Components models

%% Introduction

% In this document, several examples are run to show the operation and possibilities 
% offered by the UComp software, a tool developed for the identification and 
% forecasting of Unobserved Components models.

% The source code of this toolbox has been written in C++ and supported by the 
% functions of Armadillo library, so that its functionality has been extended 
% to this platform by using MEX files.
 
% It should be noted that the MEX function included in this set of files has 
% been built in Windows 10 and Octave 5.2, so if you are using a 
% different OS and/or Octave's version, you may not get the expected results. 
% See README.txt file, where you will get information about this.

%% UComp algorithm

% UComp is an automatic estimation and prediction tool based on UC models. These 
% models are very useful for operations such as smoothing, signal extraction, 
% automatic outlier detection, seasonal adjustments, trends, etc. See the documentation 
% forfdsa detailed information.

% The following list shows the available functions and a brief description of 
% what they do.

% 1. Functions for creating and adjusting the parameters of UComp objects to 
% control the behaviour of these functions. Time series data and its fundamental 
% period are the only compulsory inputs required.

% * UC - Overall function that runs all the rest.    
% * UCsetup - Creates an UComp object and sets all input options controlling 
%             how the rest of functions work.
% * UCmodel -  Runs UCsetup and UCestim sequentially.           

% 2. Functions that works directly with UComp objects:

% * UCestim - Identifies UC model, estimates it by Maximum Likelihood and computes 
%             forecasts.
% * UCvalidate - Validates UC model previously estimated.
% * UCfilter - Optimal Kalman filtering of UC models.
% * UCsmooth - Optimal Fixed Interval Smoother.
% * UCdisturb - Optimal Disturbance Smoother.
% * UCcomponents - Components estimation.

%% Examples: 
                                                                                                                       
%% Build MEX-file UComp.cpp (compulsory step if not Windows user)

% Please, modify the next line with the path to your Armadillo 
% (and LAPACK/BLAS libraries if necessary) include folder and 
% library names and uncomment it.

%mex -Ipath\to\armadillo\include (-Lpath\to\blas-lapack\libraries) -llapack -lblas UCompC.cpp

% In this folder is included an example of time series and his fundamental period, 
% namely Airline Passegers, from Box et al. (2015).  This data will be used to 
% show the power of the algorithms.

% In first place, data is plotted to get an overview about the properties of 
% time series. 

load 'airpassengers.mat'

t = (1949 : 1/12 : 1960.99)';
plot(t, y)
title('Air Passengers')

% As a first approach to the software, some different possibilities of entering 
% input parameters are shown. For the introduction of optional variables, it is 
% necessary to enter the name of the variable and then, the value (see the complete 
% list of inputs typing help UCsetup)

%Just required inputs
m1 = UCsetup(log(y), frequency);
%Required inputs plus optional verbose mode
m2 = UCmodel(log(y), frequency, 'verbose', true); 

% Example of wrong format input
% m3 = UCmodel(log(y), frequency, 'model', "llt/equal/arma(0,0)");

% Now, UC identification is run using different information criteria. Let's 
% start with Akaike's.

mAIC = UCmodel(log(y), frequency, 'criterion', 'aic');
fprintf("Optimal model with AIC criterion: %s", mAIC.model);
mAIC.criteria
mBIC = UCmodel(log(y), frequency, 'criterion', 'bic');
fprintf("Optimal model with BIC criterion: %s", mBIC.model);
mBIC.criteria
mAICc = UCmodel(log(y), frequency, 'criterion', 'aicc');
fprintf("Optimal model with AICc criterion: %s", mAICc.model);
mAICc.criteria

% Running UCmodel function allows the user to get information criteria and 
% forecasted values.
% Then, UCvalidate function is used to validate the estimated model.

mAIC = UCvalidate(mAIC);

% The following graph shows the values obtained from forecasting
figure;
plot(y); hold on;  plot((144:162),[y(end);exp(mAIC.yFor)]); legend('Input','Forecasted');

% In addition, estimated components can be calculated with UCcomponents

mAIC = UCcomponents(mAIC);
figure;
subplot(2,2,1); plot(mAIC.comp.Trend); title('Trend')
subplot(2,2,2); plot(mAIC.comp.Slope); title('Slope');
subplot(2,2,3); plot(mAIC.comp.Seasonal); title('Seasonal')
subplot(2,2,4); plot(mAIC.comp.Irregular); title('Irregular')

% We can also include inputs in the model, represented by a variable named u.

u = zeros(3, 144);
u(1, 100 : 120) = 1;
u(2, 50) = 1;
u(3, 30) = 1;
m4 = UC(log(y), frequency, 'u', u,'verbose', true);

% Other options are outlier detection and cycles, as shown below.

m5 = UCmodel(log(y), frequency, 'verbose', true, 'outlier', 4);

% Computing time of this output in comparison with that obtained for model m2 
% (line 14) shows that execution time increases considerably. One way to reduce 
% this time is adding the optional stepwise input as true, which allows reducing 
% the number of models to estimate.

m6 = UC(log(y), frequency, 'verbose', true, 'outlier', 4, 'stepwise', true);
fprintf("Optimal model: %s",m6.model);

% The same optimal model is obtained as in the m2 structure because the time 
% serie does not present outliers.  However, this may be checked if any outliers 
% are added artificially.

% For example, it is included two outliers in the samples 40 and 100
yMod = y;
yMod(40) = 400;
yMod(100) = 600;

figure; plot(log(yMod)); title('Airline Passengers with outliers')
m7 = UC(log(yMod), frequency, 'verbose', true, 'outlier', 4);
fprintf("Optimal model: %s", m7.model);
figure;
subplot(3,2,1); plot(m7.comp.Trend); title('Trend')
subplot(3,2,2); plot(m7.comp.Slope); title('Slope');
subplot(3,2,3); plot(m7.comp.Seasonal); title('Seasonal')
subplot(3,2,4); plot(m7.comp.Irregular); title('Irregular')
subplot(3,2,5); plot(m7.comp.AO40); title('Additive outlier sample 40')
subplot(3,2,6); plot(m7.comp.AO100); title('Additive outlier sample 100')

% In case of setting cycles, the value must be indicated in variable named _model_ 
% using the 'trend/cycle/seasonal/irregular' format. The following call to _UC_ 
% tests for the existence of a cycle of approximate four years including the previous 
% fake inputs. The results is that such cycle does not exist, since the outcome 
% is a model without such cycle.

m8 = UC(log(y),frequency, 'u', u, 'verbose', true, 'outlier', 4, 'model', 'llt/-48?/eq/?');

% To sum up, UComp is a general software for UC modelling with a wide range 
% of options that gives it high flexibility and power, especially due to the automatic 
% identification facilities. See the documentation for detailed information of 
% the inputs, outputs, functionalities and advantages of using this tool.