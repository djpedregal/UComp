// #include <iostream>
// #include <math.h>
// #include <string.h>
// #include <armadillo>
// using namespace arma;
// using namespace std;
#include "DJPTtools.h"
#include "optim.h"
#include "stats.h"
#include "boxcox.h"
#include "SSpace.h"
#include "ARMAmodel.h"

// Exponential Smoothing models
struct ETSmodel{
    // INPUTS:
    string model,                   // Model definition
           criterion,               // Information criterion for identification
           parConstraints;          // Parameter constraints: none, standard, admissible
    int userS,                      // Seasonal period supplied by user
        s,                          // Seasonal period
        h,                          // Forecasting horizon
        nSimul;                     // number of simulations for forecasting interval estimation
    vec y,                          // output data
        arma;                       // arma orders
    rowvec alphaL = {0.0, 1.0};     // limits for alpha parameter supplied by user
    rowvec betaL = {0.0, 1.0};      // limits for beta parameter supplied by user
    rowvec gammaL = {0.0, 1.0};     // limits for gamma parameter supplied by user
    rowvec phiL = {0.8, 0.98};      // limits for phi parameter supplied by user
    mat u;                          // input data
    bool identAll = false,          // Identify all 30 models
         verbose = false,           // intermediate results output
         forIntervals = false,      // Forecasting intervals
         bootstrap = false,         // bootstrap simulation for simulate function
         armaIdent;                 // identify ARMA models
    double lambda = 1.0;            // Box-Cox transformation parameter
    // OUTPUTS:
    string initialModel, error, trend, seasonal,      // Model definition
           estimOk,                 // Estimation optimization text
           compNames = "Error/Fit/Level";     // Components names (error, fit, level, seasonal, slope, exogenous, arma)
    double objFunValue,             // Objective function value
           loge2,
           logr,
           alpha,                   // alpha smoothing parameter
           beta,                    // beta smoothing parameter
           phi = 1.0,               // phi damping trend paramter
           gamma,                   // gamma smoothing parameter
           sigma2,                  // estimated variance
           prop = 0.5;              // proportion of initial condition for alpha
    int flag,                       // estimation optimization flag
        modelType;                  // Model type (0: linear; 1: matrix F; 2: vector F
    uvec missing;                   // missing index supplied initially by user
    vec g,                          // system matrix
        ns,                         // number of states for component (trend, seasonal, arma)
        p0,                         // initial values for parameters (alpha, beta, phi, gamma, states, exogenous)
        p0user,                     // p0 supplied by user
        x0,                         // Initial states
        p,                          // Estimated values for parameters (alpha, beta, phi, gamma, states)
        truep,                      // Estimated true values for parameters (alpha, beta, phi, gamma)
        grad,                       // grad at estimation point
        criteria,                   // Esimation information criteria
        nPar,                       // number of parameters per component (trend, seasonal, initial states, exogenous, ARMA)
        xn,                         // estate estimated at n
        yFor,                       // forecasts for y
        yForV,                      // variance of forecasts
        ar,                         // ar polynomial
        ma;                         // ma polynomial
    rowvec w, d;                    // system matrices
    mat F,                          // system matrix
        limits,                     // Limits for parameters
        comp,                       // Estimated components (error, fit, level, seasonal, slope)
        ySimul;                     // simulated data with simulate()
    bool exact = false,             // Analytical or numerical gradient true/false
         negative,                  // input variable with negative values
         scores = true,             // pass scores or true pars to llikETS and gradETS
         errorExit = false;         // error output
    vector<string> table;           // output table from evaluate()
};
/**************************
 * Model CLASS ETS
 ***************************/
class ETSclass{
    public:
        ETSmodel inputModel;
        ETSclass(ETSmodel);
        void interpolate();
        void estim(bool);
        void validate();
        void forecast();
        void simulate(uword, vec);
        void ident(bool);
        void components();
};
/****************************************************
// ETS functions declarations
****************************************************/
// Set model (construct ETSmodel)
void setModel(ETSmodel&, string, int);
// Check limits for parameters supplied by user
void checkLimits(rowvec&, rowvec&, rowvec&, rowvec&, string, bool&);
// pre-process of user inputs
ETSclass preProcess(vec, mat, string, int, int, bool, string, bool, rowvec, rowvec, rowvec, rowvec, string, bool, bool, int, vec, bool, double);
// post-process of user inputs
void postProcess(ETSmodel&);
// Main function
void ETS(vec, mat, string, int, int, bool, string, bool, rowvec, rowvec, rowvec, rowvec, string, bool, bool, int, vec, bool);
// System matrices for given p
void etsMatrices(ETSmodel*, vec);
// Loglik computation
double llikETS(vec&, void*);
// Gradient of Loglik
vec gradETS(vec&, void*, double&, int&);
// Parameter variances
void covPar(void*, vector<string>&, mat&);
// Minimizer for ETS (gradient function computes objective value and gradient in one run)
int quasiNewtonETS(std::function <double (vec&, void*)>,
                   std::function <vec (vec&, void*, double&, int&)>,
                   vec&, void*, double&, vec&, mat&, bool, int);
// Simulate forecasts for forecasting intervals estimation
void simulForecast45(string, vec&, rowvec, int, int, vec&, vec, double, double, vec, double&, double&);
// Initialize parameters
void initPar(ETSmodel&);
// Count number of states
//double ETScountStatesTrend(string);
// Basic system matrices
void initEtsMatrices(ETSmodel&);
// Check and divide model into components
void modelDivide(string&, string&, string&, string&, bool&);
// Parameter transformation
void trans(vec&, mat);
// Parameter un transformation
void untrans(vec&, mat);
// Parameter transformation derivatives
vec dtrans(vec&, mat);
// Returns stationary polynomial from an arbitrary one
void polyETS(vec&, double);
// Inverse of polyStationary
void invPolyETS(vec&, double);
// Find all combinations of models to estimate in identification process
void findModels(string, string, string, bool, vector<string>&);
// pretty model name
string prettyModel(string);
// AMM models forecasting and error calculation. For forecasting y = nan and a = nan
void AMM(vec&, rowvec, int, int, vec&, vec, double, double, vec, double&, double&);
// AMA models forecasting and error calculation. For forecasting y = nan and a = nan
void AMA(vec&, rowvec, int, int, vec&, vec, double, double, vec, double&, double&);
// AMN models forecasting and error calculation. For forecasting y = nan and a = nan
void AMN(vec&, rowvec, int, vec&, vec, double, double, vec, double&, double&);
// AAM models forecasting and error calculation. For forecasting y = nan and a = nan
void AAM(vec&, rowvec, int, int, vec&, vec, double, double, vec, double&, double&);
// ANM models forecasting and error calculation. For forecasting y = nan and a = nan
void ANM(vec&, rowvec, int, int, vec&, vec, double, vec, double&, double&);
// MMN models forecasting and error calculation. For forecasting y = nan and a = nan
void MMM(vec&, rowvec, int, int, vec&, vec, double, double, vec, double&, double&);
// MMN models forecasting and error calculation. For forecasting y = nan and a = nan
void MMN(vec&, rowvec, int, vec&, vec, double, double, vec, double&, double&);
// MMA models forecasting and error calculation. For forecasting y = nan and a = nan
void MMA(vec&, rowvec, int, int, vec&, vec, double, double, vec, double&, double&);
/****************************************************
// ETS functions implementations
****************************************************/
// Constructor
ETSclass::ETSclass(ETSmodel input){
    this->inputModel = input;
}
// Set model (construct ETSmodel)
void setModel(ETSmodel& inputModel, string model, int s){
    // Number of states
    // Checking correctness of string model input
    string error, trend, seasonal;
    bool errorExit = inputModel.errorExit;
    // Checking seasonality
    if (s < 2)
        model[model.length() - 1] = 'N';
    // Check model and divide into components
    modelDivide(model, error, trend, seasonal, errorExit);
    // ETSmodel inputModel;
    inputModel.errorExit = errorExit;
    inputModel.model = model;
    inputModel.error = error;
    inputModel.trend = trend;
    inputModel.seasonal = seasonal;
    inputModel.s = s;
    if (inputModel.seasonal == "N")
        inputModel.s = 0;
    if (inputModel.trend[0] != 'M' && inputModel.error != "M" && inputModel.seasonal != "M"){
        inputModel.modelType = 0;
    } else {
        inputModel.modelType = 1;
    }
    inputModel.exact = false;
    if (inputModel.modelType == 0 && inputModel.model != "ANN" && sum(inputModel.arma) == 0)
        inputModel.exact = true;
    if (inputModel.modelType != 0)
        inputModel.arma.fill(0.0);
    inputModel.ns.zeros(3);
    inputModel.prop = 0.5;
    if (error != "?" && trend != "?" && seasonal != "?"){
        inputModel.ns(0) = 1 + (trend != "N");
        inputModel.ns(1) = inputModel.s;
        if (sum(inputModel.arma) > 0)
            inputModel.ns(2) = max(inputModel.arma(0), inputModel.arma(1) + 1);
        initEtsMatrices(inputModel);
        initPar(inputModel);
    }
}
// Interpolation
void ETSclass::interpolate(){
    ETSmodel sysCopy = inputModel;
    inputModel.h = 0;
    vec y = inputModel.y;
    mat u = inputModel.u;
    string error = "M", trend = "?", seasonal = "M", model;
    if (inputModel.negative){
        error = "A";
        seasonal = "A";
    }
    if (inputModel.s < 2)
        seasonal = "N";
    // Selecting model
    /*
    if (inputModel.error != "?")
        error = inputModel.error;
    if (inputModel.trend[0] != '?')
        trend = inputModel.trend;
    if (inputModel.seasonal != "?")
        seasonal = inputModel.seasonal;
    */
    model = error + trend + seasonal;
    inputModel.model = model;
    vec y1(inputModel.missing.n_elem), y2(inputModel.missing.n_elem), aux(inputModel.y.n_elem);
    uvec col(1); col(0) = 1;
    int iMin = min(inputModel.missing), iMax = max(inputModel.missing),
        iniObs = 0, finalObs = inputModel.y.n_elem;
    uvec indFinite, poly(3, fill::ones), aux3;
    vec fit;
    // Forward interpolation
    if (iMin > 4){
        setModel(inputModel, model, inputModel.userS);
        ident(false);
        model = error + inputModel.model[0] + seasonal;
        components();
        y1 = inputModel.comp.submat(inputModel.missing, col);
    } else {
        // Missing from start
        indFinite = find_finite(y.rows(iMin, y.n_elem - 1));
        aux3 = conv(diff(indFinite), poly);
        iniObs = indFinite(min(find(aux3.rows(2, aux3.n_elem - 1) == 3))) + iMin;
        inputModel.y = y.rows(iniObs, y.n_elem - 1);
        if (u.n_rows > 0)
            inputModel.u = u.cols(iniObs, y.n_elem - 1);
        setModel(inputModel, model, inputModel.userS);
        ident(false);
        model = error + inputModel.model[0] + seasonal;
        components();
        fit = join_vert(zeros(iniObs), inputModel.comp.col(1));
        y1 = fit(inputModel.missing);
        inputModel.y = y;
        inputModel.u = u;
    }
    if (inputModel.y.n_elem - iMax - 1 > 5){
        // Backwards interpolation
        inputModel.y = reverse(inputModel.y);
        setModel(inputModel, model, inputModel.userS);
        estim(false);
        components();
        aux = reverse(inputModel.comp.col(1));
        y2 = aux(inputModel.missing);
    } else {
        // Missing at the right end
        indFinite = find_finite(y.rows(0, iMax));
        aux3 = conv(diff(indFinite), poly);
        finalObs = indFinite(max(find(aux3.rows(0, aux3.n_elem - 3) == 3)) + 1);
        inputModel.y = reverse(y.rows(0, finalObs));
        if (u.n_rows > 0)
            inputModel.u = reverse(u.cols(0, finalObs), 1);
        setModel(inputModel, model, inputModel.userS);
        estim(false);
        components();
        fit = join_vert(reverse(inputModel.comp.col(1)), zeros(finalObs));
        y2 = fit(inputModel.missing);
        inputModel.y = y;
        inputModel.u = u;
    }
    y(inputModel.missing) = (y1 + y2) / 2;
    if (iMin < 6){
        // Correcting first chunk
        uvec missInd = inputModel.missing(find(inputModel.missing < iniObs));
        y(missInd) = y2.rows(0, missInd.n_elem - 1);
    }
    if (inputModel.y.n_elem - iMax < 6){
        // Correcting last chunk
        uvec missInd = inputModel.missing(find(inputModel.missing > finalObs));
        y(missInd) = y1.rows(y1.n_elem - missInd.n_elem, y1.n_elem - 1);
    }
    // Restoring initial values and interpolating
    inputModel = sysCopy;
    inputModel.y = y;
    setModel(inputModel, inputModel.model, inputModel.userS);
}
// Estimation
void ETSclass::estim(bool verbose){
    // outputs
    double objFunValue = 0.0;
    inputModel.loge2 = 0.0;
    inputModel.logr = 0.0;
    vec grad, p = inputModel.p0, p0userCOPY = inputModel.p0user;
    mat iHess;
    // initial time
    wall_clock timer;
    timer.tic();
    // Comparing exact and numerical gradients (do not remove!!)
    if (false){
        int nFuns;
        vec grad0;
        bool EXACT = inputModel.exact;
        inputModel.exact = true;
        grad0 = gradETS(p, &inputModel, objFunValue, nFuns);
        grad0.t().print("grad 208 exact");
        inputModel.exact = false;
        grad0 = gradETS(p, &inputModel, objFunValue, nFuns);
        grad0.t().print("grad 208 false");
        inputModel.exact = EXACT;
    }
    int flag, bestFlag = 10, nAttempts = 0;
    ETSmodel best;
    double LLIK = 0.0, AIC = 0.0, BIC = 0.0, AICc = 0.0, bestAIC = 0.0,
           bestBIC = 0.0, bestLLIK = 0.0, bestAICc = 1e10, 
           bestObjFunValue = 0.0, maxGrad = 0.0;
    vec bestP, bestGrad;
    bool again = false;
    do{
        //p.t().print("p 332");
        //vec pp = p.rows(0, inputModel.limits.n_rows - 1);
        //trans(pp, inputModel.limits);
        //pp.t().print("pp 335");

        flag = quasiNewtonETS(llikETS, gradETS, p, &inputModel, objFunValue, grad, iHess, verbose,
                              inputModel.limits.n_rows);
        // Correction in case of convergence problems
        if (flag > 5)
                objFunValue = llikETS(p, &inputModel);
        if (isnan(objFunValue))
            inputModel.loge2 = datum::nan;
        // Information criteria
        uvec indNan = inputModel.missing; //find_nonfinite(inputModel.y);
        int nNan = inputModel.y.n_elem - indNan.n_elem;
                //LLIK = -0.5 * (nNan * inputModel.loge2 + 2 * inputModel.logr); ////////////////////////////////////////////
        // Hyndman lo hace como en la siguiente línea
        inputModel.sigma2 = exp(inputModel.loge2) / nNan;
                //LLIK = -0.5 * (nNan * log(inputModel.sigma2 * (nNan - p.n_elem + 1)) + 2 * inputModel.logr);
        LLIK = -0.5 * (nNan * log(inputModel.sigma2 * nNan) + 2 * inputModel.logr);
        infoCriteria(LLIK, p.n_elem, nNan, AIC, BIC, AICc);
        maxGrad = max(abs(inputModel.grad.rows(0, inputModel.limits.n_rows - 1)));
        if (nAttempts == 0 || (!isnan(objFunValue) && objFunValue < bestObjFunValue)){
            best = inputModel;
            bestAICc = AICc;
            bestAIC = AIC;
            bestBIC = BIC;
            bestLLIK = LLIK;
            bestFlag = flag;
            bestObjFunValue = objFunValue;
            bestGrad = grad;
            bestP = p;
        }
        if (false){
            if (nAttempts > 0)
                inputModel.p0user = p0userCOPY;
            if (flag > 2 && maxGrad > 1e-3 && nAttempts < 4 && inputModel.p0user.n_elem == 0){
                again = true;
                if (AICc < bestAICc){
                    best = inputModel;
                    bestAICc = AICc;
                    bestAIC = AIC;
                    bestBIC = BIC;
                    bestLLIK = LLIK;
                    bestFlag = flag;
                    bestObjFunValue = objFunValue;
                    bestGrad = grad;
                    bestP = p;
                }
                if (nAttempts == 0){
                    //inputModel.prop = 0.2;
                    inputModel.p0user = {inputModel.alphaL(1) - 0.01, inputModel.betaL(0) + 0.0001,
                                        inputModel.phiL(1) - 0.01, 1 - inputModel.alphaL(1) + 0.0001};
                } else if (nAttempts == 1){
                    //inputModel.p0user = p0userCOPY;
                    inputModel.prop = 0.3;
                } else {
                    inputModel.prop += 0.3;
                }
                initPar(inputModel);
                p = inputModel.p0;
            } else
                again = false;
            nAttempts++;
            if (isnan(AICc) && nAttempts < 4)
                again = true;
        }
    } while (again);
    if (nAttempts > 0)
        inputModel = best;
    inputModel.p0user = p0userCOPY;
    vec criteria(4);
    criteria(0) = bestLLIK;
    criteria(1) = bestAIC;
    criteria(2) = bestBIC;
    criteria(3) = bestAICc;
    this->inputModel.criteria = criteria;
    if (!isfinite(objFunValue))
        bestFlag = 0;
    // Printing results
    if (bestFlag == 1) {
      this->inputModel.estimOk = "Q-Newton: Gradient convergence.\n";
    } else if (bestFlag == 2){
      this->inputModel.estimOk = "Q-Newton: Function convergence.\n";
    } else if (bestFlag == 3){
        this->inputModel.estimOk = "Q-Newton: Parameter convergence.\n";
    } else if (bestFlag == 4){
        this->inputModel.estimOk = "Q-Newton: Maximum number of iterations reached.\n";
    } else if (bestFlag == 5){
        this->inputModel.estimOk = "Q-Newton: Maximum number of Function evaluations.\n";
    } else if (bestFlag == 6){
        this->inputModel.estimOk = "Q-Newton: Unable to decrease objective function.\n";
    } else if (bestFlag == 7){
        this->inputModel.estimOk = "Q-Newton: Objective function returns nan.\n";
    } else {
        this->inputModel.estimOk = "Q-Newton: No convergence!!\n";
    }
    if (verbose){
      double nSeconds = timer.toc();
      printf("%s", this->inputModel.estimOk.c_str());
      printf("Elapsed time: %10.5f seconds\n", nSeconds);
    }
    etsMatrices(&inputModel, bestP);
    vec paux = bestP.rows(0, sum(inputModel.nPar.rows(0, 1)) - 1);
    trans(paux, inputModel.limits);
    // Storing results in structure
    this->inputModel.p = bestP;
    this->inputModel.truep = paux;
    this->inputModel.objFunValue = bestObjFunValue;
    this->inputModel.grad = bestGrad;
    this->inputModel.flag = bestFlag;
    // Storing parameter values
    vec aux = bestP.rows(0, inputModel.limits.n_rows - 1);
    vec nParCum = cumsum(inputModel.nPar);
    trans(aux, inputModel.limits);
    bestP.rows(0, inputModel.limits.n_rows - 1) = aux;
    int pos = 1;
    inputModel.alpha = bestP(0);
    if (inputModel.trend != "N"){
        inputModel.beta = bestP(1);
        pos++;
    }
    if (inputModel.model.length() > 3){
        inputModel.phi = bestP(2);
        pos++;
    }
    if (inputModel.seasonal != "N"){
        inputModel.gamma = bestP(pos);
        pos++;
    }
    inputModel.x0 = bestP.rows(pos, nParCum(2) - 1);
    if (inputModel.u.n_rows > 0)
        inputModel.d = bestP.rows(nParCum(2), nParCum(3) - 1).t();
    if (sum(inputModel.arma) > 0){
        if (inputModel.arma(0) > 0){
            inputModel.ar = bestP.rows(nParCum(3), nParCum(3) + inputModel.arma(0) - 1);
            polyStationary(inputModel.ar);
        }
        if (inputModel.arma(1) > 0){
            inputModel.ma = bestP.rows(nParCum(3) + inputModel.arma(0), nParCum(3) + sum(inputModel.arma) - 1);
            polyStationary(inputModel.ma);
        }
    }
}
// Validation
void ETSclass::validate(){
    // First part of table
    char str[70];
    inputModel.table.clear();
    inputModel.table.push_back(" -------------------------------------------------------------\n");
    snprintf(str, 70, " Model: ETS(%s,%s,%s)", inputModel.error.c_str(), inputModel.trend.c_str(), inputModel.seasonal.c_str());
    // Model name
    if (inputModel.u.n_rows == 0 && sum(inputModel.arma) == 0)
        snprintf(str, 70, " Model: ETS(%s,%s,%s)\n", inputModel.error.c_str(), inputModel.trend.c_str(), inputModel.seasonal.c_str());
    if (inputModel.u.n_rows > 0 && sum(inputModel.arma) == 0)
        snprintf(str, 70, " Model: ETS(%s,%s,%s) + exogenous\n", inputModel.error.c_str(), inputModel.trend.c_str(), inputModel.seasonal.c_str());
    else if (inputModel.u.n_rows > 0 && sum(inputModel.arma) > 0){
        string ar = to_string((int)inputModel.arma(0)),
               ma = to_string((int)inputModel.arma(1));
        snprintf(str, 70, " Model: ETS(%s,%s,%s) + exogenous + ARMA(%s,%s)\n",
                inputModel.error.c_str(), inputModel.trend.c_str(), inputModel.seasonal.c_str(), ar.c_str(), ma.c_str());
    } else if (inputModel.u.n_rows == 0 && sum(inputModel.arma) > 0){
        string ar = to_string((int)inputModel.arma(0)),
                ma = to_string((int)inputModel.arma(1));
        snprintf(str, 70, " Model: ETS(%s,%s,%s) + ARMA(%s,%s)\n",
                inputModel.error.c_str(), inputModel.trend.c_str(), inputModel.seasonal.c_str(), ar.c_str(), ma.c_str());
    }
    inputModel.table.push_back(str);
    // Box-Cox lambda
    snprintf(str, 70, " Box-Cox lambda: %3.2f\n", inputModel.lambda);
    inputModel.table.push_back(str);
    // Parameter values
    vector<string> parNames;
    mat tp;
    if (inputModel.estimOk != "Q-Newton: No convergence!!\n")
        covPar(&inputModel, parNames, tp);
    //inputModel.table.push_back(str);
    snprintf(str, 70, " %s", inputModel.estimOk.c_str());
    inputModel.table.push_back(str);
    inputModel.table.push_back("-------------------------------------------------------------\n");
    inputModel.table.push_back("                  Param        S.E.          |T|     |Grad|\n");
    inputModel.table.push_back("-------------------------------------------------------------\n");
    // Table of numbers
    for (unsigned i = 0; i < tp.n_rows; i++){
        if (abs(tp(i, 0)) > 1e-4){
            snprintf(str, 70, "%*s: %12.4f %10.4f %12.4f %10.3e\n", 10, parNames.at(i).c_str(), tp(i, 0), tp(i, 1), tp(i, 2), tp(i, 3));
        } else {
            snprintf(str, 70, "%*s: %12.3e %10.3e %12.4f %10.3e\n", 10, parNames.at(i).c_str(), tp(i, 0), tp(i, 1), tp(i, 2), tp(i, 3));
        }
        inputModel.table.push_back(str);
    }
    inputModel.table.push_back("-------------------------------------------------------------\n");
    // Adding criteria information
    snprintf(str, 70, "  AIC: %12.4f   BIC: %12.4f   AICc: %12.4f\n", inputModel.criteria(1), inputModel.criteria(2), inputModel.criteria(3));
    inputModel.table.push_back(str);
    snprintf(str, 70, "           Log-Likelihood: %12.4f\n", inputModel.criteria(0));
    inputModel.table.push_back(str);
    inputModel.table.push_back("-------------------------------------------------------------\n");
    // Recovering innovations for tests
    components();
    postProcess(inputModel);
    //Second part of table
    inputModel.table.push_back("   Summary statistics:\n");
    inputModel.table.push_back("-------------------------------------------------------------\n");
    uvec auxx = find_finite(inputModel.comp.col(0));
    if (auxx.n_elem < 5){
      inputModel.table.push_back("  All innovations are NaN!!\n");
    } else {
      outputTable(inputModel.comp.submat(0, 0, inputModel.y.n_elem - 1, 0), inputModel.table);
    }
    inputModel.table.push_back("-------------------------------------------------------------\n");
     // Show Table
     if (inputModel.verbose){
         // for (auto i = inputs.table.begin(); i != inputs.table.end(); i++){
         //   cout << *i << " ";
         // }
         for (unsigned int i = 0; i < inputModel.table.size(); i++){
           printf("%s ", inputModel.table[i].c_str());
         }
     }
}
// Simulate system for forecasts confidence bands estimation
void ETSclass::simulate(uword n, vec x0){
    // Storing variables
    vec Xn = inputModel.xn; //, YFOR = inputModel.yFor;
    int H = inputModel.h;
    bool VAR = inputModel.forIntervals, BOOTSTRAP = inputModel.bootstrap;
    inputModel.xn = x0;
    inputModel.h = n;
    inputModel.forIntervals = false;
    inputModel.bootstrap = true;
    // Simulating
    inputModel.ySimul.set_size(n, inputModel.nSimul);
    for (int i = 0; i < inputModel.nSimul; i++){
        forecast();
        inputModel.ySimul.col(i) = inputModel.yFor;
    }
    inputModel.yFor = mean(inputModel.ySimul, 1);
    inputModel.yForV = var(inputModel.ySimul, 0, 1);
    //Restoring variables
    inputModel.xn = Xn;
    inputModel.h = H;
    //inputModel.yFor = YFOR;
    inputModel.forIntervals = VAR;
    inputModel.bootstrap = BOOTSTRAP;
}
// Error calculation
void ETSclass::forecast(){
    vec x = inputModel.xn, accum(1, fill::ones), a(inputModel.h);
    rowvec fitu(inputModel.h, fill::zeros);
    inputModel.yFor.set_size(inputModel.h);
    inputModel.yForV.set_size(inputModel.h);
    if (isnan(inputModel.objFunValue)){
        inputModel.yFor.fill(datum::nan);
        inputModel.yForV.fill(datum::nan);
        return;
    }
    inputModel.yFor.fill(0.0);
    inputModel.yForV.fill(0.0);
    bool nu = (inputModel.u.n_rows > 0);
    if (nu)
        fitu = inputModel.d * inputModel.u.tail_cols(inputModel.h);
    if (inputModel.bootstrap){
        // Bootstraping noise for simulation
        if (inputModel.comp.n_cols == 0)
            components();
        int aux;
        vec e;
        uvec ind;
        e = inputModel.comp.submat(0, 0, inputModel.y.n_elem - 1, 0);
        e = removeNans(e, aux);
        ind = conv_to<uvec>::from(randi(inputModel.h, distr_param(0, e.n_elem - 1)));
        a = e(ind);
    }
    if (inputModel.modelType == 0){  // Linear Class 1 p. 81
        if (inputModel.bootstrap){
            // Bootstrap simulation of just mean
            for (int t = 0; t < inputModel.h; t++){
                inputModel.yFor.row(t) = inputModel.w * x + a.row(t) + fitu.col(t);
                x = inputModel.F * x + inputModel.g * a.row(t);
            }
        } else {
            // Normal estimation (no simulation)
            vec Fg = inputModel.g;
            for (int i = 0; i < inputModel.h; i++){
                inputModel.yFor.row(i) = inputModel.w * x + fitu.col(i);
                inputModel.yForV.row(i) = accum;
                x = inputModel.F * x;
                accum += inputModel.w * Fg * Fg.t() * inputModel.w.t();
                Fg = inputModel.F * Fg;
            }
            inputModel.yForV *= inputModel.sigma2;
        }
    } else if (inputModel.error == "M" && inputModel.seasonal != "M" && inputModel.trend[0] != 'M'){
        // Class 2 p. 83
        if (inputModel.bootstrap){
            vec fit(1);
            // Bootstrap simulation of just mean
            for (int i = 0; i < inputModel.h; i++){
                fit = inputModel.w * x;
                inputModel.yFor.row(i) = fit * (1 + a.row(i)) + fitu.col(i);
                x = inputModel.F * x + inputModel.g * fit * a.row(i);
            }
        } else {
            // Normal estimation
            vec Fg = inputModel.g;
            vec theta(1), cj2(1, fill::ones);
            accum.fill(0);
            //theta = pow(inputModel.w * x, 2);
            for (int i = 0; i < inputModel.h; i++){
                inputModel.yFor.row(i) = inputModel.w * x + fitu.col(i);
                theta = pow(inputModel.yFor.row(i), 2) + inputModel.sigma2 * accum;
                inputModel.yForV.row(i) = (1 + inputModel.sigma2) * theta - pow(inputModel.yFor.row(i), 2);
                x = inputModel.F * x;
                accum += cj2 * theta;
                cj2 = inputModel.w * Fg * Fg.t() * inputModel.w.t();
                Fg = inputModel.F * Fg;
            }
        }
    } else if (inputModel.error == "M" && inputModel.seasonal == "M" && inputModel.trend[0] != 'M'){
        // Class 3 p. 83 approximate formulas for variance
        if (inputModel.bootstrap){
            // Bootstrap simulation of just mean
            double s, fit;
            int ns = sum(inputModel.ns.rows(0, 1)) - 1;
            vec g = inputModel.g;
            //theta = pow(inputModel.w * x, 2);
            for (int i = 0; i < inputModel.h; i++){
                if (inputModel.model == "MNM"){
                    s = x(ns);
                    inputModel.yFor(i) = (x(0) * s) * (1 + a(i)) + fitu(i);
                    x = (inputModel.F * x) % (1 + inputModel.g * a(i));
                } else {
                    // MAM or MAdM
                    s = x(ns);
                    fit = x(0) + inputModel.phi * x(1);
                    g(0) = inputModel.g(0) * fit;
                    g(1) = inputModel.g(1) * fit;
                    g(2) = inputModel.g(2) * s;
                    inputModel.yFor(i) = fit * s * (1 + a(i)) + fitu(i);
                    x = inputModel.F * x + g * a.row(i);
                }
            }
        } else {
            // Normal estimation
            vec Fg = inputModel.g;
            vec theta(1), cj2(1, fill::ones);
            accum.fill(0);
            double s;
            int ns = sum(inputModel.ns.rows(0, 1)) - 1;
            //theta = pow(inputModel.w * x, 2);
            for (int i = 0; i < inputModel.h; i++){
                if (inputModel.model == "MNM"){
                    s = x(ns);
                    inputModel.yFor(i) = x(0) * s;
                } else {
                    // MAM or MAdM
                    s = x(ns);
                    inputModel.yFor(i) = (x(0) + inputModel.phi * x(1)) * s + fitu(i);
                }
                theta = pow(inputModel.yFor.row(i), 2) + inputModel.sigma2 * accum;
                inputModel.yForV.row(i) = s * s * ((1 + inputModel.sigma2) * pow((1 + inputModel.sigma2 *
                                          pow(inputModel.g(inputModel.nPar(0)), 2)),
                                          fmod(i, inputModel.userS)) * theta - pow(inputModel.yFor.row(i), 2));
                x = inputModel.F * x;
                accum += cj2 * theta;
                cj2 = inputModel.w * Fg * Fg.t() * inputModel.w.t();
                Fg = inputModel.F * Fg;
            }
        }
    } else {
        // Classes 4 and 5. Calculate intervals by simulation always, either bootstrap or Gaussian
        vec y(inputModel.h), xn = inputModel.xn;
        if (!inputModel.bootstrap)
            a.fill(0);
        y.fill(datum::nan);
        double obj = 0.0, logr = 0.0, e = 0.0;
        int ns = sum(inputModel.ns.rows(0, 1)) - 1;
        inputModel.yForV.fill(0.0);
        double sigma = sqrt(inputModel.sigma2);
        simulForecast45(inputModel.model, y, fitu, inputModel.h, ns, xn, inputModel.g, inputModel.phi, e, a, obj, logr);
        inputModel.yFor = y;
        if (inputModel.forIntervals){     // Variance
            for (int i = 0; i < inputModel.nSimul; i++){
                if (!inputModel.bootstrap){
                    a.randn();
                    a *= sigma;
                }
                y.fill(datum::nan);
                xn = inputModel.xn;
                simulForecast45(inputModel.model, y, fitu, inputModel.h, ns, xn, inputModel.g, inputModel.phi, e, a, obj, logr);
                inputModel.yForV = (i * inputModel.yForV + pow(y - inputModel.yFor, 2)) / (i + 1);
            }
        }
    }
}
/*
// Calculate resids of arma model
void residARMA(vec& y, vec ar, vec ma){
    SSinputs data;
    data.y = y;
    //SSmodel(data);
    ARMAmodel model = ARMAmodel(data, ar.n_elem, ma.n_elem);
    model.filter(0);
    data = model.getInputs();
    y = data.v;
}
*/
// Simulate forecasts for forecasting intervals estimation
void simulForecast45(string model, vec& y, rowvec fitu, int h, int ns, vec& x, vec g, double phi, double e, vec a, double& obj, double& logr){
    if (model == "MMN" || model == "MMdN")
        MMN(y, fitu, h, x, g, phi, e, a, obj, logr);
    else if (model == "MMM" || model == "MMdM")
        MMM(y, fitu, h, ns, x, g, phi, e, a, obj, logr);
    else if (model == "MMA" || model == "MMdA")
        MMA(y, fitu, h, ns, x, g, phi, e, a, obj, logr);
    else if (model == "ANM")
        ANM(y, fitu, h, ns, x, g, e, a, obj, logr);
    else if (model == "AAM" || model == "AAdM")
        AAM(y, fitu, h, ns, x, g, phi, e, a, obj, logr);
    else if (model == "AMN" || model == "AMdN")
        AMN(y, fitu, h, x, g, phi, e, a, obj, logr);
    else if (model == "AMA" || model == "AMdA")
        AMA(y, fitu, h, ns, x, g, phi, e, a, obj, logr);
    else if (model == "AMM" || model == "AMdM")
        AMM(y, fitu, h, ns, x, g, phi, e, a, obj, logr);
}
// AMM models forecasting and error calculation. For forecasting y = nan and a = nan
void AMM(vec& y, rowvec fitu, int n, int ns, vec& x, vec g, double phi, double e, vec a, double& obj, double& logr){
    double b, s, fit;
    obj = 0.0;
    logr = 0.0;
    for (int t = 0; t < n; t++){
        s = x(ns);
        b = pow(x(1), phi);
        fit = x(0) * b;
        if (isfinite(y(t)))
            e = y(t) - fit * s - fitu(t);
        else if (a.has_nan())
            e = 0.0;
        else {
            e = a(t);
            y(t) = fit * s + e + fitu(t);
        }
        x(1) = b + g(1) * e / (s * x(0));
        x(0) = fit + g(0) * e / s;
        x.rows(3, ns) = x.rows(2, ns - 1);
        x(2) = s + g(2) * e / fit;
        obj += e * e;
    }
}
// AMA models forecasting and error calculation. For forecasting y = nan and a = nan
void AMA(vec& y, rowvec fitu, int n, int ns, vec& x, vec g, double phi, double e, vec a, double& obj, double& logr){
    double b, s, fit;
    obj = 0.0;
    logr = 0.0;
    for (int t = 0; t < n; t++){
        b = pow(x(1), phi);
        s = x(ns);
        fit = x(0) * b;
        if (isfinite(y(t)))
            e = y(t) - fit - s - fitu(t);
        else if (a.has_nan())
            e = 0.0;
        else {
            e = a(t);
            y(t) = fit + s + e + fitu(t);
        }
        x(1) = b + g(1) * e / x(0);
        x(0) = fit + g(0) * e;
        x.rows(3, ns) = x.rows(2, ns - 1);
        x(2) = s + g(2) * e;
        obj += e * e;
    }
}
// AMN models forecasting and error calculation. For forecasting y = nan and a = nan
void AMN(vec& y, rowvec fitu, int n, vec& x, vec g, double phi, double e, vec a, double& obj, double& logr){
    double b, fit;
    obj = 0.0;
    logr = 0.0;
    for (int t = 0; t < n; t++){
        b = pow(x(1), phi);
        fit = x(0) * b;
        if (isfinite(y(t)))
            e = y(t) - fit - fitu(t);
        else if (a.has_nan())
            e = 0.0;
        else {
            e = a(t);
            y(t) = fit + e + fitu(t);
        }
        x(1) = b + g(1) * e / x(0);
        x(0) = fit + g(0) * e;
        obj += e * e;
    }
}
// AAM models forecasting and error calculation. For forecasting y = nan and a = nan
void AAM(vec& y, rowvec fitu, int n, int ns, vec& x, vec g, double phi, double e, vec a, double& obj, double& logr){
    double s, fit;
    obj = 0.0;
    logr = 0.0;
    for (int t = 0; t < n; t++){
        s = x(ns);
        fit = x(0) + phi * x(1);
        if (isfinite(y(t)))
            e = y(t) - fit * s - fitu(t);
        else if (a.has_nan())
            e = 0.0;
        else {
            e = a(t);
            y(t) = fit * s + e + fitu(t);
        }
        x(1) += g(1) * e / x(0);
        x(0) = fit + g(0) * e / s;
        x.rows(3, ns) = x.rows(2, ns - 1);
        x(2) = s + g(2) * e / fit;
        obj += e * e;
    }
}
// ANM models forecasting and error calculation. For forecasting y = nan and a = nan
void ANM(vec& y, rowvec fitu, int n, int ns, vec& x, vec g, double e, vec a, double& obj, double& logr){
    double s, fit;
    obj = 0.0;
    logr = 0.0;
    for (int t = 0; t < n; t++){
        s = x(ns);
        fit = x(0);
        if (isfinite(y(t)))
            e = y(t) - fit * s- fitu(t);
        else if (a.has_nan())
            e = 0.0;
        else {
            e = a(t);
            y(t) = fit * s + e + fitu(t);
        }
        x(0) += g(0) * e / s;
        x.rows(2, ns) = x.rows(1, ns - 1);
        x(1) = s + g(1) * e / fit;
        obj += e * e;
    }
}
// MMN models forecasting and error calculation. For forecasting y = nan and a = nan
void MMM(vec& y, rowvec fitu, int n, int ns, vec& x, vec g, double phi, double e, vec a, double& obj, double& logr){
    double b, s, fit;
    obj = 0.0;
    logr = 0.0;
    for (int t = 0; t < n; t++){
        b = pow(x(1), phi);
        s = x(ns);
        fit = x(0) * b * s;
        if (isfinite(y(t)))
            e = (y(t) - fitu(t)) / fit - 1;
        else if (a.has_nan())
            e = 0.0;
        else {
            e = a(t);
            y(t) = fit * (1 + e) + fitu(t);
        }
        x(1) = b * (1 + g(1) * e);
        x(0) = x(0) * b * (1 + g(0) * e);
        x.rows(3, ns) = x.rows(2, ns - 1);
        x(2) = s * (1 + g(2) * e);
        obj += e * e;
        logr += log(abs(fit));
    }
}
// MMN models forecasting and error calculation. For forecasting y = nan and a = nan
void MMN(vec& y, rowvec fitu, int n, vec& x, vec g, double phi, double e, vec a, double& obj, double& logr){
    double b, fit;
    obj = 0.0;
    logr = 0.0;
    for (int t = 0; t < n; t++){
        b = pow(x(1), phi);
        fit = x(0) * b;
        if (isfinite(y(t)))
            e = (y(t) - fitu(t)) / fit - 1;
        else if (a.has_nan())
            e = 0.0;
        else {
            e = a(t);
            y(t) = fit * (1 + e) + fitu(t);
        }
        x(1) = b *(1 + g(1) * e);
        x(0) = fit * (1 + g(0) * e);
        obj += e * e;
        logr += log(abs(fit));
    }
}
// MMA models forecasting and error calculation. For forecasting y = nan and a = nan
void MMA(vec& y, rowvec fitu, int n, int ns, vec& x, vec g, double phi, double e, vec a, double& obj, double& logr){
    double b, s, fit;
    obj = 0.0;
    logr = 0.0;
    for (int t = 0; t < n; t++){
        b = pow(x(1), phi);
        s = x(ns);
        fit = x(0) * b + s;
        if (isfinite(y(t)))
            e = (y(t) - fitu(t)) / fit - 1;
        else if (a.has_nan())
            e = 0.0;
        else {
            e = a(t);
            y(t) = fit * (1 + e) + fitu(t);
        }
        x(1) = b + g(1) * fit * e / x(0);
        x(0) = x(0) * b + g(0) * fit * e;
        x.rows(3, ns) = x.rows(2, ns - 1);
        x(2) = s + g(2) * fit * e;
        obj += e * e;
        logr += log(abs(fit));
    }
}
// Error calculation
void ETSclass::components(){
    // Estimated components (error, fit, level, seasonal, slope, exogenous, arma)
    // Positioning alpha, beta, gamma and phi
    ETSmodel* m = &inputModel;
    etsMatrices(m, m->p);
    uword n = m->y.n_elem, posSeas = 0;
    bool seas = (m->s > 1), slope = (m->trend != "N"), arma = (sum(m->arma) > 0);
    vec x = m->x0;
    bool nu = (inputModel.u.n_rows > 0);
    mat fitu;
    // Components names
    m->compNames = "Error/Fit/Level";
    char name[15];
    if (m->seasonal != "N")
        m->compNames += "/Seasonal";
    if (m->trend != "N")
        m->compNames += "/Slope";
    if (nu){
            
            
            
            
            
        string auxbeta;
        for (int i = 0; i < (int)m->u.n_rows; i++){
                auxbeta = to_string(i + 1);
                snprintf(name, 15, "/Beta(%s)", auxbeta.c_str());
                m->compNames += name;
        }
            
            
        // for (int i = 0; i < (int)m->u.n_rows; i++){
        //     snprintf(name, 15, "/Beta(%d)", i + 1);
        //     m->compNames += name;
        // }
    }
    if (arma){
        string ar = to_string((int)m->arma(0)),
                ma = to_string((int)m->arma(1));
        snprintf(name, 15, "/ARMA(%s,%s)", ar.c_str(), ma.c_str());
        m->compNames += name;
    }
    // inputs?
    if (nu)
        fitu = repmat(inputModel.d.t(), 1, m->u.n_cols) % inputModel.u;
    else {
        fitu.set_size(1, inputModel.y.n_elem + inputModel.h);
        fitu.fill(0);
    }
    // Initializing
    m->comp.set_size(n + m->h, 3 + seas + slope + m->u.n_rows + arma);
    m->comp.fill(datum::nan);
    m->comp.submat(0, 1, n - 1, 1) = m->y;
    if (seas)
        posSeas = slope + 1;
    if (m->modelType == 0){
        vec e(1), fit(1);
        int ind = m->ns(0) + m->ns(1);
        // Linear model
        for (uword t = 0; t < n + m->h; t++){
            fit = m->w * x + sum(fitu.col(t));
            if (is_finite(m->comp(t, 1))){
                e = m->y.row(t) - fit;
                x = m->F * x + m->g * e;
                m->comp(t, 0) = e(0);
            } else {
                x = m->F * x;
            }
            m->comp(t, 1) = fit(0);
            m->comp(t, 2) = x(0);
            if (seas)
                m->comp(t, 3) = x(posSeas);
            if (slope)
                m->comp(t, 3 + seas) = x(1);
            if (nu)
                m->comp.submat(t, 3 + seas + slope, t, 3 + seas + slope + m->u.n_rows - 1) = fitu.col(t).t();
            if (arma)
                m->comp(t, 3 + seas + slope + m->u.n_rows) = x(ind) - e(0);
        }
    } else {
        // Non linear model
        double b, s, e, fit = 0.0;
        int ns = x.n_elem - 1;
        vec g = m->g;
        if (m->error == "A"){
            // Additive error
            for (uword t = 0; t < n + m->h; t++){
                if (m->model == "AMN" || m->model == "AMdN") {
                    b = pow(x(1), m->phi);
                    fit = x(0) * b + sum(fitu.col(t));
                    if (is_finite(m->comp(t, 1))){
                        e = m->y(t) - fit;
                        m->comp(t, 0) = e;
                    } else {
                        e = 0.0;
                    }
                    x(1) = b + g(1) * e / x(0);
                    x(0) = fit + g(0) * e;
                } else if (m->model == "ANM"){
                    s = x(ns);
                    fit = x(0);
                    if (is_finite(m->comp(t, 1))){
                        e = m->y(t) - fit * s - sum(fitu.col(t));
                        m->comp(t, 0) = e;
                    } else
                        e = 0.0;
                    x(0) += g(0) * e / s;
                    x.rows(2, ns) = x.rows(1, ns - 1);
                    x(1) = s + g(1) * e / fit;
                    fit *= s + sum(fitu.col(t));
                } else if (m->model == "AMA" || m->model == "AMdA"){
                    b = pow(x(1), m->phi);
                    s = x(ns);
                    fit = x(0) * b;
                    if (is_finite(m->comp(t, 1))){
                        e = m->y(t) - fit - s - sum(fitu.col(t));
                        m->comp(t, 0) = e;
                    } else
                        e = 0.0;
                    x(1) = b + g(1) * e / x(0);
                    x(0) = fit + g(0) * e;
                    x.rows(3, ns) = x.rows(2, ns - 1);
                    x(2) = s + g(2) * e;
                    fit += s + sum(fitu.col(t));
                } else if (m->model == "AAM" || m->model == "AAdM"){
                    s = x(ns);
                    fit = x(0) + m->phi * x(1);
                    if (is_finite(m->comp(t, 1))){
                        e = m->y(t) - fit * s - sum(fitu.col(t));
                        m->comp(t, 0) = e;
                    } else
                        e = 0.0;
                    x(1) += g(1) * e / x(0);
                    x(0) = fit + g(0) * e / s;
                    x.rows(3, ns) = x.rows(2, ns - 1);
                    x(2) = s + g(2) * e / fit;
                    fit *= s + sum(fitu.col(t));
                } else if (m->model == "AMM" || m->model == "AMdM"){
                    s = x(ns);
                    b = pow(x(1), m->phi);
                    fit = x(0) * b;
                    if (is_finite(m->comp(t, 1))){
                        e = m->y(t) - fit * s - sum(fitu.col(t));
                        m->comp(t, 0) = e;
                    } else
                        e = 0.0;
                    x(1) = b + g(1) * e / (s * x(0));
                    x(0) = fit + g(0) * e / s;
                    x.rows(3, ns) = x.rows(2, ns - 1);
                    x(2) = s + g(2) * e / fit;
                    fit *= s + sum(fitu.col(t));
                }
                m->comp(t, 1) = fit;
                m->comp(t, 2) = x(0);
                if (seas)
                    m->comp(t, 3) = x(posSeas);
                if (slope)
                    m->comp(t, 3 + seas) = x(1);
                if (nu)
                    m->comp.submat(t, 3 + seas + slope, t, 3 + seas + slope + m->u.n_rows - 1) = fitu.col(t).t();
            }
        } else {
            // Multiplicative error
            for (uword t = 0; t < n + m->h; t++){
                if (m->model == "MNN"){
                    fit = x(0);
                    if (is_finite(m->comp(t, 1))){
                        e = (m->y(t) - sum(fitu.col(t))) / fit - 1;
                        m->comp(t, 0) = e;
                    } else
                        e = 0.0;
                    x(0) = x(0) * (1 + g(0) * e);
                } else if (m->model == "MAN" || m->model == "MAdN"){
                    fit = x(0) + m->phi * x(1);
                    if (is_finite(m->comp(t, 1))){
                        e = (m->y(t) - sum(fitu.col(t))) / fit - 1;
                        m->comp(t, 0) = e;
                    } else
                        e = 0.0;
                    x(1) = m->phi * x(1) + g(1) * fit * e;
                    x(0) = fit * (1 + g(0) * e);
                } else if (m->model == "MMN" || m->model == "MMdN"){
                    b = pow(x(1), m->phi);
                    fit = x(0) * b;
                    if (is_finite(m->comp(t, 1))){
                        e = (m->y(t) - sum(fitu.col(t))) / fit - 1;
                        m->comp(t, 0) = e;
                    } else
                        e = 0.0;
                    x(1) = b *(1 + g(1) * e);
                    x(0) = fit * (1 + g(0) * e);
                } else if (m->model == "MNA"){
                    s = x(ns);
                    fit = x(0) + s;
                    if (is_finite(m->comp(t, 1))){
                        e = (m->y(t) - sum(fitu.col(t))) / fit - 1;
                        m->comp(t, 0) = e;
                    } else
                        e = 0.0;
                    x(0) = x(0) + g(0) * fit * e;
                    x.rows(2, ns) = x.rows(1, ns - 1);
                    x(1) = s + g(1) * fit * e;
                } else if (m->model == "MAA" || m->model == "MAdA"){
                    b = m->phi * x(1);
                    s = x(ns);
                    fit = x(0) + b + s;
                    if (is_finite(m->comp(t, 1))){
                        e = (m->y(t) - sum(fitu.col(t))) / fit - 1;
                        m->comp(t, 0) = e;
                    } else
                        e = 0.0;
                    x(1) = b + g(1) * fit * e;
                    x(0) += b + g(0) * fit * e;
                    x.rows(3, ns) = x.rows(2, ns - 1);
                    x(2) = s + g(2) * fit * e;
                } else if (m->model == "MNM"){
                    s = x(ns);
                    fit = x(0) * s;
                    if (is_finite(m->comp(t, 1))){
                        e = (m->y(t) - sum(fitu.col(t))) / fit - 1;
                        m->comp(t, 0) = e;
                    } else
                        e = 0.0;
                    x(0) = x(0) * (1 + g(0) * e);
                    x.rows(2, ns) = x.rows(1, ns - 1);
                    x(1) = s * (1 + g(1) * e);
                } else if (m->model == "MAM" || m->model == "MAdM"){
                    b = m->phi * x(1);
                    s = x(ns);
                    fit = x(0) + b;
                    if (is_finite(m->comp(t, 1))){
                        e = (m->y(t) - sum(fitu.col(t))) / (fit * s) - 1;
                        m->comp(t, 0) = e;
                    } else
                        e = 0.0;
                    //x(1) = b + g(1) * fit * e;
                    //x(0) = fit * (1 + g(0) * e);
                    //x.rows(3, ns) = x.rows(2, ns - 1);
                    //x(2) = s * (1 + g(2) * e);
                    //fit *= s;
                    g(0) = inputModel.g(0) * fit;
                    g(1) = inputModel.g(1) * fit;
                    g(2) = inputModel.g(2) * s;
                    fit *= s; // * (1 + a(i)) + fitu(i);
                    x = inputModel.F * x + g * e;
                } else if (m->model == "MMM" || m->model == "MMdM"){
                    b = pow(x(1), m->phi);
                    s = x(ns);
                    fit = x(0) * b * s;
                    if (is_finite(m->comp(t, 1))){
                        e = (m->y(t) - sum(fitu.col(t))) / fit - 1;
                        m->comp(t, 0) = e;
                    } else
                        e = 0.0;
                    x(1) = b * (1 + g(1) * e);
                    x(0) = x(0) * b * (1 + g(0) * e);
                    x.rows(3, ns) = x.rows(2, ns - 1);
                    x(2) = s * (1 + g(2) * e);
                } else if (m->model == "MMA" || m->model == "MMdA"){
                    b = pow(x(1), m->phi);
                    s = x(ns);
                    fit = x(0) * b + s;
                    if (is_finite(m->comp(t, 1))){
                        e = (m->y(t) - sum(fitu.col(t))) / fit - 1;
                        m->comp(t, 0) = e;
                    } else
                        e = 0.0;
                    x(1) = b + g(1) * fit * e / x(0);
                    x(0) = x(0) * b + g(0) * fit * e;
                    x.rows(3, ns) = x.rows(2, ns - 1);
                    x(2) = s + g(2) * fit * e;
                }
                fit += sum(fitu.col(t));
                m->comp(t, 1) = fit;
                m->comp(t, 2) = x(0);
                if (seas)
                    m->comp(t, 3) = x(posSeas);
                if (slope)
                    m->comp(t, 3 + seas) = x(1);
                if (nu)
                    m->comp.submat(t, 3 + seas + slope, t, 3 + seas + slope + m->u.n_rows - 1) = fitu.col(t).t();
            }
        }
    }
    if (m->missing.n_elem > 0){
        uvec cero(1);
        cero.fill(0);
        //if (m->missing.n_elem == m->y.n_elem)
        //    m->comp(m->missing, cero).fill(datum::nan);
        //else {
            //uvec aux = find_nonfinite(m->y);
            m->comp(find_nonfinite(m->y), cero).fill(datum::nan);
        //}
    }
}
// Identification
void ETSclass::ident(bool verbose){
    wall_clock timer;
    timer.tic();
    // Finding models
    vector<string> allModels;
    string error, trend, seasonal;
    if (inputModel.error == "?"){
        if (inputModel.negative)
            error = "A";
        else
            error = "A/M";
    } else {
        error = inputModel.error;
    }
    if (inputModel.trend == "?"){
        if (inputModel.negative)
            trend = "N/A/Ad";
        else
            trend = "N/A/Ad/M/Md";
    } else {
        trend = inputModel.trend;
    }
    if (inputModel.seasonal == "?"){
        if (inputModel.negative)
            seasonal = "N/A";
        else
            seasonal = "N/A/M";
    } else {
        seasonal = inputModel.seasonal;
    }
    findModels(error, trend, seasonal, inputModel.identAll, allModels);
    // output if verbose
    if (verbose){
        if (inputModel.missing.n_elem > 0){
            printf("--------------------------------------------------------\n");
            printf("   Identification with %1i missing data.\n", (int)inputModel.missing.n_elem);
        }
        printf("--------------------------------------------------------\n");
        printf("    Model            AIC           BIC          AICc\n");
        printf("--------------------------------------------------------\n");
    }
    // Estimation loop
//    bool ARMAESTIM = inputModel.armaIdent;
//    inputModel.armaIdent = false;
    setModel(inputModel, allModels[0], inputModel.userS);
    ETSclass m1(inputModel);
    m1.estim(false);
    if (verbose){
        printf("  %*s: %13.4f %13.4f %13.4f\n", 8, prettyModel(allModels[0]).c_str(),
                m1.inputModel.criteria(1), m1.inputModel.criteria(2), m1.inputModel.criteria(3));
    }
    inputModel = m1.inputModel;
    uword crit;
    if (inputModel.criterion == "aicc")
        crit = 3;
    else if (inputModel.criterion == "aic")
        crit = 1;
    else
        crit = 2;
    for (uword i = 1; i < allModels.size(); i++){
        setModel(m1.inputModel, allModels[i], inputModel.userS);
        //cout << "model 1286: " << prettyModel(allModels[i]).c_str() << endl;
        m1.estim(false);
        if (verbose){
            printf("  %*s: %13.4f %13.4f %13.4f\n", 8, prettyModel(allModels[i]).c_str(),
                   m1.inputModel.criteria(1), m1.inputModel.criteria(2), m1.inputModel.criteria(3));
        }
        if (m1.inputModel.criteria[crit] < inputModel.criteria[crit])
            inputModel = m1.inputModel;
    }
    // Testing for ARMA, only linear models (modelType = 0)
    if (inputModel.modelType == 0 && inputModel.armaIdent){
        components();
        vec beta0, orders;
        uword maxAR = max(8, inputModel.s + 2);
        int maxOrder = min(inputModel.s - 1, 5);
        if (maxOrder < 2)
            maxOrder = 5;
        if (inputModel.comp.n_rows > 4 * maxAR)
            selectARMA(inputModel.comp.col(0), maxOrder, maxAR, "bic", orders, beta0);
        if (sum(orders) > 0){
            m1.inputModel = inputModel;
            m1.inputModel.arma = orders;
            setModel(m1.inputModel, m1.inputModel.model, inputModel.userS);
            m1.estim(false);
            if (verbose){
                string modelARMA = '+' + prettyModel(inputModel.model);
                printf(" %*s: %13.4f %13.4f %13.4f\n", 8, modelARMA.c_str(),
                        m1.inputModel.criteria(1), m1.inputModel.criteria(2), m1.inputModel.criteria(3));
            }
            if (m1.inputModel.criteria[crit] < inputModel.criteria[crit])
                inputModel = m1.inputModel;
        }
    }
    if (verbose){
        double nSeconds = timer.toc();
        printf("--------------------------------------------------------\n");
        printf("  Identification time: %10.5f seconds\n", nSeconds);
        printf("--------------------------------------------------------\n");
    }
//    inputModel.armaIdent = ARMAESTIM;
}
// post-process of user inputs
void postProcess(ETSmodel& input){
    input.y.rows(input.missing).fill(datum::nan);
    if (input.comp.n_rows > 0){
        uvec col(1);
        col(0) = 0;
        input.comp(input.missing, col).fill(datum::nan);
    }
}
// Pre-process of user inputs
ETSclass preProcess(vec y, mat u, string model, int s, int h,
                    bool verbose, string criterion, bool identAll,
                    rowvec alphaL, rowvec betaL, rowvec gammaL, rowvec phiL,
                    string parConstraints, bool forIntervals, bool bootstrap,
                    int nSimul, vec arma, bool armaIdent, vec p0, double lambda){
    // y:       otuput data (one time series)
    // u:       input data (excluding constant)
    // model:   string with three or four letters with model for error, trend and seasonal
    // s:       seasonal period
    // h:       forecasting horizon (if inputs it is recalculated as the length differences
    //          between u and y
    // verbose: shows estimation intermediate results
    // criterion: information criterion to use in identification
    // identALL: Whether to estimate all models
    // alphaL:  limits for alpha parameter
    // betaL:   limits for beta parameter
    // gammaL:  limits for gamma parameter
    // phiL:    limits for alphipha parameter
    // parConstraints: Constraints in parameters: none, standard, admissible
    // forIntervals: forecast variance calculation
    // nSimul:  number of simulations for bootstrap forecast simulation
    // arma:    ARMA(p, q) orders
    // armaIdent: identification of ARMA models on/off
    // lambda: Box-Cox transformation constant

    //Checking model and y
    bool negative = false;
    bool errorExit = false;
    if (model.length() == 0)
        model = "???";
    upper(model);
    if (model.length() > 3)
        model[2] = 'd';
    if (nanMin(y) <= 0){
        negative = true;
        if (model[0] == 'M'){
            printf("%s", "ERROR: Cannot run model on time series with negative or zero values!!!\n");
            errorExit = true;
        }
    }
    lower(criterion);
    lower(parConstraints);
    // Checking s
    if (s < 2){
        s = 0;
    }
    // Correcting h in case there are inputs
    if (u.n_cols > 0){
        h = u.n_cols - y.n_elem;
        if (h < 0){
            printf("%s", "ERROR: Inputs should be at least as long as the ouptut!!!\n");
            errorExit = true;
        }
    }
    // Correcting parameter limits
    checkLimits(alphaL, betaL, gammaL, phiL, parConstraints, errorExit);
    // Correcting initial parameters
    if (p0.n_elem > 0 && (any(p0 < 0) || any(p0 > 1) || p0(1) > p0(0) || p0(3) > 1 - p0(0))){
        printf("%s", "ERROR: Initial parameters incorrect, please check!!!\n");
        printf("%s", "p0 = (alpha, beta, phi, gamma)\n");
        printf("%s", "0 < alpha < 1\n");
        printf("%s", "0 <  beta < alpha\n");
        printf("%s", "0 <  phi  < 1\n");
        printf("%s", "0 < gamma < 1 - alpha\n");
        errorExit = true;
    }
    // Creating model
    ETSmodel input;
    input.initialModel = input.model;
    input.missing = find_nonfinite(y);
    input.userS = s;
    input.h = h;
    input.negative = negative;
    input.identAll = identAll;
    input.alphaL = alphaL;
    input.betaL = betaL;
    input.gammaL = gammaL;
    input.phiL = phiL;
    input.parConstraints = parConstraints;
    input.y = y;
    input.u = u;
    input.verbose = verbose;
    input.forIntervals = forIntervals;
    input.nSimul = nSimul;
    input.bootstrap = bootstrap;
    input.arma = arma;
    input.p0user = p0;
    // Checking lambda
    if (lambda != 9999.9 && abs(lambda) > 1)
        lambda = sign(lambda);
    if (lambda == 9999.9){
        vec periods;
        if (input.s > 1)
            periods = input.s / regspace(1, floor(input.s / 2));
        else {
            periods.resize(1);
            periods(0) = 1.0;
        }
        lambda = testBoxCox(y, periods);
    }
    input.lambda = lambda;
    input.y = BoxCox(input.y, lambda);

    if (armaIdent || model[0] == '?' || model[1] == '?' || model[2] == '?' || model[model.length() - 1] == '?')
        input.arma.fill(0);
//    if (sum(arma > 0))
//        armaIdent = true;
    input.armaIdent = armaIdent;
    input.criterion = criterion;
    // Interpolation
    if ((float)input.missing.n_elem / (float)input.y.n_elem > 0.4){
        printf("%s", "ERROR: Too many missing values!!!\n");
        errorExit = true;
    }
    input.errorExit = errorExit;
    setModel(input, model, s);
    ETSclass m(input);
    if (m.inputModel.errorExit)
        return m;
    if (m.inputModel.missing.n_elem > 0)
        m.interpolate();
    return m;
}
// Main function
void ETS(vec y, mat u, string model, int s, int h,
             bool verbose, string criterion, bool identAll,
             rowvec alphaL, rowvec betaL, rowvec gammaL, rowvec phiL,
             string parConstraints, bool forIntervals, bool bootstrap,
             int nSimul, vec arma, bool armaIdent, vec p0, double lambda){
    // y:       otuput data (one time series)
    // u:       input data (excluding constant)
    // model:   string with three or four letters with model for error, trend and seasonal
    // s:       seasonal period
    // h:       forecasting horizon (if inputs it is recalculated as the length differences
    //          between u and y
    // verbose: shows estimation intermediate results
    // criterion: information criterion to use in identification
    // identALL: Whether to estimate all models
    // alphaL:  limits for alpha parameter
    // betaL:   limits for beta parameter
    // gammaL:  limits for gamma parameter
    // phiL:    limits for alphipha parameter
    // parConstraints: Constraints in parameters: none, standard, admissible
    // forIntervals: forecast variance calculation
    // nSimul:  number of simulations for bootstrap forecast simulation
    // arma:    ARMA(p, q) orders
    // armaIdent: identification of ARMA models on/off
    // lambda: Box-Cox transformation constant

    ETSmodel input;
    ETSclass m(input);
    m = preProcess(y, u, model, s, h, verbose, criterion, identAll, alphaL, betaL, gammaL, phiL,
                   parConstraints, forIntervals, bootstrap, nSimul, arma, armaIdent, p0, lambda);
    if (m.inputModel.errorExit)
        return;
    // Estimating
    if (m.inputModel.error == "?" || m.inputModel.trend == "?" || m.inputModel.seasonal == "?" || m.inputModel.armaIdent)
        m.ident(verbose);
    else {
        m.estim(verbose);
    }
    //m.inputModel.p.t().print("p 1498");
    m.validate();
    m.forecast();
    m.components();
    //cout << m.inputModel.compNames << endl;
    m.simulate(24, m.inputModel.xn);

    postProcess(m.inputModel);
    //m.inputModel.comp.print("comp 1458");
    // Put back missing values /////////////////////////////////////////////
    //m.inputModel.y.rows(m.inputModel.missing).fill(datum::nan);



    //m.inputModel.yFor.print("line 1269");
    //m.inputModel.yForV.print("line 1269");

    //m.inputModel.ySimul.save("/Users/diego.pedregal/Google Drive/C++/ETS/additional/simul.dat", raw_ascii);
    //m.inputModel.comp.print("Components:");
    //m.inputModel.comp.save("/Users/diego.pedregal/Google Drive/C++/ETS/additional/comp.dat", raw_ascii);
    //m.inputModel.yFor.t().print("yFor 712");

    /*
        m.inputModel.yFor.t().print("yFOR 568");
        rowvec conf = m.inputModel.yFor.t() + 2 * sqrt(m.inputModel.yForV.t());
     ;   m.inputModel.yForV.t().print("yforV");
        conf.print("conf(95%) 570");
        */
}
// Check limits for parameters supplied by user
void checkLimits(rowvec& alphaL, rowvec& betaL, rowvec& gammaL, rowvec& phiL, string parConstraints, bool& errorExit){
    // Check whether limits are inside boundaries
    if (alphaL(1) < 0 || alphaL(1) > 1)
        alphaL(1) = 1;
    if (alphaL(0) < 0 || alphaL(0) > 1)
        alphaL(0) = 0;
    if (betaL(1) < 0 || betaL(1) > 1)
        betaL(1) = 1;
    if (betaL(0) < 0 || betaL(0) > 1)
        betaL(0) = 0;
    if (gammaL(1) < 0 || gammaL(1) > 1)
        gammaL(1) = 1;
    if (gammaL(0) < 0 || gammaL(0) > 1)
        gammaL(0) = 0;
    if (phiL(1) < 0 || phiL(1) > 1)
        phiL(1) = 0.98;
    if (phiL(0) < 0 || phiL(0) > 1)
        phiL(0) = 0.8;
    if (parConstraints[0] == 's'){
        // Set 0 < beta < alpha
        betaL(1) = alphaL(1);
        // set 0 < gamma < 1 - alpha
        gammaL(1) = 1 - alphaL(0);
    }
    // Check insconsistencies
    if (alphaL(0) + 0.02 >= alphaL(1)){
        printf("%s", "ERROR: Wrong limits for alpha parameter!!\n");
        errorExit = true;
    }
    if (betaL(0) + 0.02 >= betaL(1)){
        printf("%s", "ERROR: Wrong limits for beta parameter!!\n");
        errorExit = true;
    }
    if (gammaL(0) + 0.02 >= gammaL(1)){
        printf("%s", "ERROR: Wrong limits for gamma parameter!!\n");
        errorExit = true;
    }
    if (phiL(0) + 0.02 >= phiL(1)){
        printf("%s", "ERROR: Wrong limits for phi parameter!!\n");
        errorExit = true;
    }
}
// System matrices for given p
void etsMatrices(ETSmodel* m, vec p){
    int pari = 0;
    mat limits = m->limits;
    vec aux = p.rows(0, limits.n_rows - 1), aux0;
    bool recalculate = false;
    aux0 = aux;
    if (m->scores)
        trans(aux, limits);
    // Selecting appropriate limits for STANDARD parameter constraints
    if (m->scores && m->parConstraints[0] == 's'){
        if (m->nPar(0) > 1){  // there is beta < alpha
            if (aux(1) >= aux(0)){
                recalculate = true;
                limits(1, 1) = aux(0);
                if (limits(1, 1) < limits(1, 0)){
                    if (m->verbose)
                        printf("WARNING: Inadmissible value for beta parameter!!\n");
                    aux(1) = datum::nan;
                }
            }
        }
        if (m->s > 1){        // there is gamma < 1 - alpha
            if (aux(m->nPar(0)) >= 1 - aux(0)){
                recalculate = true;
                limits(m->nPar(0), 1) = 1 - aux(0);
                if (limits(m->nPar(0), 1) < limits(m->nPar(0), 0)){
                    if (m->verbose)
                        printf("WARNING: Inadmissible value for gamma parameter!!\n");
                    aux(m->nPar(0)) = datum::nan;
                }
            }
        }
        if (recalculate){
            aux = aux0;
            trans(aux, limits);
            m->limits = limits;
        }
    } else if (m->scores && m->parConstraints[0] == 'a'){
        // Admissible contraints. Still to do (equation 10.5 in page 156)
        double alpha = 0.5, beta = 0.01, gamma = 0.01, phi = 0.97;
        vec poly(13);
        poly.fill(alpha + beta - alpha * phi);
        //poly(0) = 1;
        poly(0) = alpha + beta - phi;
        poly(11) += gamma - 1;
        poly(12) = phi * (1 - alpha - gamma);


        poly.fill(-0.3);
        poly(0) = 1.25;
        poly(11) = 4.0;
        poly(12) = -1.02;

        poly.t().print("poly 551");
        polyETS(poly, 1);
        poly.t().print("stationary poly");
        invPolyETS(poly, 1);
        poly.t().print("poly back");


    }
    p.rows(0, m->limits.n_rows - 1) = aux;

    //p.t().print("p 1408");
    //limits.print("limits");


    // Allocating trend parameters
    int nPar = m->nPar(0);
    if (m->model.length() > 3){
        nPar--;
    }
    if (m->nPar(0) > 0){
        m->g.rows(pari, pari + nPar - 1) = p.rows(pari, pari + nPar - 1);
        pari += nPar;
    }
    // Ad or Md trend
    if (m->nPar(0) > 2){
        m->phi = p(2);
        pari++;
        if (m->modelType < 2){
            m->F(0, 1) = p(2);
            m->F(1, 1) = p(2);
            m->w(1) = p(2);
        }
    }
    // Seasonal parameters
    if (m->nPar(1) > 0){
        m->g(nPar) = p(pari);
        pari++;
    }
    // Initial states
    int ns = m->F.n_cols; //ns(0) + m->ns(1);
    //m->p0.rows(pari, pari + ns - 1) = p.rows(pari, pari + ns - 1);
    m->x0 = p.rows(pari, pari + ns - 1);
    pari += ns;
    // Exogenous parameters
    if (m->u.n_rows > 0){
        m->d = p.rows(pari, pari + m->u.n_rows - 1).t();
        pari += m->u.n_rows;
    }
    // ARMA parameters
    if (m->modelType == 0 && sum(m->arma) > 0){
        vec aux, arPoly(m->ns(2), fill::zeros), maPoly(m->ns(2), fill::zeros);
        maPoly(0) = 1.0;
        if (m->arma(0) > 0){  // AR model
            aux = p.rows(pari, pari + m->arma(0) - 1);
            if (m->scores)
                polyStationary(aux);
            arPoly.rows(0, m->arma(0) - 1) = aux;
        }
        if (m->arma(1) > 0){    // MA model
            aux = p.rows(pari + m->arma(0), pari + m->arma(0) + m->arma(1) - 1);
            if (m->scores)
                polyStationary(aux);
            maPoly.rows(1, m->arma(1)) = aux;
        }
        mat F(m->ns(2), m->ns(2), fill::zeros);
        if (m->ns(2) > 1)
            F.diag(1).ones();
        F.col(0) = -arPoly;
        int ind1 = m->ns(0) + m->ns(1), ind2 = m->F.n_cols - 1;
        m->F.submat(ind1, ind1, ind2, ind2) = F;
        rowvec rowF = m->F.submat(ind1, ind1, ind1, ind2);
        m->F.submat(0, ind1, 0, ind2) = rowF * m->g(0);
        if (m->trend != "N")
            m->F.submat(1, ind1, 1, ind2) = rowF * m->g(1);
        if (m->seasonal != "N")
            m->F.submat(m->ns(0), ind1, m->ns(0), ind2) = rowF * m->g(m->ns(0));
        // g matrix
        m->g.rows(ind1, ind2) = maPoly;
        // w matrix
        m->w.cols(ind1, ind2) = rowF;
    }


    //m->w.print("w 1469");
}
// Loglik computation
double llikETS(vec& p, void* opt_data){
    // Converting void* to ETSmodel*
    ETSmodel* m = (ETSmodel*)opt_data;
    // Positioning alpha, beta, gamma and phi
    etsMatrices(m, p);
    int n = m->y.n_elem;
    vec x = m->x0, a(1);
    a(0) = datum::nan;
    int ns = m->x0.n_elem - 1;
    double obj = 0.0;
    rowvec fitu(m->y.n_elem + m->h);
    bool nu = (m->u.n_rows > 0);
    if (nu)
        fitu = m->d * m->u;
    else
        fitu.fill(0.0);
    if (m->modelType == 0){
        vec e(1);
        // Linear model
        for (int t = 0; t < n; t++){
            if (isfinite(m->y(t)))
                e = m->y.row(t) - m->w * x - fitu.col(t);
            else
                e(0) = 0.0;
            x = m->F * x + m->g * e;
            obj += e(0) * e(0);
        }
        obj = log(obj);
        if (!isnan(obj))
            m->loge2 = obj;
    } else {
        vec g = m->g;
        double e = 0.0, fit, b, s, logr = 0.0, phi = 1.0;
        if (m->model.length() > 3){
            phi = m->phi;
        }
        if (m->error == "A"){
            // Additive error
            if (m->model == "AMN" || m->model == "AMdN") {
                AMN(m->y, fitu, n, x, g, phi, e, a, obj, logr);
                /*
                for (int t = 0; t < n; t++){
                    b = pow(x(1), phi);
                    fit = x(0) * b;
                    if (isfinite(m->y(t)))
                        e = m->y(t) - fit;
                    else
                        e = 0.0;
                    x(1) = b + g(1) * e / x(0);
                    x(0) = fit + g(0) * e;
                    obj += e * e;
                }
                */
            } else if (m->model == "ANM"){
                ANM(m->y, fitu, n, ns, x, g, e, a, obj, logr);
                /*
                for (int t = 0; t < n; t++){
                    s = x(ns);
                    fit = x(0);
                    if (isfinite(m->y(t)))
                        e = m->y(t) - fit * s;
                    else
                        e = 0.0;
                    x(0) += g(0) * e / s;
                    x.rows(2, ns) = x.rows(1, ns - 1);
                    x(1) = s + g(1) * e / fit;
                    obj += e * e;
                }
                */
            } else if (m->model == "AMA" || m->model == "AMdA"){
                AMA(m->y, fitu, n, ns, x, g, phi, e, a, obj, logr);
                /*
                for (int t = 0; t < n; t++){
                    b = pow(x(1), phi);
                    s = x(ns);
                    fit = x(0) * b;
                    if (isfinite(m->y(t)))
                        e = m->y(t) - fit - s;
                    else
                        e = 0.0;
                    x(1) = b + g(1) * e / x(0);
                    x(0) = fit + g(0) * e;
                    x.rows(3, ns) = x.rows(2, ns - 1);
                    x(2) = s + g(2) * e;
                    obj += e * e;
                }
                */
            } else if (m->model == "AAM" || m->model == "AAdM"){
                AAM(m->y, fitu, n, ns, x, g, phi, e, a, obj, logr);
                /*
                for (int t = 0; t < n; t++){
                    s = x(ns);
                    fit = x(0) + phi * x(1);
                    if (isfinite(m->y(t)))
                        e = m->y(t) - fit * s;
                    else
                        e = 0.0;
                    x(1) += g(1) * e / x(0);
                    x(0) = fit + g(0) * e / s;
                    x.rows(3, ns) = x.rows(2, ns - 1);
                    x(2) = s + g(2) * e / fit;
                    obj += e * e;
                }
                */
            } else if (m->model == "AMM" || m->model == "AMdM"){
                AMM(m->y, fitu, n, ns, x, g, phi, e, a, obj, logr);
                /*
                for (int t = 0; t < n; t++){
                    s = x(ns);
                    b = pow(x(1), phi);
                    fit = x(0) * b;
                    if (isfinite(m->y(t)))
                        e = m->y(t) - fit * s;
                    else
                        e = 0.0;
                    x(1) = b + g(1) * e / (s * x(0));
                    x(0) = fit + g(0) * e / s;
                    x.rows(3, ns) = x.rows(2, ns - 1);
                    x(2) = s + g(2) * e / fit;
                    obj += e * e;
                }
                */
            }
            obj = log(obj);
            if (!isnan(obj)){
                m->loge2 = obj;
                m->logr = 0;
            }
        } else {
            // Multiplicative error
            if (m->model == "MNN"){
                for (int t = 0; t < n; t++){
                    fit = x(0);
                    if (isfinite(m->y(t)))
                        e = (m->y(t) - fitu(t)) / fit - 1;
                    else
                        e = 0.0;
                    x(0) = x(0) * (1 + g(0) * e);
                    obj += e * e;
                    logr += log(abs(fit));
                }
            } else if (m->model == "MAN" || m->model == "MAdN"){
                for (int t = 0; t < n; t++){
                    fit = x(0) + phi * x(1);
                    if (isfinite(m->y(t)))
                        e = (m->y(t) - fitu(t)) / fit - 1;
                    else
                        e = 0.0;
                    x(1) = phi * x(1) + g(1) * fit * e;
                    x(0) = fit * (1 + g(0) * e);
                    obj += e * e;
                    logr += log(abs(fit));
                }
            } else if (m->model == "MMN" || m->model == "MMdN"){
                MMN(m->y, fitu, n, x, g, phi, e, a, obj, logr);
                /*
                for (int t = 0; t < n; t++){
                    b = pow(x(1), phi);
                    fit = x(0) * b;
                    if (isfinite(m->y(t)))
                        e = m->y(t) / fit - 1;
                    else
                        e = 0.0;
                    x(1) = b *(1 + g(1) * e);
                    x(0) = fit * (1 + g(0) * e);
                    obj += e * e;
                    logr *= log(fit);
                }
                */
            } else if (m->model == "MNA"){
                for (int t = 0; t < n; t++){
                    s = x(ns);
                    fit = x(0) + s;
                    if (isfinite(m->y(t)))
                        e = (m->y(t) - fitu(t)) / fit - 1;
                    else
                        e = 0.0;
                    x(0) = x(0) + g(0) * fit * e;
                    x.rows(2, ns) = x.rows(1, ns - 1);
                    x(1) = s + g(1) * fit * e;
                    obj += e * e;
                    logr += log(abs(fit));
                }
            } else if (m->model == "MAA" || m->model == "MAdA"){
                for (int t = 0; t < n; t++){
                    b = phi * x(1);
                    s = x(ns);
                    fit = x(0) + b + s;
                    if (isfinite(m->y(t)))
                        e = (m->y(t) - fitu(t)) / fit - 1;
                    else
                        e = 0.0;
                    x(1) = b + g(1) * fit * e;
                    x(0) += b + g(0) * fit * e;
                    x.rows(3, ns) = x.rows(2, ns - 1);
                    x(2) = s + g(2) * fit * e;
                    obj += e * e;
                    logr += log(abs(fit));
                }
            } else if (m->model == "MNM"){
                for (int t = 0; t < n; t++){
                    s = x(ns);
                    fit = x(0) * s;
                    if (isfinite(m->y(t)))
                        e = (m->y(t) - fitu(t)) / fit - 1;
                    else
                        e = 0.0;
                    x(0) = x(0) * (1 + g(0) * e);
                    x.rows(2, ns) = x.rows(1, ns - 1);
                    x(1) = s * (1 + g(1) * e);
                    obj += e * e;
                    logr += log(abs(fit));
                }
            } else if (m->model == "MAM" || m->model == "MAdM"){
                for (int t = 0; t < n; t++){
                    b = phi * x(1);
                    s = x(ns);
                    fit = x(0) + b;
                    if (isfinite(m->y(t)))
                        e = (m->y(t) - fitu(t)) / (fit * s) - 1;
                    else
                        e = 0.0;
                    //g(0) = m->g(0) * fit;
                    //g(1) = m->g(1) * fit;
                    //g(2) = m->g(2) * s;
                    //fit *= s; // * (1 + a(i)) + fitu(i);
                    //x = m->F * x + g * e;
                    x(1) = b + g(1) * fit * e;
                    x(0) = fit * (1 + g(0) * e);
                    x.rows(3, ns) = x.rows(2, ns - 1);
                    x(2) = s * (1 + g(2) * e);
                    obj += e * e;
                    logr += log(abs(fit * s));
                }
            } else if (m->model == "MMM" || m->model == "MMdM"){
                MMM(m->y, fitu, n, ns, x, g, phi, e, a, obj, logr);
                /*
                for (int t = 0; t < n; t++){
                    b = pow(x(1), phi);
                    s = x(ns);
                    fit = x(0) * b * s;
                    if (isfinite(m->y(t)))
                        e = m->y(t) / fit - 1;
                    else
                        e = 0.0;
                    x(1) = b * (1 + g(1) * e);
                    x(0) = x(0) * b * (1 + g(0) * e);
                    x.rows(3, ns) = x.rows(2, ns - 1);
                    x(2) = s * (1 + g(2) * e);
                    obj += e * e;
                    logr *= log(fit);
                }
                */
            } else if (m->model == "MMA" || m->model == "MMdA"){
                MMA(m->y, fitu, n, ns, x, g, phi, e, a, obj, logr);
                /*
                for (int t = 0; t < n; t++){
                    b = pow(x(1), phi);
                    s = x(ns);
                    fit = x(0) * b + s;
                    if (isfinite(m->y(t)))
                        e = m->y(t) / fit - 1;
                    else
                        e = 0.0;
                    x(1) = b + g(1) * fit * e / x(0);
                    x(0) = x(0) * b + g(0) * fit * e;
                    x.rows(3, ns) = x.rows(2, ns - 1);
                    x(2) = s + g(2) * fit * e;
                    obj += e * e;
                    logr *= log(fit);
                }
                */
            }
            //m->loge2 = log(obj);
            //m->logr = log(logr);
            double lobj = log(obj);
            obj = lobj + 2 * logr / n;
            if (!isnan(obj)){
                m->loge2 = lobj;
                m->logr = logr;
            }
        }
    }
    if (!x.has_nan())
        m->xn = x;
    return obj;
}
// Gradient of Loglik
vec gradETS(vec& p, void* opt_data, double& obj, int& nFuns){
    // Converting void* to ETSmodel*
    ETSmodel* m = (ETSmodel*)opt_data;
    // Positioning alpha, beta, gamma and phi
    etsMatrices(m, p);
    obj = 0.0;
    int nPar = sum(m->nPar);
    nFuns = 0;
    vec grad(nPar, fill::zeros);
    if (m->exact){
        // Exact gradient
        rowvec gradr = grad.t();
        int n = m->y.n_elem,
            ns = sum(m->ns),
            nParG = m->nPar(0) + m->nPar(1);
        vec e(1),
            x = m->x0;
        rowvec fitu(m->y.n_elem + m->h),
               dw(ns, fill::zeros),
               de(nPar, fill::zeros),
               de_x(ns, fill::zeros);
        bool nu = (m->u.n_rows > 0),
             Ad = false;
        if (m->model.length() > 3)
            Ad = true;
        if (nu)
            fitu = m->d * m->u;
        else
            fitu.fill(0.0);
        mat dg(ns, nParG, fill::zeros),
            dx(m->x0.n_elem, nParG, fill::zeros),
            dF(ns, ns, fill::zeros);
        de_x = -m->w;
        vec aux1, aux2;
        aux1 = p.rows(0, nParG - 1);
        aux2 = dtrans(aux1, m->limits);
        if (Ad){   // Ad trend
            dw(1) = aux2(2);
            dF(0, 1) = aux2(2);
            dF(1, 1) = aux2(2);
            if (m->seasonal != "N")
                dg(2, 3) = aux2(3);
            aux2(2) = 0.0;
        } else {
            if (m->seasonal != "N")
                dg(m->ns(0), m->ns(0)) = aux2(m->ns(0));
        }
        dg(0, 0) = aux2(0);
        if (m->trend != "N")
            dg(1, 1) = aux2(1);
        //dg.rows(0, nParG - 1) = diagmat(aux2);
        mat F_gw = m->F - m->g * m->w;
        for (int t = 0; t < n; t++){
            if (isfinite(m->y(t))){
                e = m->y.row(t) - m->w * x - fitu.col(t);
                de.cols(0, nParG - 1) = -m->w * dx;
                if(Ad)      //Ad trend
                    de.col(2) -= dw * x;
                dx = m->F * dx + dg * e(0) + m->g * de.cols(0, nParG - 1);
                if (Ad)    // Ad trend
                    dx.col(2) += dF * x;
                de.cols(nPar - ns - m->u.n_rows, nPar - m->u.n_rows - 1) = de_x;
                if (nu)
                    de.cols(nPar - m->u.n_rows, nPar - 1) = -m->u.col(t).t();
                gradr +=  e * de;
                de_x *= F_gw;
                x = m->F * x + m->g * e;
                obj += e(0) * e(0);
            } else {
                x = m->F * x;
            }
        }
        grad = 2 * gradr.t() / obj;
        obj = log(obj);
        if (!isnan(obj))
            m->loge2 = obj;
        nFuns++;
        m->xn = x;
    } else {
        // Numerical gradient
        vec p0, F1 = p;
        obj = llikETS(p, opt_data);
        for (int i = 0; i < nPar; i++){
            p0 = p;
            p0(i) += 1e-8;
            F1(i) = llikETS(p0, opt_data);
        }
        grad = (F1 - obj) / 1e-8;
        nFuns += nPar + 1;
    }
    m->grad = grad;
    return grad;
}
// Parameter variances
void covPar(void* optData, vector<string>& parNames, mat& tablePar){
    ETSmodel* m = (ETSmodel*)optData;
    uword nPar = m->p.n_elem;
    vec grad(nPar), p = m->p, p0 = p, inc(nPar);
    mat Hess(nPar, nPar);
    vec grad0 = m->grad;
    char name[9];
    inc.fill(1e-5);
    double llikValue2 = 0, llikValue0;
    // True parameters
    uword pos = 1;
    p(0) = m->alpha;
    snprintf(name, 9, "Alpha");
    parNames.push_back(name);
    if (m->trend != "N"){
        p(1) = m->beta;
        snprintf(name, 9, "Beta");
        parNames.push_back(name);
        pos++;
    }
    if (m->model.length() > 3){
        p(2) = m->phi;
        snprintf(name, 9, "Phi");
        parNames.push_back(name);
        pos++;
    }
    if (m->s > 1){
        p(pos) = m->gamma;
        snprintf(name, 9, "Gamma");
        parNames.push_back(name);
        pos += m->ns(0) + m->ns(1) + m->ns(2) + 1;
    }
    int count;
    vec aux;
    if (m->u.n_rows > 0){
        string auxbeta;
        pos += m->u.n_rows;
        for (count = 0; count < (int)m->u.n_rows; count++){
                auxbeta = to_string(count + 1);
                snprintf(name, 9, "Beta(%s)", auxbeta.c_str());
                parNames.push_back(name);
        }
        // pos += m->u.n_rows;
        // for (count = 0; count < (int)m->u.n_rows; count++){
        //     snprintf(name, 9, "Beta(%d)", count + 1);
        //     parNames.push_back(name);
        // }
    }
    if (m->ar.n_elem > 0){
        aux = p.rows(pos, pos + m->ar.n_elem - 1);
        polyStationary(aux);
        p.rows(pos, pos + m->ar.n_elem - 1) = aux;
        pos += m->ar.n_elem;
        string auxar;
        for (count = 0; count < (int)m->ar.n_elem; count++){
                auxar = to_string(count + 1);
                snprintf(name, 9, "AR(%s)", auxar.c_str());
                parNames.push_back(name);
        }
        // pos += m->ar.n_elem;
        // for (count = 0; count < (int)m->ar.n_elem; count++){
        //     snprintf(name, 9, "AR(%d)", count + 1);
        //     parNames.push_back(name);
        // }
    }
    if (m->ma.n_elem > 0){
        aux = p.rows(pos, pos + m->ma.n_elem - 1);
        polyStationary(aux);
        p.rows(pos, pos + m->ma.n_elem - 1) = aux;
        // for (count = 0; count < (int)m->ma.n_elem; count++){
        //   snprintf(name, 9, "MA(%d)", count + 1);
        //   parNames.push_back(name);
        // }
        string auxma;
        for (count = 0; count < (int)m->ma.n_elem; count++){
            auxma = to_string(count + 1);
            snprintf(name, 9, "MA(%s)", auxma.c_str());
            parNames.push_back(name);
        }
    }
    p0 = p;
    // rest
    m->scores = false;
    llikValue0 = llikETS(p0, m);
    Hess.fill(0);
    for (uword i = 0; i < nPar; i++){
        p0 = p;
        p0.row(i) += inc(i);
        grad0(i) = llikETS(p0, m);
    }
    for (uword i = 0; i < nPar; i++){
        for (uword j = i; j < nPar; j++){
            p0 = p;
            p0.row(i) += inc(i);
            p0.row(j) += inc(j);
            llikValue2 = llikETS(p0, m);
            Hess(i, j) = as_scalar((llikValue2 - grad0.row(i) - grad0.row(j) + llikValue0)
                                   / inc(i) / inc(j));
        }
    }
    if (nPar > 1){
        Hess = Hess + trimatu(Hess, 1).t();
    }
    // Parameters avoiding initial states
    uvec ind = regspace<uvec>(0, m->nPar(0) + m->nPar(1) - 1);
    if (m->u.n_rows > 0 || sum(m->arma) > 0){
        ind = join_vert(ind, regspace<uvec>(sum(m->nPar.rows(0, 2)), p.n_elem - 1));
    }
    Hess *= m->y.n_elem * 0.5;
    Hess = pinv(Hess);
    vec diagH = Hess.diag();
    diagH = diagH(ind);
    m->scores = true;
    tablePar.set_size(ind.n_elem, 4);
    tablePar.col(0) = p(ind);
    tablePar.col(1) = sqrt(abs(diagH));
    tablePar.col(2) = tablePar.col(0) / tablePar.col(1);
    tablePar.col(3) = abs(m->grad(ind));
    if (max(tablePar.col(3)) < 1e-8)
        m->estimOk = "Q-Newton: Gradient convergence.\n";
}
// Minimizer for ETS (gradient function computes objective value and gradient in one run)
int quasiNewtonETS(std::function <double (vec& x, void* inputs)> objFun,
                   std::function <vec (vec& x, void* inputs, double& obj, int& nFuns)> gradFun,
                   vec& xNew, void* inputs, double& objNew, vec& gradNew, mat& iHess, bool verbose,
                   int nparConst){
  int nx = xNew.n_elem, flag = 0, nOverallFuns, nFuns = 0, nIter = 0;
  double objOld, alpha_i;
  vec gradOld(nx), xOld = xNew, d(nx);
  // crit: Criteria to stop quasi-Newton:
        // Gradient
        // Difference in obj function
        // Difference in parameter values
        // Maximum number of iterations
        // Maximum number of function evaluations
  vec crit(5); crit(0) = 1e-8; crit(1) = 1e-12; crit(2) = 1e-6; crit(3) = 1000; crit(4) = 20000;

  uvec indMax, indMin, ind;

  iHess.eye(nx, nx);
  //objNew = objFun(xNew, inputs);
  gradNew = gradFun(xNew, inputs, objNew, nFuns);
  nOverallFuns = nFuns;
  // Head of table
  if (verbose){
      printf(" Iter FunEval  Objective       Step\n");
      printf("%5.0i %5.0i %12.5f %12.5f\n", nIter, nOverallFuns, objNew, 1.0);
  }
  // Main loop
  do{
    nIter++;
    // Search direction
    d = -iHess * gradNew;
    // Line Search
    xOld = xNew; gradOld = gradNew; objOld = objNew;
    alpha_i = 0.5;
    lineSearch(objFun, alpha_i, xNew, objNew, gradNew, d, nIter, nFuns, inputs);
    nOverallFuns += nFuns;
    // Constraining parameters to boundaries
    indMax = find(xNew.rows(0, nparConst - 1) >= 100);
    indMin = find(xNew.rows(0, nparConst - 1) <= -100);
    xNew.elem(indMax).fill(100);
    xNew.elem(indMin).fill(-100);
    ind = join_vert(indMax, indMin);
    gradNew = gradFun(xNew, inputs, objNew, nFuns);
    gradNew.elem(ind).fill(0);
    nOverallFuns += nFuns;
    // Verbose
    if (verbose){
        printf("%5.0i %5.0i %12.5f %12.5f\n", nIter, nOverallFuns, objNew, alpha_i);
    }
    // Stop Criteria
    flag = stopCriteria(crit, mean(abs(gradNew)), objOld - objNew,
                        mean(abs(xOld - xNew) / abs(xOld)), nIter, nOverallFuns);
    if (flag > 5){
        objNew = objOld;
        gradNew = gradOld;
        xNew = xOld;
    }
    // Inverse Hessian BFGS update
    if (!flag){
      bfgs(iHess, gradNew - gradOld, xNew - xOld, nx, nIter);
    }
  } while (!flag);
  return flag;
}
// Initialize parameters
void initPar(ETSmodel& m){
    m.nPar.zeros(5);
    vec p0(4);
    bool noUser = true;
    if (m.p0user.n_elem > 0){
        noUser = false;
        p0 = m.p0user;
    }
    if (true){    // Old initial conditions
            if (noUser)
                p0(0) = m.prop * m.alphaL(0) + (1 - m.prop) * m.alphaL(1);
            //if (m.model == "AMA" || m.model == "AMM" || m.model == "MMA")
            //    p0(0) = 0.1 * m.alphaL(0) + 0.9 * m.alphaL(1);
            //else
            //    p0(0) = sum(m.alphaL) / 2;
            // Limits for parameters
            m.limits.ones(4, 2);
            m.limits.row(0) = m.alphaL;
            // Calculating number of parameters
            m.nPar(0) = 2;
            if (m.trend[0] == 'N'){
                m.nPar(0) = 1;
            } else {
                m.limits.row(1) = m.betaL;
                if (noUser){
                    //p0(1) = m.betaL(0) + 0.01;
                    //p0(1) = 0.9 * m.betaL(0) + 0.1 * m.betaL(1);
                    p0(1) = 0.6 * m.betaL(0) + 0.4 * p0(0);
                }
            }
            if (m.model.length() > 3){
                m.nPar(0) = 3;
                m.limits.row(2) = m.phiL;
                if (noUser){
                    p0(2) = m.phiL(1) - 0.01;
                    //p0(2) = 0.1 * m.phiL(0) + 0.9 * m.phiL(1);
                }
            }
            if (m.seasonal != "N"){
                m.nPar(1) = 1;
                m.limits.row(m.nPar(0)) = m.gammaL;
                if (noUser){
                    //p0(m.nPar(0)) = m.gammaL(0) + 0.01;
                    //p0(m.nPar(0)) = 1 - p0(0) + 0.001;
                    p0(m.nPar(0)) = 0.1 * m.gammaL(0) + 0.9 * (1 - p0(0));
                } else {
                    p0(m.nPar(0)) = m.p0user(3);
                    //p0(m.nPar(0)) = 0.9 * m.gammaL(0) + 0.1 * m.gammaL(1);
                }
            }
    } else {   // Hyndman initial conditions
        double aux = max(m.s, 1);
        p0(0) = m.alphaL(0) + m.prop * (m.alphaL(1) - m.alphaL(0)) / aux;
        // Limits for parameters
        m.limits.ones(4, 2);
        m.limits.row(0) = m.alphaL;
        // Calculating number of parameters
        m.nPar(0) = 2;
        if (m.trend[0] == 'N'){
            m.nPar(0) = 1;
        } else {
            m.betaL(1) = min(m.betaL(1), p0(0));
            m.limits.row(1) = m.betaL;
            p0(1) = m.betaL(0) + 0.1 * (m.betaL(1) - m.betaL(0));
        }
        if (m.model.length() > 3){
            m.nPar(0) = 3;
            m.limits.row(2) = m.phiL;
            p0(2) = m.phiL(0) + 0.99 * (m.phiL(1) - m.phiL(0));
        }
        if (m.seasonal != "N"){
            m.nPar(1) = 1;
            m.gammaL(1) = min(m.gammaL(1), 1 - p0(0));
            m.limits.row(m.nPar(0)) = m.gammaL;
            p0(m.nPar(0)) = m.gammaL(0) + 0.05 * (m.gammaL(1) - m.gammaL(0));
        }
    }
    m.limits = m.limits.rows(0, sum(m.nPar) - 1);
    p0 = p0.rows(0, sum(m.nPar.rows(0, 1)) - 1);
    // Setting parameter values
    // untrans(p0, m.limits);
    // Adding initial states
    m.nPar(2) = sum(m.ns);
    vec p0s(sum(m.ns), fill::zeros);
    if (m.y.n_elem < 10 || m.y.n_elem < 4 * (uword)m.s || m.y.rows(0, 9).has_nan()){     // My initial conditions
       if (m.ns(0) > 0){
            // Trend
            p0s.row(0) = m.y(0);
        }
        if (m.ns(0) > 1 && m.s < 2){
            if (m.trend[0] == 'A')
                p0s.row(1) = m.y(1) - m.y(0);
            else
                p0s.row(1) = 1 + (m.y(1) - m.y(0)) / m.y(0);
        }
        if (m.s > 1){
            vec aux1, aux2, aux3;
            float m1, m2;
            aux1 = m.y.rows(0, m.s - 1);
            aux2 = m.y.rows(m.s, 2 * m.s - 1);
            m1 = nanMean(aux1);
            m2 = nanMean(aux2);
            if (m.ns(0) > 1){   // slope initial
                if (m.trend[0] == 'A')
                    p0s.row(1) = m2 - m1;
                else
                    p0s.row(1) = 1 + (m2 - m1) / m1;
            }
            // Seasonal
            if (m.seasonal == "M"){
                aux3 = (aux1 / m1 + aux2 / m2) / 2;
                p0s.rows(m.ns(0), m.ns(0) + m.ns(1) - 1) = reverse(aux3);
                p0s.elem(find_nonfinite(p0s)).zeros();
            } else {
                aux3 = (aux1 - m1 + aux2 - m2) / 2;
                p0s.rows(m.ns(0), m.ns(0) + m.ns(1) - 1) = reverse(aux3);
                p0s.elem(find_nonfinite(p0s)).ones();
            }
        }
    } else {    // Hyndman's initial conditions
        vec seasAdj = m.y.rows(0, 9);
        if (m.s> 1){
            // Seasonal (normalization needed)
            vec movPar(m.s * 2), aux1;
            int n = min((int)m.y.n_elem / m.s * m.s, 20 * m.s) - 1 - m.s;
            movPar.fill(1 / ((double)m.s * 2));
            aux1 = conv(m.y, movPar);   // Moving average
            mat aux;
            if (m.seasonal == "A")  // seasonal
                aux = m.y.rows(m.s, n) - aux1.rows(m.s * 2, n + m.s);
            else
                aux = m.y.rows(m.s, n) / aux1.rows(m.s * 2, n + m.s);
            aux.reshape(m.s, aux.n_elem / m.s);
            aux = aux.t();
            rowvec factors = reverse(nanMean(aux));   // seasonal factors
            p0s.rows(m.ns(0), m.ns(0) + m.ns(1) - 1) = factors.t();
            p0s.elem(find_nonfinite(p0s)).ones();
            mat seas = repmat(factors.t(), ceil(10 / (double)m.s), 1);
            if (m.seasonal == "A")
                seasAdj -= seas.rows(0, 9);
            else
                seasAdj /= seas.rows(0, 9);
        }
        // Trend
        mat X = join_horiz(ones(10, 1), regspace(1, 10));
        vec beta, stdBeta, eOut;
        double BIC, AIC, AICc;
        regress(seasAdj, X, beta, stdBeta, eOut, BIC, AIC, AICc);
        p0s.row(0) = beta(0);
        // Slope
        if (m.ns(0) > 1){
             if (m.trend[0] == 'A')
                 p0s.row(1) = beta(1);
             else
                 p0s.row(1) = 1 + beta(1) / beta(0);
        }
        uvec ind = find_finite(m.y);
        // case annual or sign of initial trend different to data
        if (m.s < 2 || sign(m.y(ind(0))) != sign(beta(0))){
            p0s(0) = m.y(ind(0));
            if (m.ns(0) > 1){
                if (m.trend[0] == 'A')
                    p0s.row(1) = m.y(ind(1)) - m.y(ind(0)) / (ind(1) - ind(0));
                else
                    p0s.row(1) = 1 + m.y(ind(1)) / m.y(ind(0));
            }
        }
    }
    m.p0 = join_cols(p0, p0s);
    // inputs
    if (m.u.n_cols > 0){
        m.nPar(3) = m.u.n_rows;
        vec Beta, stdBeta, eOut;
        double BIC, AIC, AICc;
        regress(m.y - nanMean(m.y), m.u.cols(0, m.y.n_rows - 1).t(), Beta, stdBeta, eOut, BIC, AIC, AICc);
        m.p0 = join_cols(m.p0, Beta);
    }
    // ARMA
    if (sum(m.arma)){
        m.nPar(4) = sum(m.arma);
        vec aux(m.nPar(4) + m.nPar(2), fill::zeros);
        m.p0 = join_cols(m.p0, aux);
    }
    ////// ADDED CORRECTION for initial alpha, beta, gamma running llik function
    uvec indPar(m.nPar(0) + m.nPar(1));
    uword npar = 1;
    if (m.trend[0] != 'N'){
        indPar(npar) = 1;
        npar++;
    }
    if (m.trend.length() > 1){
        indPar(npar) = 2;
        npar++;
    }
    if (m.seasonal[0] != 'N'){
        indPar(npar) = 3;
    }
    if (true){  // One sort of initial conditions for alpha, beta, phi, gamma
        /*
        if (noUser && m.seasonal == "N" && sum(m.arma) > 0){
            vec p0aux = {0.5, 0.1, 0.97, 0.1};
            p0 = p0aux.rows(indPar);
            untrans(p0, m.limits);
        } else
        */
        if (noUser){
            vec pauxi = m.p0, pauxCol, pBest = m.p0;
            rowvec alphaL, betaL, gammaL, phiL;
            // Pesos de intervalos de rango de cada parámetro
            mat pesos = {{0.5,  0.5,  0.7,  0.7, 0.3, 0.3},
                         {0.2,  0.2,  0.14, 0.14, 0.3, 0.3},
                         {0.95, 0.95, 0.95, 0.95, 0.95, 0.95},
                         {0.2, 0.8,   0.33,  1.33, 0.14, 0.56}};
    //        mat pesos = {{0.5,  0.5,  0.5, 0.5, 0.7,  0.7, 0.3, 0.3},
    //                     {0.2,  0.2,  0.6, 0.6, 0.14, 0.14, 0.3, 0.3},
    //                     {0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95},
    //                     {0.2, 0.8,   0.2,  0.8, 0.33,  1.33, 0.14, 0.56}};
            alphaL = (1 - pesos.row(0)) * m.alphaL(0) + pesos.row(0) * m.alphaL(1);
            betaL = (1 - pesos.row(1)) * m.betaL(0) + pesos.row(1) % alphaL;
            phiL = (1 - pesos.row(2)) * m.phiL(0) + pesos.row(2) * m.phiL(1);
            gammaL = (1 - pesos.row(3)) * m.gammaL(0) + pesos.row(3) % (1 - alphaL);
            mat p0all = join_vert(alphaL, betaL, phiL, gammaL);
            //p0all.print("p0all 2434");
            p0all = p0all.rows(indPar);
            double llikVal, optVal = 1e6;
            for (uword i = 0; i < p0all.n_rows; i++){
                pauxCol = p0all.col(i);
                untrans(pauxCol, m.limits);
                //pauxi = join_cols(pauxCol, p0s);
                pauxi.rows(0, pauxCol.n_rows - 1) = pauxCol;
                //pauxi.t().print("pauxi 2488");
                llikVal = llikETS(pauxi, &m);
                if (llikVal < optVal){
                    optVal = llikVal;
                    pBest = pauxi;
                }
            }
            m.p0 = pBest;
        }
    }
/*        if (noUser)
            m.p0 = p0;
        else
            m.p0 = join_cols(m.p0user.rows(indPar), p0s);
    } else {
        m.p0 = join_cols(p0, p0s);
    }
*/
}
// Initial ETS matrices for linear model
void initEtsMatrices(ETSmodel& m){
    int ns, nsTS;
    nsTS = m.ns(0) + m.ns(1);
    ns = nsTS + (m.modelType == 0) * m.ns(2);
    if (m.modelType == 2){
        m.w.zeros(1);
        m.F.zeros(ns, 1);
        m.g.zeros(ns);
    } else {
        m.w.zeros(ns);
        m.F.zeros(ns, ns);
        m.g.zeros(ns);
        m.w(0) = 1.0;
        m.F(0, 0) = 1.0;
        if (m.trend == "A" || m.trend == "Ad"){
            m.w(1) = 1.0;
            m.F(0, 1) = 1.0;
            m.F(1, 1) = 1.0;
        }
        if (m.modelType < 2 && m.seasonal != "N"){
            m.w(ns - 1) = 1.0;
            m.F(span(m.ns(0) + 1, nsTS - 1), span(m.ns(0), nsTS - 1)) = eye(m.s - 1, m.s);
            m.F(m.ns(0), nsTS - 1) = 1.0;
        }
    }
    // inputs
    if (m.u.n_rows > 0)
        m.d.set_size(m.u.n_rows);
}
// Check and divide model into components
void modelDivide(string& model, string& error, string& trend, string& seasonal, bool& errorExit){
    error = model[0];
    if (model.length() == 3){
        trend = model[1];
        seasonal = model[2];
    } else {
        trend = model.substr(1, 2);
        seasonal = model[3];
    }
    if ((error != "M" && error != "A" && error != "?") ||
        (trend != "N" && trend[0] != 'M' && trend[0] != 'A' && trend != "?") ||
        (seasonal != "N" && seasonal != "A" && seasonal != "M" && seasonal != "?")){
        printf("ERROR: Invalid model name!!\n");
        errorExit = true;
    }
}
// Parameter transformation
void trans(vec& p, mat limits){
  p = exp(p);
  p = limits.col(0) + (limits.col(1) - limits.col(0)) % p / (1 + p);
}
// Parameter un transformation
void untrans(vec& p, mat limits){
    p = log((p - limits.col(0)) / (limits.col(1) - p));
}
// Parameter transformation derivatives
vec dtrans(vec& p, mat limits){
    p = exp(p);
    return p % (limits.col(1) - limits.col(0)) / ((1 + p) % (1 + p));
}
// Returns stationary polynomial from an arbitrary one
void polyETS(vec& PAR, double bound){
  // (1 + PAR(1) * B + PAR(2) *B^2 + ...) y(t) = a(t)
  mat limits(PAR.n_elem, 2);
  limits.col(0).fill(-bound);
  limits.col(1).fill(bound);
  trans(PAR, limits);
  pacfToAr(PAR);
  PAR = -PAR;
}
// Inverse of polyStationary
void invPolyETS(vec& PAR, double bound){
  // (1 + PAR(1) * B + PAR(2) *B^2 + ...) y(t) = a(t)
  mat limits(PAR.n_elem, 2);
  limits.col(0).fill(-bound);
  limits.col(1).fill(bound);
  PAR = -PAR;
  arToPacf(PAR);
  untrans(PAR, limits);
}
// Find all combinations of models to estimate in identification process
void findModels(string error, string trend, string seasonal, bool identAll, vector<string>& allModels){
    int nTrendModels, nSeasonalModels, nErrorModels;
    vector <string> trendModels, seasonalModels, errModels;
    // Possible trends
    chopString(trend, "/", trendModels);
    nTrendModels = trendModels.size();
    // Possible seasonals
    chopString(seasonal, "/", seasonalModels);
    nSeasonalModels = seasonalModels.size();
    // Possible error models
    chopString(error, "/", errModels);
    nErrorModels = errModels.size();
    // All possible models
    // int count = 0;
    string cModel;
    for (int k = 0; k < nErrorModels; k++){
        for (int i = 0; i < nTrendModels; i++){
            for (int j = 0; j < nSeasonalModels; j++){
                cModel = errModels[k];
                cModel.append(trendModels[i]).append(seasonalModels[j]);
                if (identAll || (trendModels[i][0] != 'M' && cModel != "ANM" && cModel != "AAM" && cModel != "AAdM")){
                    allModels.push_back(cModel);
                }
            }
        }
    }
}
// pretty model name
string prettyModel(string model){
    string pModel;
    if (model.length() == 3)
        pModel = " (" + model.substr(0, 1) + "," + pModel + model.substr(1, 1) + "," + model.substr(2, 1) + ")";
    else
        pModel = "(" + model.substr(0, 1) + "," + pModel + model.substr(1, 2) + "," + model.substr(3, 1) + ")";
    return pModel;
}
