/*
 Basic Structural models
 Additional inputs for R, MATLAB, python:
    TVP
    trendOptions
    seasonalOptions
    irregularOptions
*/
#include <iostream>
#include <math.h>
#include <string.h>
#include <armadillo>
using namespace arma;
using namespace std;
#include "DJPTtools.h"
#include "optim.h"
#include "stats.h"
#include "boxcox.h"
#include "SSpace.h"
#include "ARIMASSmodel.h"
#include "ARMAmodel.h"

struct BSMmodel{
    // INPUTS:
    string model = "llt/none/equal/arma(0,0)",  // model to fit
        criterion = "aic";                      // identification criterion
    bool stepwise,              // stepwise identification or brute one
         tTest,                 // unit roots test or not for identification
         arma;                  // check arma models for irregular component
    vec periods,                // vector of periods for harmonics
        TVP;                    // vector of zeros and ones to allocate position of TVP parameters
    bool MSOE = false,          // MSOE model or BSM
         PTSnames = false;      // PTS model names or UC
    // OUTPUTS:
    string trend,               // type of trend
        cycle,                  // type of cycle
        seasonal,               // type of seasonal component
        irregular,              // type of irregular
        cycle0,                 // type of cycle without numbers
        compNames = "",         // components names
        trendOptions = "none/rw/llt/dt",          // trend options to select amongst (none/rw/llt/td/dt/srw)
        seasonalOptions = "none/equal/different", // seasonal options to select amongst (none/equal/different/linear)
        irregularOptions = "none/arma(0,0)";      // irregular components (none/arma(p,q))
    int ar = 0, ma = 0;         // AR and MA orders of irregular component
    double seas,                // seasonal period
           lambda = 1.0;              // Box-Cox transformation parameter
    vec rhos,                   // vector indicating whether period is cyclical or seasonal
        ns,                     // number of states in components (trend, cycle, seasonal, irregular)
        nPar,                   // number of parameters in components (trend, cycle, seasonal, irregular)
        p0Return,               // initial parameters user understandable
        typePar,                // type of parameter (0: variance;
                                //         -1: damped of trend;
                                //          1: cycle rhos;
                                //          2: cycle periods;
                                //          3: ARMA;
                                //          4: inputs)
        eps,                    // observation perturbation
        beta0ARMA,              // initial estimates of ARMA model (without variance)
        constPar;               // constrained parameters (0: not constrained;
                //                         1: concentrated-out;
                //                         2: variance constrained to 0;
                //                         3: alpha constrained to 0 or 1)
    uvec harmonics;             // vector with the indices of harmonics selected
    mat comp,                   // estimated components
        compV,                  // variance of components
        typeOutliers,           // Matrix with type of outliers and sample of each outlier
        cycleLimits;            // limits for period of cycle estimation
    bool pureARMA = false,      // Pure ARMA model flag
         Drift = false;         // trend with drift
    vector<string> parNames;    // Parameter names
};
/**************************
 * Model CLASS BSM
 ***************************/
class BSMclass : public SSmodel{
private:
    BSMmodel inputs;
    // Set model
    void setModel(string, vec, vec, bool);
    // Count states and parameters of BSM model
    void countStates(vec, string, string, string, string);
    // Fix matrices in standard BSM models (all except variances)
    void initMatricesBsm(vec, vec, string, string, string, string);
    // Initializing parameters of BSM model
    void initParBsm();
    // Optimization routine
    int quasiNewtonBSM(std::function <double (vec&, void*)>, 
                       std::function <vec (vec&, void*, double, int&)>,
                       vec&, void*, double&, vec&, mat&, bool);
    // Estimation of a family of UC models
    void estimUCs(vector<string>, uvec, double&, bool, double, int);
public:
    // Constructors
    BSMclass();
    BSMclass(SSinputs, BSMmodel);
    // Convert to MSOE
    void bsm2msoe();
    // Parameter names
    void parLabels();
    // Interpolation of initial NaN values
    void interpolate(int);
    //Estimation
    void estim(bool);
    void estim(vec, bool);
    // Identification
    void ident(string, bool);
    // Outlier detection
    void estimOutlier(vec, bool);
    // Check whether re-estimation is necessary
    void checkModel(uvec);
    // Components
    void components();
    // Covariance of parameters (inverse of hessian)
    mat parCov(vec&);
    // Finding true parameter values out of transformed parameters
    vec parameterValues(vec);
    // Validation of BSM models
    void validate(bool);
    // Disturbance smoother (to recover just trend and epsilons)
    void disturb();
    // Get data
    BSMmodel getInputs(){
        parLabels();
        return inputs;
    }
    // Set data
    void setInputs(BSMmodel inputs){
        this->inputs = inputs;
    }
    //Print inputs on screen
    // void showInputs();
};
/***************************************************
 * Auxiliar function declarations
 ****************************************************/
// Main aux function
void BSMaux(vec, mat, string, int, double, bool, string, vec, bool, bool,
         vec, bool, string, string, string, vec, double, SSinputs&, BSMmodel&);
// Main function
void BSM(vec, mat, string, int, double, bool, string, vec, bool, bool,
         vec, bool, string, string, string, vec, double);
// Convert UC model to PTS
string UC2PTS(string);
// States names for filtering and smoothing
string stateNames(BSMmodel);
// Pre-processing
bool preProcess(vec, mat&, string&, int&, double&, string&,
                vec, vec, int&, string, string, string, vec&, double&);
// Variance matrices in standard BSM
void bsmMatrices(vec, SSmatrix*, void*);
// Variance matrices in standard BSM for true parameters
void bsmMatricesTrue(vec, SSmatrix*, void*);
// Extract trend seasonal and irregular of model in a string
void splitModel(string, string&, string&, string&, string&);
// SS form of trend models
void trend2ss(int, mat*, mat*);
// SS form of seasonal models
void bsm2ss(int, int, vec, vec, mat*, mat*);
// Remove elements of vector in n adjacent points
uvec selectOutliers(vec&, int, float);
// Create dummy variable for outliers 0: AO, 1: LS, 2: SC
void dummy(uword, uword, rowvec&);
// combining UC models
void findUCmodels(string, string, string, string, vector<string>&);
// Corrects model, cycle string, periods and rhos for modelling cycles
void modelCorrect(string&, string&, string&, vec&, vec&);
// Calculate limits for cycle periods for estimation
void calculateLimits(int, vec, vec, mat&, double);
// Find first observation of n non-nan contiguous values
int findFirst(vec, int);
// Show SS model
void showSS(SSmatrix);
// Show BSMmodel
void showBSM(BSMmodel);

/****************************************************
 // BSM implementations for univariate UC models
 ****************************************************/
// Constructors
BSMclass::BSMclass(){
}
BSMclass::BSMclass(SSinputs data, BSMmodel inputs) : SSmodel(data){
    SSmodel::inputs = data;
    this->inputs = inputs;
    this->inputs.rhos = ones(size(inputs.periods));
    this->inputs.cycleLimits.resize(1, 1);
    this->inputs.cycleLimits(0, 0) = datum::nan;
    vec reserve = inputs.constPar;
    setModel(inputs.model, inputs.periods, inputs.rhos, true);
    if (!reserve.has_nan() && reserve.n_elem > 0)
        this->inputs.constPar = reserve;
    this->inputs.harmonics = regspace<uvec>(0, inputs.periods.n_elem - 1);
}
// main aux function
void BSMaux(vec y, mat u, string model, int h, double outlier, bool tTest, string criterion,
         vec periods, bool verbose, bool stepwise, vec p0, bool arma, string trendOptions,
         string seasonalOptions, string irregularOptions, vec TVP, double lambda,
         SSinputs& inputsSS, BSMmodel& inputsBSM){
    // Correcting dimensions of u (k x n)
    size_t k = u.n_rows;
    size_t n = u.n_cols;
    if (k > n){
        u = u.t();
    }
    if (k == 1 && n == 2){
        u.resize(0);
    }
    int iniObs = max(periods);
    // Pre-processing
    bool errorExit = preProcess(y, u, model, h, outlier, criterion, periods, p0, iniObs,
                                trendOptions, seasonalOptions, irregularOptions, TVP, lambda);
    if (sum(TVP) > 0)
        outlier = 0;
    if (errorExit)
        printf("%d", errorExit);
    // End of pre-processing
    inputsSS.y = y.rows(iniObs, y.n_elem - 1);
    mat uIni;
    if (iniObs > 0 && u.n_rows > 0){ // && command == "estimate"){
        inputsSS.u = u.cols(iniObs, u.n_cols - 1);
        uIni = u.cols(0, iniObs - 1);
    } else {
        inputsSS.u= u;
    }
    inputsBSM.model = model;
    inputsBSM.periods = periods;
    inputsBSM.rhos = ones(periods.n_elem);
    inputsSS.h = h;
    inputsBSM.tTest = tTest;
    inputsBSM.criterion = criterion;
    inputsBSM.trendOptions = trendOptions;
    inputsBSM.seasonalOptions = seasonalOptions;
    inputsBSM.irregularOptions = irregularOptions;
    inputsBSM.TVP = TVP;
    inputsSS.p0 = p0;
    inputsSS.outlier = outlier;
    inputsSS.verbose = verbose;
    inputsBSM.seas = max(periods);
    inputsBSM.stepwise = stepwise;
    inputsBSM.harmonics = regspace<uvec>(0, periods.n_elem - 1);
    inputsBSM.arma = arma;

    // BoxCox transformation
    if (lambda == 9999.9)
        lambda = testBoxCox(y, periods);
    inputsBSM.lambda = lambda;
    inputsSS.y = BoxCox(inputsSS.y, inputsBSM.lambda);
}
// Main function
void BSM(vec y, mat u, string model, int h, double outlier, bool tTest, string criterion,
         vec periods, bool verbose, bool stepwise, vec p0, bool arma, string trendOptions,
         string seasonalOptions, string irregularOptions, vec TVP, double lambda){
    // y:         otuput data (one time series)
    // u:         input data (excluding constant)
    // model:     string with three or four letters with model for error, trend and seasonal
    // h:         forecasting horizon (if inputs it is recalculated as the length differences
    //            between u and y
    // outlier:   standard deviation for outlier detection
    // tTest:     use of unit root tests for stationarity detection true/false
    // criterion: information criterion to use in identification (aic / bic / aicc)
    // periods:   seasonal period
    // verbose:   shows estimation intermediate results
    // stepwise:  use of stepwise faster identification procedure true / false
    // p0:        initial values for parameters to start search
    // arma:      testing for arma innovations true / false
    // trendOptions:     set of trends to choose among (none/rw/llt/irw/dt/td)
    // seasonalOptions:  set of seasonal models to choose among (none/equal/different)
    // irregularOptions: set of irregular models to choose amongst (none/arma(0,0)/arma(0,1)/...)
    // TVP:       vector of zeros and ones where to allocate TVP parameters for inputs
    // lambda:    BoxCox transformation value (9999.9 for estimation)


    /*
    // Correcting dimensions of u (k x n)
    size_t k = u.n_rows;
    size_t n = u.n_cols;
    if (k > n){
        u = u.t();
    }
    if (k == 1 && n == 2){
        u.resize(0);
    }
    int iniObs = max(periods);
    // Setting inputs
    SSinputs inputsSS;
    BSMmodel inputsBSM;
    // Pre-processing
    bool errorExit = preProcess(y, u, model, h, outlier, criterion, periods, p0, iniObs,
                                trendOptions, seasonalOptions, irregularOptions, TVP, lambda);
    if (sum(TVP) > 0)
        outlier = 0;
    if (errorExit)
        printf("%d", errorExit);
    // End of pre-processing
    inputsSS.y = y.rows(iniObs, y.n_elem - 1);
    mat uIni;
    if (iniObs > 0 && u.n_rows > 0){ // && command == "estimate"){
        inputsSS.u = u.cols(iniObs, u.n_cols - 1);
        uIni = u.cols(0, iniObs - 1);
    } else {
        inputsSS.u= u;
    }
    inputsBSM.model = model;
    inputsBSM.periods = periods;
    inputsBSM.rhos = ones(periods.n_elem);
    inputsSS.h = h;
    inputsBSM.tTest = tTest;
    inputsBSM.criterion = criterion;
    inputsBSM.trendOptions = trendOptions;
    inputsBSM.seasonalOptions = seasonalOptions;
    inputsBSM.irregularOptions = irregularOptions;
    inputsBSM.TVP = TVP;
    inputsBSM.MSOE = false;
    inputsSS.p0 = p0;
    inputsSS.outlier = outlier;
    inputsSS.verbose = verbose;
    inputsBSM.seas = max(periods);
    inputsBSM.stepwise = stepwise;
    inputsBSM.harmonics = regspace<uvec>(0, periods.n_elem - 1);
    inputsBSM.arma = arma;

    // BoxCox transformation
    if (lambda == 9999.9)
        lambda = testBoxCox(y, periods);
    inputsBSM.lambda = lambda;
    inputsSS.y = BoxCox(inputsSS.y, inputsBSM.lambda);
    */

    SSinputs inputsSS;
    BSMmodel inputsBSM;
    BSMaux(y, u, model, h, outlier, tTest, criterion, periods, verbose, stepwise, p0, arma, trendOptions,
        seasonalOptions, irregularOptions, TVP, lambda, inputsSS, inputsBSM);
    // Building model
    BSMclass m = BSMclass(inputsSS, inputsBSM);

    // Estimating
    BSMmodel inputs = m.getInputs();
    if (inputs.irregular == "?" || inputs.trend == "?" || inputs.seasonal == "?" || inputs.arma)
        m.ident("both", verbose);
    else {
        m.estim(verbose);
    }
    // next lines to correct cycles
    BSMmodel aux2 = m.getInputs();
    if (aux2.cycle[0] != 'n' && aux2.cycle != "?"){
        string model1 = aux2.model, cycle = aux2.cycle, cycle0 = aux2.cycle0;
        vec periods = aux2.periods, rhos = aux2.rhos;
        modelCorrect(model, cycle, inputs.cycle0, periods, rhos);
        aux2.model= model1, aux2.cycle= cycle, aux2.cycle0= cycle0;
        aux2.periods = periods, aux2.rhos = rhos;
        m.setInputs(aux2);
    }
    printf("empiezo validated:\n");
    m.validate(true);
    printf("empiezo forecast:\n");
    m.forecast();
    printf("empiezo components:\n");
    m.components();
    //BSMmodel aux = m.getInputs();
    //aux.comp.print("components 358");
    printf("empiezo filter:\n");
    m.filter();
    printf("empiezo distturb:\n");
    m.disturb();
}
// Pre-processing
bool preProcess(vec y, mat& u, string& model, int& h, double& outlier, 
                string& criterion, vec periods, vec p0, int& iniObs,
                string trendOptions, string seasonalOptions, string irregularOptions,
                vec& TVP, double& lambda){
    // Looking for first observation for estimation
    // if (y.has_nan()){
    //     uword nskip = 2 * periods.n_elem + 3;
    //     uvec indFinite, poly(3, fill::ones), aux3;
    //     indFinite = find_finite(y);
    //     aux3 = conv(diff(indFinite), poly);
    //     //iniObs = indFinite(min(find(aux3.rows(2, aux3.n_elem - 1) == 3))) + iMin;
    //     iniObs = indFinite(min(find(aux3.rows(nskip - 1, aux3.n_elem - 1) == nskip))) + iMin;
    // } else {
    //     iniObs = 0;
    // }
    // vec p(1);
    // p.fill(datum::nan);
    // Correcting inputs by user
    iniObs = findFirst(y, iniObs);
    lower(criterion);
    deblank(model);
    lower(model);
    lower(trendOptions);
    lower(irregularOptions);
    lower(seasonalOptions);
    // checking trendOptions
    vector<string> aux;
    bool alright = false; uword ind = 0;
    chopString(trendOptions, "/", aux);
    do{
        alright = false;
        if (aux[ind] == "none")
            alright = true;
        else if (aux[ind] == "rw")
            alright = true;
        else if (aux[ind] == "irw")
            alright = true;
        else if (aux[ind] == "llt")
            alright = true;
        else if (aux[ind] == "dt")
            alright = true;
        else if (aux[ind] == "srw")   // Hyndman damped
            alright = true;
        else if (aux[ind] == "td")
            alright = true;
        ind++;
    } while(alright == true && ind < aux.size());
    if (alright == false){
        printf("%s", "ERROR: trendOptions wrongly set (none/rw/irw/llt/dt/td/hd)!!!\n");
        return true;
    }
    // checking seasonalOptions
    ind = 0;
    if (max(periods) == 1)
        seasonalOptions = "none";
    chopString(seasonalOptions, "/", aux);
    do{
        alright = false;
        if (aux[ind] == "none")
            alright = true;
        else if (aux[ind] == "equal")
            alright = true;
        else if (aux[ind] == "different")
            alright = true;
        else if (aux[ind] == "linear")
            alright = true;
        ind++;
    } while(alright == true && ind < aux.size());
    if (alright == false){
        printf("%s", "ERROR: seasonalOptions wrongly set (none/equal/different/linear)!!!\n");
        return true;
    }
    // checking irregularOptions
    alright = false; ind = 0;
    chopString(irregularOptions, "/", aux);
    do{
        alright = false;
        if (aux[ind] == "none")
            alright = true;
        else if (aux[ind].find("arma(") != std::string::npos)
            alright = true;
        ind++;
    } while(alright == true && ind < aux.size());
    if (alright == false){
        printf("%s", "ERROR: irregularOptions wrongly set (none/arma(p,q))!!!\n");
        return true;
    }
    // Checking TVP
    if (any(TVP > 1)){
        printf("%s", "ERROR: TVP vector should contain zeros and/or ones only!!!\n");
        return true;
    }
    if (u.n_rows > 0){
        if (TVP.n_elem == 0)
            TVP.zeros(u.n_rows);
        else if (TVP.n_elem < u.n_rows){
            vec aux2(u.n_rows - TVP.n_elem, fill::zeros);
            TVP = join_vert(TVP, aux2);
        } else if (TVP.n_elem > u.n_rows){
            TVP = TVP.rows(0, u.n_rows - 1);
        }
    } else {
        TVP = {};
    }
    // Checking lambda
    if (lambda != 9999.9 && abs(lambda) > 1)
        lambda = sign(lambda);
    // Checking forecasting horizon
    int initPer;
    initPer = max(periods);
    h = abs(h);
    if (h == 9999){
        if (initPer == 1)
            initPer = 5;
        h = 2 * initPer;
    }
    if (outlier == -9999)
        outlier = 0.0;
    // Correcting h in case there are inputs
    if (u.n_rows > 0){
        h = u.n_cols - y.n_elem;
        if (h < 0){
            printf("%s", "ERROR: Inputs should be at least as long as the ouptut!!!\n");
            return true;
        }
    }
    // Checking periods
    if (min(periods) < 1){
        printf("%s", "ERROR: All periods should be higher or equal than zero!!!\n");
        return true;
    }
    // Removing nans at beginning or end
    /*
     if (!y.row(0).is_finite() || !y.row(y.n_elem - 1).is_finite()){
     uvec ind = find_finite(y);
     int minInd = min(ind), maxInd = max(ind);
     y = y.rows(minInd, maxInd);
     if (u.n_rows > 0){
     u = u.cols(minInd, maxInd);
     }
     }
     */
    // Checking periods
    //    if (is.ts(y) && is.na(periods) && frequency(y) > 1){
    //        periods = frequency(y) / (1 : floor(frequency(y) / 2))
    //    } else if (is.ts(y) && is.na(periods)){
    //        periods = 1
    //    } else if (!is.ts(y) && is.na(periods)){
    //        stop("Input \"periods\" should be supplied!!")
    //    }
    if (model.find("/") == string::npos){
        printf("%s", "ERROR: Incorrect number of components (trend/seasonal/irregular)!!!\n");
        return true;
    }
    vector<string> comps;
    chopString(model, "/", comps);
    // Number of components
    if (comps.size() < 3 || comps.size() > 4){
        printf("%s", "ERROR: Incorrect number of components (trend/seasonal/irregular)!!!\n");
        return true;
    }
    // Adding cycle in case of T/S/I model specification
    if (comps.size() == 3){
        model = comps[0] + "/none/" + comps[1] + "/" + comps[2];
        chopString(model, "/", comps);
    }
    // Setting seasonal to none for annual data
    if (max(periods) == 1){
        if (comps.size() == 3){
            comps[1] = "none";
            model = comps[0] + "/" + comps[1] + "/" + comps[2];
        }
        if (comps.size() == 4){
            comps[2] = "none";
            model = comps[0] + "/" + comps[1] + "/" + comps[2] + "/" + comps[3];
        }
    }
    // Checking model
    if (comps[1][0] != 'n' && comps[2][0] == 'l'){
        printf("%s", "ERROR: Cycle can estimated only with trigonometric seasonal components!!!\n");
        return true;
    }
    if (comps[0][0] == 'n' && comps[1][0] == 'n' && comps[2][0] == 'n' && comps[3][0] == 'n'){
        printf("%s", "ERROR: No correct model specified!!!\n");
        return true;
    }
    if (model.find("?") == string::npos && p0(0) != -9999.9){
        p0.set_size(1);
        p0(0) = -9999.9;
    }
    if (model.find("?") == string::npos && isnan(p0(0))){
        p0.set_size(1);
        p0(0) = datum::nan;
    }
    if (model.find("arma") != string::npos && model.find("(") == string::npos){
        model = model + "(0,0)";
        chopString(model, "/", comps);
    }
    if (model.find("arma") != string::npos && model[model.length() - 1] != ')'){
        model = model + ")";
        chopString(model, "/", comps);
    }
    // Checking components
    string opt = "?nirlds";
    if (opt.find(comps[0][0]) == string::npos){
        printf("%s", "ERROR: Incorrect TREND model (? / none / irw / td / llt / dt / srw)!!!\n");
        return true;
    }
    opt = "?n+-0123456789";
    if (opt.find(comps[1][0]) == string::npos){
        printf("%s", "ERROR: Incorrect CYCLE model (? / none / +-integer)!!!\n");
        return true;
    }
    opt = "?nedl";
    if (opt.find(comps[2][0]) == string::npos){
        printf("%s", "ERROR: Incorrect SEASONAL model (? / none / equal / different / linear)!!!\n");
        return true;
    }
    if (comps[3][0] == 'a' && model.find("arma") == string::npos){
        printf("%s", "ERROR: Incorrect IRREGULAR model (? / none / arma(p, q))!!!\n");
        return true;
    }
    opt = "?na";
    if (opt.find(comps[3][0]) == string::npos){
        printf("%s", "ERROR: Incorrect IRREGULAR model (? / none / arma(p, q))!!!\n");
        return true;
    }
    // Set rhos
    //vec rhos(periods.n_elem, fill::ones);
    // Checking cycle
    if (comps[1][0] == '?'){
        initPer *= -4;
        if (initPer == -4)
            initPer = -8;
        comps[1] = to_string(initPer) + '?';
    } else if (comps[1][0] != '+' && comps[1][0] != '-' && comps[1][0] != 'n')
        comps[1] = '+' + comps[1];
    model = comps[0] + "/" + comps[1] + "/" + comps[2] + "/" + comps[3];
    // Checking criterion
    if (criterion != "aic" && criterion != "bic" && criterion != "aicc"){
        criterion = "aic";
    }
    // Building structures
    // SSinputs inputsSS;
    // BSMmodel inputsBSM;
    // inputsSS.y = y;
    // inputsSS.u = u;   // Inputs data (mat)
    // inputsBSM.model = model;
    // inputsSS.h = h;
    // inputsBSM.tTest = tTest;
    // inputsBSM.criterion = criterion;
    // inputsSS.outlier = outlier;
    // vec aux(1); aux(0) = outlier;
    // if (aux.has_nan()){
    //     outlier = 0;
    // }
    return false;
    // inputsBSM.periods = periods;
    // inputsSS.verbose = verbose;
    // inputsBSM.stepwise = stepwise;
    // inputsSS.p0 = p0;
    // inputsBSM.arma = arma;
    //inputsBSM.errorExit = errorExit;
    //inputsBSM.rhos = rhos;
    
    //vec VOID(1); VOID.fill(datum::nan);
    // inputsSS.p = VOID;
    // inputsSS.grad = VOID;
    // inputsSS.criteria = VOID;
    // inputsSS.d_t = 0;
    // inputsSS.innVariance = VOID(0);
    // inputsSS.objFunValue = VOID(0);
    // inputsSS.cLlik = true;
    // inputsSS.Iter = 0;
    // inputsSS.nonStationaryTerms = VOID(0);
    // inputsBSM.ns = VOID;
    // inputsBSM.nPar = VOID;
    // if (harmonics.has_nan()){
    //     inputsBSM.harmonics.resize(1);
    //     inputsBSM.harmonics(0) = 0;
    // } else {
    //     inputsBSM.harmonics = conv_to<uvec>::from(harmonics);
    // }
    // inputsBSM.constPar = VOID;
    // inputsBSM.typePar = VOID;
    // inputsBSM.typeOutliers = {-1, -1};
    // inputsBSM.cycleLimits = VOID;
    
    //inputsBSM.seas = max(periods);
    //inputsSS.p = {datum::nan};        // Estimated parameters (vec)
    // inputsSS.v = {datum::nan};        // Estimated innovations (vec)
    // inputsSS.yFit = {datum::nan};     // Fitted values (vec)
    // inputsSS.yFor = {datum::nan};     // Point forecasts (vec)
    // inputsSS.F = {datum::nan};        // Innovations variance (vec)
    // inputsSS.FFor = {datum::nan};     // Variance of forecasts (vec)
    // inputsBSM.comp = {datum::nan};    // Estimated components (mat)
    // inputsBSM.compV = {datum::nan};   // Variance of estimated components (mat)
    // inputsSS.a = {datum::nan};        // Estimated states (mat)
    // inputsSS.P = {datum::nan};        // Estimated variance of states (mat)
    // inputsSS.eta = {datum::nan};      // Estimated transition perturbations (mat)
    // inputsBSM.eps = {datum::nan};     // Estimated observed perturbations (vec)
    // inputsSS.criteria = {datum::nan}; // Likelihood and information criteria at optimum (vec)
    // inputsBSM.cycleLimits = {datum::nan};
    // inputsBSM.rhos = inputsBSM.periods; inputsBSM.rhos.fill(1);
    
    //BSMclass m(inputsSS, inputsBSM);
    //return m;
    //if (errorExit)
    //    return m;
    //   if (y.has_nan())
    //        m.interpolate();
    //return m;
    
    // Building UComp system
    //BSMclass sysBSM(inputsSS, inputsBSM);
}
// // Interpolation
// void BSMclass::interpolate(){
//     BSMmodel sysCopy = inputs;
//     SSinputs ssCopy = SSmodel::inputs;
//     SSmodel::inputs.h = 0;
//     SSmodel::inputs.verbose = false;
//     SSmodel::inputs.h = 0;
//     vec y = SSmodel::inputs.y;
//     mat u = SSmodel::inputs.u;
//     string seasTypes, model;
//     vector<string>models;
//     if (max(inputs.periods) == 1){
//         seasTypes = "none";
//         //model1 = "llt/none/none/arma(0,0)";
//         //model2 = "rw/none/none/arma(0,0)";
//     } else {
//         seasTypes = "equal";
//         //model1 = "llt/none/equal/arma(0,0)";
//         //model2 = "llt/none/equal/arma(0,0)";
//     }
//     //inputs.model = model1;
//     double minCrit;
//     vec y1(inputs.missing.n_elem), y2(inputs.missing.n_elem), aux(SSmodel::inputs.y.n_elem);
//     uvec col(1); col(0) = 1;
//     int iMin = min(inputs.missing), iMax = max(inputs.missing),
//         iniObs = 0, finalObs = SSmodel::inputs.y.n_elem;
//     uvec indFinite, poly(3, fill::ones), aux3;
//     vec fit;
//     // Forward interpolation
//     if (iMin > 4){
//         //setModel(inputs.model, inputs.periods, inputs.rhos, false);
//         ///////// no funciona identificación
//         findUCmodels("llt/rw", "none", seasTypes, "arma(0,0)", models);
//         estimUCs(models, inputs.harmonics, minCrit, false, 1e12, SSmodel::inputs.u.n_rows);
//         model = inputs.model;
//         SSmodel::smooth(false);
//         y1 = SSmodel::inputs.yFit(inputs.missing);
//     } else {
//         // Missing from start
//         indFinite = find_finite(y.rows(iMin, y.n_elem - 1));
//         aux3 = conv(diff(indFinite), poly);
//         iniObs = indFinite(min(find(aux3.rows(2, aux3.n_elem - 1) == 3))) + iMin;
//         SSmodel::inputs.y = y.rows(iniObs, y.n_elem - 1);
//         if (u.n_rows > 0)
//             SSmodel::inputs.u = u.cols(iniObs, y.n_elem - 1);
//         findUCmodels("llt/rw", "none", seasTypes, "arma(0,0)", models);
//         estimUCs(models, inputs.harmonics, minCrit, false, 1e12, SSmodel::inputs.u.n_rows);
//         model = inputs.model;
//         SSmodel::smooth(false);
//         fit = join_vert(zeros(iniObs), inputs.comp.col(1));
//         y1 = fit(inputs.missing);
//         SSmodel::inputs.y = y;
//         SSmodel::inputs.u = u;
//     }
//     bool VERBOSE = SSmodel::inputs.verbose;
//     if (y.n_elem - iMax - 1 > 5){
//         // Backwards interpolation
//         SSmodel::inputs.y = reverse(SSmodel::inputs.y);
//         setModel(model, inputs.periods, inputs.rhos, false);
//         SSmodel::inputs.verbose = false;
//         SSmodel::estim();
//         SSmodel::inputs.verbose = VERBOSE;
//         SSmodel::smooth(false);
//         aux = reverse(SSmodel::inputs.yFit);
//         y2 = aux(inputs.missing);
//     } else {
//         // Missing at the right end
//         indFinite = find_finite(y.rows(0, iMax));
//         aux3 = conv(diff(indFinite), poly);
//         finalObs = indFinite(max(find(aux3.rows(0, aux3.n_elem - 3) == 3)) + 1);
//         SSmodel::inputs.y = reverse(y.rows(0, finalObs));
//         if (u.n_rows > 0)
//             SSmodel::inputs.u = reverse(u.cols(0, finalObs), 1);
//         setModel(model, inputs.periods, inputs.rhos, false);
//         SSmodel::inputs.verbose = false;
//         SSmodel::estim();
//         SSmodel::inputs.verbose = VERBOSE;
//         SSmodel::smooth(false);
//         fit = join_vert(reverse(SSmodel::inputs.yFit), zeros(finalObs));
//         y2 = fit(inputs.missing);
//         SSmodel::inputs.y = y;
//         SSmodel::inputs.u = u;
//     }
//     y(inputs.missing) = (y1 + y2) / 2;
//     if (iMin < 6){
//         // Correcting first chunk
//         uvec missInd = inputs.missing(find(inputs.missing < iniObs));
//         y(missInd) = y2.rows(0, missInd.n_elem - 1);
//     }
//     if (y.n_elem - iMax < 6){
//         // Correcting last chunk
//         uvec missInd = inputs.missing(find(inputs.missing > finalObs));
//         y(missInd) = y1.rows(y1.n_elem - missInd.n_elem, y1.n_elem - 1);
//     }
//     // Restoring initial values and interpolating
//     SSmodel::inputs = ssCopy;
//     inputs = sysCopy;
//     SSmodel::inputs.y = y;
//     setModel(inputs.model, inputs.periods, inputs.rhos, false);
// }

// Estim SSOE system
//void BSMclass::estimSSOE(){
//}
// Set model (part of constructor)
void BSMclass::setModel(string model, vec periods, vec rhos, bool runFromConstructor){
    string trend, cycle, seasonal, irregular;
    vec ns(7), nPar(7), typePar, noVar, constPar;
    mat cycleLimits;
    splitModel(model, trend, cycle, seasonal, irregular);
    // Checking cycle model and correcting from string input
    if (cycle[0] != 'n' && cycle != "?"){
        modelCorrect(model, cycle, inputs.cycle0, periods, rhos);
    }
    this->inputs.trend = trend;
    this->inputs.cycle = cycle;
    this->inputs.seasonal = seasonal;
    this->inputs.irregular = irregular;
    if (cycle[0] != 'n' && inputs.cycleLimits.has_nan()){
        calculateLimits(SSmodel::inputs.y.n_elem, periods, rhos, cycleLimits, inputs.seas);
        this->inputs.cycleLimits = cycleLimits;
    } else if (cycle[0] != 'n') {
        cycleLimits = inputs.cycleLimits;
    }
    // Checking for arma identification
    if (irregular != "?"){
        inputs.arma = 0;
    }
    // Checking for constant input from user and removing in that case
    if (SSmodel::inputs.u.n_rows > 0){
        uvec rowCnt = find(sum(SSmodel::inputs.u - 1, 1) == 0);
        SSmodel::inputs.u.shed_rows(rowCnt);
    }
    // Initializing matrices
    if (trend != "?" && cycle != "?" && seasonal != "?" && irregular != "?"){  // One model
        initMatricesBsm(periods, rhos, trend, cycle, seasonal, irregular);
        this->inputs.model = model;
        if (cycle[0] != 'n'){
            this->inputs.periods = periods;
            this->inputs.rhos = rhos;
        }
        this->SSmodel::inputs.userInputs = &this->inputs;
        // User function to fill the changing matrices
        this->SSmodel::inputs.userModel = bsmMatrices;
        // Initializing parameters of BSM model
        if (!runFromConstructor)
            SSmodel::inputs.p0(0) = -9999.9;
        typePar = this->inputs.typePar;
        // inputs.beta0ARMA.reset();
        initParBsm();
        // Convert to MSOE form
        if (inputs.MSOE)
            bsm2msoe();
    }
    // Making coherent h and size(u)
    if (SSmodel::inputs.u.n_elem > 0){
        SSmodel::inputs.h =  SSmodel::inputs.u.n_cols - SSmodel::inputs.y.n_elem;
    }
}
// Convert bsm system to msoe
void BSMclass::bsm2msoe(){
    bool seas = false;
    if (inputs.seasonal[0] != 'n')
        seas = true;
    uword nsAdd = seas + 1, ns = SSmodel::inputs.system.T.n_rows;
    // Change R
    SSmodel::inputs.system.R = join_vert(SSmodel::inputs.system.R, zeros(nsAdd, SSmodel::inputs.system.R.n_cols));
    // Change T and Z
    mat aux(ns + nsAdd, ns + nsAdd, fill::zeros);
    aux.submat(0, 0, ns - 1, ns - 1) = SSmodel::inputs.system.T;
    SSmodel::inputs.system.T = aux;
    SSmodel::inputs.system.T.row(ns) = SSmodel::inputs.system.T.row(0);
    SSmodel::inputs.system.Z = join_horiz(SSmodel::inputs.system.Z, zeros(1, nsAdd));
    SSmodel::inputs.system.Z(ns) = 1.0;
    if (seas){
        if (inputs.seasonal[0] == 'l'){
            SSmodel::inputs.system.T.row(ns + 1) = SSmodel::inputs.system.T.row(inputs.ns(0));
            SSmodel::inputs.system.Z.cols(0, inputs.ns(0) + inputs.ns(1) + inputs.ns(2) - 1).fill(0.0);
            SSmodel::inputs.system.Z(ns + 1) = 1.0;
        } else {
            SSmodel::inputs.system.Z.cols(0, inputs.ns(0) - 1).fill(0.0);
            SSmodel::inputs.system.T.submat(ns + 1, inputs.ns(0), ns + 1, ns - 1) = SSmodel::inputs.system.Z.cols(inputs.ns(0), ns - 1);
        }
    } else {
        SSmodel::inputs.system.Z.cols(0, inputs.ns(0) - 1).fill(0.0);
    }
//    showSS(SSmodel::inputs.system);
//    cout << "model: " << inputs.model << endl;
//    cout << "here" << endl;
}
// Print inputs on screen
// void BSMclass::showInputs(){
//   cout << "**************************" << endl;
//   cout << "Start of BSM system:" << endl;
//   cout << "Model: " << inputs.model << endl;
//   cout << "criterion: " << inputs.criterion << endl;
//   cout << "trend: " << inputs.trend << endl;
//   cout << "cycle: " << inputs.cycle << endl;
//   cout << "seasonal: " << inputs.seasonal << endl;
//   cout << "irregular: " << inputs.irregular << endl;
//   cout << "cycle0: " << inputs.cycle0 << endl;
//   cout << "ar: " << inputs.ar << endl;
//   cout << "ma: " << inputs.ma << endl;
//   inputs.periods.t().print("periods:");
//   inputs.rhos.t().print("rhos:");
//   inputs.ns.t().print("ns:");
//   inputs.nPar.t().print("nPar:");
//   inputs.typePar.t().print("typePar:");
//   inputs.beta0ARMA.t().print("beta0ARMA:");
//   inputs.constPar.t().print("constPar");
//   inputs.harmonics.t().print("harmonics:");
//   inputs.typeOutliers.t().print("typeOutliers:");
//   inputs.cycleLimits.print("cycleLimits:");
//   cout << "stepwise: " << inputs.stepwise << endl;
//   cout << "tTest: " << inputs.tTest << endl;
//   cout << "arma: " << inputs.arma << endl;
//   inputs.eps.t().print("eps:");
//   cout << "End of BSM system:" << endl;
//   cout << "**************************" << endl;
// }
// Interpolation of initial NaN values
void BSMclass::interpolate(int iniObs){
    BSMmodel sysCopy = inputs;
    SSinputs ssCopy = SSmodel::inputs;
    SSmodel::inputs.h = 0;
    uvec missing = find_nonfinite(SSmodel::inputs.y.rows(0, iniObs));
    SSmodel::inputs.y = reverse(SSmodel::inputs.y);
    if (SSmodel::inputs.u.n_rows > 0)
        SSmodel::inputs.u = reverse(SSmodel::inputs.u, 1);
    int lastObs = findFirst(SSmodel::inputs.y, sum(inputs.ns));
    SSmodel::inputs.y = SSmodel::inputs.y.rows(lastObs, SSmodel::inputs.y.n_elem - 1);
    if (SSmodel::inputs.u.n_rows > 0)
        SSmodel::inputs.u = SSmodel::inputs.u.cols(lastObs, SSmodel::inputs.u.n_cols - 1);
    estim(false);
    SSmodel::smooth(false);
    vec yFit = reverse(SSmodel::inputs.yFit);
    inputs = sysCopy;
    SSmodel::inputs = ssCopy;
    SSmodel::inputs.y(missing) = yFit(missing);
    // BSMmodel sysCopy = inputs;
    // SSinputs ssCopy = SSmodel::inputs;
    // SSmodel::inputs.h = 0;
    // SSmodel::inputs.y = reverse(SSmodel::inputs.y);
    // if (SSmodel::inputs.u.n_rows > 0)
    //     SSmodel::inputs.u = reverse(SSmodel::inputs.u, 1);
    // int lastObs = findFirst(SSmodel::inputs.y, sum(inputs.ns));
    // SSmodel::inputs.y = SSmodel::inputs.y.rows(lastObs, SSmodel::inputs.y.n_elem - 1);
    // estim(false);
    // SSmodel::smooth(false);
    // vec yFit = reverse(SSmodel::inputs.yFit);
    // SSmodel::inputs.y = reverse(SSmodel::inputs.y);
    // uvec missing = find_nonfinite(SSmodel::inputs.y.rows(0, iniObs));
    // inputs = sysCopy;
    // SSmodel::inputs = ssCopy;
    // SSmodel::inputs.y(missing) = yFit(missing + lastObs);
    
}
// Estimation: runs estim(p) or ident()
void BSMclass::estim(bool VERBOSE){
    bool verboseCopy = SSmodel::inputs.verbose;
    SSmodel::inputs.verbose = VERBOSE;
    if (inputs.trend != "?" && inputs.cycle != "?" && inputs.seasonal != "?" && inputs.irregular != "?"){
        // Particular model
        if (SSmodel::inputs.outlier == 0){
            // Without outlier detection
            uvec harmonics = inputs.harmonics;
            estim(SSmodel::inputs.p0, VERBOSE);
            checkModel(harmonics);
        } else {
            // With outlier detection
            estimOutlier(SSmodel::inputs.p0, VERBOSE);
        }
    } else {
        // Some or all the components to identify
        string cycle = inputs.cycle;
        string cycle0 = inputs.cycle0;
        size_t found = cycle.find('?');
        if (found != string::npos && inputs.arma){  // cycle has ?
            BSMmodel old = inputs;
            SSinputs oldSS = SSmodel::inputs;
            // First estimation with cycle
            inputs.cycle = inputs.cycle0;
            ident("head", VERBOSE);
            SSinputs bestSS = SSmodel::inputs;
            BSMmodel bestBSM = inputs;
            inputs = old;
            SSmodel::inputs = oldSS;
            // Second estimation without cycle
            inputs.cycle = "none";
            strReplace("?", "", inputs.cycle0);
            ident("tail", VERBOSE);
            // Now decide which is best
            int crit = 1;
            if (inputs.criterion == "bic"){
                crit = 2;
            } else if (inputs.criterion == "aicc"){
                crit = 3;
            }
            if (SSmodel::inputs.criteria(crit) > bestSS.criteria(crit)){
                SSmodel::inputs = bestSS;
                inputs = bestBSM;
            }
            inputs.cycle = cycle;
            inputs.cycle0 = cycle0;
        } else {
            // Estimation as is
            ident("both", VERBOSE);
        }
    }
    SSmodel::inputs.verbose = verboseCopy;
}
// Check whether re-estimation is necessary
void BSMclass::checkModel(uvec harmonics){
    // Repeat estimation of one model in case of anomalies
    string ok = SSmodel::inputs.estimOk;
    bool add = (inputs.model[0] == 'd');
    bool printed = false;
    // If no convergence and llt or dt trend model, then slope p0 more rigid
    if ((ok[10] == 'M' || ok[10] == 'U' || ok[10] == 'O' || ok[10] == 'N') &&
        (inputs.model[0] == 'l' || inputs.model[0] == 'd')){
        // Next 5 lines in every exception
        if (SSmodel::inputs.verbose){
            printf("    --\n");
            printf("    Estimation problems, trying again...\n");
            printf("    --\n");
            printed = true;
        }
        SSinputs old = SSmodel::inputs;
        setModel(inputs.model, inputs.periods(harmonics), inputs.rhos(harmonics), false);
        bool VERBOSE = old.verbose;
        SSmodel::inputs.verbose = false;
        SSmodel::inputs.p0(1 + add) = -6.2325;
        // Estimation of particular model
        if (SSmodel::inputs.outlier == 0){
            // Without outlier detection
            estim(SSmodel::inputs.p0, VERBOSE);
        } else {
            // With outlier detection
            estimOutlier(SSmodel::inputs.p0, VERBOSE);
        }
        if (!old.criteria.has_nan() &&
            (old.criteria(1) < SSmodel::inputs.criteria(1))){
            SSmodel::inputs = old;
            SSmodel::inputs.verbose = VERBOSE;
        }
    }
    // Repeat estimation of one model in case of anomalies
    ok = SSmodel::inputs.estimOk;
    //add = (inputs.model[0] == 'd');
    // If no convergence and llt or dt trend model, then level p0 more rigid
    if ((ok[10] == 'M' || ok[10] == 'U' || ok[10] == 'O' || ok[10] == 'N') &&
        (inputs.model[0] == 'l' || inputs.model[0] == 'd')){
        // Next 5 lines in every exception
        if (SSmodel::inputs.verbose && !printed){
            printf("    --\n");
            printf("    Estimation problems, trying again...\n");
            printf("    --\n");
            printed = true;
        }
        SSinputs old = SSmodel::inputs;
        setModel(inputs.model, inputs.periods(harmonics), inputs.rhos(harmonics), false);
        bool VERBOSE = old.verbose;
        SSmodel::inputs.verbose = false;
        SSmodel::inputs.p0(0 + add) = -6.2325;
        // Estimation of particular model
        if (SSmodel::inputs.outlier == 0){
            // Without outlier detection
            estim(SSmodel::inputs.p0, VERBOSE);
        } else {
            // With outlier detection
            estimOutlier(SSmodel::inputs.p0, VERBOSE);
        }
        if (!old.criteria.has_nan() &&
            (old.criteria(1) < SSmodel::inputs.criteria(1))){
            SSmodel::inputs = old;
            SSmodel::inputs.verbose = VERBOSE;
        }
    }
    inputs.harmonics = harmonics;
}
void BSMclass::estim(vec p, bool VERBOSE){
    bool verboseCopy = SSmodel::inputs.verbose;
    SSmodel::inputs.verbose = VERBOSE;
    double objFunValue;
    vec grad;
    mat iHess;
    int flag; //, nPar, k;
    SSmodel::inputs.p0 = p;
    wall_clock timer;
    timer.tic();
    if (SSmodel::inputs.augmented){
        SSmodel::inputs.llikFUN = llikAug;
    } else {
        SSmodel::inputs.llikFUN = llik;
    }
    // This next line is a bit dangerous
    SSmodel::inputs.userInputs = &inputs;
    flag = quasiNewtonBSM(SSmodel::inputs.llikFUN, gradLlik, p, &(SSmodel::inputs),
                          objFunValue, grad, iHess, SSmodel::inputs.verbose);
    uvec indNan = find_nonfinite(SSmodel::inputs.y);
    int nNan2pi = SSmodel::inputs.y.n_elem - indNan.n_elem;
    int nTrue;
    if (SSmodel::inputs.augmented){
        nTrue = nNan2pi - SSmodel::inputs.u.n_rows - SSmodel::inputs.system.T.n_rows;
        uvec stat;
        isStationary(SSmodel::inputs.system.T, stat);
        SSmodel::inputs.nonStationaryTerms = SSmodel::inputs.system.T.n_rows - stat.n_elem;
        // Correction for DT trend models
        // if (inputs.model[0] == 'd' && stat.n_elem > 0 && stat[0] == 1){
        //   SSmodel::inputs.nonStationaryTerms++;
        // }
    } else {
        if (SSmodel::inputs.d_t < (int)(SSmodel::inputs.system.T.n_rows + 10)){
            // Colapsed KF
            nTrue = nNan2pi - 1 - SSmodel::inputs.d_t;
        } else {
            // KF did not colapsed
            nTrue = nNan2pi - 1 - SSmodel::inputs.system.T.n_rows;
        }
    }
    double LLIK, AIC, BIC, AICc;
    // Exception when function is nan
    if (flag > 6){
        objFunValue = datum::nan;
    }
    LLIK = -0.5 * (log(2*datum::pi) * nNan2pi + nTrue * objFunValue);
    infoCriteria(LLIK, p.n_elem - SSmodel::inputs.cLlik + SSmodel::inputs.u.n_rows + SSmodel::inputs.nonStationaryTerms,
                 nNan2pi, AIC, BIC, AICc);
    vec criteria(4);
    criteria(0) = LLIK;
    criteria(1) = AIC;
    criteria(2) = BIC;
    criteria(3) = AICc;
    SSmodel::inputs.criteria = criteria;
    if (flag == 1) {
        SSmodel::inputs.estimOk = "Q-Newton: Gradient convergence\n";
    } else if (flag == 2){
        SSmodel::inputs.estimOk = "Q-Newton: Function convergence\n";
    } else if (flag == 3){
        SSmodel::inputs.estimOk = "Q-Newton: Parameter convergence\n";
    } else if (flag == 4){
        SSmodel::inputs.estimOk = "Q-Newton: Maximum Number of iterations reached\n";
    } else if (flag == 5){
        SSmodel::inputs.estimOk = "Q-Newton: Maximum Number of function evaluations\n";
    } else if (flag == 6){
        SSmodel::inputs.estimOk = "Q-Newton: Unable to decrease objective function\n";
    } else if (flag == 7){
        SSmodel::inputs.estimOk = "Q-Newton: Objective function returns nan\n";
        objFunValue = datum::nan;
    } else {
        SSmodel::inputs.estimOk = "Q-Newton: No convergence!!\n";
    }
    if (SSmodel::inputs.verbose){
        double nSeconds = timer.toc();
        printf("%s", SSmodel::inputs.estimOk.c_str());
        printf("Elapsed time: %10.5f seconds\n", nSeconds);
    }
    SSmodel::inputs.p = p;
    SSmodel::inputs.objFunValue = objFunValue;
    SSmodel::inputs.grad = grad;
    // Eliminating cycle periods
    uvec aux = find(inputs.rhos > 0);
    inputs.rhos = inputs.rhos(aux);
    inputs.periods = inputs.periods(aux);
    SSmodel::inputs.v.reset();
    inputs.harmonics = regspace<uvec>(0, inputs.periods.n_elem - 1);
    SSmodel::inputs.verbose = verboseCopy;
}
/*
// Forecast
void BSMclass::forecast(){
    SSmodel::forecast();
    // Correcting for Box-Cox lambda
    inputs.yFor = SSmodel::inputs.yFor;
    inputs.yForV.reshape(SSmodel::inputs.FFor.n_rows, 2);
    vec aux = sqrt(SSmodel::inputs.FFor);
    inputs.yForV.col(1) = invBoxCox(inputs.yFor + aux, inputs.lambda);
    inputs.yForV.col(0) = invBoxCox(inputs.yFor - aux, inputs.lambda);
    inputs.yFor = invBoxCox(inputs.yFor, inputs.lambda);
}
// Filter
void BSMclass::filter(){
    SSmodel::filter();
    // Correcting for Box-Cox lambda
    inputs.yFit = SSmodel::inputs.yFit;
    inputs.yFitV.reshape(SSmodel::inputs.yFit.n_rows, 2);
    vec aux = sqrt(SSmodel::inputs.F);
    inputs.yFitV.col(1) = invBoxCox(inputs.yFit + aux, inputs.lambda);
    inputs.yFitV.col(0) = invBoxCox(inputs.yFit - aux, inputs.lambda);
    inputs.yFit = invBoxCox(inputs.yFit, inputs.lambda);
}
// Smooth
void BSMclass::smooth(bool outlier){
    SSmodel::smooth(outlier);
    // Correcting for Box-Cox lambda
    inputs.yFit = SSmodel::inputs.yFit;
    inputs.yFitV.reshape(SSmodel::inputs.yFit.n_rows, 2);
    vec aux = sqrt(SSmodel::inputs.F);
    inputs.yFitV.col(1) = invBoxCox(inputs.yFit + aux, inputs.lambda);
    inputs.yFitV.col(0) = invBoxCox(inputs.yFit - aux, inputs.lambda);
    inputs.yFit = invBoxCox(inputs.yFit, inputs.lambda);
}
*/
// Estimation of a family of UC models
void BSMclass::estimUCs(vector <string> allUCModels, uvec harmonics,
                        double& minCrit, bool VERBOSE, 
                        double oldMinCrit, int nuInit){
    // Estim a number of UC models and select the best according to minCrit
    //       The best is compared to oldMinCrit that is the current best system
    //       and the overall best is put into SSmodel::inputs and inputs
    //       If there is no previous model to compare to set oldMinCrit to 1e12
    double curCrit, 
    AIC, 
    BIC, 
    AICc;
    SSinputs bestSS = SSmodel::inputs;
    BSMmodel bestBSM = inputs;
    if (isnan(oldMinCrit)){
        oldMinCrit = 1e12;
    }
    minCrit = oldMinCrit;
    bool inputsArma = inputs.arma;
    for (unsigned int i = 0; i < allUCModels.size(); i++){
        SSmodel::inputs.p0 = -9999.9;
        int wide = 30;
        if (inputs.PTSnames)
            wide = 8;
        bool arma = inputs.arma;
        setModel(allUCModels[i], inputs.periods(harmonics), inputs.rhos(harmonics), false);
        inputs.arma = arma;
        // Cleaning variables for outliers starting anew
        if (SSmodel::inputs.u.n_elem > 0){
            if (nuInit > 0){
                SSmodel::inputs.u = SSmodel::inputs.u.rows(0, nuInit - 1);
            } else {
                SSmodel::inputs.u.resize(0);
            }
        }
        // Model estimation
        estim(SSmodel::inputs.verbose);
        AIC = SSmodel::inputs.criteria(1);
        BIC = SSmodel::inputs.criteria(2);
        AICc = SSmodel::inputs.criteria(3);
        // Avoid selecting a model with problems
        if (AIC == -datum::inf || AIC == datum::inf){
            AIC = BIC = AICc = datum::nan;
        }
        if (VERBOSE){
            string MODEL = allUCModels[i];
            if (inputs.PTSnames){
                MODEL = UC2PTS(allUCModels[i]);
                printf(" %*s: %13.4f %13.4f %13.4f\n", wide, MODEL.c_str(), AIC, BIC, AICc);
            } else {
                printf(" %*s: %8.4f %8.4f %8.4f\n", wide, MODEL.c_str(), AIC, BIC, AICc);
            }
        }
        if (inputs.criterion == "aic"){
            curCrit = AIC;
        } else if (inputs.criterion == "bic"){
            curCrit = BIC;
        } else {
            curCrit = AICc;
        }
        if ((curCrit < minCrit && !isnan(curCrit))){  // || i == 0){
            minCrit = curCrit;
            bestSS = SSmodel::inputs;
            bestBSM = inputs;
        }
    }
    SSmodel::inputs = bestSS;
    inputs = bestBSM;
    inputs.arma = inputsArma;
}
// Identification
void BSMclass::ident(string show, bool VERBOSE){
    bool verboseCopy = SSmodel::inputs.verbose;
    SSmodel::inputs.verbose = VERBOSE;
    wall_clock timer;
    timer.tic();
    // Interpolation in case of missing
    // if (inputs.missing.n_elem > 0)
    //     interpolate();
    double season,
    maxLag,
    outlierCopy = SSmodel::inputs.outlier;
    string inputTrend = inputs.trend, 
        inputCycle = inputs.cycle,
        inputSeasonal = inputs.seasonal, 
        inputIrregular = inputs.irregular, 
        model,
        trendTypes, 
        cycTypes, 
        seasTypes, 
        irrTypes, 
        restRW; 
    int trueTrend,
    nuInit = SSmodel::inputs.u.n_rows;
    //bool VERBOSE = SSmodel::inputs.verbose;
    // Controling estimation with or without outliers
    if (outlierCopy > 0){
        SSmodel::inputs.outlier = 0;
    }
    // Controlling verbose output
    SSmodel::inputs.verbose = false;
    vec periods = inputs.periods(find(inputs.rhos > 0));
    season = max(periods);
    if (season == 1){
        inputSeasonal = "none";
        inputs.seasonal = "none";
    }
    maxLag = floor(season / 2);
    // Trend tests
    if (SSmodel::inputs.y.n_rows < 15){
        inputs.tTest = false;
    }
    if (inputs.stepwise && inputTrend == "?" && inputs.tTest){
        vec lagsAdf(2);
        double lagsAdfMax;
        lagsAdf(0) = 2 * season + 2;
        lagsAdf(1) = 10;
        lagsAdfMax = max(lagsAdf);
        if (lagsAdfMax < SSmodel::inputs.y.n_rows / 2){
            inputs.tTest = false;
        } else {
            trueTrend = adfTests(SSmodel::inputs.y, max(lagsAdf), "bic");
            if (trueTrend == 0){    // No trend detected
                inputTrend = "none";
                trendTypes = "none";
            } else if (trueTrend == 1){
                inputTrend = "SOME";
                trendTypes = "rw";
            }
        }
    }
    // Seasonal test
    string isSeasonal;
    if (inputSeasonal[0] == 'n'){
        inputSeasonal = "none";
        isSeasonal = "none";
    } else {
        isSeasonal = "true";
    }
    // Selecting harmonics
    uvec harmonics = regspace<uvec>(0, periods.n_elem - 1);
    uvec harmonics0 = harmonics;
    if (inputSeasonal[0] != 'n' && inputSeasonal[0] != 'l' && !inputs.MSOE){
        vec betaHR;
        selectHarmonics(SSmodel::inputs.y, SSmodel::inputs.u, periods, harmonics, betaHR, isSeasonal);
        if (harmonics.n_rows == 0){
            inputSeasonal = "none";
            harmonics = harmonics0;
            periods = inputs.periods;
        }
    }
    if (season == 4 && harmonics.n_rows > 0){
        harmonics.reset();
        harmonics = regspace<uvec>(0, 1);
    }
    inputs.harmonics = harmonics;
    // UC identification
    double minCrit; // = 1e12, minCrit1;
    if (VERBOSE && (show == "head" || show == "both")){
        printf("------------------------------------------------------------\n");
        if (SSmodel::inputs.outlier < 0){
            printf(" Identification started WITH outlier detection\n");
        } else {
            if (inputs.PTSnames)
                printf(" Identification of PTS models:\n");
            else
                printf(" Identification started WITHOUT outlier detection\n");
        }
        printf("------------------------------------------------------------\n");
        if (inputs.PTSnames)
            printf("    Model            AIC           BIC          AICc\n");
        else
            printf("          Model                       AIC      BIC     AICc\n");
        printf("------------------------------------------------------------\n");
    }
    // Finding models to identify
    vector<string> allUCModels;
    size_t pos;
    bool runAll = !inputs.stepwise;
    if (season == 1 || inputSeasonal != "?"){
        runAll = true;
    }
    if (!runAll){
        if (isSeasonal[0] == 't'){
            seasTypes = "equal/different";
        } else if (isSeasonal[0] == 'd'){
            seasTypes = "none/equal/different";
        } else if (isSeasonal[0] == 'f'){
            seasTypes = "none";
        }
    }
    if (inputIrregular == "?"){
        irrTypes = "none/arma(0,0)";
    } else {          // Model with one irregular
        irrTypes = inputIrregular;
        runAll = true;
    }
    if (inputCycle == "?"){
        cycTypes = "none/" + inputs.cycle0;
    } else {
        cycTypes = inputCycle;
    }
    if (inputTrend == "?"){
        trendTypes = "none/rw";
    } else if (inputTrend[0] != 'S'){
        trendTypes = inputTrend;
        runAll = true;
    } else if (inputTrend[0] == 'S'){
        runAll = false;
    }
    if (runAll){    // no stepwise
        if (inputTrend == "?"){
            if (inputIrregular != "none" && inputIrregular != "arma(0,0)"
                    && inputIrregular != "?"){
                // Avoiding identification problems between arma(p,q) and dt trend
                trendTypes = inputs.trendOptions;
                strReplace("/dt", "", trendTypes);
                strReplace("dt/", "", trendTypes);
                strReplace("/srw", "", trendTypes);
                strReplace("srw/", "", trendTypes);
            } else {
                trendTypes = inputs.trendOptions;
            }
        }
        if (inputSeasonal == "?")
            seasTypes = inputs.seasonalOptions;
        else
            seasTypes = inputSeasonal;
        if (inputIrregular == "?")
            irrTypes = inputs.irregularOptions;
        else
            irrTypes = inputIrregular;
        findUCmodels(trendTypes, cycTypes, seasTypes, irrTypes, allUCModels);
        estimUCs(allUCModels, harmonics, minCrit, VERBOSE, 1e12, nuInit);
    } else {                  // stepwise
        if (inputSeasonal[0] == 'n'){
            // Annual or non seasonal data
            if (inputTrend == "?" || inputTrend == "SOME")
                trendTypes = trendTypes + "/llt/dt";
            seasTypes = "none";
            findUCmodels(trendTypes, cycTypes, seasTypes, irrTypes, allUCModels);
            estimUCs(allUCModels, harmonics, minCrit, VERBOSE, 1e12, nuInit);
        } else {
            // Seasonal data
            if (inputSeasonal != "?"){
                //   seasTypes = "none/equal/different";
                // } else {
                seasTypes = inputSeasonal;
            }
            // Best of rw or none trends
            findUCmodels(trendTypes, cycTypes, seasTypes, irrTypes, allUCModels);
            estimUCs(allUCModels, harmonics, minCrit, VERBOSE, 1e12, nuInit);
            allUCModels.clear();
            if (inputs.model.substr(0, 1) == "n"){
                // case if best trend is none
                findUCmodels("dt", cycTypes, seasTypes, "arma(0,0)", allUCModels);
                estimUCs(allUCModels, harmonics, minCrit, VERBOSE, minCrit, nuInit);
            } else {
                // case rw or llt is best: estimate some dts
                pos = inputs.model.find("/", 0);
                // Extract non trend part of the best model so far
                restRW = inputs.model.substr(pos, inputs.model.size() - pos);
                // Estimate the best model changing the trend to LLT
                findUCmodels("llt", cycTypes, seasTypes, irrTypes, allUCModels);
                estimUCs(allUCModels, harmonics, minCrit, VERBOSE, minCrit, nuInit);
                // Now take the non trend part of the best model so far and use the DT trend instead
                pos = inputs.model.find("/", 0);
                restRW = inputs.model.substr(pos, inputs.model.size() - pos);
                allUCModels.clear();
                allUCModels.push_back("dt" + restRW);
                estimUCs(allUCModels, harmonics, minCrit, VERBOSE, minCrit, nuInit);
            }
        }
    }
    // Checking for identification total failure
    bool succeed = true;
    if (SSmodel::inputs.p.has_nan() || isnan(minCrit)){
        succeed = false;
        // Setting up default model
        setModel("rw/none/none/none", inputs.periods(harmonics), inputs.rhos(harmonics), false);
        SSmodel::inputs.p.set_size(1);
        SSmodel::inputs.p.fill(0);
        llikAug(SSmodel::inputs.p, &(SSmodel::inputs));
    }
    if (VERBOSE && !succeed){
        printf("                      Identification failed!!\n");
        printf("              Unable to find a proper model!!\n");
    }
    // Selecting best ARMA
    if (inputs.arma && succeed){
        string armaModel, modelNew;
        vec beta0, orders(2);
        orders.fill(0);
        if (inputIrregular == "?" && succeed){
            string inputIrregular2;
            splitModel(inputs.model, inputTrend, inputCycle, inputSeasonal, inputIrregular2);
            if (inputIrregular == "?" && SSmodel::inputs.y.n_elem > 30){
                int maxSearch = season + 4;
                if (maxSearch > 28){
                    maxSearch = 28;
                }
                maxLag = 5;
                if (season == 1){
                    maxSearch = 8;
                }
                if ((float)SSmodel::inputs.y.n_elem - (float)SSmodel::inputs.system.T.n_rows - 3 - (float)maxLag - (float)maxSearch > 3 * (float)season){
                    filter();
                    uvec ind = find_finite(SSmodel::inputs.v);
                    // if ((float)SSmodel::inputs.v.n_elem - (float)SSmodel::inputs.system.T.n_rows - 2 > 3 * (float)season){
                    if ((float)ind.n_elem - (float)SSmodel::inputs.system.T.n_rows - 2 > 3 * (float)season){
                        selectARMA(SSmodel::inputs.v.rows(SSmodel::inputs.system.T.n_rows + 2, SSmodel::inputs.v.n_elem - 1),
                                   maxLag, maxSearch, "bic", orders, beta0);
                        inputs.beta0ARMA = beta0;
                    }
                }
            }
            // Model with ARMA
            if (sum(orders) > 0){    // ARMA identified
                armaModel.append(to_string((int)orders(0))).append(",").append(to_string((int)orders(1)));
                // Reformulating the irregular model
                string tModel, sModel, cModel, iModel;
                splitModel(inputs.model, tModel, cModel, sModel, iModel);
                // if (pureARMA)
                //   tModel = "none";
                modelNew.append(tModel).append("/").append(cModel).append("/").append(sModel).append("/").append("arma(").append(armaModel).append(")");
                allUCModels.clear();
                allUCModels.push_back(modelNew);
                // Estimating potential best model
                estimUCs(allUCModels, harmonics, minCrit, VERBOSE, minCrit, nuInit);
            }
        }
        // Selecting best pure ARMA (in case best model is trend + noise of any kind so far)
        pos = inputs.model.find("/none/none/");
        if (pos < 5 && inputs.model[0] != 'n' && SSmodel::inputs.y.n_elem > 30){
            // Searching for pure ARMA when trend + noise has been detected previously
            int maxSearch = season + 4;
            if (maxSearch > 28){
                maxSearch = 28;
            }
            maxLag = 5;
            if (season == 1){
                maxSearch = 8;
            }
            vec orders1(2);
            orders1.fill(0);
            vec beta1;
            if (SSmodel::inputs.y.n_rows - maxLag - maxSearch > 3 * season){
                selectARMA(SSmodel::inputs.y, maxLag, maxSearch, "bic", orders1, beta1);
            }
            inputs.beta0ARMA = beta1;
            if (sum(orders1) > 0){
                armaModel = "";
                modelNew = "";
                armaModel.append(to_string((int)orders1(0))).append(",").append(to_string((int)orders1(1)));
                modelNew.append("none/none/none/arma(").append(armaModel).append(")");
                allUCModels.clear();
                allUCModels.push_back(modelNew);
                // Estimating potential best model
                estimUCs(allUCModels, harmonics, minCrit, VERBOSE, minCrit, nuInit);
            }
            if (inputs.model[0] == 'n'){
                beta0.reset();
                beta0 = beta1;
                orders = orders1;
            }
            inputs.beta0ARMA = beta1;
        }
        if (VERBOSE && outlierCopy > 0){
            printf("------------------------------------------------------------\n");
            printf(" Final model WITH outlier detection\n");
            printf("------------------------------------------------------------\n");
        }
    }
    // bool correct = true;
    if (outlierCopy > 0){
        SSmodel::inputs.outlier = -abs(outlierCopy);
        allUCModels.clear();
        allUCModels.push_back(inputs.model);
        estimUCs(allUCModels, harmonics, minCrit, VERBOSE, minCrit, nuInit);
    }
    if (VERBOSE && (show == "tail" || show == "both")){
        double nSeconds = timer.toc();
        printf("------------------------------------------------------------\n");
        printf("  Identification time: %10.5f seconds\n", nSeconds);
        printf("------------------------------------------------------------\n");
    }
    // Final estimation (genunine with nans in case of missing data)
    //if (inputs.missing.n_elem > 0){
    //     SSmodel::inputs.y(inputs.missing).fill(datum::nan);
    //     estim();
    //}
    SSmodel::inputs.verbose = VERBOSE;
    // Updating inputs
    if (inputSeasonal[0] == 'n'){
        inputs.periods = {1};
        harmonics = {0};
        inputs.rhos = {1};
    }
    inputs.harmonics = harmonics;
    inputs.rhos = inputs.rhos(harmonics);
    SSmodel::inputs.outlier = -abs(outlierCopy);
    if (harmonics.n_elem > 0){
        inputs.periods = inputs.periods(harmonics);
    } else {
        inputs.periods.resize(1);
        inputs.periods.fill(1);
    }
    SSmodel::inputs.verbose = verboseCopy;
}
// Outlier detection a la Harvey and Koopman
void BSMclass::estimOutlier(vec p0, bool VERBOSE){
    bool verboseCopy = SSmodel::inputs.verbose;
    SSmodel::inputs.verbose = VERBOSE;
    // Havey, A.C. and Koopman, S.J. (1992), Diagnostic checking of unobserved
    //        components time series models , JBES, 10, 377-389.
    int n = SSmodel::inputs.y.n_elem - 1, //nNan,
        nu = SSmodel::inputs.u.n_rows,
        lu = 0;
    //bool VERBOSE = SSmodel::inputs.verbose;
    SSmodel::inputs.verbose = false;
    vec periodsCopy = inputs.periods,
        rhosCopy = inputs.rhos;
    // Length of u's
    if (nu == 0){
        lu = n + SSmodel::inputs.h + 1;
    } else {
        lu = SSmodel::inputs.u.n_cols;
    }
    wall_clock timer;
    timer.tic();
    // Initial estimation without checking oultiers
    SSmodel::inputs.p0 = p0;
    estim(SSmodel::inputs.p0, VERBOSE);
    inputs.periods = periodsCopy;
    inputs.rhos = rhosCopy;
    // Storing initial model clean
    SSinputs bestSS = SSmodel::inputs;
    BSMmodel bestBSM = inputs;
    // Forward Addition loop
    // Disturbances estimation
    disturb();
    // AO
    vec eps;
    if (inputs.nPar(3) == 1){
        eps = abs(inputs.eps);
    } else {
        eps = join_vert(zeros<vec>(n - SSmodel::inputs.v.n_elem + 1),
                        abs(SSmodel::inputs.v / sqrt(SSmodel::inputs.F)));
        eps.replace(datum::nan, 0);
    }
    uvec indAO = find(eps > 2.3);
    vec  valAO = eps(indAO);
    uvec sortInd;
    // Correction in case there are too many AO's
    if (valAO.n_elem > 10){
        sortInd = sort_index(valAO, "descend");
        sortInd = sortInd.rows(0, 9);
        indAO = indAO(sortInd);
        valAO = valAO(sortInd);
    }
    // LS
    uvec indLS;
    vec  valLS;
    if (inputs.nPar(0) > 0){ // && SSmodel::inputs.eta.row(0).max() > 2.5){
        valLS = abs(SSmodel::inputs.eta.row(0).t());
        indLS = selectOutliers(valLS, 3, 2.5);
        eps = abs(SSmodel::inputs.eta.row(0).t());
        valLS = eps(indLS);
    }
    // SC
    uvec indSC;
    vec  valSC;
    if (inputs.ns(0) > 1 && inputs.nPar(0) > 0){ // && SSmodel::inputs.eta.row(1).max() > 3){
        valSC = abs(SSmodel::inputs.eta.row(1).t());
        indSC = selectOutliers(valSC, 3, 3.0);
        eps = abs(SSmodel::inputs.eta.row(1).t());
        valSC = eps(indSC);
    }
    // All outliers together
    inputs.typeOutliers = join_vert(join_vert(zeros(size(indAO)), ones(size(indLS))), 2 * ones(size(indSC)));
    uvec ind = join_vert(join_vert(indAO, indLS), indSC);
    vec  val = join_vert(join_vert(valAO, valLS), valSC);
    // Sorting vectors
    if (ind.n_elem > 0){
        // Sorting and removing less significant in case of many outliers
        sortInd = sort_index(val, "descend");
        val = val(sortInd);
        ind = ind(sortInd);
        inputs.typeOutliers = inputs.typeOutliers(sortInd);
        if (ind.n_elem > 20){
            val = val.rows(0, 19);
            ind = ind.rows(0, 19);
            inputs.typeOutliers = inputs.typeOutliers.rows(0, 19);
        }
        // Sorting now by date
        sortInd = sort_index(ind);
        val = val(sortInd);
        ind = ind(sortInd);
        inputs.typeOutliers = inputs.typeOutliers(sortInd);
    }
    // Removing duplicated outliers of different types
    vec uniqueInd = unique(conv_to<mat>::from(ind));
    if (uniqueInd.n_elem < ind.n_elem){
        uvec indAux(uniqueInd.n_elem);
        vec  valAux(uniqueInd.n_elem);
        mat  outlAux(uniqueInd.n_elem, 1);
        uvec ii;
        int j = 0;
        for (uword i = 0; i < uniqueInd.n_elem; i++){
            ii = find(uniqueInd(i) == ind);
            if (ii.n_elem > 1){
                j = ii(val(ii).index_max());
            } else {
                j = ii(0);
            }
            valAux(i) = val(j);
            outlAux(i, 0) = inputs.typeOutliers(j);
            indAux(i) = ind(j);
        }
        ind = indAux;
        inputs.typeOutliers = outlAux;
        val = valAux;
    }
    // done
    bool cLlikCopy = SSmodel::inputs.cLlik,
         augmentedCopy = SSmodel::inputs.augmented,
         exactCopy = SSmodel::inputs.exact;
    if (ind.n_elem > 0){
        // matrix of potential inputs
        mat uNew(ind.n_elem, lu);
        uNew.fill(0);
        rowvec ui(lu);
        ui.fill(0);
        for (unsigned int i = 0; i < uNew.n_rows; i++){
            dummy(ind(i), inputs.typeOutliers(i), ui);
            uNew.row(i) = ui;
        }
        if (nu > 0){
            SSmodel::inputs.u = join_vert(SSmodel::inputs.u, uNew);
        } else {
            SSmodel::inputs.u = uNew;
        }
        // Re-estimation with inputs and all outliers in model
        SSmodel::inputs.cLlik = true;
        SSmodel::inputs.augmented = true;
        SSmodel::inputs.exact = false;
        if (VERBOSE){
            SSmodel::inputs.verbose = true;
        }
        estim(SSmodel::inputs.p0, VERBOSE);
        inputs.periods = periodsCopy;
        inputs.rhos = rhosCopy;
        vec obj(1); obj(0) = SSmodel::inputs.objFunValue;
        if (obj.is_finite()){
            // Model with all initial outliers converged
            SSmodel::inputs.verbose = false;
            // Backward deletion step
            uvec remove;
            int ns = SSmodel::inputs.system.T.n_rows,
                nuAll;
            int count = 0;
            do{
                nuAll = nu + ind.n_elem;
                vec t = abs(SSmodel::inputs.betaAug.rows(ns + nu, ns + nuAll - 1) /
                    sqrt(SSmodel::inputs.betaAugVar.rows(ns + nu, ns + nuAll - 1)));
                remove = find(t < abs(SSmodel::inputs.outlier));
                if (remove.n_elem > 0){
                    // Removing inputs
                    SSmodel::inputs.u.shed_rows(nu + remove);
                    inputs.typeOutliers.shed_rows(remove);
                    ind.shed_rows(remove);
                    // if (SSmodel::inputs.u.n_rows == 0 && inputs.model[0] != 'd'){
                    //   SSmodel::inputs.augmented = false;
                    //   SSmodel::inputs.exact = true;
                    // }
                    if (SSmodel::inputs.u.n_rows == 0){
                        SSmodel::inputs = bestSS;
                        inputs = bestBSM;
                    } else {
                        // Final estimation
                        estim(SSmodel::inputs.p0, VERBOSE);
                        inputs.periods = periodsCopy;
                        inputs.rhos = rhosCopy;
                    }
                }
                count++;
            } while (count < 4 && ind.n_elem > 0 && remove.n_elem > 0);
        }
        if (ind.n_elem > 0){
            inputs.typeOutliers.insert_cols(1, conv_to<mat>::from(ind));
        }
        // Final check
        vec best(1);
        if (inputs.criterion == "aic"){
            obj(0) = SSmodel::inputs.criteria(1);
            best(0) = bestSS.criteria(1);
        } else if (inputs.criterion == "bic"){
            obj(0) = SSmodel::inputs.criteria(2);
            best(0) = bestSS.criteria(2);
        } else {
            obj(0) = SSmodel::inputs.criteria(3);
            best(0) = bestSS.criteria(3);
        }
        if ((!obj.is_finite()) || (obj(0) > best(0))){
            // Model with outliers did not converge or is worse than initial
            SSmodel::inputs = bestSS;
            inputs = bestBSM;      
        }
    }
    // Restoring initial values
    SSmodel::inputs.verbose = VERBOSE;
    SSmodel::inputs.cLlik = cLlikCopy;
    SSmodel::inputs.augmented = augmentedCopy;
    SSmodel::inputs.exact = exactCopy;
    uvec aux = find(inputs.rhos > 0);
    inputs.rhos = inputs.rhos(aux);
    inputs.periods = inputs.periods(aux);
    SSmodel::inputs.verbose = verboseCopy;
}
// Components
void BSMclass::components(){
    SSmodel::smooth(true);
    int nCycles = sum(inputs.rhos < 0), k = SSmodel::inputs.u.n_rows;
    inputs.comp.set_size(4 + nCycles + k, SSmodel::inputs.yFit.n_rows);
    inputs.comp.fill(datum::nan);
    inputs.compV = inputs.comp;
    uword ny = SSmodel::inputs.y.n_elem - 1;
    vec nsCum = cumsum(inputs.ns);
    uvec remove(inputs.comp.n_rows, fill::zeros);
    inputs.compNames = "";
    uword ns = sum(inputs.ns), ind = 0;
    // Level
    if (inputs.ns(0) > 0 && SSmodel::inputs.system.T(0, 0) != 0){
        if (SSmodel::inputs.a.n_rows > ns)
            ind = ns;
        inputs.comp.row(0) = SSmodel::inputs.a.row(ind);
        inputs.compV.row(0) = SSmodel::inputs.P.row(ind);
        //if (sum(abs(inputs.comp.row(0).cols(0, ny))) > 1e-7)
        inputs.compNames += "Level/";
    } else
        remove(0) = 1;
    // Slope
    if (inputs.ns(0) > 1){
        inputs.comp.row(1) = SSmodel::inputs.a.row(1);
        inputs.compV.row(1) = SSmodel::inputs.P.row(1);
        //if (sum(abs(inputs.comp.row(1).cols(0, ny))) > 1e-7)
        inputs.compNames += "Slope/";
        //else
        //    remove(1) = 1;
    } else
        remove(1) = 1;
    // Seasonal
    if (inputs.ns(2) > 0){
        if (inputs.seasonal[0] != 'l'){
            urowvec ind = regspace<urowvec>(nsCum(1), 2, nsCum(2) - 1);
            inputs.comp.row(2) = sum(SSmodel::inputs.a.rows(ind));
            inputs.compV.row(2) = sum(SSmodel::inputs.P.rows(ind));
        } else {
            ind = nsCum(1);
            if (SSmodel::inputs.a.n_rows > ns)
                ind = SSmodel::inputs.a.n_rows - 1;
            inputs.comp.row(2) = SSmodel::inputs.a.row(ind);
            inputs.compV.row(2) = SSmodel::inputs.P.row(ind);
        }
        inputs.compNames += "Seasonal/";
    } else
        remove(2) = 1;
    // Irregular
    if (inputs.ns(3) == 0){     // White noise
        inputs.comp.row(3).cols(0, ny) = SSmodel::inputs.y.t() - SSmodel::inputs.yFit.rows(0, ny).t();
        //vec F = SSmodel::inputs.y;
        // uvec ind = find_finite(SSmodel::inputs.y);
        // inputs.compV()
        // inputs.compV.row(3).cols(0, ny) = SSmodel::inputs.F.rows(0, ny).t();
        // 
        // 
        // 
        // inputs.compV.row(3).cols(0, ny) = SSmodel::inputs.F.rows(0, ny).t();
        rowvec v = inputs.comp.row(3).cols(0, ny);
        uvec aux = find_finite(v);
        //inputs.comp.submat(3, 0, 3, SSmodel::inputs.y.n_elem - 1).replace(datum::nan, 0);
        if (sum(abs(v(aux))) > 1e-7)
            inputs.compNames += "Irregular/";
        else
            remove(3) = 1;
    } else {                    // ARMA
        inputs.comp.row(3) = SSmodel::inputs.a.row(nsCum(2));
        // inputs.compV.row(3) = SSmodel::inputs.P.row(nsCum(2));
        rowvec v = inputs.comp.row(3).cols(0, ny);
        uvec aux = find_finite(v);
        //inputs.comp.submat(3, 0, 3, SSmodel::inputs.y.n_elem - 1).replace(datum::nan, 0);
        if (sum(abs(aux)) > 1e-7){
            inputs.compNames += "ARMA(" + to_string(inputs.ar) +
                "," + to_string(inputs.ma) + ")/";
        } else
            remove(3) = 1;
    }
    // NaNs in irregular
    //if (inputs.missing.n_elem > 0){
    //    vec aux = inputs.comp.row(3);
    //    aux(inputs.missing).fill(datum::nan);
    //    inputs.comp.row(3) = aux;
    //}
    // Cycle
    if (inputs.ns(1) > 0){
        for (int i = 0; i < nCycles; i++){
            inputs.comp.row(4 + i) = SSmodel::inputs.a.row(nsCum(0) + 2 * i);
            inputs.compV.row(4 + i) = SSmodel::inputs.P.row(nsCum(0) + 2 * i);
            //if (sum(abs(inputs.comp.row(4 + i).cols(0, ny))) > 1e-7)
            inputs.compNames += "Cycle" + to_string(i + 1) + "/";
            //else
            //    remove(4 + i) = 1;
        }
    }
    // Inputs
    if (k > 0){
        if (SSmodel::inputs.system.Z.n_rows == 1){
            for (int i = 0; i < k; i++){
                inputs.comp.row(4 + nCycles + i) = SSmodel::inputs.system.D(i) *
                    SSmodel::inputs.u.submat(i, 0, i, SSmodel::inputs.u.n_cols - 1);
            }
        } else {
            for (int i = 0; i < k; i++){
                inputs.comp.row(4 + nCycles + i) =
                        SSmodel::inputs.a.row(nsCum(5) + i) % SSmodel::inputs.u.row(i);
            }
        }
        // Components names
        int nOut = 0;
        if (inputs.typeOutliers.n_rows > 0 && inputs.typeOutliers(0, 1) != -1)
            nOut = inputs.typeOutliers.n_rows;
        int nU = k - nOut;
        if (nU > 0){
            for (int i = 0; i < nU; i++)
                inputs.compNames += "Reg(" + to_string(i + 1) + ")/";
        }
        if(nOut > 0){
            string namei;
            for (int i = 0; i < nOut; i++){
                namei = "AO";
                if (inputs.typeOutliers(i, 0) == 1){
                    namei = "LS";
                } else if(inputs.typeOutliers(i, 0) == 2){
                    namei = "SC";
                }
                inputs.compNames += namei + to_string((uword)(inputs.typeOutliers(i, 1) + 1)) + "/";
            }
        }
    }
    //inputs.comp.submat(0, 0, inputs.comp.n_rows - 1, SSmodel::inputs.y.n_elem - 1).replace(datum::nan, 0);
    //inputs.comp.submat(0, 0, inputs.comp.n_rows - 1, SSmodel::inputs.y.n_elem - 1).replace(datum::inf, 0);
    if (sum(remove) > 0){
        inputs.comp.shed_rows(find(remove == 1));
        inputs.compV.shed_rows(find(remove == 1));
    }
    /*
    // Correcting for lambda
    mat aux = sqrt(compV);
    aux = inputs.comp + aux;
    inputs.compPlus = invBoxCoxMat(inputs.comp + aux, inputs.lambda);
    inputs.compMinus = invBoxCoxMat(inputs.comp - aux, inputs.lambda);
    inputs.comp = invBoxCoxMat(inputs.comp, inputs.lambda);
    inputs.compNames = inputs.compNames.substr(0, inputs.compNames.length() - 1);
    */
}
// Covariance of parameters (inverse of hessian)
mat BSMclass::parCov(vec& returnP){
    vec reserveP = SSmodel::inputs.p;
    // Finding true parameter values
    SSmodel::inputs.p = parameterValues(SSmodel::inputs.p);
    // Hessian and covariance of parameters
    int k = SSmodel::inputs.p.n_elem;
    uvec nn = find_finite(SSmodel::inputs.y);
    bool reserveCLLIK = SSmodel::inputs.cLlik;
    uvec isVar = find(inputs.typePar == 0); //, nonConstrained = find(inputs.constPar == 0);
    if (reserveCLLIK){
        SSmodel::inputs.userModel = bsmMatricesTrue;
        SSmodel::inputs.cLlik = false;
    //    SSmodel::inputs.p(isVar) =log(exp(2 * SSmodel::inputs.p(isVar)) * SSmodel::inputs.innVariance) / 2;
    }
    returnP = SSmodel::inputs.p;
    mat hess = hessLlik(&(SSmodel::inputs));
    hess *= 0.5 * (nn.n_elem);   // - SSmodel::inputs.nonStationaryTerms - reserveP.n_elem + 1);
    SSmodel::inputs.cLlik = reserveCLLIK;
    SSmodel::inputs.p = reserveP;
    SSmodel::inputs.userModel = bsmMatrices;
    mat iHess(k, k);
    iHess.fill(datum::nan);
    uvec indHess = find(inputs.constPar < 1);
    if (hess.is_finite()){
        if (SSmodel::inputs.cLlik){
            iHess.submat(indHess, indHess) = pinv(hess.submat(indHess, indHess));
        } else {
            iHess = pinv(hess);
        }
        iHess.diag() = abs(iHess.diag());
    }
    return iHess;
}
// Finding true parameter values out of transformed parameters
vec BSMclass::parameterValues(vec p){
    vec nparCum = cumsum(inputs.nPar);
    int nCycles = sum(inputs.rhos < 0);
    // Transforming all variances
    vec parValues(p.n_elem);
    vec isVar(p.n_elem);
    uvec aux; aux = find(inputs.typePar == 0);
    parValues(aux) = exp(2 * p(aux));
    isVar.fill(0);
    isVar(aux).fill(1);
    // Transforming the rest of parameters
    // Trend
    if (inputs.nPar(0) == 3){                     // Damped trend
        double alpha = p(0);
        constrain(alpha, regspace<vec>(0, 1)); //exp(p(0)) / (1+ exp(p(0)));
        parValues(0) = alpha;
    }
    //Cycle
    if (inputs.nPar(1) > 0){
        // Rhos
        int pos;
        aux = regspace<uvec>(nparCum(0), nparCum(1) - 1);
        vec pCycle = SSmodel::inputs.p(aux);
        vec pp = pCycle(span(0, nCycles - 1));
        constrain(pp, regspace<vec>(0, 1)); //exp(p(0)) / (1+ exp(p(0)));
        pos = nparCum(0) + nCycles;
        parValues(span(nparCum(0), pos - 1)) = pp;
        // Periods
        int nn = sum(inputs.periods < 0);
        if (nn > 0){
            aux = find(inputs.periods < 0);
            pp = pCycle(span(nCycles, nCycles - 1 + nn));
            constrain(pp, inputs.cycleLimits.rows(aux)); //exp(p(0)) / (1+ exp(p(0)));
            parValues(span(pos, pos - 1 + nn)) = pp;
            pos = pos + nn;
        }
        // Variances
        // parValues(span(pos, nparCum(1) - 1)) = exp(2 * pCycle(span(nCycles + nn, 2 * nCycles + nn - 1)));
    }
    // ARMA
    if (inputs.ar >0 || inputs.ma > 0) {  // ARMA model
        uvec ind;
        vec polyAux;
        isVar(span(nparCum(2) + 1, isVar.n_elem - 1)).fill(0);
        if (inputs.ar > 0){
            ind = regspace<uvec>(nparCum(2) + 1, nparCum(2) + inputs.ar);
            polyAux = p(ind);
            polyStationary(polyAux);
            parValues(ind) = polyAux; //(span(1, polyAux.n_elem - 1));
        }
        if (inputs.ma > 0){
            ind = regspace<uvec>(nparCum(2) + inputs.ar + 1, nparCum(2) + inputs.ar + inputs.ma);
            polyAux = p(ind);
            polyStationary(polyAux);
            parValues(ind) = polyAux; //(span(1, polyAux.n_elem - 1));
        }
    }
    // Inputs
    //if (nu > 0){
    //    ind1 = SSmodel::inputs.betaAug.n_elem - nu;
    //    parValues.rows(nparCum(3), nparCum(3) + nu - 1) = SSmodel::inputs.betaAug.rows(ind1, ind1 + nu - 1);
    //}
    // pureARMA
    if (inputs.pureARMA){
        // setting constant value
        parValues(nparCum(5) - 1) = SSmodel::inputs.p(nparCum(5) - 1);
    }
    if (SSmodel::inputs.cLlik){
        parValues(find(isVar)) *= SSmodel::inputs.innVariance;
    }
    return parValues;
}
// Parameter names
void BSMclass::parLabels(){
    inputs.parNames.clear();
    // Trend
    if (inputs.trend[0] == 'd' || inputs.trend[0] == 's'){
        inputs.parNames.push_back("Damping");
    }
    if (inputs.trend[0] != 'n' && inputs.trend[0] != 'i')
        inputs.parNames.push_back("Level");
    if (inputs.trend[0] != 'n' && inputs.trend[0] != 'r' && inputs.trend[0] != 't'){
        inputs.parNames.push_back("Slope");
    }
    vec nsCum = cumsum(inputs.ns);
    vec nparCum = cumsum(inputs.nPar);
    // Cycle
    int nCycles = sum(inputs.rhos < 0);
    uvec aux;
    vec pCycle, typeParC;
    //if (inputs.cycle[0] != 'n'){
    if (nCycles > 0 && SSmodel::inputs.p.n_elem > 0){
        aux = regspace<uvec>(nparCum(0), nparCum(1) - 1);
        pCycle = SSmodel::inputs.p(aux);
        typeParC = inputs.typePar(aux);
        int count;
        char name[20];
        for (count = 0; count < nCycles; count++){
            snprintf(name, 20, "Rho(%d)", count + 1);
            inputs.parNames.push_back(name);
        }
        for (count = 0; count < nCycles; count++){
            if (inputs.periods(count) < 0){
                snprintf(name, 20, "Period(%d)", count + 1);
                inputs.parNames.push_back(name);
            }
        }
        for (count = 0; count < nCycles; count++){
            snprintf(name, 20, "Var(%d)", count + 1);
            inputs.parNames.push_back(name);
        }
    }
    // Seasonal
    if (inputs.seasonal[0] == 'l')
        inputs.parNames.push_back("Seas");
    if (inputs.seasonal[0] == 'e')
        inputs.parNames.push_back("Seas(All)");
    if (inputs.seasonal[0] == 'd'){
        char seasNames[20];
        for (unsigned int i = sum(inputs.rhos < 0); i < inputs.periods.n_elem; i++){
            snprintf(seasNames, 20, "Seas(%1.1f)", inputs.periods(i));
            inputs.parNames.push_back(seasNames);
        }
    }
    // Irregular
    if (inputs.irregular[0] != 'n')
        inputs.parNames.push_back("Irregular");
    if (inputs.irregular != "arma(0,0)" && inputs.irregular[0] != 'n'){
        char arNames[20];
        for (int i = 0; i < inputs.ar; i++){
            snprintf(arNames, 20, "AR(%d)", i + 1);
            inputs.parNames.push_back(arNames);
        }
        for (int i = 0; i < inputs.ma; i++){
            snprintf(arNames, 20, "MA(%d)", i + 1);
            inputs.parNames.push_back(arNames);
        }
    }
    // Inputs
    int nOut = inputs.typeOutliers.n_rows;
    int nu = SSmodel::inputs.u.n_rows;
    if (sum(inputs.TVP) > 0){
        char betas[20];
        uvec ind = find(inputs.TVP);
        for (int i = 0; i < sum(inputs.TVP); i++){
            snprintf(betas, 20, "TVP(%d)", (int)ind(i) + 1);
            inputs.parNames.push_back(betas);
        }
        for (int i = 0; i < nu - nOut; i++){
            snprintf(betas, 20, "State(%d)", i + 1);
            inputs.parNames.push_back(betas);
        }
    } else if (nu - nOut > 0){
        char betas[20];
        for (int i = 0; i < nu - nOut; i++){
            snprintf(betas, 20, "Beta(%d)", i + 1);
            inputs.parNames.push_back(betas);
        }
    }
    // Outliers
    if (nOut > 0){
        char betas[20], typeO[5];
        for (int i = 0; i < nOut; i++){
            if (inputs.typeOutliers(i, 0) == 0){
                snprintf(typeO, 5, "AO");
            } else if (inputs.typeOutliers(i, 0) == 1){
                snprintf(typeO, 5, "LS");
            } else if (inputs.typeOutliers(i, 0) == 2){
                snprintf(typeO, 5, "SC");
            }
            snprintf(betas, 20, "%s%0.0f", typeO, inputs.typeOutliers(i, 1) + 1);
            inputs.parNames.push_back(betas);
        }
    }
    if (inputs.pureARMA){
        inputs.parNames.push_back("Const");
    }
}
// Validation of BSM models
void BSMclass::validate(bool showTable){
    // SSpace validate
    SSmodel::validate(false, sum(inputs.nPar));
    vec scores;
    mat iHess = parCov(scores);
    SSmodel::inputs.covp = iHess;
    // Parameter names
    parLabels();
    // Parameter values
    int nu = SSmodel::inputs.u.n_rows;
    vec p, stdP, stdPBSM;
    stdPBSM = sqrt(abs(iHess.diag()));
    if (nu == 0 ){
        p = scores;
        stdP = stdPBSM;
    } else {
        if (SSmodel::inputs.system.Z.n_rows == 1){
            uword ind1 = SSmodel::inputs.betaAug.n_elem - nu;
            p = join_vert(scores, SSmodel::inputs.betaAug.rows(ind1, ind1 + nu - 1));
            stdP = join_vert(stdPBSM, sqrt(SSmodel::inputs.betaAugVar.rows(ind1, ind1 + nu - 1)));
        } else {
            uword ind1 = SSmodel::inputs.aEnd.n_rows - nu;
            p = join_vert(scores, SSmodel::inputs.aEnd.rows(ind1, SSmodel::inputs.aEnd.n_rows - 1));
            vec dPEnd = SSmodel::inputs.PEnd.diag();
            stdP = join_vert(stdPBSM, sqrt(dPEnd.rows(ind1, SSmodel::inputs.aEnd.n_rows - 1)));
        }
    }
    vec parValues = p;
    // Calculating t stats and pValues
    char str[70];
    vec t = stdP;
    vec pValue = abs(p / stdP);
    // Creating table
    string fullModel;
    if (SSmodel::inputs.u.n_rows == 0){
        fullModel = inputs.model;
    } else {
        fullModel = inputs.model + " + inputs";
    }
    if (SSmodel::inputs.system.Z.n_rows > 1){
        fullModel += " + TVP";
    }
    string MODEL = fullModel;
    if (inputs.PTSnames)
        MODEL = UC2PTS(fullModel);
    snprintf(str, 70, " Model: %s\n", MODEL.c_str());
    auto it = SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 1, str);
    snprintf(str, 70, " Box-Cox lambda: %3.2f\n", inputs.lambda);
    SSmodel::inputs.table.insert(it, str);
    //if (SSmodel::inputs.cLlik)
    //    SSmodel::inputs.table.insert(it, " Concentrated Maximum-Likelihood\n");
    //else
    //    SSmodel::inputs.table.insert(it, " Maximum-Likelihood\n");
    vec col1;
    int insert = 0;
    // Periods
    vec periods1 = inputs.periods(find(inputs.rhos > 0));
    if (inputs.seasonal[0] == 'l')
        periods1.reset();
    int lPer = periods1.n_elem;
    if (lPer > 0 && periods1(0) > 1 && inputs.nPar(2) > 0){
        string line;
        snprintf(str, 70, " Periods: %5.1f", periods1(0));
        line = str;
        for (int i = 1; i < lPer; i++){
            snprintf(str, 70, " /%5.1f", periods1(i));
            line += str;
        }
        SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 3, line + "\n");
        insert++;
    } else {
        SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 3, " Periods: \n");
        insert++;
    }
    if (any(inputs.constPar == 1)){
        snprintf(str, 70, " (*)  concentrated out parameters\n");
        SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 4 + insert, str);
        insert++;
    }
    if (any(inputs.constPar > 1)){
        snprintf(str, 70, " (**) constrained parameters during estimation\n");
        SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 4 + insert, str);
        insert++;
    }
    SSmodel::inputs.table.at(5 + insert) = "                     Param   asymp.s.e.        |T|     |Grad| \n";
    col1 = parValues;
    // Pretty numbers for constrained parameters
    if (inputs.nPar(0) == 3 && inputs.constPar(0) > 0){  // DT trend
        if (p(0) < -100)
            col1(0) = 0;
        if (p(0) > 100)
            col1(0) = 1;
    }
    uvec indConst = find(inputs.constPar == 2);
    if (indConst.n_elem > 0){
        col1(indConst).fill(0);
    }
    uvec ind = find(inputs.constPar > 1);
    t(ind).fill(datum::nan);
    pValue(ind).fill(datum::nan);
    SSmodel::inputs.grad(ind).fill(datum::nan);
    vec gradBetas(nu);
    gradBetas.fill(datum::nan);
    vec grad = join_vert(SSmodel::inputs.grad, gradBetas);
    // stars for constrained parameters
    vector<string> col2;
    string chari;
    vec constPar;
    if (nu > 0){
        if (SSmodel::inputs.system.Z.n_rows == 1)
            constPar = join_vert(inputs.constPar, ones(nu, 1));
        else
            constPar = join_vert(inputs.constPar, zeros(nu, 1));
    } else {
        constPar = inputs.constPar;
    }
    for (uword i = 0; i < constPar.n_elem; i++){
        if (constPar(i) == 0)
            chari = "  ";
        else if (constPar(i) == 1)
            chari = "* ";
        else
            chari = "**";
        col2.push_back(chari);
    }
    // Adding spaces for betas
    for (int i = 0; i < nu; i++){
        col2.push_back("  ");
    }
    // for (unsigned i = 0; i < p.n_elem; i++){
    //     snprintf(str, 70, "%s:  \n", inputs.parNames.at(i).c_str());
    // }
    for (unsigned i = 0; i < sum(inputs.nPar) + SSmodel::inputs.u.n_rows; i++){
        if (abs(col1(i)) > 1e-3 || abs(col1(i)) == 0 || abs(col1(i)) == 1){
            if (isnan(pValue(i))){
                snprintf(str, 70, "%*s: %12.4f%2s \n", 12, inputs.parNames.at(i).c_str(), col1(i), col2.at(i).c_str());
            } else {
                if (constPar(i) > 0 || isnan(grad(i))){
                    snprintf(str, 70, "%*s: %12.4f%2s %10.4f %10.4f \n", 12, inputs.parNames.at(i).c_str(), col1(i), col2.at(i).c_str(), t(i), pValue(i));
                } else {
                    snprintf(str, 70, "%*s: %12.4f%2s %10.4f %10.4f %10.2e\n", 12, inputs.parNames.at(i).c_str(), col1(i), col2.at(i).c_str(), t(i), pValue(i), abs(grad(i)));
                }
            }
        } else {
            if (isnan(pValue(i))){
                snprintf(str, 70, "%*s: %12.2e%2s \n", 12, inputs.parNames.at(i).c_str(), col1(i), col2.at(i).c_str());
            } else {
                if (constPar(i) > 0 || isnan(grad(i))){
                    snprintf(str, 70, "%*s: %12.2e%2s %10.2e %10.4f \n", 12, inputs.parNames.at(i).c_str(), col1(i), col2.at(i).c_str(), t(i), pValue(i));
                } else {
                    snprintf(str, 70, "%*s: %12.2e%2s %10.2e %10.4f %10.2e\n", 12, inputs.parNames.at(i).c_str(), col1(i), col2.at(i).c_str(), t(i), pValue(i), abs(grad(i)));
                }
            }
        }
        SSmodel::inputs.table.at(i + 7 + insert) = str;
    }
    // mat coef = join_horiz(join_horiz(col1, t), pValue);
    SSmodel::inputs.coef = col1;
    // if (showTable){
    //     for (auto i = SSmodel::inputs.table.begin(); i != SSmodel::inputs.table.end(); i++){
    //         cout << *i << " ";
    //     }
    // }
    if (showTable){
        for (unsigned int i = 0; i < SSmodel::inputs.table.size(); i++){
            printf("%s ", SSmodel::inputs.table[i].c_str());
        }
    }
}
// Disturbance smoother (to recover just trend and epsilons)
void BSMclass::disturb(){
    inputs.eps.zeros(SSmodel::inputs.y.n_elem);
    if (inputs.irregular[0] == 'a' && inputs.ar == 0 && inputs.ma == 0){
        // Modification adding the observation noise as a final state
        SSinputs  copiaSS  = SSmodel::getInputs();
        // Modifying system
        int nsAll = sum(inputs.ns) + 1;
        // int nu = SSmodel::inputs.u.n_rows;
        bool TVP = (copiaSS.system.Z.n_rows > 1);
        mat T(nsAll, nsAll), R(nsAll, nsAll), Q(nsAll, nsAll), Z(1, nsAll); //, D(1, nu);
        if (TVP)
            Z.resize(copiaSS.u.n_cols, nsAll);
        T.fill(0);
        R.eye();
        Q.fill(0);
        Z.fill(0);
        nsAll -= 2;
        T(span(0, nsAll), span(0, nsAll)) = copiaSS.system.T;
        //R(span(0, nsAll), span(0, nsAll)) = copiaSS.system.R;
        R(span(0, nsAll), span(0, nsAll + 1)) = copiaSS.system.R;
        R(nsAll + 1, nsAll + 1) = 1;
        Q = copiaSS.system.Q;
        Q(nsAll + 1, nsAll + 1) = copiaSS.system.H(0, 0);
        //Q(span(0, nsAll), span(0, nsAll)) = copiaSS.system.Q;
        if (TVP){
            Z.cols(0, nsAll) = copiaSS.system.Z;
            Z.col(nsAll + 1).fill(copiaSS.system.C(0, 0));
        } else {
            Z(0, span(0, nsAll)) = copiaSS.system.Z;
            Z(0, nsAll + 1) = copiaSS.system.C(0, 0);
        }
        // copying into copiaSS
        copiaSS.system.T = T;
        copiaSS.system.R = R;
        copiaSS.system.Q = Q;
        copiaSS.system.Z = Z;
        copiaSS.system.H(0, 0) = 0;
        copiaSS.system.C(0, 0) = 0;
        // Creating new system
        SSmodel copia = SSmodel(copiaSS);
        copia.disturb();
        // Saving in system the disturbances (just trend and irregular)
        copiaSS = copia.getInputs();
        inputs.eps = copiaSS.eta.row(copiaSS.eta.n_rows - 1).t();
        SSmodel::inputs.eta = copiaSS.eta.rows(span(0, inputs.ns(0) - 1));
    } else {
        // No need of system modification, just run disturb()
        SSmodel::disturb();
        if (inputs.nPar(2) > 1)
            inputs.eps = SSmodel::inputs.eta.row(SSmodel::inputs.eta.n_rows - 1).t();
        SSmodel::inputs.eta = SSmodel::inputs.eta.rows(span(0, inputs.ns(0) - 1));
    }
}
// Gauss-Newton Minimum searcher
// First with numerical gradient function and second with
//                gradient supplied by user
int BSMclass::quasiNewtonBSM(std::function <double (vec& x, void* inputsFake)> objFun,
                             std::function <vec (vec& x, void* inputsFake, double obj, int& nFuns)> gradFun,
                             vec& xNew, void* inputsFake, double& objNew, vec& gradNew, mat& iHess,
                             bool verbosef){
    // Code for inputs.constPar
    // 0: not constrained; 1: concentrated-out; 2: zero variance; 3: alpha constrained
    int nx = xNew.n_elem, 
        flag = 0, 
        nOverallFuns, 
        nFuns = 0, 
        nIter = 0;
    double objOld, alpha_i;
    vec gradOld(nx), 
        xOld = xNew, 
        d(nx);
    vec crit(5); crit(0) = 1e-6; crit(1) = 1e-7; crit(2) = 1e-5; crit(3) = 1000; crit(4) = 10000;
    iHess.eye(nx, nx);
    uvec isVar = find(inputs.typePar == 0); //, nonConstrained = find(inputs.constPar == 0);
    bool cLlik = (sum(inputs.constPar) > 0);
    int newTry = 0;
    // Initial concentrated variance
    uvec concentratedOutPar = find(inputs.constPar == 1);
    uvec initialConcentratedPar = concentratedOutPar;
    if (cLlik){
        xNew(concentratedOutPar).fill(0);
    }
    // Calculating objective function and gradient
    objNew = objFun(xNew, inputsFake);
    vec xUncon = xNew;
    // xUncon is xNew de-converted to original variances (xNew is ratio of variances)
    if (cLlik){
        xUncon(isVar) = log(exp(2 * xNew(isVar)) * SSmodel::inputs.innVariance) / 2;
    }
    if (inputs.pureARMA){
        gradNew = gradFun(xNew, inputsFake, objNew, nFuns);
    } else {
        gradNew = gradFun(xUncon, inputsFake, objNew, nFuns);
    }
    nOverallFuns = nFuns + 1;
    if (cLlik){
        gradNew(concentratedOutPar).fill(0);
    }
    // Head of table
    if (verbosef){
        printf(" Iter FunEval  Objective       Step\n");
        printf("%5.0i %5.0i %12.5f %12.5f\n", nIter, nOverallFuns, objNew, 1.0);
    }
    // Main loop
    uvec zeroVar, largestVar, allVar = find(inputs.typePar == 0); //, nonConst;
    vec newVar, variances, maxVar, d_old, critPar(1); //, critGrad(1);
    double innVar, objBest = 1e6;
    bool diagHess = false;
    int counter = 0;    // Regulates diagonal or full hessian
    vec xBest;
    // Main loop
    do{
        nIter++;
        // Search direction
        d = -iHess * gradNew;
        d(find(inputs.constPar > 0)).fill(0);
        if (counter < 6 && as_scalar(abs(d.t() * gradNew)) > 0.01)
            diagHess = true;
        // Line Search
        xOld = xNew; gradOld = gradNew; objOld = objNew;
        alpha_i = 0.5;
        // storing in case objNew becomes nan
        innVar = SSmodel::inputs.innVariance;
        d_old = d;
        lineSearch(objFun, alpha_i, xNew, objNew, gradNew, d, nIter, nFuns, inputsFake);
        // Correcting when function becomes nan
        if (isnan(objNew)){   // Linesearch failed
            xNew = xOld;
            objNew = objOld;
            objOld = datum::nan;
            gradNew = gradOld;
            SSmodel::inputs.innVariance = innVar;
            alpha_i = 1;
            d = d_old;
        }
        nOverallFuns = nOverallFuns + nFuns;
        xUncon = xNew;
        if (cLlik){
            xUncon(isVar) = log(exp(2 * xNew(isVar)) * SSmodel::inputs.innVariance) / 2;
        }
        // Checking for zero variances
        zeroVar = find((((xUncon % (inputs.constPar == 0) % (inputs.typePar == 0)) < -10) +
            ((abs(gradNew) % (inputs.constPar == 0)) % (inputs.typePar == 0) < 0.0001)) == 2);
        if (zeroVar.n_elem > 0){
            xNew(zeroVar).fill(-300);
            inputs.constPar(zeroVar).fill(2);
        }
        // Checking for boundaries in trend damping
        if (inputs.nPar(0) > 2 && inputs.constPar(0) == 0){    // DT trend
            if (xNew(0) > 20){
                xNew(0) = 300;
                inputs.constPar(0) = 3;
            }
            if (xNew(0) < -4){
                xNew(0) = -300;
                inputs.constPar(0) = 3;
            }
        }
        // Changing concentrated-out parameter (code 1)
        if (cLlik){
            variances = exp(2 * xUncon) % (inputs.typePar == 0); //  % (inputs.constPar == 0);
            if (inputs.nPar(0) == 3){      // DT trend
                variances(0) = -300;
            }
            largestVar = (variances).index_max();
            maxVar = variances(largestVar);
            if (concentratedOutPar(0) != largestVar(0)){
                inputs.constPar(concentratedOutPar).fill(0);
                concentratedOutPar = largestVar;
                inputs.constPar(concentratedOutPar).fill(1);
                newVar = exp(2 * xNew(concentratedOutPar));
                xNew(allVar) = log(exp(2 * xNew(allVar)) / newVar(0)) / 2;
                xNew(find(inputs.constPar == 2)).fill(-300);
                diagHess = true;
            }
        }
        if (inputs.pureARMA){
            gradNew = gradFun(xNew, inputsFake, objNew, nFuns);
        } else {
            gradNew = gradFun(xUncon, inputsFake, objNew, nFuns);
        }
       // Correcting gradient for constrained parameters
        if (cLlik){
            gradNew(find(inputs.constPar)).fill(0);
        }
        nOverallFuns += nFuns;
        // Verbose
        if (verbosef){
            printf("%5.0i %5.0i %12.5f %12.5f\n", nIter, nOverallFuns, objNew, alpha_i);
        }
        // Stop Criteria
        flag = stopCriteria(crit, max(abs(gradNew)), objOld - objNew, 1e5, nIter, nOverallFuns);
        // Inverse Hessian BFGS update
        if (!flag){
            bfgs(iHess, gradNew - gradOld, xNew - xOld, nx, nIter);
            if (diagHess){
                diagHess = false;
                iHess = diagmat(iHess);
            }
        }
        // Try other initial conditions because non decreasing or nan function
        // Provisions when  problems with optimisation
        if (flag > 5){
            objNew = objOld;
            gradNew = gradOld;
            xNew = xOld;
        }
        if (flag > 105 && newTry < 4){
            newTry++;
            flag = 0;
            if (SSmodel::inputs.verbose){
                // cout << "    Trying new point..." << endl;
                printf("    Trying new point...\n");
            }
            xNew(allVar) = round(xNew(allVar)); //  % inputs.typePar;
            if (inputs.nPar(0) > 2){
                xNew(0) = 2;
            }
            if (objNew < objBest){
                objBest = objNew;
                xBest = xNew;
            } else {
                objNew = objBest;
                xNew = xBest;
            }
            xNew(allVar).fill(-newTry);
            objNew = objFun(xNew, inputsFake);
            xUncon = xNew;
            // xUncon is xNew de-converted to original variances (xNew is ratio of variances)
            if (cLlik)
                xUncon(isVar) = log(exp(2 * xNew(isVar)) * SSmodel::inputs.innVariance) / 2;


            if (inputs.pureARMA){
                gradNew = gradFun(xNew, inputsFake, objNew, nFuns);
            } else {
                gradNew = gradFun(xUncon, inputsFake, objNew, nFuns);
            }
            nOverallFuns += nFuns;
            if (cLlik){
                gradNew(concentratedOutPar).fill(0);
            }
            iHess.eye(nx, nx);
        }
        counter++;
    } while (!flag);
    SSmodel::inputs.flag = flag;
    SSmodel::inputs.Iter = nIter;
    SSmodel::inputs.pTransform = xUncon;
    return flag;
}
// Get states names
string stateNames(BSMmodel sys){
    string namesStates = "Level";
    uword j;
    if (sys.ns(0) > 1)
        namesStates += "/Slope";
    if (sys.ns(1) > 0){
        j = 1;
        for (uword i = 0; i < sys.ns(1); i = i + 2){
            namesStates += "/Cycle" + to_string(j) + 
                           "/Cycle" + to_string(j) + "*";
            j++;
        }
    }
    if (sys.ns(2) > 0){
        if (sys.seasonal[0] != 'l'){
            uword nharm = ceil(sys.ns(2) / 2);
            vec periods = sys.periods.tail_rows(nharm);
            j = 1;
            for (uword i = 0; i < sys.ns(2); i = i + 2){
                namesStates += "/Seasonal" + to_string(j);
                if (sys.periods(j - 1) != 2)
                    namesStates += "/Seasonal" + to_string(j) + "*";
                j++;
            }
        } else {
            for (uword i = 0; i < sys.ns(2); i++){
                namesStates += "/Seasonal" + to_string(i + 1);
            }
        }
    }
    if (sys.ns(3) > 0){
        for (uword i = 0; i < sys.ns(3); i++){
            if (i == 0)
                namesStates += "/Irregular" + to_string(i + 1);
            else
                namesStates += "/Irr*";
        }
    }
    if (sys.ns(6) > 0){
        for (uword i = 0; i < sys.TVP.n_elem; i++){
            namesStates += "/TVP(" + to_string(i + 1) + ")";
        }
    }
    return namesStates;
}
// Count states and parameters of BSM model
void BSMclass::countStates(vec periods, string trend, string cycle, string seasonal, string irregular){
    // string trend, string cycle, string seasonal, string irregular, 
    // int nu, vec P, vec rhos, vec& ns, vec& nPar, int& arOrder, 
    // int& maOrder, bool& exact
    inputs.ns = zeros(7);
    inputs.nPar = inputs.ns;
    SSmodel::inputs.exact = true;
    if (SSmodel::inputs.augmented){
        SSmodel::inputs.exact = false;
        SSmodel::inputs.cLlik = true;
    }
    // Trend
    if (trend[0] == 'l'){         // LLT trend
        inputs.ns(0) = 2;
        inputs.nPar(0) = 2;
    } else if (trend[0] == 'd' || trend[0] == 's'){  // Damped or Hyndman damped trend
        inputs.ns(0) = 2;
        inputs.nPar(0) = 3;
        SSmodel::inputs.exact = false;
        SSmodel::inputs.cLlik = true;
        SSmodel::inputs.augmented = false;
    } else if(trend[0] == 'r'){   // RW trend
        inputs.ns(0) = 1;
        inputs.nPar(0) = 1;
    } else if(trend[0] == 'i' || trend[0] == 't'){   // IRW trend or trend with drift
        inputs.ns(0) = 2;
        inputs.nPar(0) = 1;
    } else {                      // No trend
        inputs.ns(0) = 1;
        inputs.nPar(0) = 0;
    }
    if (trend[0] == 't')
        inputs.Drift = true;
    else
        inputs.Drift = false;
    // Cycle
    int nCycles = 0;
    if (cycle[0] != 'n'){
        string cycle0 = cycle;
        strReplace("+", "", cycle0);
        strReplace("-", "", cycle0);
        nCycles = cycle.length() - cycle0.length();
        inputs.ns(1) = nCycles * 2;
        inputs.nPar(1) = inputs.ns(1) + sum(periods < 0);
        SSmodel::inputs.exact = false;
        SSmodel::inputs.augmented = false;
    }
    // Seasonal
    int minus;
    int nHarm = periods.n_elem - nCycles;
    if (any(periods == 2))
        minus = 1;
    else
        minus = 0;
    if (seasonal[0] == 'e'){          // All equal
        inputs.ns(2) = nHarm * 2 - minus;
        inputs.nPar(2) = 1;
    } else if (seasonal[0] == 'd'){  // All different
        inputs.ns(2) = nHarm * 2 - minus;
        inputs.nPar(2) = nHarm;
    } else if (seasonal[0] == 'l'){   // Linear
        inputs.ns(2) = inputs.seas - 1;
        inputs.nPar(2) = 1;
    } else {                        // No seasonal
        inputs.ns(2) = 0;
        inputs.nPar(2) = 0;
    }
    // Irregular
    inputs.ar = 0;
    inputs.ma = 0;
    if (irregular[0] == 'a'){      // ARMA
        int ind1 = irregular.find("(");
        int ind2 = irregular.find(",");
        int ind3 = irregular.find(")");
        inputs.ar = stoi(irregular.substr(ind1 + 1, ind2 - ind1 - 1));
        inputs.ma = stoi(irregular.substr(ind2 + 1, ind3 - ind2 - 1));
        if (inputs.ar == 0 && inputs.ma == 0){   // Just noise
            inputs.ns(3) = 0;
            inputs.nPar(3) = 1;
        } else {                      // ARMA
            inputs.ns(3) = max(inputs.ar, inputs.ma + 1);
            inputs.nPar(3) = inputs.ar + inputs.ma + 1;
            SSmodel::inputs.exact = false;
            SSmodel::inputs.augmented = false;
        }
    } else if (irregular[0] == 'n'){
        inputs.ns(3) = 0;
        inputs.nPar(3) = 0;
    }
    // inputs
    int nu = SSmodel::inputs.u.n_rows;
    if (nu > 0){
        SSmodel::inputs.exact = false;
        SSmodel::inputs.cLlik = true;
        if (sum(inputs.TVP) > 0){
            SSmodel::inputs.augmented = false;
            inputs.ns(6) = nu;
            inputs.nPar(6) = sum(inputs.TVP);
        } else {
            SSmodel::inputs.augmented = true;
        }
    }
    // Checking pureARMA model without inputs or with just constant
    inputs.pureARMA = false;
    if (trend[0] == 'n' && cycle[0] == 'n' && seasonal[0] == 'n' && irregular[0] == 'a' && SSmodel::inputs.u.n_rows == 0
            && (inputs.ar > 0 || inputs.ma > 0)){
        inputs.pureARMA = true;
        SSmodel::inputs.augmented = false;
        inputs.nPar(5) = 1;
    }
}
// Fix matrices in standard BSM models (all except variances)
void BSMclass::initMatricesBsm(vec periods, vec rhos, string trend, string cycle, string seasonal, string irregular){
    int nsCol, nsColTVP;
    countStates(periods, trend, cycle, seasonal, irregular);
    // Initializing system matrices
    int nsAll = sum(inputs.ns);
    SSmodel::inputs.system.T.eye(nsAll, nsAll);
    nsCol = sum(inputs.ns(span(0, 2))) + 1;
    nsColTVP = nsCol + inputs.ns(6);
    //SSmodel::inputs.system.R.eye(nsAll, nsCol);
    SSmodel::inputs.system.R.resize(nsAll, nsColTVP);
    SSmodel::inputs.system.R.submat(0, 0, nsAll - 1, nsCol - 1) = eye(nsAll, nsCol);
    uword nTVP = sum(inputs.TVP);
    if (nTVP > 0){
        uword nu = SSmodel::inputs.u.n_rows;
        SSmodel::inputs.system.R.submat(nsAll - nu, nsColTVP - nu, nsAll - 1, nsColTVP - 1) = eye(nu, nu);
    }
    SSmodel::inputs.system.Q.zeros(nsColTVP, nsColTVP);
    SSmodel::inputs.system.Gam = SSmodel::inputs.system.D = SSmodel::inputs.system.S = 0.0;
    SSmodel::inputs.system.Z.zeros(1, nsAll);
    SSmodel::inputs.system.C.ones(1, 1);
    SSmodel::inputs.system.H.zeros(1, 1);
    // Trends
    if (inputs.ns(0) > 0){
        trend2ss(inputs.ns(0), &SSmodel::inputs.system.T, &SSmodel::inputs.system.Z);
    }
    // Cycles
    uvec aux;
    if (inputs.ns(1) > 0){
        aux = find(rhos < 0);
        bsm2ss(inputs.ns(0), inputs.ns(1), abs(periods(aux)), abs(rhos(aux)), 
               &SSmodel::inputs.system.T, &SSmodel::inputs.system.Z);
    }
    // Seasonal
    if (inputs.ns(2) > 0){
        if (seasonal[0] != 'l'){
            aux = find(rhos > 0);
            bsm2ss(inputs.ns(0) + inputs.ns(1), inputs.ns(2), abs(periods(aux)),
                   abs(rhos(aux)), &SSmodel::inputs.system.T, &SSmodel::inputs.system.Z);
        } else {
            uword i1 = inputs.ns(0) + inputs.ns(1), i2 = i1 + inputs.ns(2) - 1;
            SSmodel::inputs.system.Z(i1 + inputs.ns(2) - 1) = 1.0;
            //SSmodel::inputs.system.Z(SSmodel::inputs.system.Z.n_elem - 1) = 1.0;
            SSmodel::inputs.system.T(span(i1 + 1, i2), span(i1, i2)) = eye(inputs.seas - 2, inputs.seas - 1);
            SSmodel::inputs.system.T(span(i1, i1), span(i1, i2)).fill(-1.0);
            //SSmodel::inputs.system.T(i1, i2) = 1.0;
            //SSmodel::inputs.system.T(i1, i1) = 0.0;
        }
    }
    // Irregular as ARMA
    if (inputs.ar > 0 || inputs.ma > 0){   // ARMA
        SSmodel::inputs.system.C.zeros(1, 1);
        SSmodel::inputs.system.Z.col(nsCol - 1) = 1.0;
    }
    // Inputs in case of pure regression
    if (SSmodel::inputs.u.n_elem > 0 && sum(inputs.nPar.rows(0, 3)) == 1 && !inputs.pureARMA){
        SSmodel::inputs.system.T(0, 0) = 0;
    }
    // Inputs as TVP
    if (sum(inputs.TVP) > 0){
        SSmodel::inputs.system.Z = repmat(SSmodel::inputs.system.Z, SSmodel::inputs.u.n_cols, 1);
        SSmodel::inputs.system.Z.cols(nsAll - SSmodel::inputs.u.n_rows, nsAll -1) = SSmodel::inputs.u.t();
    }
    // Pure ARMA
    if (inputs.pureARMA){
        // SSmodel::inputs.system.D = SSmodel::inputs.u;
        SSmodel::inputs.system.T(0, 0) = 0;
    }
}
// Initializing parameters of BSM model
void BSMclass::initParBsm(){
    int nTrue = sum(inputs.nPar);
    bool userP0 = true;
    uvec indNaN = find_nonfinite(SSmodel::inputs.p0);
    if ((SSmodel::inputs.p0(0) == -9999.9) || (indNaN.n_elem == SSmodel::inputs.p0.n_elem)){
        userP0 = false;
    }
    SSmodel::inputs.p0.resize(nTrue);
    uvec aux, aux1;
    vec p0 = SSmodel::inputs.p0;
    inputs.typePar = zeros(nTrue);
    // Trends
    SSmodel::inputs.p0.fill(-1.15);
    if (inputs.nPar(0) == 3){           // DT trend
        SSmodel::inputs.p0(0) = 2;                 // alpha
        SSmodel::inputs.p0(2) = -1.5;              // slope
        inputs.typePar(0) = -1;
    } else if (inputs.nPar(0) == 2){    // LLT trend
        SSmodel::inputs.p0(1) = -1.5;              // slope
    } else if (inputs.nPar(0) == 1 && inputs.ns(0) > 1){   // IRW or trend with drift
        if (inputs.Drift)                          // level of trend with drift
            SSmodel::inputs.p0(0) = -1.15;
        else
            SSmodel::inputs.p0(0) = -1.5;              // slope
    }
    // Cycles
    if (inputs.nPar(1) > 0){
        // Cycle inputs.rhos
        int nRhos = sum(inputs.rhos < 0), pos;
        pos = inputs.nPar(0) + nRhos;
        aux = regspace<uvec>(inputs.nPar(0), 1, pos - 1);
        SSmodel::inputs.p0(aux).fill(2);
        inputs.typePar(aux).fill(1);
        // Cycle inputs.periods
        aux1 = find(inputs.periods < 0);
        int nPer = aux1.n_elem;
        aux = regspace<uvec>(pos, 1, pos + nPer - 1);
        SSmodel::inputs.p0(aux) = -inputs.periods(aux1);
        vec aaa = SSmodel::inputs.p0(aux);
        unconstrain(aaa, inputs.cycleLimits.rows(aux1));
        SSmodel::inputs.p0(aux) = aaa;
        inputs.typePar(aux).fill(2);   // Marking inputs.periods in overall parameter vector
        // Cycle variances
        pos += nRhos;
        aux = regspace<uvec>(pos, 1, inputs.nPar(0) + inputs.nPar(1) - 1);
    }
    // ARMA models
    vec periods = inputs.periods(find(inputs.rhos > 0));
    if (periods.n_rows == 0){
        aux = 0.0;
    } else {
        aux = max(periods);
    }
    vec stdBeta, e, orders(2), betaHR;
    //double BIC, AIC, AICc;
    // Estimating initial conditions for ARMA from innovations
    if (inputs.nPar(3) > 1){
        uword ini = sum(inputs.nPar(span(0, 2))) + 1;
        aux = regspace<uvec>(ini, 1, sum(inputs.nPar.rows(0, 3)) - 1);
        inputs.typePar(aux).fill(3);
        orders(0) = inputs.ar;
        orders(1) = inputs.ma; //nPar(3) - inputs.ar - 1;
        // "Intelligent" initial conditions for ARMA
        //harmonicRegress(SSmodel::inputs.y, SSmodel::inputs.u, periods, betaHR, stdBeta, e);
        //linearARMA(e, orders, inputs.beta0ARMA, stdBeta);
        // 0 initial conditions for ARMA
        inputs.beta0ARMA.zeros(inputs.ar + inputs.ma);
        vec beta0aux, beta0aux1;
        uvec ind;
        vec armas = p0(ind);
        // Testing for unit roots in ARMA parameters chosen by user
        if (userP0 && !armas.has_nan()){
            ind = find(inputs.typePar == 3);
            inputs.beta0ARMA = p0(ind);
            vec absRoots, uno = {1};
            if (inputs.ar > 0){
                // AR model
                // Checking for non-stationary polynomial
                beta0aux = inputs.beta0ARMA(span(0, inputs.ar - 1));
                beta0aux1 = -beta0aux;
                absRoots = abs(roots(join_vert(uno, -beta0aux1)));
                if (any(absRoots >= 1)){
                    myError("\n\nERROR: Non-stationary model for AR initial conditions!!!\n");
                }
            }
            if (inputs.ma > 0){
                // MA model
                // Checking for non-invertible polynomial
                beta0aux = inputs.beta0ARMA(span(inputs.ar, inputs.ar + inputs.ma - 1));
                // Bringing MA polynomial to invertibility
                absRoots = abs(roots(join_vert(uno, beta0aux)));
                if (any(absRoots >= 1)){
                    myError("\n\nERROR: Non-invertible model for MA initial conditions!!!\n");
                }
            }
        }
        // Converting to estimation space
        // AR pars
        if (inputs.ar > 0){
            beta0aux = inputs.beta0ARMA(span(0, inputs.ar - 1));
            beta0aux1 = -beta0aux;
            // Correction for non-stationary polynomial
            arToPacf(beta0aux1);
            ind = find(abs(beta0aux1) >= 1);
            if (ind.n_elem > 0){
                beta0aux1(ind) = sign(beta0aux1(ind)) * 0.96;
                pacfToAr(beta0aux1);
                beta0aux = -beta0aux1;
            }
            invPolyStationary(beta0aux);
            beta0aux.elem(find_nonfinite(beta0aux)).zeros();
            aux = regspace<uvec>(ini, 1, ini + inputs.ar - 1);
            SSmodel::inputs.p0(aux) = beta0aux;
        }
        // MA pars
        if (inputs.ma > 0){
            beta0aux = inputs.beta0ARMA(span(inputs.ar, inputs.ar + inputs.ma - 1));
            // Bringing MA polynomial to invertibility
            maInvert(beta0aux);
            inputs.beta0ARMA(span(inputs.ar, inputs.ar + inputs.ma - 1)) = beta0aux;
            // Parameterising polynomial to be invertible
            invPolyStationary(beta0aux);
            aux = regspace<uvec>(ini + inputs.ar, 1, ini + inputs.ar + inputs.ma - 1);
            SSmodel::inputs.p0(aux) = beta0aux;
        }
    }
    // inputs
    int nu = SSmodel::inputs.u.n_rows;
    if (sum(inputs.TVP) > 0){
        uword ini = sum(inputs.nPar.rows(0, 5));
        aux = regspace<uvec>(ini, ini + sum(inputs.TVP) - 1);
        SSmodel::inputs.p0(aux).fill(-1.5);
    } else {
        int ns = SSmodel::inputs.betaAug.n_rows;
        if (nu > 0 && ns > 1){
            SSmodel::inputs.system.D = SSmodel::inputs.betaAug.rows(ns - nu, ns - 1);
        }
    }
    // Pure ARMA
    if (inputs.pureARMA){
        // setting value for constant
        SSmodel::inputs.system.D = nanMean(SSmodel::inputs.y);
        SSmodel::inputs.p0(nTrue - 1) = SSmodel::inputs.system.D(0, 0);
        inputs.typePar(nTrue - 1) = 5;
    }
    // Choosing initial concentrated variance
    inputs.constPar = zeros(nTrue);
    if (SSmodel::inputs.cLlik){
        if (inputs.nPar(3) > 0){         // arma(0,0) or arma(p,q)
            inputs.constPar(inputs.nPar(0) + inputs.nPar(1) + inputs.nPar(2)) = 1;
        } else if (inputs.nPar(3) == 0){   // no irregular component
            uvec minIndex = find(inputs.typePar == 0);
            inputs.constPar(minIndex(0)) = 1;
        }
        SSmodel::inputs.p0(find(inputs.constPar)).fill(0);
    }
    // User supplied initial values (NA)
    if (userP0){
        // type of parameter (0: variance;
        //        -1: damped of trend;
        //         1: cycle rhos;
        //         2: cycle periods;
        //         3: ARMA;
        // Converting user initial parameters to UComp initial
        inputs.p0Return = p0;
        // variances
        // concentrated out variance
        vec conc = p0(find(inputs.constPar));
        if (conc.has_nan()){
            conc = 1;
        }
        if (conc(0) < 1e-6){
            myError("\n\nERROR: Cannot select such small value for concentrated out variance!!!\n");
        }
        uvec ind2 = find(inputs.typePar == 0);
        vec variances = p0(ind2) / conc(0);
        if (any(variances < 0)){
            myError("\n\nERROR: Initial conditions for variances must be non-negative!!!\n");
        }
        variances(find(variances == 0)).fill(1e-70);
        variances = log(variances) / 2;
        uvec ind = find_nonfinite(variances);
        if (ind.n_elem > 0){
            // Replacing nan value selected by computer
            variances(ind) = exp(2 * SSmodel::inputs.p0(ind)) * conc;
        }
        SSmodel::inputs.p0(ind2) = variances;
        // Rhos
        ind = join_vert(find(inputs.typePar == -1), find(inputs.typePar == 1));
        vec pp;
        if (ind.n_elem > 0){
            pp = p0(ind);
            if (any(pp > 1) || any(pp < 0)){
                myError("\n\nERROR: Initial conditions for damping parameters must be between 0 and 1!!!\n");
            }
            vec lim1(pp.n_elem); lim1.fill(0);
            vec lim2(pp.n_elem); lim2.fill(1);
            mat limit = join_rows(lim1, lim2);
            unconstrain(pp, limit);
            ind2 = find_nonfinite(pp);
            if (ind2.n_elem > 0){
                pp(ind2) = SSmodel::inputs.p0(ind(ind2));
            }
            SSmodel::inputs.p0(ind) = pp;
        }
    } else {
        // type of parameter (0: variance;
        //        -1: damped of trend;
        //         1: cycle rhos;
        //         2: cycle periods;
        //         3: ARMA;
        // Converting initial parameters to user understandable
        inputs.p0Return = SSmodel::inputs.p0;
        // concentrated out variance
        inputs.p0Return.rows(find(inputs.constPar)).fill(0);
        // variances
        uvec ind = find(inputs.typePar == 0);
        inputs.p0Return(ind) = exp(2 * SSmodel::inputs.p0(ind));
        // Rhos
        ind = join_vert(find(inputs.typePar == -1), find(inputs.typePar == 1));
        vec pp;
        if (ind.n_elem > 0){
            pp = SSmodel::inputs.p0(ind);
            constrain(pp, regspace<vec>(0, 1));
            inputs.p0Return(ind) = pp;
        }
        // cycle periods
        uvec aux = find(inputs.periods < 0);
        ind = find(inputs.typePar == 2);
        if (ind.n_elem > 0){
            pp = SSmodel::inputs.p0(ind);
            constrain(pp, inputs.cycleLimits.rows(aux));
            inputs.p0Return(ind) = pp;
        }
        // ARMA
        ind = find(inputs.typePar == 3);
        if (ind.n_elem > 0){
            inputs.p0Return(ind) = inputs.beta0ARMA;
        }
    }
}
/*************************************************************
 //  * Implementation of auxiliar functions
 //  ************************************************************/
 // Variance matrices in standard BSM on top of fixed structure
 void bsmMatrices(vec p, SSmatrix* model, void* userInputs){
     BSMmodel* inp = (BSMmodel*)userInputs;
     vec nsCum = cumsum(inp->ns);
     vec nparCum = cumsum(inp->nPar);
     // Trend
     if (inp->nPar(0) == 2 && inp->ns(0) == 2){        // LLT
         model->Q(0, 0) = exp(2 * p(0));
         model->Q(1, 1) = exp(2 * p(1));
     } else if (inp->nPar(0) == 1 && inp->ns(0) == 1){  // RW trend
         model->Q(0, 0) = exp(2 * p(0));
     } else if (inp->nPar(0) == 3){                     // Damped trend
         constrain(p(0), regspace<vec>(0, 1)); //exp(p(0)) / (1+ exp(p(0)));
         model->Q(0, 0) = exp(2 * p(1));
         model->Q(1, 1) = exp(2 * p(2));
         model->T(1, 1) = p(0);
         if (inp->trend[0] == 's' && inp->MSOE){
             model->T(0, 1) = p(0);
             model->T(nsCum(nsCum.n_elem - 1), 1) = p(0);
         }
     } else if (inp->nPar(0) == 1 && inp->ns(0) == 2){    // IRW or trend with drift
         if (inp->Drift)
            model->Q(0, 0) = exp(2 * p(0));   // Trend with drift
         else
            model->Q(1, 1) = exp(2 * p(0));   // IRW
     } else if (inp->nPar(0) == 0 && inp->ns(0) == 1){  // No trend
         model->Q(0, 0) = 0;
     }
     // Cycle
     uvec ind, ind1;
     vec aux, aux1, periods, rhos, variances;
     uword pos;
     if (inp->nPar(1) > 0){
         // Rhos
         int nRhos = sum(inp->typePar == 1);
         pos = inp->nPar(0) + nRhos;
         ind = regspace<uvec>(inp->nPar(0), 1, pos - 1);
         vec pInd = p(ind);
         constrain(pInd, regspace<vec>(0, 1));
         p(ind) = pInd;
         rhos = pInd;
         // Cycle periods
         int nPerEstim = sum(inp->typePar == 2);
         periods = inp->periods(span(0, nRhos - 1));
         if (nPerEstim > 0){
             ind = regspace<uvec>(pos, 1, pos + nPerEstim - 1);
             pos += nPerEstim;
             ind1 = find(inp->periods < 0);
             pInd = p(ind);
             constrain(pInd, inp->cycleLimits.rows(ind1));
             p(ind) = pInd;
             periods(ind1) = pInd;
         }
         ind1 = regspace<uvec>(pos, 1, nparCum(1) - 1);
         variances = exp(2 * p(ind1));
         // Matrices for cycles
         bsm2ss(inp->ns(0), inp->ns(1), abs(periods), rhos, &model->T, &model->Z);
         ind = regspace<uvec>(nsCum(0), 1, nsCum(1) - 1);
         aux = vectorise(repmat(variances.t(), 2, 1));
         aux1 = aux(span(0, inp->ns(1) - 1));
         model->Q(ind, ind) = diagmat(aux1);
     }
     // Seasonal
     if (inp->nPar(2) > 0){                               // With seasonal
         // Index of states for seasonal
         ind = regspace<uvec>(nsCum(1), 1, nsCum(2) - 1);
         if (inp->seasonal[0] == 'l')
             model->Q(ind(0), ind(0)) = exp(2 * p(nparCum(1)));
         else if (inp->nPar(2) == 1){                         // Equal variances
             model->Q(ind, ind) = eye(inp->ns(2), inp->ns(2)) *
                 exp(2 * p(nparCum(1)));
         } else {                                        // Different variances
             ind1 = regspace<uvec>(nparCum(1), 1, nparCum(2) - 1);
             aux = vectorise(repmat(exp(2 * p(ind1)).t(), 2, 1));
             aux1 = aux(span(0, inp->ns(2) - 1));
             model->Q(ind, ind) = diagmat(aux1);
         }
     }
     // Irregular
     if (inp->nPar(3) == 1){              // Irregular no ARMA model
         model->H = exp(2 * p(nparCum(3) - 1));
     } else if (inp->ar > 0 || inp->ma > 0) {  // ARMA model
         SSmatrix mARMA;
         ARMAinputs iARMA;
         iARMA.ar = inp->ar;
         iARMA.ma = inp->ma;
         int aux4;
         uvec aux2 = regspace<uvec>(nparCum(2), 1, nparCum(3) - 1);
         initMatricesArma(inp->ar, inp->ma, aux4, mARMA);
         armaMatrices(p(aux2), &mARMA, &iARMA);
         uvec ind2 = regspace<uvec>(nsCum(2), 1, nsCum(3) - 1);
         model->T(ind2, ind2) = mARMA.T;
         uvec nsCol(1); nsCol(0) = nsCum(2); //sum(inp->ns(span(0, 1)));
         model->R(ind2, nsCol) = mARMA.R;
         model->Q(nsCol, nsCol) = mARMA.Q;
     }
     if (inp->pureARMA){
         model->D(0, 0) = p(nparCum(5) - 1);
     }
     // inputs
     //showSS(*model);
     if (sum(inp->TVP) > 0){
         uvec aux1 = nsCum(5) + find(inp->TVP == 1) - 1;
         uvec aux2 = regspace<uvec>(nparCum(5), 1, nparCum(6) - 1);
         model->Q(aux1, aux1) = diagmat(exp(2 * p(aux2)));
     }
     //showSS(*model);
 }
// Variance matrices in standard BSM on top of fixed structure for true parameters
void bsmMatricesTrue(vec p, SSmatrix* model, void* userInputs){
     BSMmodel* inp = (BSMmodel*)userInputs;
     vec nsCum = cumsum(inp->ns);
     vec nparCum = cumsum(inp->nPar);
     // Trend
     if (inp->nPar(0) == 2 && inp->ns(0) == 2){        // LLT
         model->Q(0, 0) = p(0);
         model->Q(1, 1) = p(1);
     } else if (inp->nPar(0) == 1 && inp->ns(0) == 1){  // RW trend
         model->Q(0, 0) = p(0);
     } else if (inp->nPar(0) == 3){                     // Damped trend
         //constrain(p(0), regspace<vec>(0, 1)); //exp(p(0)) / (1+ exp(p(0)));
         model->Q(0, 0) = p(1);
         model->Q(1, 1) = p(2);
         model->T(1, 1) = p(0);
         if (inp->trend[0] == 's' && inp->MSOE){
             model->T(0, 1) = p(0);
             model->T(nsCum(nsCum.n_elem - 1), 1) = p(0);
         }
     } else if (inp->nPar(0) == 1 && inp->ns(0) == 2){    // IRW or trend with drift
         if (inp->Drift)
             model->Q(0, 0) = p(0);    // Trend with drift
         else
            model->Q(1, 1) = p(0);     // IRW
     } else if (inp->nPar(0) == 0 && inp->ns(0) == 1){  // No trend
         model->Q(0, 0) = 0;
     }
     // Cycle
     uvec ind, ind1;
     vec aux, aux1, periods, rhos, variances;
     uword pos;
     if (inp->nPar(1) > 0){
         // Rhos
         int nRhos = sum(inp->typePar == 1);
         pos = inp->nPar(0) + nRhos;
         ind = regspace<uvec>(inp->nPar(0), 1, pos - 1);
         vec pInd = p(ind);
         //constrain(pInd, regspace<vec>(0, 1));
         //p(ind) = pInd;
         rhos = pInd;
         // Cycle periods
         int nPerEstim = sum(inp->typePar == 2);
         periods = inp->periods(span(0, nRhos - 1));
         if (nPerEstim > 0){
             ind = regspace<uvec>(pos, 1, pos + nPerEstim - 1);
             pos += nPerEstim;
             ind1 = find(inp->periods < 0);
             pInd = p(ind);
             //constrain(pInd, inp->cycleLimits.rows(ind1));
             //p(ind) = pInd;
             periods(ind1) = pInd;
         }
         ind1 = regspace<uvec>(pos, 1, nparCum(1) - 1);
         variances = p(ind1);
         // Matrices for cycles
         bsm2ss(inp->ns(0), inp->ns(1), abs(periods), rhos, &model->T, &model->Z);
         ind = regspace<uvec>(nsCum(0), 1, nsCum(1) - 1);
         aux = vectorise(repmat(variances.t(), 2, 1));
         aux1 = aux(span(0, inp->ns(1) - 1));
         model->Q(ind, ind) = diagmat(aux1);
     }
     // Seasonal
     if (inp->nPar(2) > 0){                               // With seasonal
         // Index of states for seasonal
         ind = regspace<uvec>(nsCum(1), 1, nsCum(2) - 1);
         if (inp->seasonal[0] == 'l')
             model->Q(ind(0), ind(0)) = p(nparCum(1));
         else if (inp->nPar(2) == 1){                         // Equal variances
             model->Q(ind, ind) = eye(inp->ns(2), inp->ns(2)) *
                 p(nparCum(1));
         } else {                                        // Different variances
             ind1 = regspace<uvec>(nparCum(1), 1, nparCum(2) - 1);
             aux = vectorise(repmat(p(ind1).t(), 2, 1));
             aux1 = aux(span(0, inp->ns(2) - 1));
             model->Q(ind, ind) = diagmat(aux1);
         }
     }
     // Irregular
     if (inp->nPar(3) == 1){              // Irregular no ARMA model
         model->H = p(nparCum(3) - 1);
     } else if (inp->ar > 0 || inp->ma > 0) {  // ARMA model
         SSmatrix mARMA;
         ARMAinputs iARMA;
         iARMA.ar = inp->ar;
         iARMA.ma = inp->ma;
         int aux4;
         uvec aux2 = regspace<uvec>(nparCum(2), 1, nparCum(3) - 1);
         initMatricesArma(inp->ar, inp->ma, aux4, mARMA);
         armaMatricesTrue(p(aux2), &mARMA, &iARMA);
         uvec ind2 = regspace<uvec>(nsCum(2), 1, nsCum(3) - 1);
         model->T(ind2, ind2) = mARMA.T;
         uvec nsCol(1); nsCol(0) = nsCum(2); //sum(inp->ns(span(0, 1)));
         model->R(ind2, nsCol) = mARMA.R;
         model->Q(nsCol, nsCol) = mARMA.Q;
     }
     if (inp->pureARMA){
         model->D(0, 0) = p(nparCum(5) - 1);
     }
     // inputs
     if (sum(inp->TVP) > 0){
         uvec aux1 = nsCum(5) + find(inp->TVP == 1) - 1;
         uvec aux2 = regspace<uvec>(nparCum(5), 1, nparCum(6) - 1);
         model->Q(aux1, aux1) = diagmat(p(aux2));
     }
}
// Remove elements of vector in n adjacent points
uvec selectOutliers(vec& val, int nTogether, float limit){
    int n = val.n_elem - 1,
        indMax;
    uvec indAround;
    bool next = true;
    uvec times;
    do{
        indMax = index_max(val);
        if (val(indMax) > limit){
            times.resize(times.n_elem + 1);
            times(times.n_elem - 1) = indMax;
            val.rows(std::max(0, (int)indMax - nTogether), std::min(n, (int)indMax + nTogether)).fill(0);
        } else {
            next = false;
        }
    } while (next);
    return times;
}
// Create dummy variable for outliers 0: AO, 1: LS, 2: SC
void dummy(uword indMax, uword typeO, rowvec& u){
    int n = u.n_elem;
    u.fill(0);
    if (typeO == 0){
        u(indMax) = 1.0;
    } else if (typeO == 1){
        u.cols(indMax, n - 1).fill(1.0);
    } else if (typeO == 2){
        u.cols(indMax, n - 1) = regspace(1, n - indMax).t();
    }
}
// Extract trend seasonal and irregular of model in a string
void splitModel(string model, string& trend, string& cycle, string& seasonal, string& irregular){
    int ind1, ind2, ind3;
    string aux1, aux2;
    
    lower(model);
    deblank(model);
    ind1 = model.find("/");
    aux1 = model.substr(ind1 + 1);
    ind2 = aux1.find("/");
    aux2 = aux1.substr(ind2 + 1);
    ind3 = aux2.find("/");
    trend = model.substr(0, ind1);
    cycle = aux1.substr(0, ind2);
    seasonal = aux2.substr(0, ind3);
    irregular = aux2.substr(ind3 + 1);
}
// SS form of trend models
void trend2ss(int ns, mat* T, mat* Z){
    if (ns > 1){
        (*T)(0, 1) = 1;
    }
    (*Z)(0) = 1;
}
// SS form of seasonal models
void bsm2ss(int ns0, int nsSeas, vec P, vec rhos, mat* T, mat* Z){
    bool minus = 1 - any(P == 2);
    uvec aux1 = regspace<uvec>(0, 2, nsSeas - 1) + ns0;
    (*Z).cols(aux1) = ones(Z->n_rows, aux1.n_elem);
    vec sinf = sin(2 * (datum::pi) / P) % rhos;
    vec cosf = cos(2 * (datum::pi) / P) % rhos;
    vec oneZero(2); oneZero(0) = 1; oneZero(1) = 0;
    vec oneOne(2); oneOne.fill(1);
    vec sines = kron(sinf, oneZero);
    vec cosines = kron(cosf, oneOne);
    int nDiag = nsSeas - 1 - minus;
    uvec aux3 = regspace<uvec>(0, 1, nDiag);
    mat aux = diagmat(cosines) + diagmat(sines(aux3), 1) + diagmat(-sines(aux3), -1);
    uvec aux2 = regspace<uvec>(0, 1, nDiag + minus);
    (*T)(aux2 + ns0, aux2 + ns0) = aux(aux2, aux2);
}
// Combining models for components
void findUCmodels(string trend, string cycle, string seasonal, string irregular, vector<string>& allModels){
    int nTrendModels, nCycleModels, nSeasonalModels, nIrregularModels;
    vector <string> trendModels, cycleModels, seasonalModels, irrModels;
    // Possible trends
    chopString(trend, "/", trendModels);
    nTrendModels = trendModels.size();
    // Possible cycles
    chopString(cycle, "/", cycleModels);
    nCycleModels = cycleModels.size();
    // Possible seasonals
    chopString(seasonal, "/", seasonalModels);
    nSeasonalModels = seasonalModels.size();
    // Possible irregulars
    chopString(irregular, "/", irrModels);
    nIrregularModels = irrModels.size();
    // All possible models
    // int count = 0;
    string cModel;
    for (int i = 0; i < nTrendModels; i++){
        for (int l = 0; l < nCycleModels; l++){
            for (int j = 0; j < nSeasonalModels; j++){
                for (int k = 0; k < nIrregularModels; k++){
                    if (trendModels[i] == "none" && cycleModels[l] == "none" && seasonalModels[j] == "none" && irrModels[k] == "none"){
                    } else {
                        cModel = trendModels[i];
                        cModel.append("/").append(cycleModels[l]).append("/").append(seasonalModels[j]).append("/").append(irrModels[k]);
                        allModels.push_back(cModel);
                    }
                }
            }
        }
    }
}
// Corrects model, cycle string, periods and rhos for modelling cycles
void modelCorrect(string& model, string& cycle, string& cycle0, vec& periods, vec& rhos){
    size_t pos0, pos1; //, pos2;
    vec number(1);
    cycle0 = cycle;
    // Is there any "?" in cycle
    pos1 = cycle.find("?");
    if (pos1 < cycle.length()){
        strReplace(cycle, "?", model);
        cycle = "?";
        strReplace("?", "", cycle0);
    }
    // Adapt periods and rhos to cycle specification
    pos1 = 0;
    vec mOne(1);
    mOne(0) = -1;
    do {
        pos0 = pos1;
        pos1 = min(cycle0.find('+', pos1 + 1), cycle0.find('-', pos1 + 1));
        number(0) = stod(cycle0.substr(pos0, pos1 - pos0));
        periods = join_vert(number, periods);
        rhos = join_vert(mOne, rhos);
    } while(pos1 != string::npos);
    // Chechking validity of cycles
    vec pCycles = periods(find(rhos < 0));
    double s = max(periods(find(rhos > 0)));
    if (any(abs(pCycles) < 1.5 * s) || any(abs(pCycles) <= 2)){
        myError("\n\nERROR: Cycle period too small!!");
    }
    uvec sIndex = sort_index(abs(periods), "descend");
    periods = sign(periods(sIndex)) % sort(abs(periods), "descend");
    rhos = rhos(sIndex);
    // Eliminating duplicities
    sIndex = find_unique(periods);
    periods = periods(sIndex);
    rhos = rhos(sIndex);
}
// Calculate limits for cycle periods for estimation
void calculateLimits(int n, vec periods, vec rhos, mat& cycleLimits, double s){
    double media;
    vec pCycles;
    int nCycles = sum(rhos < 0);
    cycleLimits.resize(nCycles, 2);
    pCycles = periods(span(0, nCycles - 1));
    //double s = max(periods(span(nCycles, periods.n_elem - 1)));
    // Sorting intermediate limits
    if (any(abs(pCycles) < 1.5 * s) || any(abs(pCycles) <= 2)){
        myError("\n\nERROR: Cycle period too small!!");
    }
    for (int i = 1; i < nCycles; i++){
        if (pCycles(i) > 0){
            cycleLimits(i, 1) = pCycles(i);
        }
        if (pCycles(i - 1) > 0){
            cycleLimits(i - 1, 0) = pCycles(i - 1);
        }
        if (pCycles(i) > 0 && pCycles(i - 1) < 0){
            cycleLimits(i - 1, 0) = pCycles(i) + 1;
        } else if (pCycles(i) < 0 && pCycles(i - 1) > 0){
            cycleLimits(i, 1) = pCycles(i - 1) - 1;
        } else if (pCycles(i) < 0 && pCycles(i - 1) < 0){
            media = (-pCycles(i) - pCycles(i - 1)) / 2;
            cycleLimits(i, 1) = media;
            cycleLimits(i - 1, 0) = media + 1;
        }
    }
    // Correcting extreme limits
    cycleLimits(0, 1) = pCycles(0);
    cycleLimits(nCycles - 1, 0) = pCycles(nCycles - 1);
    if (pCycles(0) < 0){
        vec aux(2);
        if (s == 1){
            aux(0) = n / 1.5;
            aux(1) = 70;
            cycleLimits(0, 1) = min(aux);
        } else if (s == 4){
            aux(0) =  1.5 * abs(cycleLimits(0, 1));
            aux(1) = 24;
            cycleLimits(0, 1) = max(aux);
            aux(0) = cycleLimits(0, 1);
            aux(1) = n / 1.5;
            cycleLimits(0, 1) = min(aux);
        } else {
            aux(0) = n / 1.5;
            aux(1) = 70 * s;
            cycleLimits(0, 1) = min(aux);
        }
        if (-pCycles(0) <= cycleLimits(0, 0)){
            myError("\n\nERROR: Initial condition for cycle too small!!!\n\n");
        } else if (-pCycles(0) >= cycleLimits(0, 1)){
            myError("\n\nERROR: Initial condition for cycle too big!!!\n\n");
        }
    }
    int nCycles1 = nCycles - 1;
    if (pCycles(nCycles1) < 0){
       cycleLimits(nCycles1, 0) = s * 1.5;
        if (-pCycles(nCycles1) <= cycleLimits(nCycles1, 0)){
            myError("\n\nERROR: Initial condition for cycle too small!!!\n\n");
        } else if (-pCycles(nCycles1) >= cycleLimits(nCycles1, 1)){
            myError("\n\nERROR: Initial condition for cycle too big!!!\n\n");
        }
    }
}
// Find first observation of n non-nan contiguous values
int findFirst(vec y, int n){
    uvec indFinite, poly(n, fill::ones), aux3;
    indFinite = find_finite(y);
    aux3 = conv(diff(indFinite), poly);
    int iniObs = indFinite(min(find(aux3.rows(n - 1, aux3.n_elem - 1) == n)));
    return iniObs;
}
// Translates names from UC to PTS
string UC2PTS(string modelUC){
    vector<string> aux;
    chopString(modelUC, "/", aux);
    // noise
    string model = "(A,";
    if (aux[3] == "none")
        model = "(N,";
    // trend
    if (aux[0] == "rw" || aux[0] == "none")
            model += "N,";
    else if (aux[0] == "srw")
            model += "Ad,";
    else if (aux[0] == "llt")
            model += "A,";
    else if (aux[0] == "td")
            model += "L,";
    // seasonal
    if (aux[2] == "none")
            model += "N)";
    else if (aux[2] == "equal")
            model += "E)";
    else if (aux[2] == "different")
            model += "D)";
    else if (aux[2] == "linear")
            model += "L)";
    return model;
}
// Show SSmodel
void showSS(SSmatrix m){
    printf("*********** SS system start *********\n");
    m.T.print("Matrix T:");
    m.R.print("Matrix R:");
    m.Q.print("Matrix Q:");
    mat RQR = m.R * m.Q * m.R.t();
    RQR.print("Matrix RQR:");
    if (m.Z.n_rows > 10)
        m.Z.rows(0, 9).print("First 10 rows of matrix Z:");
    else
        m.Z.print("Matrix Z:");
    printf("*********** SS system end *********\n");
}
// Show BSMmodel
void showBSM(BSMmodel m){
    printf("*********** BSM model start *********\n");
    printf("model: %s\n", m.model.c_str());
    printf("criterion: %s\n", m.criterion.c_str());
    printf("stepwise: %10i\n", m.stepwise);
    printf("tTest: %10i\n", m.tTest);
    printf("arma: %10i\n", m.arma);
    m.periods.t().print("periods:");
    printf("trend: %s\n", m.trend.c_str());
    printf("seasonal: %s\n", m.seasonal.c_str());
    printf("irregular: %s\n", m.irregular.c_str());
    printf("compNames: %s\n", m.compNames.c_str());
    printf("ar: %10i\n", m.ar);
    printf("ma: %10i\n", m.ma);
    m.rhos.t().print("rhos:");
    m.ns.t().print("ns:");
    m.nPar.t().print("nPar:");
    m.p0Return.t().print("p0Return:");
    m.typePar.t().print("typePar:");
    m.eps.t().print("eps:");
    m.beta0ARMA.t().print("beta0ARMA:");
    m.constPar.t().print("constPar:");
    m.harmonics.t().print("harmonics:");
    m.comp.print("comp:");
    m.typeOutliers.t().print("typeOutliers:");
    m.cycleLimits.print("cycleLimits:");
    printf("pureARMA: %10i\n", m.pureARMA);
    for (uword i = 0; i < m.parNames.size(); i++){
        printf("%s / ", m.parNames[i].c_str());
    }
    printf("\n*********** BSM model end *********");
}
