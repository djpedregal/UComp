// Hacer version R, MATLAB y Python
// #include <iostream>
// #include <math.h>
// #include <string.h>
// #include <armadillo>
// using namespace arma;
// using namespace std;
// #include "DJPTtools.h"
// #include "optim.h"
// #include "stats.h"
// #include "boxcox.h"
// #include "SSpace.h"
// #include "ARMAmodel.h"
#include "ETSmodel.h"

// Tobit Exponential Smoothing models
struct TETSmodel{
    ETSmodel m;
    vec Ymax,       // Censorship from above
        Ymin;       // Censorship from below
};
/**************************
* Model CLASS TETS
***************************/
class TETSclass{
public:
    TETSmodel data;
    TETSclass(ETSmodel m, vec Ymin, vec Ymax){
        TETSmodel data;
        data.m = m;
        data.Ymax = Ymax;
        data.Ymin = Ymin;
        this->data = data;
    };
//    ETSmodel m;
//    vec Ymax, Ymin;
//    bool errorExit = false;
//    TETSclass(ETSmodel m, vec Ymax, vec Ymin, bool errorExit){
//        this->m = m;
//        this->Ymax = Ymax;
//        this->Ymin = Ymin;
//        this->errorExit = errorExit;
//    };
//    void interpolate();
    void estim(bool);
    void validate();
    void forecast();
    // void simulate(uword, vec);
    void ident(bool);
    void components();
};
/****************************************************
// TETS functions declarations
****************************************************/
// pre-process of user inputs
TETSclass preProcess(vec, mat, string, int, int, bool, string, bool, rowvec, rowvec, rowvec, rowvec,
                     string, bool, bool, int, vec, bool, vec, double, vec, vec);
// Loglik computation
double llikTETS(vec&, void*);
// Gracient of Loglik
vec gradTETS(vec&, void*, double&, int&);
// One step ahead prediction
void oneStep(double, vec&, vec&, vec&, double, double, double, bool,
             vec&, vec&, double&, double&);
// System matrices for given p
// void tetsMatrices(TETSmodel*, vec);
/****************************************************
// TETS functions implementations
****************************************************/
// Estimation
void TETSclass::estim(bool verbose){
    // outputs
    double objFunValue = 0.0;
    data.m.loge2 = 0.0;
    data.m.logr = 0.0;
    vec grad, p = data.m.p0, p0userCOPY = data.m.p0user;
    mat iHess;
    // initial time
    wall_clock timer;
    timer.tic();
    int flag, bestFlag = 10, nAttempts = 0;
    TETSmodel best;
    double LLIK = 0.0, AIC = 0.0, BIC = 0.0, AICc = 0.0, bestAIC = 0.0,
        bestBIC = 0.0, bestLLIK = 0.0, bestAICc = 1e10,
        bestObjFunValue = 0.0, maxGrad = 0.0;
    vec bestP, bestGrad;
    bool again = false;
    do{


        // data.m.F.print("F 348");
        // data.m.g.print("g");
        // data.m.w.print("w");
        // data.m.p0.t().print("p0");
        // p.t().print("p");
        // LLIK = llikTETS(p, &data.m);


        flag = quasiNewtonETS(llikTETS, gradTETS, p, &data.m, objFunValue, grad, iHess, verbose,
                              data.m.limits.n_rows);
        // Correction in case of convergence problems
        if (flag > 5)
            objFunValue = llikTETS(p, &data.m);
        if (isnan(objFunValue))
            data.m.loge2 = datum::nan;
        // Information criteria
        uvec indNan = data.m.missing; //find_nonfinite(data.m.y);
        int nNan = data.m.y.n_elem - indNan.n_elem;
            //LLIK = -0.5 * (nNan * data.m.loge2 + 2 * data.m.logr); ////////////////////////////////////////////
        // Hyndman lo hace como en la siguiente línea
        //data.m.sigma2 = exp(data.m.loge2) / nNan;
            //LLIK = -0.5 * (nNan * log(data.m.sigma2 * (nNan - p.n_elem + 1)) + 2 * data.m.logr);
        data.m.sigma2 = exp(2 * p(p.n_elem - 1));
        LLIK = -nNan * objFunValue; //-0.5 * (nNan * log(data.m.sigma2 * nNan) + 2 * data.m.logr);
        infoCriteria(LLIK, p.n_elem, nNan, AIC, BIC, AICc);
        maxGrad = max(abs(data.m.grad.rows(0, data.m.limits.n_rows - 1)));
        if (nAttempts == 0 || (!isnan(objFunValue) && objFunValue < bestObjFunValue)){
            best.m = data.m;
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
                data.m.p0user = p0userCOPY;
            if (flag > 2 && maxGrad > 1e-3 && nAttempts < 4 && data.m.p0user.n_elem == 0){
                again = true;
                if (AICc < bestAICc){
                    best.m = data.m;
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
                    //data.m.prop = 0.2;
                    data.m.p0user = {data.m.alphaL(1) - 0.01, data.m.betaL(0) + 0.0001,
                                     data.m.phiL(1) - 0.01, 1 - data.m.alphaL(1) + 0.0001};
                } else if (nAttempts == 1){
                    //data.m.p0user = p0userCOPY;
                    data.m.prop = 0.3;
                } else {
                    data.m.prop += 0.3;
                }
                initPar(data.m);
                p = data.m.p0;
            } else
                again = false;
            nAttempts++;
            if (isnan(AICc) && nAttempts < 4)
                again = true;
        }
    } while (again);
    if (nAttempts > 0)
        data.m = best.m;
    data.m.p0user = p0userCOPY;
    vec criteria(4);
    criteria(0) = bestLLIK;
    criteria(1) = bestAIC;
    criteria(2) = bestBIC;
    criteria(3) = bestAICc;
    this->data.m.criteria = criteria;
    if (!isfinite(objFunValue))
        bestFlag = 0;
    // Printing results
    if (bestFlag == 1) {
        this->data.m.estimOk = "Q-Newton: Gradient convergence.\n";
    } else if (bestFlag == 2){
        this->data.m.estimOk = "Q-Newton: Function convergence.\n";
    } else if (bestFlag == 3){
        this->data.m.estimOk = "Q-Newton: Parameter convergence.\n";
    } else if (bestFlag == 4){
        this->data.m.estimOk = "Q-Newton: Maximum number of iterations reached.\n";
    } else if (bestFlag == 5){
        this->data.m.estimOk = "Q-Newton: Maximum number of Function evaluations.\n";
    } else if (bestFlag == 6){
        this->data.m.estimOk = "Q-Newton: Unable to decrease objective function.\n";
    } else if (bestFlag == 7){
        this->data.m.estimOk = "Q-Newton: Objective function returns nan.\n";
    } else {
        this->data.m.estimOk = "Q-Newton: No convergence!!\n";
    }
    if (verbose){
        double nSeconds = timer.toc();
        printf("%s", this->data.m.estimOk.c_str());
        printf("Elapsed time: %10.5f seconds\n", nSeconds);
    }
    etsMatrices(&data.m, bestP);
    vec paux = bestP.rows(0, sum(data.m.nPar.rows(0, 1)) - 1);
    trans(paux, data.m.limits);
    // Storing results in structure
    this->data.m.p = bestP;
    this->data.m.truep = paux;
    this->data.m.objFunValue = bestObjFunValue;
    this->data.m.grad = bestGrad;
    this->data.m.flag = bestFlag;
    // Storing parameter values
    vec aux = bestP.rows(0, data.m.limits.n_rows - 1);
    vec nParCum = cumsum(data.m.nPar);
    trans(aux, data.m.limits);
    bestP.rows(0, data.m.limits.n_rows - 1) = aux;
    int pos = 1;
    data.m.alpha = bestP(0);
    if (data.m.trend != "N"){
        data.m.beta = bestP(1);
        pos++;
    }
    if (data.m.model.length() > 3){
        data.m.phi = bestP(2);
        pos++;
    }
    if (data.m.seasonal != "N"){
        data.m.gamma = bestP(pos);
        pos++;
    }
    data.m.x0 = bestP.rows(pos, nParCum(2) - 1);
    if (data.m.u.n_rows > 0)
        data.m.d = bestP.rows(nParCum(2), nParCum(3) - 1).t();
    if (sum(data.m.arma) > 0){
        if (data.m.arma(0) > 0){
            data.m.ar = bestP.rows(nParCum(3), nParCum(3) + data.m.arma(0) - 1);
            polyStationary(data.m.ar);
        }
        if (data.m.arma(1) > 0){
            data.m.ma = bestP.rows(nParCum(3) + data.m.arma(0), nParCum(3) + sum(data.m.arma) - 1);
            polyStationary(data.m.ma);
        }
    }
}
// Validation
void TETSclass::validate(){
    ETSclass m(data.m);
    bool VERBOSE = m.inputModel.verbose;
    m.inputModel.verbose = false;
    // ETS Validation
    m.inputModel.p = m.inputModel.p.rows(0, m.inputModel.p.n_elem - 2);
    m.validate();
    components();
    // data.m.comp = m.inputModel.comp;
    // data.m.compNames = m.inputModel.compNames;
    // Manipulating output table
    // Replacing second part of table
    for (size_t i = 0; i < m.inputModel.table.size(); ++i) {
        std::string line = m.inputModel.table[i];
        if (line.find("Summary") != std::string::npos) {
            break;
        }
        data.m.table.push_back(line);
    }
    //Second part of table
    data.m.table.push_back("   Summary statistics:\n");
    data.m.table.push_back("-------------------------------------------------------------\n");
    uvec auxx = find_finite(data.m.comp.col(0));
    if (auxx.n_elem < 5){
        data.m.table.push_back("  All innovations are NaN!!\n");
    } else {
        outputTable(data.m.comp.submat(0, 0, data.m.y.n_elem - 1, 0), data.m.table);
    }
    data.m.table.push_back("-------------------------------------------------------------\n");
    string firstLine = data.m.table[1];
    data.m.table[1].replace(7, 1, " TOBIT T");
    if (data.m.verbose){
        for (unsigned int i = 0; i < data.m.table.size(); i++){
            printf("%s ", data.m.table[i].c_str());
        }
    }
    m.inputModel.verbose = VERBOSE;
}
// Identification
void TETSclass::ident(bool verbose){
    wall_clock timer;
    timer.tic();
    // Error variance
    vec evar(1);
    evar(0) = data.m.p0(data.m.p0.n_elem - 1);
    // Finding models
    vector<string> allModels;
    string error, trend, seasonal;
    if (data.m.error == "?"){
        // if (data.m.negative)
            error = "A";
        // else
        //     error = "A";
    } else {
        error = data.m.error;
    }
    if (data.m.trend == "?"){
        // if (data.m.negative)
            trend = "N/A/Ad";
        // else
        //     trend = "N/A/Ad/M/Md";
    } else {
        trend = data.m.trend;
    }
    if (data.m.seasonal == "?"){
        // if (data.m.negative)
            seasonal = "N/A";
        // else
        //     seasonal = "N/A/M";
    } else {
        seasonal = data.m.seasonal;
    }
    findModels(error, trend, seasonal, data.m.identAll, allModels);
    // output if verbose
    if (verbose){
        if (data.m.missing.n_elem > 0){
            printf("--------------------------------------------------------\n");
            printf("   Identification with %1i missing data.\n", (int)data.m.missing.n_elem);
        }
        printf("--------------------------------------------------------\n");
        printf("    Model            AIC           BIC          AICc\n");
        printf("--------------------------------------------------------\n");
    }
    // Estimation loop
    //    bool ARMAESTIM = inputModel.armaIdent;
    //    inputModel.armaIdent = false;
    setModel(data.m, allModels[0], data.m.userS);
    data.m.p0 = join_vert(data.m.p0, evar);
    TETSclass m1(data.m, data.Ymin, data.Ymax);
    m1.estim(false);
    if (verbose){
        printf("  %*s: %13.4f %13.4f %13.4f\n", 8, prettyModel(allModels[0]).c_str(),
               m1.data.m.criteria(1), m1.data.m.criteria(2), m1.data.m.criteria(3));
    }
    data.m = m1.data.m;
    uword crit;
    if (data.m.criterion == "aicc")
        crit = 3;
    else if (data.m.criterion == "aic")
        crit = 1;
    else
        crit = 2;
    for (uword i = 1; i < allModels.size(); i++){
        setModel(m1.data.m, allModels[i], data.m.userS);
        //cout << "model 1286: " << prettyModel(allModels[i]).c_str() << endl;
        m1.data.m.p0 = join_vert(m1.data.m.p0, evar);
        m1.estim(false);
        if (verbose){
            printf("  %*s: %13.4f %13.4f %13.4f\n", 8, prettyModel(allModels[i]).c_str(),
                   m1.data.m.criteria(1), m1.data.m.criteria(2), m1.data.m.criteria(3));
        }
        if (m1.data.m.criteria[crit] < data.m.criteria[crit])
            data.m = m1.data.m;
    }
   if (verbose){
        double nSeconds = timer.toc();
        printf("--------------------------------------------------------\n");
        printf("  Identification time: %10.5f seconds\n", nSeconds);
        printf("--------------------------------------------------------\n");
    }
    //    inputModel.armaIdent = ARMAESTIM;
}
// Components estimation
void TETSclass::components(){
    // Estimated components (error, fit, level, seasonal, slope, exogenous, arma)
    // Positioning alpha, beta, gamma and phi
    ETSmodel* m = &data.m;
    etsMatrices(m, m->p);
    uword n = m->y.n_elem, posSeas = 0;
    bool seas = (m->s > 1), slope = (m->trend != "N"), arma = (sum(m->arma) > 0);
    vec x = m->x0;
    bool nu = (data.m.u.n_rows > 0);
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
    }
    if (arma){
        string ar = to_string((int)m->arma(0)),
            ma = to_string((int)m->arma(1));
        snprintf(name, 15, "/ARMA(%s,%s)", ar.c_str(), ma.c_str());
        m->compNames += name;
    }
    // inputs?
    if (nu)
        fitu = repmat(data.m.d.t(), 1, m->u.n_cols) % data.m.u;
    else {
        fitu.set_size(1, data.m.y.n_elem + data.m.h);
        fitu.fill(0);
    }
    // Initializing
    m->comp.set_size(n + m->h, 3 + seas + slope + m->u.n_rows + arma);
    m->comp.fill(datum::nan);
    m->comp.submat(0, 1, n - 1, 1) = m->y;
    if (seas)
        posSeas = slope + 1;
    double sigma = sqrt(exp(2 * data.m.p(data.m.p.n_elem - 1))),
           cdfMin, cdfMax;
    vec future(data.m.h); future.fill(datum::nan);
    vec y = join_vert(m->y, future), Fx, aux(m->h, fill::zeros),
        Ymin = join_vert(data.Ymin, aux),
        Ymax = join_vert(data.Ymax, aux);
    if (m->modelType == 0){
        vec e(1), Za(1);
        int ind = m->ns(0) + m->ns(1);
        // double lMin, lMax, cdfMin, cdfMax, pUn, pdfMin, pdfMax,
        //     lt, ct, ctpUn, yhat;
        // Linear model
        for (uword t = 0; t < n + m->h; t++){
            // if (t > 143){
            //     cout << "t423: " << t << endl;
            // }
            Za = m->w * x + sum(fitu.col(t));
            Fx = m->F * x;
            oneStep(y(t), Za, Fx, m->g, sigma, Ymin(t),
                    Ymax(t), t < n, e, x, cdfMin, cdfMax);
            // if (isfinite(y(t)) && t < n){
                // if (t >142){
                //     cout << "t400: " << t << endl;
                // }
            //     lMin = (data.Ymin(t) - Za(0)) / sigma;
            //     lMax = (data.Ymax(t) - Za(0)) / sigma;
            //     cdfMax = 1 - normcdf(lMax);
            //     cdfMin = normcdf(lMin);
            //     pUn = 1 - cdfMax - cdfMin;
            //     pdfMin = normpdf(lMin);
            //     pdfMax = normpdf(lMax);
            //     if (pUn < 1e-5){
            //         lt = 0;
            //         ctpUn = 0;
            //         yhat = 0;
            //     } else {
            //         lt = (pdfMax - pdfMin) / pUn;  // Inversa del ratio de Mills
            //         yhat = pUn * (Za(0) - sigma * lt);
            //     }
            //     if (!isfinite(lMin) && isfinite(lMax)){
            //         ct = -lMax * pdfMax;
            //         yhat = yhat + cdfMax * data.Ymax(t);
            //     } else if (!isfinite(lMax) && isfinite(lMin)){
            //         ct = lMin * pdfMin;
            //         yhat = yhat + cdfMin * data.Ymin(t);
            //     } else if (!isfinite(lMax) && !isfinite(lMin)){
            //         ct = 0;
            //     } else {
            //         ct = lMin * pdfMin - lMax * pdfMax;
            //         yhat = yhat + cdfMin * data.Ymin(t) + cdfMax * data.Ymax(t);
            //     }
            //     if (pUn >= 1e-5)
            //         ctpUn = ct / pUn;
            // // if (isfinite(y(t))){
            //     e(0) = y(t) - yhat;
            //     x = m->F * x + (pUn / (1 + ctpUn - lt * lt)) * m->g * e;
            //     if (y(t) == data.Ymax(t) || y(t) == data.Ymin(t))
            //         e(0) = datum::nan;
            // } else {
            //     e(0) = datum::nan;
            //     x = m->F * x;
            // }
            // Storing information
            m->comp(t, 0) = e(0);
            m->comp(t, 1) = Za(0);
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
    }
    if (m->missing.n_elem > 0){
        uvec cero(1);
        cero.fill(0.0);
        m->comp(find_nonfinite(y), cero).fill(datum::nan);
    }
}
// Forecast function
void TETSclass::forecast(){
    ETSclass m1(data.m);
    m1.forecast();
    if (data.m.bootstrap)
        data.m.ySimul = m1.inputModel.ySimul;
    data.m.yFor = m1.inputModel.yFor;
    data.m.yForV = m1.inputModel.yForV;
}
// Main function
void TETS(vec y, mat u, string model, int s, int h,
          bool verbose, string criterion, bool identAll,
          rowvec alphaL, rowvec betaL, rowvec gammaL, rowvec phiL,
          string parConstraints, bool forIntervals, bool bootstrap,
          int nSimul, vec arma, bool armaIdent, vec p0, double lambda,
          vec Ymax, vec Ymin){
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
    // Ymax: Censorship from above
    // Ymin: Censorship from below

    ETSmodel m1;
    TETSclass m(m1, Ymin, Ymax);
//    ETSmodel input;
//    TETSclass m(input, Ymax, Ymin, false);
    m = preProcess(y, u, model, s, h, verbose, criterion, identAll, alphaL, betaL, gammaL, phiL,
                   parConstraints, forIntervals, bootstrap, nSimul, arma, armaIdent, p0, lambda,
                   Ymax, Ymin);
    if (m.data.m.errorExit)
        return;
    // Estimating
    if (m.data.m.error == "?" || m.data.m.trend == "?" || m.data.m.seasonal == "?" || m.data.m.armaIdent)
        m.ident(verbose);
    else {
        m.estim(verbose);
    }
    m.validate();
    m.forecast();
    m.components();
//    m.simulate(24, m.m.xn);

//    postProcess(m.m);
}
// Pre-process of user inputs
TETSclass preProcess(vec y, mat u, string model, int s, int h,
                    bool verbose, string criterion, bool identAll,
                    rowvec alphaL, rowvec betaL, rowvec gammaL, rowvec phiL,
                    string parConstraints, bool forIntervals, bool bootstrap,
                    int nSimul, vec arma, bool armaIdent, vec p0, double lambda,
                    vec Ymax, vec Ymin){
    ETSmodel input_;
    ETSclass m_(input_);
    bool errorExit = false, p0ini = false;
    // checking for correct p0 dimension
    if (p0.n_elem > 0 && p0.n_elem < 4){
        printf("%s", "ERROR: p0 should have 4 or 5 elements (alpha/phi/beta/gamma/sigma2)!!!\n");
        errorExit = true;
    }
    // Checking for multiplicative components
    uword ml = model.length();
    if (ml > 0){
        upper(model);
        for (uword i = 0; i < ml; i++){
            if (model[i] == 'M')
                errorExit = true;
        }
        if (errorExit){
            printf("%s", "ERROR: multiplicative components not allowed!!!\n");
        }
    }
    TETSmodel input;
    double p0sigma = 0.5;
    if (!errorExit){
        vec p0pre = p0;
        if (p0.n_elem == 5){
            p0ini = true;
            p0sigma = p0(p0.n_elem - 1);
            p0pre = p0.rows(0, p0.n_elem - 2);
        }
        // end of p0 correction
        m_ = preProcess(y, u, model, s, h, verbose, criterion, identAll, alphaL, betaL, gammaL, phiL,
                        parConstraints, forIntervals, bootstrap, nSimul, arma, armaIdent, p0pre, lambda);
        input.m = m_.inputModel;
        errorExit = input.m.errorExit;
        // Additional checks specific of TETS
        uword n = y.n_elem; // + h;
        if (Ymax.has_nan() && Ymax.n_elem > 1){
            printf("%s", "ERROR: Ymax should contain only valid values!!!\n");
            errorExit = true;
        } else if (Ymax.has_nan()){
            Ymax.resize(n);
            Ymax.fill(datum::inf);
        } else if (Ymax.n_elem == 1){
            double aux = Ymax(0);
            Ymax.resize(n);
            Ymax.fill(aux);
        // } else if (Ymax.n_elem != n){
        //     printf("%s", "ERROR: Ymax should include future values!!!\n");
        //     errorExit = true;
        } else if (Ymax.n_rows != y.n_rows) {
            printf("%s", "ERROR: Ymax size should be the same as time series!!!\n");
            errorExit = true;
        }
        if (Ymin.has_nan() && Ymin.n_elem > 1){
            printf("%s", "ERROR: Ymin should contain only valid values!!!\n");
            errorExit = true;
        } else if (Ymin.has_nan()){
            Ymin.resize(n);
            Ymin.fill(-datum::inf);
        } else if (Ymin.n_elem == 1){
            double aux = Ymin(0);
            Ymin.resize(n);
            Ymin.fill(aux);
        // } else if (Ymin.n_elem != n){
        //     printf("%s", "ERROR: Ymin should include future values!!!\n");
        //     errorExit = true;
        } else if (Ymin.n_rows != y.n_rows) {
            printf("%s", "ERROR: Ymin size should be the same as time series!!!\n");
            errorExit = true;
        }
        if (any(Ymax <= Ymin)){
            printf("%s", "ERROR: Ymax, Ymin or both incorrect!!!\n");
            errorExit = true;
        }
    }
    // Creating model
    TETSclass m(input.m, Ymin, Ymax);
    m.data.Ymax = Ymax;
    m.data.Ymin = Ymin;
    m.data.m.errorExit = errorExit;
    // Correction for censoring in case of interpolated ETS
    if (input.m.missing.n_elem > 0){
        uvec ind = input.m.missing.rows(find(input.m.y.rows(input.m.missing) > Ymax(input.m.missing)));
        input.m.y.rows(ind) = Ymax(ind);
        ind = input.m.missing.rows(find(input.m.y.rows(input.m.missing) < Ymin(input.m.missing)));
        input.m.y.rows(ind) = Ymin(ind);
    }
    // Adding error variance as additional parameter at the end of vector
    vec evar(1);
    if (p0ini){
        // m.data.m.p0.insert_rows(m.data.m.p0.n_elem, log(p0sigma) / 2);
        evar(0) = p0sigma;
        m.data.m.p0 = join_vert(m.data.m.p0, log(evar) / 2);
    } else {
        vec dy = diff(m.data.m.y);
        dy = dy.rows(s, dy.n_elem - 1) - dy.rows(0, dy.n_elem - s - 1);
        evar(0) = var(dy.elem(find_finite(dy)));
        // m.data.m.p0.insert_rows(m.data.m.p0.n_elem, log(p0sigma) / 2);
        m.data.m.p0 = join_vert(m.data.m.p0, log(evar) / 2);
    }
    return m;
}
// Loglik computation
double llikTETS(vec& p, void* opt_data){
    // Converting void* to ETSmodel*
    TETSmodel* aux = (TETSmodel*)opt_data;
    vec Ymin = aux->Ymin;
    vec Ymax = aux->Ymax;
    ETSmodel* m = &aux->m;
    // Positioning alpha, beta, gamma and phi
    // etsMatrices(&*(ETSmodel*)&m, p.head(p.n_elem - 1));
    etsMatrices(m, p.head(p.n_elem - 1));
    double sigma = sqrt(exp(2 * p(p.n_elem - 1)));
    int n = m->y.n_elem;
    vec x = m->x0, a(1);
    a(0) = datum::nan;
    // int ns = m->x0.n_elem - 1;
    double obj = 0.0;
    rowvec fitu(m->y.n_elem + m->h);
    bool nu = (m->u.n_rows > 0);
    if (nu)
        fitu = m->d * m->u;
    else
        fitu.fill(0.0);
    if (m->modelType == 0){
        vec e(1), Za(1), Fx, llik(m->y.n_elem, fill::value(1.0));
        double ehat, cdfMin, cdfMax;
        // double lMin, lMax, cdfMin, cdfMax, pUn, pdfMin, pdfMax,
        //        lt, ct, ctpUn, yhat, ehat;
        // Linear model
        for (int t = 0; t < n; t++){
            Za = m->w * x + sum(fitu.col(t));
            Fx = m->F * x;
            oneStep(m->y(t), Za, Fx, m->g, sigma, Ymin(t),
                    Ymax(t), true, e, x, cdfMin, cdfMax);
            // if (!isfinite(m->y(t))){
            //     x *= m->F;
            // }
            // lMin = (Ymin(t) - Za(0)) / sigma;
            // lMax = (Ymax(t) - Za(0)) / sigma;
            // cdfMax = 1 - normcdf(lMax);
            // cdfMin = normcdf(lMin);
            // pUn = 1 - cdfMax - cdfMin;
            // pdfMin = normpdf(lMin);
            // pdfMax = normpdf(lMax);
            // if (pUn < 1e-5){
            //     lt = 0;
            //     ctpUn = 0;
            //     yhat = 0;
            // } else {
            //     lt = (pdfMax - pdfMin) / pUn;  // Inversa del ratio de Mills
            //     yhat = pUn * (Za(0) - sigma * lt);
            // }
            // if (!isfinite(lMin) && isfinite(lMax)){
            //     ct = -lMax * pdfMax;
            //     yhat = yhat + cdfMax * Ymax(t);
            // } else if (!isfinite(lMax) && isfinite(lMin)){
            //     ct = lMin * pdfMin;
            //     yhat = yhat + cdfMin * Ymin(t);
            // } else if (!isfinite(lMax) && !isfinite(lMin)){
            //     ct = 0;
            // } else {
            //     ct = lMin * pdfMin - lMax * pdfMax;
            //     yhat = yhat + cdfMin * Ymin(t) + cdfMax * Ymax(t);
            // }
            // if (pUn >= 1e-5)
            //     ctpUn = ct / pUn;
            // if (isfinite(m->y(t))){
            //     e(0) = m->y(t) - yhat;
            //     x = m->F * x + pUn / (1 + ctpUn - lt * lt) * m->g * e;
            // }
            // Log-likelihood
            // llikt = 1.0;
            if (m->y(t) <= Ymin(t)){
                llik(t) = cdfMin;
            } else if (m->y(t) >= Ymax(t)){
                llik(t) = cdfMax;
            } else {
                ehat = (m->y(t) - Za(0)) / sigma;
                llik(t) = 1.0 / sigma * normpdf(ehat);
            }
        }
        uvec ll = find(llik > 0);
        obj = -sum(log(llik(ll))) / m->y.n_elem;
    }
    if (!x.has_nan())
        m->xn = x;
    return obj;
}
// Gradient of Loglik
vec gradTETS(vec& p, void* opt_data, double& obj, int& nFuns){
    // Converting void* to ETSmodel*
    TETSmodel* aux = (TETSmodel*)opt_data;
    ETSmodel* m = &aux->m;
    // // Positioning alpha, beta, gamma and phi
    obj = 0.0;
    // int nPar = sum(m->nPar);
    int nPar = p.n_elem;
    nFuns = 0;
    vec grad(nPar, fill::zeros);
    // Numerical gradient
    vec p0, F1 = p;
    obj = llikTETS(p, opt_data);
    for (int i = 0; i < nPar; i++){
        p0 = p;
        p0(i) += 1e-8;
        F1(i) = llikTETS(p0, opt_data);
    }
    grad = (F1 - obj) / 1e-8;
    nFuns += nPar + 1;
    m->grad = grad;
    return grad;
}
// One step agead prediction
void oneStep(double yt, vec& Za, vec& Fx, vec& g, double sigma, double ymin, double ymax,
             bool tln, vec& e, vec& x, double& cdfMin, double&cdfMax){
    e.resize(1);
    if (isfinite(yt) && tln){
        // if (t >142){
        //     cout << "t400: " << t << endl;
        // }
        double lMin, lMax, pUn, pdfMin, pdfMax,
            lt, ct, ctpUn, yhat;
        lMin = (ymin - Za(0)) / sigma;
        lMax = (ymax - Za(0)) / sigma;
        cdfMax = 1 - normcdf(lMax);
        cdfMin = normcdf(lMin);
        pUn = 1 - cdfMax - cdfMin;
        pdfMin = normpdf(lMin);
        pdfMax = normpdf(lMax);
        if (pUn < 1e-5){
            lt = 0;
            ctpUn = 0;
            yhat = 0;
        } else {
            lt = (pdfMax - pdfMin) / pUn;  // Inversa del ratio de Mills
            yhat = pUn * (Za(0) - sigma * lt);
        }
        if (!isfinite(lMin) && isfinite(lMax)){
            ct = -lMax * pdfMax;
            yhat = yhat + cdfMax * ymax;
        } else if (!isfinite(lMax) && isfinite(lMin)){
            ct = lMin * pdfMin;
            yhat = yhat + cdfMin * ymin;
        } else if (!isfinite(lMax) && !isfinite(lMin)){
            ct = 0;
        } else {
            ct = lMin * pdfMin - lMax * pdfMax;
            yhat = yhat + cdfMin * ymin + cdfMax * ymax;
        }
        if (pUn >= 1e-5)
            ctpUn = ct / pUn;
        // if (isfinite(y(t))){
        e(0) = yt - yhat;
        x = Fx + (pUn / (1 + ctpUn - lt * lt)) * g * e;
        if (yt == ymax || yt == ymin)
            e(0) = datum::nan;
    } else {
        e(0) = datum::nan;
        x = Fx;
    }
}
