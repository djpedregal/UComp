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
#include "ARIMASSmodel.h"
#include "ARMAmodel.h"

// ARIMA models
struct ARIMAmodel{
    // INPUTS:
    vec y,                // output data
        par,              // parameter estimates
        par0,             // initial estimates for parameters
        par0Std,          // standard deviation of par0
        orders;           // model orders (p,d,q)x(P,D,Q)_s
    mat u,                // input data
        ySimul;           // bootstrap simulations
    uword maxP = 3,       // max order regular AR
          maxQ = 3,       // max order regular MA
          maxPs = 1,      // max order seasonal AR
          maxQs = 1,      // max order seasonal MA
          maxD = 2,       // max regular diffs
          maxDs = 1,      // max seasonal diffs
          s = 12;         // seasonal period
    double lambda = 1.0,  // Box-Cox lambda parameter
           cnst = 9999.9; // constant on/off or identified (1/0/9999.9)
    bool verbose = false, // intermediate results on/off
        bootstrap = false, // forecasting intervals bootstrap
        // identFAST = true,    // Identification by ML or GM (fast is GM)
        restrictObs = false; // few observations
    int h = 18,           // Forecasting horizon
        nSimul = 5000;    // simulations bootstrap
    string criterion = "bic";
    // OUTPUTS:
    double BIC = 1e10,   // BIC of estimated model
           AIC = 1e10,
           AICc = 1e10,
           IC = 1e10;
    uword p = 0,         // identified regular AR order
          q = 0,         // identified regular MA order
          ps = 0,        // identified seasonal AR order
          qs = 0,        // identified seasonal MA order
          d = 0,         // identified regular diffs
          ds = 0;        // identified seasonal diffs
    vec yFor, FFor,      // forecasts and variance
        yh,              // interpolated y in ind missing data
        v,               // innovations
        a,               // noise from longAR
        xn,              // end state vector for forecasting and simulations
        betaAug,         // regression parameters
        betaAugVar;      // varaince of regression parameters
    uvec ind;            // missing value indices
    vector<string> table;           // output table from evaluate()
    bool errorExit = false, // errors in user inputs
        pureRegression = false,  // Just a regression
        tooFew = false,  // Too few observations
        IDENT = true,    // Identification on/off
        IDENTd = true;   // identification of differences on/off
    REGmodel mr;         // regression model
    mat covBeta,         // covariance matrix of beta in regression
        models;          // identified models (BIC, p, d, q, ps, ds, qs)
};

/**************************
 * Model CLASS ARIMA
 ***************************/
class ARIMAclass{
    public:
        ARIMAmodel m;
        ARIMASS mSS;
        SSinputs inputsSS;
        // ARIMAclass(ARIMAmodel);
        ARIMAclass(ARIMAmodel m){
            this->m = m;
        }
        void print();
        // void interpolate();
        void findDiff();
        void identGM();
        // void identIS();
        void estim(bool);
        void filter();
        void forecast();
        void validate();
};

/****************************************************
// ARIMA functions declarations
****************************************************/
// ARIMA main function
void ARIMA(vec, mat, vec, double, uword, int, bool, double, vec, bool, int, string);
// void ARIMA(vec, mat, uword, vec, vec, double, bool, int);
// preprocess inputs to ARIMA function
ARIMAclass preProcess(vec, mat, vec, double, uword, int,
                      bool, double, vec, bool, int, string);
// Hannan Risannen estimation of ARMA models
void HR(vec, mat, vec, uword, uword, uword, uword,
        uword, uword, double&, vec&, vec&, double&,
        uvec&, uvec&, string);
// Estimation of autocovariances
vec acov(vec, uword);
// Calculate N, a and X for HR procedure
void NaXHR(vec, uword, uword, uword&, vec&);
// Estimation of ARIMA models
void estimModel(ARIMAclass*, uword, uword, uword, uword, double&, vec&, vec&, double&);
// Interpolation with ARIMA(0,1,1)x(0,1,1)s
void interpol(vec&, uword);
// Estimation of a long AR model from autocovariances (it includes leading 1)
void longAR(vec, uword, uword, uword, vec&, vec&, uword&);
// Filtering time series
vec filter(vec, vec, vec);
// Differences polynomial
vec polyDiff(vec, vec);
// Differencing time series
vec diff(vec, vec, vec);
// Integrating constants
mat vIntConst(uword, uword, uword, uword);
// Differencing time series
vec vDiff(vec, uword, uword, uword);
mat vDiff(mat, uword, uword, uword);
// Parameter names
void parNames(uword, uword, uword, uword, uword, int, double, vector<string>&);
// Regression with nice output
void regressTable(vec, mat, vec, vec, vec, double, double, double, vector<string>&);
// Testing for stationarity and invertibility
// void testSI(vec&, uword, uword, uword, uword);
// Checking whether a polynomial has a unit root
bool unitRoot(vec, double);
// Variance of data within b standard deviations
double varNaN(vec, float);
// Join six uword values in a rowvec
void join(mat&, uword&, double, uword, uword, uword, uword, uword, uword);
/****************************************************
// ARIMA functions implementations
****************************************************/
// ARIMA main function
void ARIMA(vec y, mat u, vec orders, double cnst, uword s, int h,
           bool verbose, double lambda, vec maxOrders, bool bootstrap,
           int nSimul, string criterion){
        // y:       otuput data (one time series)
        // u:       input data (excluding constant)
        // orders:  (p, d, q, P, D, Q)
        // cnst:    constant included or not (as a drift if model with differences)
        // s:       seasonal period
        // h:       forecasting horizon (if inputs it is recalculated as the length differences
        //          between u and y
        // verbose: shows intermediate results
        // lambda:  lambda for Box-Cox transformation (9999.9 for estimation)
        // maxOrders: maximum ARIMA orders for model identification (as orders)
        ARIMAmodel input;
        ARIMAclass m(input);
        m = preProcess(y, u, orders, cnst, s, h, verbose, lambda,
                       maxOrders, bootstrap, nSimul, criterion);
        // bool IDENT = false;
        // if (sum(orders) == 0 && !m.m.pureRegression && !m.m.tooFew)
        //     IDENT = true;
        // else
        //     maxOrders = orders;  // estimate just ONE model
        if (m.m.errorExit)
            return;
        // if (m.m.IDENT)
        //     m.findDiff();
        // else
        //     maxOrders = orders;  // estimate just ONE model
        // m.identGM();
        m.identGM();

//        if (m.m.u.n_rows > 0){
//            m.m.u.cols(0, 5).print("u 141");
//        }

        m.validate();
        m.forecast();
}
// Constructor
// ARIMAclass::ARIMAclass(ARIMAmodel m){
//     this->m = m;
// }
// Printing model
// void ARIMAclass::print(){
//     cout << "ARIMA(" << m.p << "," << m.d << "," << m.q << ")" <<
//          "x(" << m.ps << "," << m.ds << "," << m.qs << ")" << endl;
//     cout << "Season: " << m.s << endl;
//     cout << "Maximum orders: " << endl;
//     cout << "ARIMA(" << m.maxP << "," << m.maxD << "," << m.maxQ << ")" <<
//          "x(" << m.maxPs << "," << m.maxDs << "," << m.maxQs << ")" << endl;
//     if (m.cnst > 0){
//         cout << "With constant" << endl;
//     } else {
//         cout << "Without constant" << endl;
//     }
//     m.par0.t().print("par0");
// }
// Ident procedure based on HR
void ARIMAclass::findDiff(){
    // Finds differencing order by minimising variance
    // This is not the procedure in TRAMO
    uword d = 0, ds = 0;
    // uvec ind = find_finite(m.y);
    vec y, aux(2);
    y = m.y; // - mean(m.y(find_finite(m.y)));
    // vec y = m.y, aux(2);
    double variance1 = var(y(find_finite(y))), variance0 = variance1, variance2, minObs;
    aux(0) = m.s * 3;
    aux(1) = 8.0;
    minObs = max(aux);
    vec diffs = {0.0, 0.0}, s = {1.0, 0.0};
    s(1) = m.s;
    uword count = 0;
    vec dy;
    bool sigue = true;
    //// Removing inputs effects ////
    if (m.u.n_rows > 0){
        mat u;
        u = m.u.cols(0, m.y.n_elem - 1);
        // u = vDiff(u, m.d, m.ds, m.s);
        // Remove constant variables from inputs
        if (m.cnst == 1)
            u.shed_row(u.n_rows - 1);
    // } else {
    //     u.reset();
    // }
        vec beta, stdBeta, ahat;
        double AIC, BIC, AICc;
    // if (m.cnst != 0 || u.n_rows > 0){
        // Removing u effects, except constant
        mat X(u.n_cols, u.n_rows + (m.cnst > 0), fill::ones);
        // if (u.n_rows > 0)
            X.cols(0, u.n_rows - 1) = u.t();
        //        REGmodel mr;
        // vec stdBeta;
        regress(y, X, beta, stdBeta, ahat, BIC, AIC, AICc);
        // m.betaAug = beta;
        // m.betaAugVar = pow(stdBeta, 2);
        // Removing u effects, except constant
        y = ahat + beta.back() * (m.cnst > 0);
    }
    if (true){  /////////////////////// Este es el bueno!!!!!!!!!!!!!!
        // Finding differences based on variance of differenced series //
        do{
            diffs(0) = d + 1;
            diffs(1) = ds;
            dy = diff(y, diffs, s);
            variance1 = varNaN(dy, 3.8);
            diffs(0) = d;
            if (m.s > 1){
                diffs(1) = ds + 1;
                dy = diff(y, diffs, s);
                variance2 = varNaN(dy, 3.8);
            } else {
                variance2 = variance0 + 1;
            }
            if (variance2 < variance0 && variance2 < variance1){
                variance0 = variance2;
                ds++;
            } else if (variance1 < variance0 && variance1 < variance2){
                variance0 = variance1;
                d++;
            } else {
                sigue = false;
            }
            count++;
        } while (sigue && count < 10 && dy.n_elem > minObs);
    } else if (true){
        double BIC, AIC, AICc, ttest;
        uword lag = 6;
        if (m.s > 4)
            lag = m.s;
        // Augmented Dickey-Fuller tests for regular unit roots
        do{
            ttest = adfTest(y, regspace(1, lag), BIC, AIC, AICc);
            if (ttest > -2){
                d++;
                y = vDiff(y, d, ds, m.s);
            }
        }while (ttest > -2 && d < m.maxD);
        // Maravall tests for seasonal unit roots
        uword bestAR = 1;
        if (m.s > 1){
            bool goOn = false;
            vec a, phi, par0Std, phiaux;
            mat u;
            uword maxPQ = max(m.maxP + m.maxPs * m.s * 1.5, m.maxQ + m.maxQs * m.s * 1.5);
            uvec lagP0, lagQ0;
            double cnst;
            do{
                goOn = false;
                longAR(y, m.s, m.maxP, m.maxQ, a, phiaux, bestAR);
                m.a = a;
                HR(y, u, a, 1, 1, 1, 1, m.s, maxPQ, BIC, phi, par0Std, cnst, lagP0, lagQ0, m.criterion);
                if (m.s > 1 && phi(1) > 0.97 && phi(3) > -0.97 && abs(-phi(1) - phi(3)) > 0.15){
                    ds++;
                    goOn = true;
                    y = vDiff(y, d, ds, m.s);
                }
            }while(goOn && ds < m.maxDs);
        }
        // m.d = d;
        // m.ds = ds;
        // return;
    } else if (false){
        // ADF Test with variance differences
        vec dy;
        double BIC, AIC, AICc, ttest;
        uword lag = 6;
        if (m.s > 4)
            lag = m.s;
        // Augmented Dickey-Fuller tests for regular unit roots
        bool goOn = false;
        // double meany;
        do{
            // meany = mean(y.elem(find_finite(y)));
            goOn = false;
            ttest = adfTest(y, regspace(1, lag), BIC, AIC, AICc);
            // y += meany;
            if (ttest > -2){
                dy = vDiff(y, d + 1, ds, m.s);
                variance1 = var(dy);
                if (variance1 < variance0 || d == 0){ // variance criterion
                    d++;
                    y = dy;
                    variance0 = variance1;
                    goOn = true;
                }
            }
        }while (goOn && d < m.maxD);
        // Variance of differenced series
        if (m.s > 1){
            do{
                y = vDiff(y, 0, ds + 1, m.s);
                variance1 = var(y);
                if (variance1 < variance0){
                    ds++;
                }
            }while(variance1 < variance0 && ds < m.maxDs);
        }
    } else if (true){
        // Difference orders by regression in Maravall pp. 178ss
        mat u;
        vec dy;
        cx_vec sqrtAR;
        vec a, phi, par0Std, beta, stdBeta, modAR(2);
        uword ps = 1, qs = 1, maxPQ = max(m.maxP + m.maxPs * m.s * 1.5, m.maxQ + m.maxQs * m.s * 1.5);
        double BIC, cnst;
        uvec lagP0, lagQ0;
        if (m.s < 2){
            ps = 0;
            qs = 0;
        }
        // First model
        HR(y, u, a, 2, 0, ps, 0, m.s, maxPQ, BIC, phi, par0Std, cnst, lagP0, lagQ0, m.criterion);
        // phi.t().print("phi inicial 336");
        d = ds = 0;
        sqrtAR = sqrt(phi(0) * phi(0) + 4 * phi(1));
        modAR(0) = abs((phi(0) + sqrtAR(0)) * 0.5);
        modAR(1) = abs((phi(0) - sqrtAR(0)) * 0.5);
        if (any(modAR > 0.97)){
            d++;
        }
        if (m.s > 1 && abs(phi(2)) > 0.97){
            ds++;
        }
        // Estimating a
        if (d + ds == 0){
            m.d = m.ds = 0;
            m.a = y;
            return;
        }
        // Subsequent models
        bool goOn = false;
        vec phiaux;
        uword bestAR;
        do{
            goOn = false;
            diffs(0) = d;
            diffs(1) = ds;
            y = diff(y, diffs, s);
            longAR(y, m.s, m.maxP, m.maxQ, a, phiaux, bestAR);
            m.a = a;
            HR(y, u, a, 1, 1, ps, qs, m.s, maxPQ, BIC, phi, par0Std, cnst, lagP0, lagQ0, m.criterion);
            // phi.t().print("phi siguiente 372");
            // a.t().print("a370");
            if (phi(0) > 0.97 && phi(2) > -0.97 && abs(-phi(0) - phi(2)) > 0.15){
                d++;
                goOn = true;
            }
            if (m.s > 1 && phi(1) > 0.97 && phi(3) > -0.97 && abs(-phi(1) - phi(3)) > 0.15){
                ds++;
                goOn = true;
            }
        }while(goOn && d < m.maxD && ds < m.maxDs);
    } else if (false){
        // Finding differences based on variance of differenced series
        // and ADF tests for d>1
        double BIC, AIC, AICc, ttest;
        vec aux(2); aux(0) = 6; aux(1) = y.n_elem - 12;
        uword lag = min(aux);
        if (lag < 0)
            lag = 1;
        bool ddiff = false;
        if (m.s > 4)
            lag = m.s;
        do{
            ddiff = false;
            diffs(1) = ds;
            if (d < 1){
                diffs(0) = d + 1;
                dy = diff(y, diffs, s);
                variance1 = varNaN(dy, 3.8);
            } else {
                diffs(0) = d;
                dy = diff(y, diffs, s);
                ttest = adfTest(dy, regspace(1, lag), BIC, AIC, AICc);
                if (ttest > -2){
                    ddiff = true;
                }
            }
            diffs(0) = d;
            if (m.s > 1){
                diffs(1) = ds + 1;
                dy = diff(y, diffs, s);
                variance2 = varNaN(dy, 3.8);
            } else {
                variance2 = variance0 + 1;
            }
            if (variance2 < variance0 && variance2 < variance1){
                variance0 = variance2;
                ds++;
            } else if (ddiff){
                d++;
            } else if (variance1 < variance0 && variance1 < variance2){
                variance0 = variance1;
                d++;
            } else {
                sigue = false;
            }
            count++;
        } while (sigue && count < 10 && dy.n_elem > minObs);
    }
    m.d = min(d, m.maxD);
    m.ds = min(ds, m.maxDs);
}
// Ident procedure based on HR
void ARIMAclass::identGM(){
    // Gómez, V., Maravall, A. (2001), Automatic Modelling Methods for Univariate Series,
    //     Chapter 7 in Peña, D., Tiao, G.C., Tsay, R.S., A Course in Time Series Analysis,
    //     John Willey & Sons.
    // and Sventunkov, I -> Adam procedure
    // Differencing
    // ARIMAclass mCOPY(m);
    if (m.tooFew){
        // printf("Too few data 904\n");
        return;
    }
    double CNSTuser = m.cnst;
    uword ibest = 0;
    vec diffs = {0.0, 0.0}, s = {1.0, 0.0};
    diffs(0) = m.d;
    diffs(1) = m.ds;
    s(1) = m.s;
    vec y;
    if (m.ind.n_rows > 0)    // Missing values
        m.y.rows(m.ind) = m.yh;
    if (m.IDENTd){
        findDiff();
    }
    y = vDiff(m.y, m.d, m.ds, m.s);
    m.y.rows(m.ind).fill(datum::nan);
    mat u;
    // Initial estimates of exogenous, including constant
    // exogenous are eliminated from output
    if (m.u.n_cols > 0){
        u = m.u.cols(0, m.y.n_elem - 1);
        u = vDiff(u, m.d, m.ds, m.s);
        // Remove constant variables from inputs
        if (m.cnst == 1)
            u.shed_row(u.n_rows - 1);
    } else {
        u.reset();
    }
    vec beta, stdBeta, ahat;
    double AIC, BIC, AICc;
    if (m.cnst != 0 || u.n_rows > 0){
        // Removing u effects, except constant
        mat X(y.n_elem, u.n_rows + (m.cnst > 0), fill::ones);
        if (u.n_rows > 0)
            X.cols(0, u.n_rows - 1) = u.t();
        //        REGmodel mr;
        vec stdBeta;
        regress(y, X, beta, stdBeta, ahat, BIC, AIC, AICc);
        // stdBeta = sqrt(covBeta.diag());
        m.betaAug = beta;
        m.betaAugVar = pow(stdBeta, 2);
        // m.covBeta = covBeta;
        // Removing u effects, except constant
        y = ahat + beta.back() * (m.cnst > 0);
    }
    u.reset();
    // Rest
    uword p = 0, q = 0, ps = 0, qs = 0, bestAR = 1;
    vec aux(3), a, par0, par0Std, phiaux;
    double cnst = m.cnst;
    uvec lagP0, lagQ0, lagP, lagQ;
    if (m.maxQ + m.maxQs > 0){    // && m.identFAST){
        longAR(y, m.s, m.maxP, m.maxQ, a, phiaux, bestAR);
    }
    double BICbest = 1e10; //, BIC;
    uword iP, iQ, maxPQ;
    mat BICmat(5, 35, fill::zeros);
    uword ind;
    maxPQ = max(m.maxP + m.maxPs * m.s * 1.5, m.maxQ + m.maxQs * m.s * 1.5);
    // if (m.tooFew)
    //     maxPQ = 0;
    // else
    //     maxPQ = 1;
    BICmat.col(0).fill(1e10);
    // Selecting seasonal order with regular AR(3) (p. 184)
    p = 3;
    q = 0;
    BICmat.col(1).fill(p);
    BICmat.col(2).fill(q);
    if (m.IDENT){   // ident among many models
        // Searching for the seasonal model
        p = q = ps = qs = 0;
        HR(y, u, a, 0, 0, 0, 0, m.s, maxPQ, BIC, par0, par0Std, cnst, lagP0, lagQ0, m.criterion);
        if (BIC < BICbest){
            BICbest = BIC;
            BICmat(0, 0) = BIC;
            BICmat(0, 1) = 0;
            BICmat(0, 5) = cnst;
        }
        if (bestAR < 3){
            m.maxQ = m.maxQs = 0;
        }
        uword arOrderTemporal = 3;
        if (arOrderTemporal > m.maxP)
            arOrderTemporal = m.maxP;
        double mmaxP = m.maxP, mmaxQ = m.maxQ, mmaxPs = m.maxPs, mmaxQs = m.maxQs;
        if (m.s > 1){
            for (iP = 0; iP <= mmaxPs; iP++){
                mmaxQs = m.maxQs;
                for (iQ = 0; iQ <= mmaxQs; iQ++){
                    HR(y, u, a, arOrderTemporal, 0, iP, iQ, m.s, maxPQ, BIC, par0, par0Std, cnst, lagP0, lagQ0, m.criterion);
                    if (BIC > BICbest){
                        if(iP == 0 && iQ == 0) {
                            arOrderTemporal = 0;
                        } else {
                            // if (mmaxQs == m.maxQs)   // Pure MA
                            mmaxQs = iQ;
                            // mmaxPs = iP;
                        }
                    } else {
                        BICbest = BIC;
                        ind = BICmat.col(0).index_max();
                        BICmat(ind, 0) = BIC;
                        BICmat(ind, 1) = arOrderTemporal;
                        BICmat(ind, 3) = iP;
                        BICmat(ind, 4) = iQ;
                        ps = iP;
                        qs = iQ;
                        BICmat(ind, 5) = cnst;
                        BICmat(ind, 6) = par0.n_elem;
                        if (par0.n_elem > 0){
                            BICmat(ind, span(7, 7 + par0.n_elem - 1)) = par0.t();
                            BICmat(ind, span(7 + par0.n_elem, 7 + 2 * par0.n_elem - 1)) = par0Std.t();
                        }
                    }
                }
            }
        }
        // Selecting regular order for best seasonal in previous
        //        BICmat.cols(0,8).print("BICmat");

        // a.t().print("a656");

        for (iP = 0; iP <= mmaxP; iP++){
            mmaxQ = m.maxQ;
            for (iQ = 0; iQ <= mmaxQ; iQ++){
                // if (m.identFAST){
                HR(y, u, a, iP, iQ, ps, qs, m.s, maxPQ, BIC, par0, par0Std, cnst, lagP0, lagQ0, m.criterion);
                if (BIC > BICbest){
                    // if (mmaxQ == m.maxQ)   // Pure MA
                    mmaxQ = iQ;
                    // mmaxP = iP;
                } else {
                    BICbest = BIC;
                    ind = BICmat.col(0).index_max();
                    BICmat(ind, 0) = BIC;
                    BICmat(ind, 1) = iP;
                    BICmat(ind, 2) = iQ;
                    BICmat(ind, 3) = ps;
                    BICmat(ind, 4) = qs;
                    p = iP;
                    q = iQ;
                    BICmat(ind, 5) = cnst;
                    BICmat(ind, 6) = par0.n_elem;
                    if (par0.n_elem > 0){
                        BICmat(ind, span(7, 7 + par0.n_elem - 1)) = par0.t();
                        BICmat(ind, span(7 + par0.n_elem, 7 + 2 * par0.n_elem - 1)) = par0Std.t();
                    }
                }
            }
        }
        // Selecting the seasonal order based on previous step
        //        BICmat.cols(0,8).print("BICmat");
        mmaxPs = m.maxPs;
        mmaxQs = m.maxQs;
        if (m.s > 1){
            if (p != arOrderTemporal || q != 0){
                for (iP = 0; iP <= mmaxPs; iP++){
                    mmaxQs = m.maxQs;
                    for (iQ = 0; iQ <= mmaxQs; iQ++){
                        // if (m.identFAST){
                        HR(y, u, a, p, q, iP, iQ, m.s, maxPQ, BIC, par0, par0Std, cnst, lagP0, lagQ0, m.criterion);
                        if (BIC > BICbest){
                            // if (mmaxQs == m.maxQs)   // Pure MA
                            mmaxQs = iQ;
                            // mmaxPs = iP;
                        } else {
                            BICbest = BIC;
                            ind = BICmat.col(0).index_max();
                            BICmat(ind, 0) = BIC;
                            BICmat(ind, 1) = p;
                            BICmat(ind, 2) = q;
                            BICmat(ind, 3) = iP;
                            BICmat(ind, 4) = iQ;
                            ps = iP;
                            qs = iQ;
                            BICmat(ind, 5) = cnst;
                            BICmat(ind, 6) = par0.n_elem;
                            if (par0.n_elem > 0){
                                BICmat(ind, span(7, 7 + par0.n_elem - 1)) = par0.t();
                                BICmat(ind, span(7 + par0.n_elem, 7 + 2 * par0.n_elem - 1)) = par0Std.t();
                            }
                        }
                    }
                }
            }
        }
        // Examining 5 best models
        double tol = 0.05;
        BICmat = BICmat.rows(sort_index(BICmat.col(0)));
        // BICmat.cols(0,8).print("BICmat580");
        for (uword i = 1; i < 5; i++){
            if (BICmat(0, 0) > BICmat(i, 0) - tol && BICmat(i, 4) > 0 && BICmat(0, 4) > BICmat(i, 4)){
                ibest = i;
            }
        }
        m.BIC = BICmat(ibest, 0);
        m.p = BICmat(ibest, 1);
        m.q = BICmat(ibest, 2);
        m.ps = BICmat(ibest, 3);
        m.qs = BICmat(ibest, 4);
        m.cnst = BICmat(ibest, 5);
        uword npar = BICmat(ibest, 6);
        if (npar > 0){
            m.par0 = BICmat(ibest, span(7, 7 + npar - 1)).t();
            m.par0Std = BICmat(ibest, span(7 + npar, 7 + 2 * npar - 1)).t();
        }
    } else {      // initial conditions for just one model
        HR(y, u, a, m.p, m.q, m.ps, m.qs, m.s, maxPQ, BIC, par0, par0Std, cnst, lagP0, lagQ0, m.criterion);
        m.BIC = BIC;
        if (par0.n_elem > 0){
            m.par0 = par0;
            m.par0Std = par0Std;
        }
        m.cnst = cnst;
    }
    m.orders(0) = m.p;
    m.orders(1) = m.d;
    m.orders(2) = m.q;
    m.orders(3) = m.ps;
    m.orders(4) = m.ds;
    m.orders(5) = m.qs;
    if (m.p + m.ps > 0)
        m.par0.rows(0, m.p + m.ps - 1) *= -1;
    // Checking for unsignificant last parameters (changing orders)
    bool IDENT = m.IDENT, IDENTd = m.IDENTd;
    vec AR(m.p), ARS(m.ps), MA(m.q), MAS(m.qs);
    if (IDENT){
        vec ttt = abs(m.par0 / m.par0Std);
        uword ac = 0;
        double lim = 1.1, count;
        bool repeat = false;
        if (m.p > 0){
            count = m.p - 1;
            AR = ttt.rows(0, count);
            ac += m.p;
            if (AR(count) < lim){
                m.p--;
                repeat = true;
            }
            // while (count >= 0 && AR(count) < lim){
            //     m.p--;
            //     count--;
            //     repeat = true;
            // }
        }
        if (m.ps > 0){
            count = m.ps - 1;
            ARS = ttt.rows(ac, ac + count);
            ac += m.ps;
            if (ARS(count) < lim){
                m.ps--;
                repeat = true;
            }
            // while (count >= 0 && ARS(count) < lim){
            //     m.ps--;
            //     count--;
            //     repeat = true;
            // }
        }
        if (m.q > 0){
            count = m.q - 1;
            MA = ttt.rows(ac, ac + count);
            ac += m.q;
            if (MA(count) < lim){
                m.q--;
                repeat = true;
            }
            // while (count >= 0 && MA(count) < lim){
            //     m.q--;
            //     count--;
            //     repeat = true;
            // }
        }
        if (m.qs > 0){
            count = m.qs - 1;
            MAS = ttt.rows(ac, ac + count);
            if (MAS(count) < lim){
                m.qs--;
                repeat = true;
            }
            // while (count >= 0 && MAS(count) < lim){
            //     m.qs--;
            //     count--;
            //     repeat = true;
            // }
        }
        if (repeat){
            m.IDENT = m.IDENTd = false;
            identGM();
            m.IDENT = IDENT;
            m.IDENTd = IDENTd;
        }
    }
    // Checking stationarity and invertibility of initial conditions
    if (m.ps > 0 && IDENT){
        vec AR(m.ps + 1);
        AR(0) = 1.0;
        AR.rows(1, m.ps) = m.par0.rows(m.p, m.p + m.ps - 1);
        if (unitRoot(AR, 0.97)){
            m.ps--;
            m.ds++;
            m.IDENT = false;
            identGM();
            m.IDENT = IDENT;
        }
    }
    if (m.p > 0 && IDENT){
        vec AR(m.p + 1);
        AR(0) = 1.0;
        AR.rows(1, m.p) = m.par0.rows(0, m.p - 1);
        if (unitRoot(AR, 0.97)){
            m.p--;
            m.d++;
            m.IDENT = false;
            identGM();
            m.IDENT = IDENT;
        }
    }
    if (m.qs > 0){
        vec MA(m.qs);
        MA.rows(0, m.qs - 1) = m.par0.rows(m.p + m.ps + m.q, m.p + m.ps + m.q + m.qs - 1);
        maInvert(MA);
        m.par0.rows(m.p + m.ps + m.q, m.p + m.ps + m.q + m.qs - 1) = MA;
    }
    if (m.q > 0){
        vec MA(m.q);
        MA.rows(0, m.q - 1) = m.par0.rows(m.p + m.ps, m.p + m.ps + m.q - 1);
        maInvert(MA);
        m.par0.rows(m.p + m.ps, m.p + m.ps + m.q - 1) = MA;
    }
    if (!m.IDENT){
        if (CNSTuser > 0.0)
            m.cnst = 1.0;
        else
            m.cnst = 0.0;
    }
    // Models with constant
    if (m.cnst == 1 && m.IDENT){    //CNSTuser != 1){
        rowvec Ones = vIntConst(m.y.n_elem + m.h, m.d, m.ds, m.s);
        m.u = join_vert(m.u, Ones);
    }
    // Checking for pure regression
    if (m.p + m.ps + m.q + m.qs == 0 && m.u.n_rows > 0){
        m.pureRegression = true;
    }
}
// Estim ARIMA model by ML
void ARIMAclass::estim(bool validation){
    if (m.tooFew){
        m.bootstrap = false;
        mat u;
        vec nans(m.h), beta, stdBeta, e, s(1), yFor;
        nans.fill(datum::nan);
        vec y = join_vert(m.y, nans);
        s(0) = m.s;
        harmonicRegress(y, u, s, 1, beta, stdBeta, e, yFor);
        e = e.rows(0, m.y.n_rows - 1);
        uvec ind = find_finite(e);
        // Regression model
        REGmodel mr;
        mr.e = e;
        int k = beta.n_elem, n = ind.n_elem;
        vec varE = (e(ind).t() * e(ind)) / (n - k);
        vec nlv = log(varE * (n - k) / n);
        m.AIC = mr.AIC = nlv(0) + 2 * k / n;
        m.BIC = mr.BIC = nlv(0) + k * log(n) / n;
        m.AICc = mr.AICc = (mr.AIC * n + (2 * k * (1 + k)) / (n - k - 1)) / n;
        m.mr = mr;
        m.IC = m.BIC;
        if (m.criterion == "aic")
            m.IC = m.AIC;
        else if (m.criterion == "aicc")
            m.IC = m.AICc;
        // Forecasting
        m.v = e;
        m.yFor = yFor.rows(m.y.n_elem, m.y.n_elem + m.h - 1);
        m.FFor.resize(m.y.n_elem + m.h); m.FFor.fill(datum::nan);
        m.cnst = 1.0;
        m.pureRegression = true;
        return ;
    }
    if (m.pureRegression){
        // m.bootstrap = false;
        vec y = vDiff(m.y, m.d, m.ds, m.s);
        mat u;
        if (m.cnst == 1 && m.u.n_rows == 0){
            // Just white noise
            u.resize(1, m.y.n_rows); u.fill(1.0);
        }
        if (m.u.n_rows > 0){
            // Regression
            u = m.u.cols(0, m.y.n_rows - 1);
            u = vDiff(u, m.d, m.ds, m.s);
        }
//        vector<string> table;
        REGmodel mr;
        regression(y, u.t(), mr);
        m.mr = mr;
        m.v = mr.e;
        m.AIC = mr.AIC;
        m.BIC = mr.BIC;
        m.AICc = mr.AICc;
        m.IC = m.BIC;
        if (m.criterion == "aic")
            m.IC = m.AIC;
        else if (m.criterion == "aicc")
            m.IC = m.AICc;
        return ;
    }
    // Initialising system
//    SSinputs inputsSS;
    unsigned int nPar = m.p + m.ps + m.q + m.qs; // + m.cnst;
    adjustVector(m.par0, nPar, 0);
    // Inputs to class SSpace
    inputsSS.h = m.h;
    inputsSS.verbose = m.verbose;
    inputsSS.cLlik = true;
    inputsSS.exact = false;
    inputsSS.p0 = m.par0;
    inputsSS.p = m.par0;
    inputsSS.nonStationaryTerms = 0;
    // Inputs to class ARIMA
    vec orders(6);
    orders(0) = m.p;
    orders(2) = m.q;
    orders(3) = m.ps;
    orders(5) = m.qs;
    mSS.orders = orders;
    mSS.s = m.s;
    // Differencing with missing values
    if (m.ind.n_rows > 0){
        m.y.rows(m.ind) = m.yh;
        inputsSS.y = vDiff(m.y, m.d, m.ds, m.s);
        m.y.rows(m.ind).fill(datum::nan);
        vec aux = conv_to<vec>::from(m.ind) - m.d - m.ds * m.s;
        aux = aux(find(aux >= 0));
        uvec ind = conv_to<uvec>::from(aux);
        inputsSS.y.rows(ind).fill(datum::nan);
    } else {
        inputsSS.y = vDiff(m.y, m.d, m.ds, m.s);
    }
    if (m.u.n_rows > 0){  // && !validation && !m.pureRegression){
        inputsSS.u = vDiff(m.u, m.d, m.ds, m.s);
    }
    mSS.orders(1) = 0;
    mSS.orders(4) = 0;
    // White noise with or without constant
    bool whiteNoise = (sum(mSS.orders) == 0);
    if (whiteNoise){
        // Regression
        if (m.cnst){
            // ARMA(0, 0) with constant
            inputsSS.betaAug = m.betaAug; //.rows(0, m.u.n_rows - 1);
            inputsSS.betaAugVar = m.betaAugVar; //.rows(0, m.u.n_rows - 1);
        } else {
        }
    }
    // Building system
    ARIMASSclass model(inputsSS, mSS);
    mSS = model.mSS;
    // Operations on system
    if (!whiteNoise){
        model.SSmodel::estim();
        inputsSS = model.getInputs();
        // Checking for non-invertibility models
        if (m.qs > 0){
            vec MA(m.qs);
            MA.rows(0, m.qs - 1) = inputsSS.p.rows(m.p + m.ps + m.q, m.p + m.ps + m.q + m.qs - 1);
            maInvert(MA);
            inputsSS.p.rows(m.p + m.ps + m.q, m.p + m.ps + m.q + m.qs - 1) = MA;
            model.setInputs(inputsSS, mSS);
        }
        if (m.q > 0){
            vec MA(m.q);
            MA.rows(0, m.q - 1) = inputsSS.p.rows(m.p + m.ps, m.p + m.ps + m.q - 1);
            maInvert(MA);
            inputsSS.p.rows(m.p + m.ps, m.p + m.ps + m.q - 1) = MA;
            model.setInputs(inputsSS, mSS);
        }
        m.par = inputsSS.p;
    }
    if (whiteNoise){
        inputsSS = model.getInputs();
        m.par = inputsSS.p;
    }
    if (inputsSS.p.n_elem == 0){   // Model this is just differences or no model
        m.v = inputsSS.y;
        uvec ind = find_finite(m.v);
        double varV = var(m.v(ind), 1);
        inputsSS.innVariance = varV;
        double LLIK, AIC, BIC, AICc;
        vec ppp(1, fill::zeros);
        if (inputsSS.augmented)
            LLIK = llikAug(ppp, &inputsSS);
        else
            LLIK = llik(ppp, &inputsSS);
        LLIK = -0.5 * ind.n_elem * (log(2*datum::pi) + LLIK);
        infoCriteria(LLIK, inputsSS.nonStationaryTerms, m.v.n_elem,
                     AIC, BIC, AICc);
        inputsSS.criteria = {LLIK, AIC, BIC, AICc};
    }
    if (validation){
        model.SSmodel::setInputs(inputsSS);
        model.SSmodel::validate(true, nPar);
        inputsSS = model.getInputs();
        m.v = inputsSS.v * sqrt(inputsSS.innVariance);
    }
    m.AIC = inputsSS.criteria(1);
    m.BIC = inputsSS.criteria(2);
    m.AICc = inputsSS.criteria(3);
    m.IC = m.BIC;
    if (m.criterion == "aic")
        m.IC = m.AIC;
    else if (m.criterion == "aicc")
        m.IC = m.AICc;
    m.orders(1) = m.d;
    m.orders(4) = m.ds;
}
// Filter ARIMA
void ARIMAclass::filter(){
    if (m.pureRegression)
        return ;
    if (m.d > 0 || m.ds > 0){
        mSS.orders(1) = m.d;
        mSS.orders(4) = m.ds;
        inputsSS.y = m.y;
        inputsSS.u = m.u;
        ARIMASSclass model1(inputsSS, mSS);
        inputsSS = model1.getInputs();
    }
    ARIMASSclass model(inputsSS, mSS);
    model.SSmodel::filter();
    inputsSS = model.getInputs();
    m.xn = inputsSS.aEnd;
    m.v = inputsSS.v * sqrt(inputsSS.innVariance);
}
// Forecasts ARIMA
void ARIMAclass::forecast(){
    if (m.tooFew || m.h == 0)
        return ;
    if (m.pureRegression){
        vec poly = {1}, x, dy;
        mat F, du;
        if (m.d + m.ds > 0){
            vec diffs(2), svec(2);
            // differencing u
            mat aux1(m.u.n_rows, m.d + m.ds * m.s, fill::zeros);
            du = join_horiz(aux1, vDiff(m.u, m.d, m.ds, m.s));
            diffs(0) = m.d;
            diffs(1) = m.ds;
            svec(0) = 1;
            svec(1) = m.s;
            poly = polyDiff(diffs, svec);
            F = zeros(poly.n_elem - 1, poly.n_elem - 1);
            x = zeros(poly.n_elem - 1);
            if (poly.n_elem > 1){
                F.col(0) = -poly.rows(1, poly.n_elem - 1);
                if (poly.n_elem > 2){
                    F.submat(0, 1, F.n_rows - 2, F.n_rows - 1).eye();
                }
            }
            // Building state at T
            x(0) = m.y.back();
            x.back() = F.col(0).back() * x(0);
            vec aux(1);
            if (poly.n_elem > 3){
                for (uword i = F.n_rows - 2; i > 0; i--){
                    aux = F.row(i) * x;
                    x(i) = aux(0);
                }
            }
        } else {
            F = zeros(1, 1);
            x = F;
            du = m.u;
        }
        // Forecasting
        m.yFor = zeros(m.h);
        m.FFor = zeros(m.h);
        // vec uf = m.yFor, ufF = m.yFor, aux(1);
        vec uf(m.h), ufF(m.h), aux(1);
        mat XX1 = inv(du * du.t());
        // Forecasting inputs part
        if (m.u.n_rows > 0){
            uf = du.cols(m.y.n_rows, m.y.n_rows + m.h - 1).t() * m.mr.beta;
            // ufF = du.cols(m.y.n_rows, m.y.n_rows + m.h - 1).t() * XX1 * du.cols(m.y.n_rows, m.y.n_rows + m.h - 1);
            for (int i = 0; i < m.h; i++){
                // aux = du.col(m.y.n_rows + i).t() * m.mr.beta;
                // uf(i) = aux(0);
                // aux = m.u.col(m.y.n_rows + i).t() * m.mr.stdBeta * m.u.col(m.y.n_rows + i);
                aux = du.col(m.y.n_rows + i).t() * XX1 * du.col(m.y.n_rows + i);
                ufF(i) = aux(0);
            }
        }
        // Forecasting differences
        double sigma2 = var(m.v(find_finite(m.v)), 1);
        vec g(F.n_rows, fill::zeros);
        g(0) = 1;
        mat Fl = F, Flg = Fl * g;
        // aux.resize(F.n_rows);
        // aux.fill(0.0);
        ufF *= sigma2;
        vec xn = x;
        // Bootstraping
        if (m.bootstrap){
            //int nNan;
            vec x, e = m.v(find_finite(m.v)), //removeNans(inputsSS.v, nNan),
                a(1), ys(m.h), Du(m.h, fill::zeros);
            uvec ind;
            m.ySimul.resize(m.h, m.nSimul);
            // if (m.u.n_rows > 0)
            //     Du = du.cols(m.y.n_rows, m.y.n_rows + m.h - 1).t() * m.mr.beta;
            for (int simul = 0; simul < m.nSimul; simul++){
                x = xn;
                ind = conv_to<uvec>::from(randi(m.h, distr_param(0, e.n_elem - 1)));
                a = e(ind);
                for (int i = 0; i < m.h; i++){
                    x = F * x;
                    x(0) += uf(i) + a(i);
                    ys(i) = x(0);
                }
                m.ySimul.col(simul) = ys;
            }
            m.yFor = mean(m.ySimul, 1);
            m.FFor = var(m.ySimul, 0, 1);
        } else {
            aux(0) = 0.0;
            for (int i = 0; i < m.h; i++){
                x = F * x;
                m.FFor(i) = aux(0) * sigma2 + ufF(i) + sigma2;
                x(0) += uf(i);
                m.yFor(i) = x(0);
                aux += g.t() * Flg * Flg.t() * g;
                Fl *= F;
                Flg = Fl * g;
            }
        }
        return ;
    }
    // Run filter() always before forecast()
    inputsSS.system.Q(0) = inputsSS.innVariance;
    // inputsSS.p = m.par;
    filter();
    rowvec Du(m.h, fill::zeros);
    if (m.u.n_rows > 0){
        Du = inputsSS.system.D * m.u.cols(m.y.n_elem, m.y.n_elem + m.h - 1);
    }
    if (m.bootstrap){
        //int nNan;
        vec x, e = m.v(find_finite(inputsSS.v)), //removeNans(inputsSS.v, nNan),
            a(1), ys(m.h), aux(1);
        uvec ind;
        m.ySimul.resize(m.h, m.nSimul);
        for (int simul = 0; simul < m.nSimul; simul++){
            x = m.xn;
            ind = conv_to<uvec>::from(randi(m.h, distr_param(0, e.n_elem - 1)));
            a = e(ind);
            for (int i = 0; i < m.h; i++){
                aux = inputsSS.system.Z * x + Du.col(i) + a(i);
                ys(i) = aux(0); // + (i == 0) * a(i);
                x = inputsSS.system.T * x + inputsSS.system.R * a(i);
            }
            m.ySimul.col(simul) = ys;
        }
        m.yFor = mean(m.ySimul, 1);
        m.FFor = var(m.ySimul, 0, 1);
    } else {
        vec x = m.xn, aux(1);
        // Variance correction
        inputsSS.innVariance = inputsSS.innVariance * inputsSS.v.n_elem / inputsSS.y.n_elem;
        mat Pt = inputsSS.PEnd * inputsSS.innVariance;
        inputsSS.system.Q *= inputsSS.innVariance;
        m.yFor.resize(m.h);
        m.FFor.resize(m.h);
        mat RQRt = inputsSS.system.R * inputsSS.system.Q * inputsSS.system.R.t();
        vec Ft(1);
        for (int i = 0; i < m.h; i++){
            aux = inputsSS.system.Z * x + Du.col(i);
            m.yFor(i) = aux(0);
            x = inputsSS.system.T * x; // + inputsSS.system.R * a(i);
            Ft = inputsSS.system.Z * Pt * inputsSS.system.Z.t(); // + CHCt;
            Pt = inputsSS.system.T * Pt * inputsSS.system.T.t() + RQRt;
            m.FFor(i) = Ft(0);
        }
    }
}
// Model validation
void ARIMAclass::validate(){
    // First part of table
    bool VERBOSE = m.verbose;
    estim(true);
    if (m.pureRegression){
        inputsSS.betaAug = m.betaAug;
        inputsSS.betaAugVar = m.betaAugVar;
    }
    m.verbose = VERBOSE;
    char str[70];
    m.table.clear();
    m.table.push_back(" -------------------------------------------------------------\n");
    if (m.tooFew){
        snprintf(str, 70, " Too few observations!!!\n");
        m.table.push_back(str);
        if (m.s > 1)
            snprintf(str, 70, " Model: Harmonic regression\n");
        else
            snprintf(str, 70, " Model: Cuadratic interpolation\n");
    } else if (m.s > 1 && m.u.n_rows == 0){
        snprintf(str, 70, " Model: ARIMA(%d,%d,%d)x(%d,%d,%d)\n", (int)m.p, (int)m.d, (int)m.q, (int)m.ps, (int)m.ds, (int)m.qs);
    } else if (m.s > 1 && m.u.n_rows > 0){
        snprintf(str, 70, " Model: ARIMA(%d,%d,%d)x(%d,%d,%d) + exogenous\n", (int)m.p, (int)m.d, (int)m.q, (int)m.ps, (int)m.ds, (int)m.qs);
    } else if (m.s < 2 && m.u.n_rows == 0){
        snprintf(str, 70, " Model: ARIMA(%d,%d,%d)\n", (int)m.p, (int)m.d, (int)m.q);
    } else if (m.s < 2 && m.u.n_rows > 0){
        snprintf(str, 70, " Model: ARIMA(%d,%d,%d) + exogenous\n", (int)m.p, (int)m.d, (int)m.q);
    }
    m.table.push_back(str);
    // Seasonal period
    if (m.s > 1){
        snprintf(str, 70, " Period: %d\n", (int)m.s);
        m.table.push_back(str);
    }
    // Box-Cox lambda
    snprintf(str, 70, " Box-Cox lambda: %3.2f\n", m.lambda);
    m.table.push_back(str);
    // parameter names
    vector<string> names;
    parNames(m.s, m.p, m.q, m.ps, m.qs, m.u.n_rows, m.cnst, names);
    snprintf(str, 70, " %s", inputsSS.estimOk.c_str());
    m.table.push_back(str);
    if (inputsSS.augmented){
        snprintf(str, 70, " (*)  concentrated out parameters\n");
        m.table.push_back(str);
    }
    m.table.push_back("-------------------------------------------------------------\n");
    m.table.push_back("                   Param        S.E.          |T|     |Grad|\n");
    m.table.push_back("-------------------------------------------------------------\n");
    mat tp;
    if (m.u.n_rows == 0){
        tp = join_rows(inputsSS.p, inputsSS.stdP, abs(inputsSS.p / inputsSS.stdP), abs(inputsSS.grad));
    } else {
        uword ind1 = inputsSS.betaAug.n_elem - inputsSS.u.n_rows;
        if (m.pureRegression){
            vec p = inputsSS.betaAug; //.rows(ind1, inputsSS.betaAug.n_rows - 1);
            vec stdP = sqrt(inputsSS.betaAugVar);  //.rows(ind1, inputsSS.betaAug.n_rows - 1));
//            vec fil(inputsSS.u.n_rows, fill::value(-999));
            vec tt = abs(p / stdP), grad(p.n_rows); grad.fill(-100); // = join_vert(abs(inputsSS.grad.rows(0, inputsSS.p.n_rows - 1)), fil);
            tp = join_rows(p, stdP, tt, grad);
        } else {
            vec p = join_vert(inputsSS.p, inputsSS.betaAug.rows(ind1, inputsSS.betaAug.n_rows - 1));
            vec stdP = join_vert(inputsSS.stdP, sqrt(inputsSS.betaAugVar.rows(ind1, inputsSS.betaAug.n_rows - 1)));
            vec fil(inputsSS.u.n_rows, fill::value(-999));
            vec tt = abs(p / stdP), grad = join_vert(abs(inputsSS.grad.rows(0, inputsSS.p.n_rows - 1)), fil);
            tp = join_rows(p, stdP, tt, grad);
        }
    }
    vector<string> col2;
    string chari;
    for (int i = 0; i < (int)tp.n_rows; i++){
        int np = inputsSS.p.n_rows;
        if (i > (np - 1))
            chari = "*";
        else
            chari = " ";
        col2.push_back(chari);
    }
    // Table of numbers
    for (unsigned i = 0; i < tp.n_rows; i++){
        if (abs(tp(i, 0)) > 1e-4 && tp(i, 3) > -1){
            snprintf(str, 70, "%*s: %12.4f%1s %10.4f %12.4f %10.3e\n", 10, names.at(i).c_str(), tp(i, 0), col2.at(i).c_str(), tp(i, 1), tp(i, 2), tp(i, 3));
        } else if (abs(tp(i, 0)) > 1e-4 && tp(i, 3) < -1){
            snprintf(str, 70, "%*s: %12.4f%1s %10.4f %12.4f\n", 10, names.at(i).c_str(), tp(i, 0), col2.at(i).c_str(), tp(i, 1), tp(i, 2));
        } else if (abs(tp(i, 0)) <= 1e-4 && tp(i, 3) > -1){
            snprintf(str, 70, "%*s: %12.3e%1s %10.3e %12.4f %10.3e\n", 10, names.at(i).c_str(), tp(i, 0), col2.at(i).c_str(), tp(i, 1), tp(i, 2), tp(i, 3));
        } else if (abs(tp(i, 0)) <= 1e-4 && tp(i, 3) < -1){
            snprintf(str, 70, "%*s: %12.3e%1s %10.3e %12.4f\n", 10, names.at(i).c_str(), tp(i, 0), col2.at(i).c_str(), tp(i, 1), tp(i, 2));
        } else {
            snprintf(str, 70, "%*s: %12.4f%1s %10.3e %12.4f\n", 10, names.at(i).c_str(), tp(i, 0), col2.at(i).c_str(), tp(i, 1), tp(i, 2));
        }
        m.table.push_back(str);
    }
    for (unsigned i = 5 + tp.n_rows; i < inputsSS.table.size(); i++){
        snprintf(str, 70, "%s", inputsSS.table[i].c_str());
        m.table.push_back(str);
    }
    // Sheer regression
    if (m.pureRegression){
        vector<string> table;
        regressTable(m.mr, table);
        for (unsigned i = 5 + m.mr.beta.n_rows; i < table.size(); i++){
            snprintf(str, 70, "%s", table[i].c_str());
            m.table.push_back(str);
        }
    }
    // Show Table
    if (m.verbose){
        for (unsigned int i = 0; i < m.table.size(); i++){
            // cout << m.table[i].c_str();
            printf("%s ", m.table[i].c_str());
        }
    }
}
// preProcess inputs to ARIMA function
ARIMAclass preProcess(vec y, mat u, vec orders, double cnst, uword s, int h,
                      bool verbose, double lambda, vec maxOrders, bool bootstrap,
                      int nSimul, string criterion){
    bool errorExit = false, IDENT = false;
    // criterion
    lower(criterion);
    if (criterion.compare("bic") != 0 && criterion.compare("aic") != 0 && criterion.compare("aicc") != 0){
        printf("%s", "ERROR: incorrect information criterion!!!\n");
        errorExit = true;
    }
    // NaNs in u
    if (u.n_rows > 0 && u.has_nonfinite()){
        printf("%s", "ERROR: missing values not allowed in input variables!!!\n");
        errorExit = true;
    }
    // orders
    if (orders.n_elem == 0){
        adjustVector(orders, 6, 0.0);
        IDENT = true;
        cnst = 9999.9;  // cnst identification
    }
    // if (!IDENT)
    //     identFAST = false;
    if (sum(orders) > 0 && cnst == 9999.9)
        cnst = 0.0;
    if (sum(orders) == 0 && cnst == 0.0){
        printf("%s", "ERROR: no model to estimate!!!\n");
        errorExit = true;
    }
    if (orders.n_elem < 6){
        adjustVector(orders, 6, 0.0);
    }
    // maxOrders
    if (maxOrders.n_elem < 6){
        adjustVector(maxOrders, 6, 0.0);
    }
    if (sum(maxOrders - orders < 0) > 0){
        printf("%s", "ERROR: maxOrders should be always bigger than orders!!!\n");
        errorExit = true;
    }
    if (s < 2){
        maxOrders.rows(3, 5).fill(0.0);
        orders.rows(3, 5).fill(0.0);
    }
    uword maxObs = max(maxOrders(0) + maxOrders(1) + (maxOrders(3) + maxOrders(4)) * s,
                       maxOrders(2) + maxOrders(5) * s) + 2 * s + 10 * (s == 1);
    uvec ind = find_finite(y);
    bool tooFew = false, pureRegression = false;
    bool restrictObs = false;
    if (ind.n_elem < maxObs){
        restrictObs = true;
        maxOrders.rows(0, 2).fill(1.0);
        maxOrders(1) = 2.0;
        if (s > 1){
            maxOrders.rows(3, 5).fill(1.0);
        }
        maxObs = max(maxOrders(0) + maxOrders(1) + (maxOrders(3) + maxOrders(4)) * s,
                     maxOrders(2) + maxOrders(5) * s) + 2 * s + 2 * (s == 1);
        if (ind.n_elem < 8 || ind.n_elem < maxObs){
            if (ind.n_elem < 8){
                printf("%s", "Error: Too few output observations!!!\n");
                errorExit = true;
            }
            tooFew = true;
            bootstrap = false;
            pureRegression = true;
        }
    }
    if (u.n_rows > u.n_cols){
        u = u.t();
    }
    if (cnst != 0.0 && cnst != 1.0 && cnst != 9999.9){
        printf("%s", "ERROR: cnst input should be \'true\', \'false\' or \'nell\'!!!\n");
        errorExit = true;
    }
    // Remove constant variables from inputs
    if (u.n_rows > 0){
        uvec ind = find(sum(u.t()) == u.n_cols);
        u.shed_rows(ind);
        if (u.n_rows == 0){
            u.reset();
        }
    }
    // Add input for constant if necessary
    if (cnst == 1.0){
        uword nn;
        if (u.n_rows == 0){
            nn = y.n_elem + h;
        } else {
            nn = u.n_cols;
        }
        mat ONES = vIntConst(nn, orders(1),
                             orders(4), s);
        if (u.n_rows == 0){
            u = ONES;
        } else {
            u = join_vert(u, ONES);
        }
    }
    // Correcting h in case there are inputs
    if (u.n_cols > 0){
        h = u.n_cols - y.n_elem;
        if (h < 0){
            printf("%s", "ERROR: Inputs should be at least as long as the ouptut!!!\n");
            errorExit = true;
        }
    }
    // Checking for "outliers"
    // Creating model object
    ARIMAmodel input;
    input.y = y;
    input.u = u;
    input.cnst = cnst;
    input.s = s;
    input.p = orders(0);
    input.d = orders(1);
    input.q = orders(2);
    input.ps = orders(3);
    input.ds = orders(4);
    input.qs = orders(5);
    input.orders = orders;
    input.h = h;
    input.verbose = verbose;
    input.lambda = lambda;
    input.maxP = maxOrders(0);
    input.maxD = maxOrders(1);
    input.maxQ = maxOrders(2);
    input.maxPs = maxOrders(3);
    input.maxDs = maxOrders(4);
    input.maxQs = maxOrders(5);
    input.bootstrap = bootstrap;
    input.nSimul = nSimul;
    // input.identFAST = identFAST;
    input.pureRegression = pureRegression;
    input.tooFew = tooFew;
    input.IDENT = IDENT;
    input.IDENTd = IDENT;
    input.restrictObs = restrictObs;
    // BoxCox transformation
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
    // Creating class
    input.errorExit = errorExit;
    // Missing values
    ind = find_nonfinite(y);
    interpol(y, input.s);
    input.ind = ind;
    input.yh = y.rows(ind);
    ARIMAclass m(input);
//    if (m.missing.n_elem > 0)
//        m.interpolate();
    return m;
}
// Hannan Risannen estimation of ARMA models
void HR(vec y, mat u, vec a, uword p, uword q, uword ps, uword qs,
        uword s, uword maxPQ, double& BIC, vec& beta, vec& stdBeta,
        double& cnst, uvec& lagP, uvec& lagQ, string criterion){
    // Gómez, V., Maravall, A. (2001), Automatic Modelling Methods for Univariate Series,
    //     Chapter 7 in Peña, D., Tiao, G.C., Tsay, R.S., A Course in Time Series Analysis,
    //     John Willey & Sons.
    // y: data
    // p: AR order
    // q: MA order
    // if (a.n_elem < maxPQ + 10){
        // Correction for small samples
    // if (maxPQ == 1)
        maxPQ = max(p + ps * s, q + qs * s);
    // }
    vec ahat, aux(3), auxP(p + 1, fill::ones),
        auxQ(q + 1, fill::ones), seas(s + 1, fill::zeros);
    uword N;
    // lagP y lagQ
    bool convolutionalLags = false;
    if (convolutionalLags){    //  lags calculated by convolution
        seas(0) = 1.0;
        seas(s) = 1.0;
        // Finding AR and MA lags
        for (uword i = 0; i < ps; i++){
            auxP = conv(auxP, seas);
        }
        for (uword i = 0; i < qs; i++){
            auxQ = conv(auxQ, seas);
        }
        auxP(0) = 0.0;
        auxQ(0) = 0.0;
        lagP = find(auxP > 0);
        lagQ = find(auxQ > 0);
    } else {                  // just regular and seasonal lags
        lagP.set_size(0);
        lagQ.set_size(0);
        if (p > 0){
            lagP = regspace<uvec>(1, p);
            if (ps > 0){
                lagP = join_vert(lagP, regspace<uvec>(1, ps) * s);
            }
        } else if (ps > 0){
            lagP = regspace<uvec>(1, ps) * s;
        }
        if (q > 0){
            lagQ = regspace<uvec>(1, q);
            if (qs > 0){
                lagQ = join_vert(lagQ, regspace<uvec>(1, qs) * s);
            }
        } else if (qs > 0){
            lagQ = regspace<uvec>(1, qs) * s;
        }
    }
    if (a.n_elem == 0 || q + qs == 0)
        N = y.n_elem;
    else
        N = a.n_elem;
    uword nX = N - maxPQ;
    mat X(nX, lagP.n_elem + lagQ.n_elem + 1 + u.n_rows);
    X.col(lagP.n_elem + lagQ.n_elem + u.n_rows).fill(1.0);
    if (u.n_rows > 0)
        //************** La siguiente línea está mal alineada u con y
        X.cols(lagP.n_elem + lagQ.n_elem, X.n_cols - 2) = u.cols(y.n_elem - X.n_rows, y.n_elem - 1).t();
    if (p + ps > 0){
        X.cols(0, lagP.n_elem - 1) = lag(y.rows(y.n_elem - X.n_rows - maxPQ, y.n_elem - 1), lagP).tail_rows(nX);
    }
    // MA lags
    if (q + qs > 0){
        X.cols(lagP.n_elem, lagP.n_elem + lagQ.n_elem - 1) = lag(a.rows(a.n_elem  - X.n_rows - maxPQ, a.n_elem - 1), lagQ).tail_rows(nX);
    }
    // First estimates
    double AIC, AICc;
    // vec stdBeta;
    regress(y.rows(y.n_elem - N + maxPQ, y.n_elem - 1), X, beta, stdBeta, ahat, BIC, AIC, AICc);
    double test = beta.back() / stdBeta.back();
    cnst = 0.0;
    if (abs(test) > 1.75)
        cnst = 1.0;
    if (criterion == "aic")
        BIC = AIC;
    else if (criterion == "aicc")
        BIC = AICc;
    if (convolutionalLags){
        uvec lagPc = lagP, lagQc = lagQ, pars;
        if (lagP.n_elem > 0)
            lagPc = join_vert(find(lagP < s), find(lagP - floor(lagP / s) * s == 0));
        if (lagQ.n_elem > 0)
            lagQc = join_vert(find(lagQ < s), find(lagQ - floor(lagQ / s) * s == 0)) + lagPc.n_elem;
        pars = join_vert(lagPc, lagQc);
        uword npar = beta.n_elem;
        beta = beta.rows(pars);
        stdBeta = stdBeta.rows(pars);
        BIC = BIC - (npar - beta.n_elem) * log(N - maxPQ) / (N - maxPQ);
    }
}
// Interpolation with long AR
void interpol(vec& y, uword s){
    if (!y.has_nonfinite())
        return;
    uvec ind = find_nonfinite(y);
    vec a, AR, dy, yi, dPoly = {1, -1};
    uword maxAR = 6, ds = 0, bestAR = 1;
    if (s > 1){
        maxAR = 2 * s + 1;
        ds = 1;
        dPoly = zeros(s + 2);
        dPoly(0) = 1;
        dPoly(1) = -1;
        dPoly(s) = -1;
        dPoly(s + 1) = 1;
    }
    // Estimating long ar for differenced series
    dy = vDiff(y, 1, ds, s);
    yi = y.rows(0, y.n_rows - dy.n_rows);
    longAR(dy, s, maxAR, 1, a, AR, bestAR);
    // AR = conv(dPoly, AR.rows(0, min(find(AR == 0)) - 1));
    AR = AR.rows(0, min(find(AR == 0)) - 1);
    // SS ARIMA model
    SSinputs ssm;
    ssm.y = y;
    ssm.p = AR.rows(1, AR.n_rows - 1);
    ssm.h = 0;
    ARIMASS mSS;
    mSS.AR = AR;
    mSS.MA = mSS.ARS = mSS.MAS = 1;
    vec orders = {0, 1, 0, 0, 1, 0};
    orders(0) = AR.n_rows - 1;
    mSS.orders = orders;
    mSS.s = s;
    ARIMASSclass m(ssm, mSS);
    m.filter();
    ssm = m.getInputs();
    y.rows(ind) = ssm.yFit.rows(ind);
}
// Estimation of autocovariances
vec acov(vec y, uword ncoef){
    // y: time series
    // ncoef: number of autocovariance coefficients
    ncoef++;
    vec c(ncoef);
    uword n = y.n_elem - 1;
    if (y.has_nan()){
        vec prod;
        uvec ind;
        uword nnan;
        ind = find_finite(y);
        nnan = ind.n_elem;
        y -= mean(y(ind));
        for (uword i = 0; i < ncoef; i++){
            prod = y.rows(i, n) % y.rows(0, n - i);
            ind = find_finite(prod);
            c.row(i) = sum(prod(ind)) / nnan;
        }
    } else {
        y -= mean(y);
        for (uword i = 0; i < ncoef; i++){
            c.row(i) = (y.rows(i, n).t() * y.rows(0, n - i)) / y.n_elem;
        }
    }
    return c;
}
// Estimation of a long AR model from autocovariances (it includes leading 1)
void longAR(vec y, uword s, uword maxP, uword maxQ, vec& a, vec& phi, uword& bestAR){
    // y: time series
    // N: AR order
    // a: Residuals
    y -= mean(y(find_finite(y)));
    vec aux(3);
    double N;
    if (y.n_elem < 30 && s == 1){
        N = pow(log(y.n_elem), 2);
    } else {
        aux.resize(3);
        aux(0) = pow(log(y.n_elem), 2);
        aux(1) = 2 * maxP + s * (s > 1);
        aux(2) = 2 * maxQ + s * (s > 1);
        N = (uword)max(aux);
    }
    vec c = acov(y, N);
    phi.resize(N + 1);
    double sigma2, BIC, BICbest;
    uword j1, maxAR = N;
    int n = y.n_elem - 1;
    phi(0) = 1.0;
    phi(1) = -c(1) / c(0);
    sigma2 = (1 - pow(phi(1), 2)) * c(0);
    // if (s == 1){
    //     BICbest = log(sigma2) + log(n) / n;
    // } else {
        BICbest = 1e10;
    // }
    uword j = 1;
    bestAR = 1;
    do{
        j1 = j;
        j++;
        phi.row(j) = -(phi.rows(0, j1).t() * reverse(c.rows(1, j))) / sigma2;
        phi.rows(1, j1) += phi(j) * reverse(phi.rows(1, j1));
        sigma2 *= (1 - pow(phi(j), 2));
        n--;
        BIC = log(sigma2) + j * log(n) / n;
        if (BIC < BICbest && j >= s){
            BICbest = BIC;
            bestAR = j;
            maxAR = j;
        }
    } while (j < maxAR);
    a = filter(phi.rows(0, j), {1}, y);
}
// Filtering time series
vec filter(vec MA, vec AR, vec y){
    // MA: MA polynomial
    // AR: AR polynomial
    // y:  time series
    uword q = MA.n_elem - 1, p = AR.n_elem - 1, n = y.n_elem;
    vec fy;
    // MA filtering
    fy = conv(MA, y);
    // AR filtering
    if (p > 0){
        vec ARp = -AR.tail(p);
        for (uword t = p; t < n; t++){
            fy.row(t) += ARp.t() * fy.rows(t - p, t - 1);
        }
        fy = fy.rows(0, n - 1);
    } else {
        fy = fy.rows(q, fy.n_elem - q - 1);
    }
    return fy;
}
// Difference polynomial
vec polyDiff(vec diffs, vec s){
    vec poly = {1}, aux;
    for (uword i = 0; i < s.n_elem; i++){
        aux.resize((uword)s(i) + 1);
        aux.fill(0.0);
        aux(0) = 1.0;
        aux(aux.n_elem - 1) = -1.0;
        for (uword j = 0; j < diffs(i); j++){
            poly = conv(poly, aux);
        }
    }
    return poly;
}
// Differencing time series
vec diff(vec y, vec diffs, vec s){
    // y:  time series
    // diffs: difference orders
    // s: seasonal orders
    vec poly = polyDiff(diffs, s), aux;
    uword n = poly.n_elem - 1;
    aux = conv(poly, y);
    aux = aux.rows(n, aux.n_elem - n - 1);
    return aux;
}
// Integrating constant
mat vIntConst(uword n, uword d, uword D, uword s){
    // x: vec to be differenced
    // d: regular differences
    // D: seasonal differences
    // s: seasonal period
    uword nS = ceil(n / s) * s + s;
    mat x(1, nS); x.fill(1);
    for (uword i = 0; i < d; i++){
        x = cumsum(x, 1);
    }
    for (uword i = 0; i < D; i++){
        for (uword t = s; t < nS; t++){
            x.col(t) += x.col(t - s);
        }
    }
    x = x.head_cols(n);
    return x;
}
// Differencing time series
vec vDiff(vec x, uword d, uword D, uword s){
    // x: vec to be differenced
    // d: regular differences
    // D: seasonal differences
    // s: seasonal period
    vec dx = diff(x, d);
    for (uword i = 0; i < D; i++){
        dx = dx.tail_rows(dx.n_elem - s) - dx.head_rows(dx.n_elem - s);
    }
    return dx;
}
mat vDiff(mat x, uword d, uword D, uword s){
    // x: mat to be differences (series in rows)
    mat dx = diff(x, d, 1);
    for (uword i = 0; i < D; i++){
        dx = dx.tail_cols(dx.n_cols - s) - dx.head_cols(dx.n_cols - s);
    }
    return dx;
}
// Parameter names for output table
void parNames(uword s, uword p, uword q, uword ps, uword qs,
              int nu, double cnst, vector<string>& names){
    char str[12];
    names.clear();
    for (uword i = 1; i <= p; i++){
        snprintf(str, 10, "AR(%d)", (int)i);
        names.push_back(str);
    }
    for (uword i = 1; i <= ps; i++){
        snprintf(str, 10, "ARs(%d)", (int)i * (int)s);
        names.push_back(str);
    }
    for (uword i = 1; i <= q; i++){
        snprintf(str, 10, "MA(%d)", (int)i);
        names.push_back(str);
    }
    for (uword i = 1; i <= qs; i++){
        snprintf(str, 10, "MAs(%d)", (int)i * (int)s);
        names.push_back(str);
    }
    for (uword i = 0; i < nu - cnst; i++){
        int jj = 1;
        snprintf(str, 10, "Beta(%d)", (int)i + jj);
        names.push_back(str);
    }
    if (abs(cnst) > 0.0){
        snprintf(str, 10, "Cnst");
        names.push_back(str);
    }
}
// Testing for stationarity and invertibility
// void testSI(vec& pout, uword p, uword ps, uword q, uword qs){
//     if (ps > 0){
//         vec AR(ps);
//         AR.rows(0, ps - 1) = pout.rows(p, p + ps - 1);
//         maInvert(AR);
//         pout.rows(p, p + ps - 1) = AR;
//     }
//     if (p > 0){
//         vec AR(p);
//         AR.rows(0, p - 1) = pout.rows(0, p - 1);
//         maInvert(AR);
//         pout.rows(0, p - 1) = AR;
//     }
//     if (qs > 0){
//         vec MA(qs);
//         MA.rows(0, qs - 1) = pout.rows(p + ps + q, p + ps + q + qs - 1);
//         maInvert(MA);
//         pout.rows(p + ps + q, p + ps + q + qs - 1) = MA;
//     }
//     if (q > 0){
//         vec MA(q);
//         MA.rows(0, q - 1) = pout.rows(p + ps, p + ps + q - 1);
//         maInvert(MA);
//         pout.rows(p + ps, p + ps + q - 1) = MA;
//     }
// }
// Checking whether a polynomial has a unit root
bool unitRoot(vec AR, double limit){
    cx_vec arRoots;
    arRoots = roots(AR);
    return any(abs(arRoots) > limit);
}
// Variance of data within b standard deviations
double varNaN(vec y, float b){
    y = y(find_finite(y));
    y = y(find(abs(y - mean(y)) < b * stddev(y)));
    return var(y);
}
// Join seven uword values in a rowvec
// void join(mat& models, uword& mC, double BIC,
//             uword p, uword d, uword q, uword ps, uword ds, uword qs){
//     rowvec out(6);
//     models(mC, 0) = BIC;
//     models(mC, 1) = p;
//     models(mC, 2) = d;
//     models(mC, 3) = q;
//     models(mC, 4) = ps;
//     models(mC, 5) = ds;
//     models(mC, 6) = qs;
//     mC++;
// }
