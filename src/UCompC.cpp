// [[Rcpp::depends(RcppArmadillo)]] 
#include <RcppArmadillo.h>
using namespace arma;
using namespace std;
using namespace Rcpp;
#include "BSMmodel.h"
#include "ETSmodel.h"
#include "ARIMAmodel.h"
#include "TETSmodel.h"

// [[Rcpp::export]]
SEXP UCompC(SEXP commands, SEXP ys, SEXP us, SEXP models, SEXP hs, SEXP lambdas, SEXP outliers, SEXP tTests,
            SEXP criterions, SEXP periodss, SEXP rhoss, SEXP verboses, SEXP stepwises, SEXP p0s, SEXP armas, SEXP TVPs,
            SEXP seass, SEXP trendOptionss, SEXP seasonalOptionss, SEXP irregularOptionss){
        string command = CHAR(STRING_ELT(commands, 0));
        NumericVector yr(ys);
        NumericMatrix ur(us);
        string model = CHAR(STRING_ELT(models, 0));
        int h = as<int>(hs);
        double lambda = as<double>(lambdas);
        double outlier = as<double>(outliers);
        bool tTest = as<bool>(tTests);
        string criterion = CHAR(STRING_ELT(criterions, 0));
        NumericVector periodsr(periodss);
        NumericVector rhosr(rhoss);
        bool verbose = as<bool>(verboses);
        bool stepwise = as<bool>(stepwises);
        NumericVector p0r(p0s);
        bool arma = as<bool>(armas);
        NumericVector TVPr(TVPs);
        double seas = as<double>(seass);
        string trendOptions = CHAR(STRING_ELT(trendOptionss, 0));
        string seasonalOptions = CHAR(STRING_ELT(seasonalOptionss, 0));
        string irregularOptions = CHAR(STRING_ELT(irregularOptionss, 0));

        vec y(yr.begin(), yr.size(), false);
        mat u(ur.begin(), ur.nrow(), ur.ncol(), false);
        vec periods(periodsr.begin(), periodsr.size(), false);
        vec rhos(rhosr.begin(), rhosr.size(), false);
        vec p0(p0r.begin(), p0r.size(), false);
        vec TVP(TVPr.begin(), TVPr.size(), false);

        // Correcting dimensions of u (k x n)
        size_t k = u.n_rows;
        size_t n = u.n_cols;
        if (k > n){
                u = u.t();
        }
        if (k == 1 && n == 2){
                u.resize(0);
        }
        int iniObs = periods.n_elem * 2 + 2;
        // Setting inputs
        SSinputs inputsSS;
        BSMmodel inputsBSM;
        if (TVP.n_elem == 1 && TVP(0) == -9999.9)
            TVP.reset();
        // Pre-processing
        bool errorExit = preProcess(y, u, model, h, outlier, criterion, periods, p0, iniObs,
                                    trendOptions, seasonalOptions, irregularOptions, TVP, lambda);
        if (errorExit)
                return List::create(Named("model") = "error");
        if (sum(TVP) > 0)
                outlier = 0;
        // End of pre-processing
        // if (command == "estimate"){
                // Missing at beginning
        inputsSS.y = y.rows(iniObs, y.n_elem - 1);
        // } else {
        //         inputsSS.y = y;
        // }
        // mat uIni;
        // if (iniObs > 0 && u.n_rows > 0 && command == "estimate"){
        if (u.n_rows > 0) {
                // Missing at beginning
                inputsSS.u = u.cols(iniObs, u.n_cols - 1);
                // inputsSS.u = u.cols(iniObs, y.n_elem - 1);
                // uIni = u.cols(0, iniObs - 1);
        } else {
                inputsSS.u= u;
        }
        inputsBSM.model = model;
        inputsBSM.periods = periods;
        inputsBSM.rhos = rhos;
        inputsSS.h = h;
        inputsBSM.tTest = tTest;
        inputsBSM.criterion = criterion;
        //if (TVP(0) == -9999.99)
        //    TVP = {};
        inputsBSM.TVP = TVP;
        inputsBSM.MSOE = false;
        inputsBSM.PTSnames = false;
        inputsBSM.trendOptions = trendOptions;
        inputsBSM.seasonalOptions = seasonalOptions;
        inputsBSM.irregularOptions = irregularOptions;
        inputsSS.p0 = p0;
        inputsSS.outlier = outlier;
        inputsSS.verbose = verbose;
        inputsBSM.stepwise = stepwise;
        inputsBSM.arma = arma;
        inputsBSM.seas = seas;
        // BoxCox transformation
        if (lambda == 9999.9)
                lambda = testBoxCox(y, periods);
        inputsBSM.lambda = lambda;
        inputsSS.y = BoxCox(inputsSS.y, inputsBSM.lambda);
        y = BoxCox(y, inputsBSM.lambda);
        // Building model
        BSMclass sysBSM = BSMclass(inputsSS, inputsBSM);
        // Commands
        SSinputs inputs;
        BSMmodel inputs2;
        // if (command == "estimate"){
        sysBSM.estim(inputsSS.verbose);
        sysBSM.forecast();
        sysBSM.parLabels();
        // Missing at beginning
        if (iniObs > 0){
            inputs = sysBSM.SSmodel::getInputs();
            inputs2 = sysBSM.getInputs();
            inputs.y = y;
            inputs.u = u;
            sysBSM.SSmodel::setInputs(inputs);
            sysBSM.setInputs(inputs2);
        }
        inputs = sysBSM.SSmodel::getInputs();
        inputs2 = sysBSM.getInputs();
        if (inputs2.cycle[0] != 'n' && inputs2.cycle != "?"){
                string model1 = inputs2.model, cycle = inputs2.cycle, cycle0 = inputs2.cycle0;
                vec periods = inputs2.periods, rhos = inputs2.rhos;
                modelCorrect(model, cycle, inputsBSM.cycle0, periods, rhos);
                inputs2.model= model1, inputs2.cycle= cycle, inputs2.cycle0= cycle0;
                inputs2.periods = periods, inputs2.rhos = rhos;
                // sysBSM.setInputs(inputs2);
                // Estimating period of cycles (lines 3101 BSMmodel)
                // int nRhos = sum(inputs2.typePar == 1);
                // uword pos = inputs2.nPar(0) + nRhos;
                // uvec ind, ind1;
                // vec pInd;
                // int nPerEstim = sum(inputs2.typePar == 2);
                // if (nPerEstim > 0){
                //     ind = regspace<uvec>(pos, 1, pos + nPerEstim - 1);
                //     pos += nPerEstim;
                //     ind1 = find(inputs2.periods < 0);
                //     pInd = inputs.p(ind);
                //     constrain(pInd, inputs2.cycleLimits.rows(ind1));
                //     periods(ind1) = pInd;
                //     inputs2.periods = periods;
                // }
                sysBSM.setInputs(inputs2);
        }
        // Values to return
        inputs = sysBSM.SSmodel::getInputs();
        inputs2 = sysBSM.getInputs();
        if (!inputs2.succeed) {
             return List::create(Named("model") = "error");
        }
        // Converting back to R
        List output = List::create(Named("model") = inputs2.model,
                              Named("yFor") = inputs.yFor,
                              Named("h") = inputs.h,
                              Named("yForV") = inputs.FFor,
                              Named("estimOk") = inputs.estimOk,
                              Named("lambda") = inputs2.lambda,
                              Named("periods") = inputs2.periods,
                              Named("rhos") = inputs2.rhos,
                              Named("p") = inputs.p,
                              Named("p0") = inputs2.p0Return,
                              Named("parNames") = inputs2.parNames,
                              Named("ns") = sum(inputs2.ns),
                              Named("criteria") = inputs.criteria);
        if (command == "validate" || command == "all") {
                sysBSM.validate(false);
                // Values to return
                inputs = sysBSM.SSmodel::getInputs();
                inputs2 = sysBSM.getInputs();
                output("table") = inputs.table;
                output("v") = inputs.v;
                output("covp") = inputs.covp;
                output("coef") = inputs.coef;
        }
        if (command == "filter" || command == "smooth" || command == "disturb" || command == "all"){
                // sysBSM.initMatricesBsm(inputs2.periods, inputs2.rhos, inputs2.trend, inputs2.cycle, inputs2.seasonal, inputs2.irregular);
                // sysBSM.setSystemMatrices();
                if (command != "all")
                        sysBSM.validate(false);
                if (command == "filter"){
                        sysBSM.filter();
                } else if (command == "smooth") {
                        sysBSM.smooth(false);
                } else if (command == "disturb") {
                        sysBSM.disturb();
                }
                inputs = sysBSM.SSmodel::getInputs();
                inputs2 = sysBSM.getInputs();
                string statesN = stateNames(inputs2);
                if (command == "disturb"){
                        uvec missing = find_nonfinite(inputs.y);
                        inputs.eta.cols(missing).fill(datum::nan);
                        inputs2.eps(missing).fill(datum::nan);
                }
                // Nans at very beginning
                if (iniObs > 0 && command != "disturb"){
                        uvec missing = find_nonfinite(inputs.y.rows(0, iniObs));
                        mat P = inputs.P.cols(0, iniObs);
                        sysBSM.interpolate(iniObs);
                        if (command == "filter"){
                                sysBSM.filter();
                        } else if (command == "smooth"){
                                sysBSM.smooth(false);
                        }
                        inputs = sysBSM.SSmodel::getInputs();
                        inputs.P.cols(0, iniObs) = P;
                        inputs.v(missing).fill(datum::nan);
                }
                output("a") = inputs.a;
                output("P") = inputs.P;
                output("v") = inputs.v;
                output("ns") = sum(inputs2.ns);
                output("yFitV") = inputs.F;
                output("yFit") = inputs.yFit;
                output("eps") = inputs2.eps;
                output("eta") = inputs.eta;
                output("stateNames") = statesN;
        }
        if (command == "components" || command == "all"){
                // sysBSM.initMatricesBsm(inputs2.periods, inputs2.rhos, inputs2.trend, inputs2.cycle, inputs2.seasonal, inputs2.irregular);
                // sysBSM.setSystemMatrices();
                sysBSM.components();
                inputs2 = sysBSM.getInputs();
                string compNames = inputs2.compNames;
                // Nans at very beginning
                if (iniObs > 0){
                        inputs = sysBSM.SSmodel::getInputs();
                        uvec missing = find_nonfinite(inputs.y.rows(0, iniObs));
                        //vec ytrun = inputs.y.rows(0, iniObs);
                        mat P = inputs2.compV.cols(0, iniObs);
                        sysBSM.interpolate(iniObs);
                        sysBSM.components();
                        inputs2 = sysBSM.getInputs();
                        inputs2.compV.cols(0, iniObs) = P;
                        // Setting irregular to nan
                        uvec rowI(1); rowI(0) = 0;
                        if (compNames.find("Level") != string::npos)
                                rowI++;
                        if (compNames.find("Slope") != string::npos)
                                rowI++;
                        if (compNames.find("Seasonal") != string::npos)
                                rowI++;
                        if (compNames.find("Irr") != string::npos ||
                            compNames.find("ARMA") != string::npos)
                                inputs2.comp.submat(rowI, missing).fill(datum::nan);
                }
                // Values to return
                //inputs2 = sysBSM.getInputs();
                // Converting back to R
                output("comp") = inputs2.comp;
                output("compV") = inputs2.compV;
                output("m") = inputs2.comp.n_rows;
                output("compNames") = compNames;
        }
        // } else if (command == "createSystem"){
        //         // Values to return
        //         inputs2 = sysBSM.getInputs();
        //         // Converting back to R
        //         return List::create(Named("model") = inputs2.model,
        //                             Named("yFor") = inputs.yFor,
        //                             Named("yForV") = inputs.FFor,
        //                             Named("estimOk") = inputs.estimOk,
        //                             Named("lambda") = inputs2.lambda,
        //                             Named("periods") = inputs2.periods,
        //                             Named("rhos") = inputs2.rhos,
        //                             Named("p") = inputs.p,
        //                             Named("parNames") = inputs2.parNames,
        //                             Named("criteria") = inputs.criteria,
        //                             Named("ns") = sum(inputs2.ns),
        //                             Named("table") = inputs.table,
        //                             Named("p0Return") = inputs2.p0Return,
        //                             Named("parNames") = inputs2.parNames);
        // }
        return output;
}

// [[Rcpp::export]]
SEXP ETSc(SEXP commands, SEXP ys, SEXP us, SEXP models, SEXP ss, SEXP hs,
          SEXP criterions, SEXP armaIdents, SEXP identAlls, SEXP forIntervalss,
          SEXP bootstraps, SEXP nSimuls, SEXP verboses, SEXP lambdas,
          SEXP alphaLs, SEXP betaLs, SEXP gammaLs, SEXP phiLs, SEXP p0s){
    // Translating inputs to armadillo data
    //vec bas(1); bas(0) = 14.0; bas.save("bas.txt", raw_ascii);
    string command = CHAR(STRING_ELT(commands, 0));
    NumericVector yr(ys);
    mat u;
    if (Rf_isNull(us)){
        u.set_size(0, 0);
    } else {
        NumericMatrix ur(us);
        mat aux(ur.begin(), ur.nrow(), ur.ncol(), false);
        u = aux;
        if (u.n_rows > u.n_cols)
            u = u.t();
    }
    string model = CHAR(STRING_ELT(models, 0));
    int s = as<int>(ss);
    int h = as<int>(hs);
    string criterion = CHAR(STRING_ELT(criterions, 0));
    bool armaIdent = as<bool>(armaIdents);    
    bool identAll = as<bool>(identAlls);    
    bool forIntervals = as<bool>(forIntervalss);    
    bool bootstrap = as<bool>(bootstraps);    
    bool verbose = as<bool>(verboses);
    double lambda = as<double>(lambdas);
    
    int nSimul = as<int>(nSimuls);
    NumericVector alphaLr(alphaLs);
    NumericVector betaLr(betaLs);
    NumericVector gammaLr(gammaLs);
    NumericVector phiLr(phiLs);
    NumericVector p0r(p0s);
    // Second step
    vec y(yr.begin(), yr.size(), false);
    rowvec alphaL(alphaLr.begin(), alphaLr.size(), false);
    rowvec betaL(betaLr.begin(), betaLr.size(), false);
    rowvec gammaL(gammaLr.begin(), gammaLr.size(), false);
    rowvec phiL(phiLr.begin(), phiLr.size(), false);
    vec p0(p0r.begin(), p0r.size(), false);
    
    // Wrapper adaptation
    if (p0.n_elem == 1 && p0(0) == -99999){
        p0.resize(0); 
    }
    string parConstraints = "standard";
    vec arma = {0, 0};
    // Creating class
    ETSmodel input;
    // BoxCox transformation
    if (lambda == 9999.9){
        vec periods;
        if (s > 1)
            periods = s / regspace(1, floor(s / 2));
        else {
            periods.resize(1);
            periods(0) = 1.0;
        }
        lambda = testBoxCox(y, periods);
    }
    if (abs(lambda) > 1)
        lambda = sign(lambda);
    input.lambda = lambda;
    input.y = BoxCox(input.y, input.lambda);
    // Creating class
    ETSclass m(input);
    m = preProcess(y, u, model, s, h, verbose, criterion, identAll, alphaL, betaL, gammaL, phiL,
                   parConstraints, forIntervals, bootstrap, nSimul, arma, armaIdent, p0, lambda);
    if (m.inputModel.errorExit)
        return List::create(Named("errorExit") = m.inputModel.errorExit);
    // End of wrapper adaptation
    
    // Commands
    if (command == "estimate" || command == "validate"){
        if (m.inputModel.error == "?" || m.inputModel.trend == "?" || m.inputModel.seasonal == "?" || m.inputModel.armaIdent)
            m.ident(verbose);
        else {
            m.estim(verbose);
        }
        m.forecast();
        if (command== "validate")
            m.validate();
        if (bootstrap)
            m.simulate(m.inputModel.h, m.inputModel.xn);
        // Output
        return List::create(Named("p") = m.inputModel.p,
                            Named("comp") = m.inputModel.comp,
                            Named("table") = m.inputModel.table,
                            Named("compNames") = m.inputModel.compNames,
                            Named("truep") = m.inputModel.truep,
                            Named("model") = m.inputModel.model,
                            Named("criteria") = m.inputModel.criteria,
                            Named("yFor") = m.inputModel.yFor,
                            Named("yForV") = m.inputModel.yForV,
                            Named("ySimul") = m.inputModel.ySimul,
                            Named("lambda") = m.inputModel.lambda
        );
    }
    if (command== "components"){
        if (m.inputModel.error == "?" || m.inputModel.trend == "?" || m.inputModel.seasonal == "?" || m.inputModel.armaIdent)
            m.ident(false);
        else {
            m.estim(false);
        }
        m.components();
        return List::create(Named("comp") = m.inputModel.comp,
                            Named("compNames") = m.inputModel.compNames);
    }
    return List::create(Named("void") = datum::nan);
}

// [[Rcpp::export]]
SEXP ARIMAc(SEXP commands, SEXP ys, SEXP us, SEXP orderss, SEXP cnsts, SEXP ss, 
            SEXP criterions, SEXP hs, SEXP verboses, SEXP lambdas, SEXP maxOrderss, 
            SEXP bootstraps, SEXP nSimuls, SEXP fasts, SEXP identDiffs,
            SEXP identMethods){
        // Translating inputs to armadillo data
        //vec bas(1); bas(0) = 14.0; bas.save("bas.txt", raw_ascii);
        string command = CHAR(STRING_ELT(commands, 0));
        NumericVector yr(ys);
        mat u;
        if (Rf_isNull(us)){
                u.set_size(0, 0);
        } else {
                NumericMatrix ur(us);
                mat aux(ur.begin(), ur.nrow(), ur.ncol(), false);
                u = aux;
                if (u.n_rows > u.n_cols)
                        u = u.t();
        }
        vec orders;
        if (Rf_isNull(orderss)){
                orders.set_size(0);
        } else {
                NumericVector ordersr(orderss);
                vec aux(ordersr.begin(), ordersr.size(), false);
                orders = aux;
        }
        double cnst = as<double>(cnsts);
        int s = as<int>(ss);
        string criterion = CHAR(STRING_ELT(criterions, 0));
        int h = as<int>(hs);
        bool verbose = as<bool>(verboses);
        // bool fast = as<bool>(fasts);
        // bool identDiff = as<bool>(identDiffs);
        double lambda = as<double>(lambdas);
        NumericVector maxOrdersr(maxOrderss);
        bool bootstrap = as<bool>(bootstraps);    
        int nSimul = as<int>(nSimuls);
        string identMethod = CHAR(STRING_ELT(identMethods, 0));
        // Second step
        vec y(yr.begin(), yr.size(), false);
        vec maxOrders(maxOrdersr.begin(), maxOrdersr.size(), false);
        // Correcting inputs
        ARIMAmodel input;
        ARIMAclass m(input);
        // m = preProcess(y, u, orders, cnst, s, h, criterion, verbose, lambda,
        //                maxOrders, bootstrap, nSimul, identDiff, identMethod);
        m = preProcess(y, u, orders, cnst, s, h, verbose, lambda,
                       maxOrders, bootstrap, nSimul, criterion);
        if (m.m.errorExit)
                return List::create(Named("void") = datum::nan);
        // Commands
        if (command == "estimate"){
                m.identGM();
            m.estim(false);
        }
        if (command== "validate"){
                m.identGM();
                m.validate();
        }
        m.forecast();
        if (m.m.u.n_rows == 0){
                m.m.cnst = 0.0;
        }
        if (m.m.cnst == 1.0){
                uvec ind(1);
                ind(0) = m.m.u.n_rows - 1;
                m.m.u.shed_rows(ind);
                if (u.n_rows == 0){
                        u.reset();
                }
        }
        return List::create(Named("p") = m.m.par,
                            Named("yFor") = m.m.yFor,
                            Named("yForV") = m.m.FFor,
                            Named("ySimul") = m.m.ySimul,
                            Named("lambda") = m.m.lambda,
                            Named("orders") = m.m.orders,
                            Named("cnst") = m.m.cnst,
                            Named("u") = m.m.u,
                            Named("BIC") = m.m.BIC,
                            Named("AIC") = m.m.AIC,
                            Named("AICc") = m.m.AICc,
                            Named("IC") = m.m.IC,
                            Named("table") = m.m.table,
                            Named("error") = m.m.v
        );
}

// [[Rcpp::export]]
SEXP TETSc(SEXP commands, SEXP ys, SEXP us, SEXP models, SEXP ss, SEXP hs,
           SEXP criterions, SEXP armaIdents, SEXP identAlls, SEXP forIntervalss,
           SEXP bootstraps, SEXP nSimuls, SEXP verboses, SEXP lambdas,
           SEXP alphaLs, SEXP betaLs, SEXP gammaLs, SEXP phiLs, SEXP p0s,
           SEXP Ymins, SEXP Ymaxs){
        // Translating inputs to armadillo data
        //vec bas(1); bas(0) = 14.0; bas.save("bas.txt", raw_ascii);
        string command = CHAR(STRING_ELT(commands, 0));
        NumericVector yr(ys);
        mat u;
        if (Rf_isNull(us)){
                u.set_size(0, 0);
        } else {
                NumericMatrix ur(us);
                mat aux(ur.begin(), ur.nrow(), ur.ncol(), false);
                u = aux;
                if (u.n_rows > u.n_cols)
                        u = u.t();
        }
        string model = CHAR(STRING_ELT(models, 0));
        int s = as<int>(ss);
        int h = as<int>(hs);
        string criterion = CHAR(STRING_ELT(criterions, 0));
        bool armaIdent = as<bool>(armaIdents);    
        bool identAll = as<bool>(identAlls);    
        bool forIntervals = as<bool>(forIntervalss);    
        bool bootstrap = as<bool>(bootstraps);    
        bool verbose = as<bool>(verboses);
        double lambda = as<double>(lambdas);
        int nSimul = as<int>(nSimuls);
        NumericVector alphaLr(alphaLs);
        NumericVector betaLr(betaLs);
        NumericVector gammaLr(gammaLs);
        NumericVector phiLr(phiLs);
        NumericVector p0r(p0s);
        NumericVector Yminr(Ymins);
        NumericVector Ymaxr(Ymaxs);
        // Second step
        vec y(yr.begin(), yr.size(), false);
        rowvec alphaL(alphaLr.begin(), alphaLr.size(), false);
        rowvec betaL(betaLr.begin(), betaLr.size(), false);
        rowvec gammaL(gammaLr.begin(), gammaLr.size(), false);
        rowvec phiL(phiLr.begin(), phiLr.size(), false);
        vec p0(p0r.begin(), p0r.size(), false);
        vec Ymin(Yminr.begin(), Yminr.size(), false);
        vec Ymax(Ymaxr.begin(), Ymaxr.size(), false);
        
        // Wrapper adaptation
        if (p0.n_elem == 1 && p0(0) == -99999){
                p0.resize(0); 
        }
        string parConstraints = "standard";
        vec arma = {0, 0};
        // Creating class
        // TETSmodel input;
        // BoxCox transformation
        // if (lambda == 9999.9){
        //         vec periods;
        //         if (s > 1)
        //                 periods = s / regspace(1, floor(s / 2));
        //         else {
        //                 periods.resize(1);
        //                 periods(0) = 1.0;
        //         }
        //         lambda = testBoxCox(y, periods);
        // }
        // if (abs(lambda) > 1)
        //         lambda = sign(lambda);
        // input.lambda = lambda;
        // input.y = BoxCox(input.y, input.lambda);
        // Creating class
        // TETSclass m(input);
        // m = preProcess(y, u, model, s, h, verbose, criterion, identAll, alphaL, betaL, gammaL, phiL,
        //                parConstraints, forIntervals, bootstrap, nSimul, arma, armaIdent, p0, lambda);
        ETSmodel m1;
        TETSclass m(m1, Ymin, Ymax);
        //    ETSmodel input;
        //    TETSclass m(input, Ymax, Ymin, false);
        m = preProcess(y, u, model, s, h, verbose, criterion, identAll, alphaL, betaL, gammaL, phiL,
                       parConstraints, forIntervals, bootstrap, nSimul, arma, armaIdent, p0, lambda,
                       Ymax, Ymin);
        if (m.data.m.errorExit)
                return List::create(Named("errorExit") = m.data.m.errorExit);
        // End of wrapper adaptation
        
        // Commands
        if (command == "estimate" || command== "validate"){
                if (m.data.m.error == "?" || m.data.m.trend == "?" || m.data.m.seasonal == "?" || m.data.m.armaIdent)
                        m.ident(verbose);
                else {
                        m.estim(verbose);
                }
                m.forecast();
                if (command == "validate")
                    m.validate();
                if (bootstrap){
                        ETSclass mETS(m.data.m);
                        mETS.simulate(m.data.m.h, m.data.m.xn);
                        m.data.m.ySimul = mETS.inputModel.ySimul;
                }
                // Output
                return List::create(Named("p") = m.data.m.p,
                                    Named("truep") = m.data.m.truep,
                                    Named("model") = m.data.m.model,
                                    Named("criteria") = m.data.m.criteria,
                                    Named("yFor") = m.data.m.yFor,
                                    Named("yForV") = m.data.m.yForV,
                                    Named("ySimul") = m.data.m.ySimul,
                                    Named("comp") = m.data.m.comp,
                                    Named("table") = m.data.m.table,
                                    Named("compNames") = m.data.m.compNames,
                                    Named("lambda") = m.data.m.lambda
                );
        }
        if (command== "components"){
                if (m.data.m.error == "?" || m.data.m.trend == "?" || m.data.m.seasonal == "?" || m.data.m.armaIdent)
                        m.ident(false);
                else {
                        m.estim(false);
                }
                m.components();
                return List::create(Named("comp") = m.data.m.comp,
                                    Named("compNames") = m.data.m.compNames);
        }
        return List::create(Named("void") = datum::nan);
}