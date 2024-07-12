/*
 * TETSCMatlab.cpp is a bridge between MATLAB/Octave and C++ to use forecasting and estimation tools
 * developed with Armadillo library.
 *
 * Note: read the file README.txt for information about building MEX files
 *
 */

#include "cpp/defines.hpp"
#include "cpp/armaMex.hpp"
//#include "mex.hpp"
//#include "mexAdapter.hpp"
// #include <stdio.h>
#include <armadillo>
#include <string>
#include <vector>
#include <math.h>
using namespace arma;
using namespace std;
#include "cpp/TETSmodel.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    bool errorExit = false;
    string command = mxArrayToString(prhs[0]);
    vec y = armaGetPr(prhs[1]);
    mat aux = armaGetPr(prhs[2]);
    mat u;
    if (aux.n_elem != 1){
        u = aux;
        if (u.n_rows > u.n_cols)
            u = u.t();
    }
    string model = mxArrayToString(prhs[3]);
    int s = mxGetScalar(prhs[4]);
    int h = mxGetScalar(prhs[5]);
    string criterion = mxArrayToString(prhs[6]);
    bool armaIdent = mxIsLogicalScalarTrue(prhs[7]);
    bool identAll = mxIsLogicalScalarTrue(prhs[8]);
    bool forIntervals = mxIsLogicalScalarTrue(prhs[9]);
    bool bootstrap = mxIsLogicalScalarTrue(prhs[10]);
    int nSimul = mxGetScalar(prhs[11]);
    bool verbose = mxIsLogicalScalarTrue(prhs[12]);
    double lambda = mxGetScalar(prhs[13]);
    rowvec alphaL = armaGetPr(prhs[14]);
    rowvec betaL = armaGetPr(prhs[15]);
    rowvec gammaL = armaGetPr(prhs[16]);
    rowvec phiL = armaGetPr(prhs[17]);
    vec p0 = armaGetPr(prhs[18]);
    vec Ymin = armaGetPr(prhs[19]);
    vec Ymax = armaGetPr(prhs[20]);

//     // Model adaptation
//     bool negative = false;
//     if (model.length() == 0)
//         model = "???";
//     upper(model);
//     if (model.length() > 3)
//         model[2] = 'd';
//     if (nanMin(y) <= 0){
//         negative = true;
//         if (model[0] == 'M'){
//             printf("%s", "ERROR: Cannot run model on time series with negative or zero values!!!\n");
//             errorExit = true;
//          }
//     }
//     lower(criterion);
//     string parConstraints = "standard";
//     lower(parConstraints);
//     // Checking s
//     if (s < 2){
//         s = 0;
//     }
//     // Correcting h in case there are inputs
//     if (u.n_cols > 0){
//         h = u.n_cols - y.n_elem;
// 	if (h < 0){
//             printf("%s", "ERROR: Inputs should be at least as long as output!!!\n");
//             errorExit = true;
//          }
//     }
//     // Correcting parameter limits
//     checkLimits(alphaL, betaL, gammaL, phiL, parConstraints, errorExit);
//     // Correcting initial conditions
//     if (p0.n_elem == 1)
//         p0.resize(0);
//     if (p0.n_elem > 0 && (any(p0 < 0) || any(p0 > 1) || p0(1) > p0(0) || p0(3) > 1 - p0(0))){
//         printf("%s", "ERROR: Initial parameters incorrect, pleas check!!!\n");
//         errorExit = true;
//     }
//      // Creating model
//     ETSmodel input;
//     input.initialModel = input.model;
//     input.userS = s;
//     input.h = h;
//     input.negative = negative;
//     input.identAll = identAll;
//     input.alphaL = alphaL;
//     input.betaL = betaL;
//     input.gammaL = gammaL;
//     input.phiL = phiL;
//     input.parConstraints = parConstraints;
//     input.y = y;
//     input.u = u;
//     input.verbose = verbose;
//     input.forIntervals = forIntervals;
//     input.nSimul = nSimul;
//     input.bootstrap = bootstrap;
//     input.p0user = p0;
//     vec arma = {0, 0};
//     input.arma = arma;
//     input.criterion = criterion;
//     if (armaIdent || model[0] == '?' || model[1] == '?' || model[2] == '?' || model[model.length() - 1] == '?')
//         input.arma.fill(0);
//     input.armaIdent = armaIdent;
//     input.errorExit = errorExit;
//     setModel(input, model, s);
//     ETSclass m(input);
    // Wrapper adaptation
    if (p0.n_elem == 1 && p0(0) == -99999){
        p0.resize(0); 
    }
    string parConstraints = "standard";
    vec arma = {0, 0};
    // Creating class
    // ETSmodel m1;
    // TETSclass m(m1, Ymin, Ymax);
    // // BoxCox transformation
    // if (lambda == 9999.9){
    //     vec periods;
    //     if (s > 1)
    //         periods = s / regspace(1, floor(s / 2));
    //     else {
    //         periods.resize(1);
    //         periods(0) = 1.0;
    //     }
    //     lambda = testBoxCox(y, periods);
    // }
    // if (abs(lambda) > 1)
    //     lambda = sign(lambda);
    // input.lambda = lambda;
    // input.y = BoxCox(input.y, input.lambda);
    // Creating class
    ETSmodel m1;
    TETSclass m(m1, Ymin, Ymax);
    m = preProcess(y, u, model, s, h, verbose, criterion, identAll, alphaL, betaL, gammaL, phiL,
                   parConstraints, forIntervals, bootstrap, nSimul, arma, armaIdent, p0, lambda,
                   Ymax, Ymin);
    // End of wrapper adaptation
    if (m.data.m.errorExit){
        plhs[0] = armaCreateMxMatrix(0, 0);
        armaSetPr(plhs[0], {});
        return;
    }
    // Commands
    if (command == "estimate"){
        if (m.data.m.error == "?" || m.data.m.trend == "?" || m.data.m.seasonal == "?" || m.data.m.armaIdent)
            m.ident(verbose);
        else {
            m.estim(verbose);
        }
        m.forecast();
        if (bootstrap){
            ETSclass mETS(m.data.m);
            mETS.simulate(m.data.m.h, m.data.m.xn);
            m.data.m.ySimul = mETS.inputModel.ySimul;
        }
        // Back to MATLAB:
        plhs[0] = armaCreateMxMatrix(m.data.m.p.n_rows, m.data.m.p.n_cols); 
        armaSetPr(plhs[0], m.data.m.p);
        
        plhs[1] = armaCreateMxMatrix(m.data.m.truep.n_rows, m.data.m.truep.n_cols); 
        armaSetPr(plhs[1], m.data.m.truep);

        plhs[2] = mxCreateString(m.data.m.model.c_str());

        plhs[3] = armaCreateMxMatrix(m.data.m.criteria.n_rows, m.data.m.criteria.n_cols); 
        armaSetPr(plhs[3], m.data.m.criteria);

        plhs[4] = armaCreateMxMatrix(m.data.m.yFor.n_rows, m.data.m.yFor.n_cols); 
        armaSetPr(plhs[4], m.data.m.yFor);

        plhs[5] = armaCreateMxMatrix(m.data.m.yForV.n_rows, m.data.m.yForV.n_cols); 
        armaSetPr(plhs[5], m.data.m.yForV);

        plhs[6] = armaCreateMxMatrix(m.data.m.ySimul.n_rows, m.data.m.ySimul.n_cols); 
        armaSetPr(plhs[6], m.data.m.ySimul);

        plhs[7] = mxCreateDoubleScalar(1); 
        *mxGetPr(plhs[7]) = m.data.m.lambda;
    }
    if (command== "validate"){
        if (m.data.m.error == "?" || m.data.m.trend == "?" || m.data.m.seasonal == "?" || m.data.m.armaIdent)
            m.ident(false);
        else {
            m.estim(false);
        }
        m.validate();
        //Back to MATLAB
        plhs[0] = armaCreateMxMatrix(m.data.m.comp.n_rows, m.data.m.comp.n_cols); 
        armaSetPr(plhs[0], m.data.m.comp);

        vector<string> table = m.data.m.table;
        int elem = table.size();
        mxArray *tab = mxCreateCellMatrix(elem, 1);
        for (mwIndex i = 0; i<elem; i++) {
            mxArray *str = mxCreateString(table[i].c_str());
            mxSetCell(tab, i, mxDuplicateArray(str));
        }
        plhs[1] = tab; //table is back in cell format

        plhs[2] = mxCreateString(m.data.m.compNames.c_str());
    }
    if (command== "components"){
        if (m.data.m.error == "?" || m.data.m.trend == "?" || m.data.m.seasonal == "?" || m.data.m.armaIdent)
            m.ident(false);
        else {
            m.estim(false);
        }
        m.components();
        //Back to MATLAB
        plhs[0] = armaCreateMxMatrix(m.data.m.comp.n_rows, m.data.m.comp.n_cols); 
        armaSetPr(plhs[0], m.data.m.comp);

        plhs[1] = mxCreateString(m.data.m.compNames.c_str());
    }
    return;
}
