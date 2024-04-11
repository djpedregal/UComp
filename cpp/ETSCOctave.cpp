/*
 * ETSCOctave.cpp is a bridge between MATLAB/Octave and C++ to use forecasting and estimation tools
 * developed with Armadillo library.
 *
 * Note: read the file README.txt for information about building MEX files
 *
 */

#include "cpp/armaMexOct.hpp"
#include <stdio.h>
#include <armadillo>
#include <string>
#include <vector>
#include <math.h>
using namespace arma;
using namespace std;
#include "cpp/ETSmodel.h"

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
    bool verbose = mxIsLogicalScalarTrue(prhs[11]);
    int nSimul = mxGetScalar(prhs[12]);
    rowvec alphaL = armaGetPr(prhs[13]);
    rowvec betaL = armaGetPr(prhs[14]);
    rowvec gammaL = armaGetPr(prhs[15]);
    rowvec phiL = armaGetPr(prhs[16]);
    vec p0 = armaGetPr(prhs[17]);
    double lambda = mxGetScalar(prhs[18]);
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
    // End of wrapper adaptation

    // Commands
    if (command == "estimate"){
        if (m.inputModel.errorExit){   // ERROR!!!
            m.inputModel.model = "error";
        } else {
            if (m.inputModel.error == "?" || m.inputModel.trend == "?" || m.inputModel.seasonal == "?" || m.inputModel.armaIdent)
                m.ident(verbose);
            else {
                m.estim(verbose);
            }
            m.forecast();
            if (bootstrap)
                m.simulate(h, m.inputModel.xn);
        }
        // Back to MATLAB:
        plhs[0] = armaCreateMxMatrix(m.inputModel.p.n_rows, m.inputModel.p.n_cols); 
        armaSetPr(plhs[0], m.inputModel.p);
        
        plhs[1] = mxCreateString(m.inputModel.model.c_str());
        
        plhs[2] = armaCreateMxMatrix(m.inputModel.yFor.n_rows, m.inputModel.yFor.n_cols); 
        armaSetPr(plhs[2], m.inputModel.yFor);
        
        plhs[3] = armaCreateMxMatrix(m.inputModel.yForV.n_rows, m.inputModel.yForV.n_cols); 
        armaSetPr(plhs[3], m.inputModel.yForV);
        
        plhs[4] = armaCreateMxMatrix(m.inputModel.ySimul.n_rows, m.inputModel.ySimul.n_cols); 
        armaSetPr(plhs[4], m.inputModel.ySimul);
        
        plhs[5] = mxCreateDoubleScalar(1); 
        *mxGetPr(plhs[5]) = m.inputModel.lambda;
 
    } else if (command== "validate"){
        if (m.inputModel.error == "?" || m.inputModel.trend == "?" || m.inputModel.seasonal == "?" || m.inputModel.armaIdent)
            m.ident(false);
        else {
            m.estim(false);
        }
        m.validate();
        //Back to MATLAB
        plhs[0] = armaCreateMxMatrix(m.inputModel.comp.n_rows, m.inputModel.comp.n_cols); 
        armaSetPr(plhs[0], m.inputModel.comp);
        
        vector<string> table = m.inputModel.table;
        int elem = table.size();
        mxArray *tab = mxCreateCellMatrix(elem, 1);
        for (mwIndex i = 0; i<elem; i++) {
            mxArray *str = mxCreateString(table[i].c_str());
            mxSetCell(tab, i, mxDuplicateArray(str));
        }
        plhs[1] = tab; //table is back in cell format

        plhs[2] = mxCreateString(m.inputModel.compNames.c_str());
    } else if (command== "components"){
        if (m.inputModel.error == "?" || m.inputModel.trend == "?" || m.inputModel.seasonal == "?" || m.inputModel.armaIdent)
            m.ident(false);
        else {
            m.estim(false);
        }
        m.components();
        //Back to MATLAB
        plhs[0] = armaCreateMxMatrix(m.inputModel.comp.n_rows, m.inputModel.comp.n_cols); 
        armaSetPr(plhs[0], m.inputModel.comp);

        plhs[1] = mxCreateString(m.inputModel.compNames.c_str());
    }
}
