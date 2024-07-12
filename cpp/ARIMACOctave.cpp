/*
 * ARIMACOctave.cpp is a bridge between MATLAB/Octave and C++ to use forecasting and estimation tools
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
#include "cpp/ARIMAmodel.h"

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
    vec orders = armaGetPr(prhs[3]);
    if (orders.n_rows == 1 && orders(0) == -99999)
        orders.resize(0);
    double cnst = mxGetScalar(prhs[4]);
    double sAux = mxGetScalar(prhs[5]);
    int s = (int)sAux;
    string criterion = mxArrayToString(prhs[6]);
    int h = mxGetScalar(prhs[7]);
    bool verbose = mxIsLogicalScalarTrue(prhs[8]);
    double lambda = mxGetScalar(prhs[9]);
    vec maxOrders = armaGetPr(prhs[10]);
    bool bootstrap = mxIsLogicalScalarTrue(prhs[11]);
    int nSimul = mxGetScalar(prhs[12]);
    bool fast = mxIsLogicalScalarTrue(prhs[13]);
    bool identDiff = mxIsLogicalScalarTrue(prhs[14]);
    string identMethod = mxArrayToString(prhs[15]);
    // Correcting inputs
    ARIMAmodel input;
    ARIMAclass m(input);
    // m = preProcess(y, u, orders, cnst, s, h, criterion, verbose, lambda,
    //                maxOrders, bootstrap, nSimul, identDiff, identMethod);
    m = preProcess(y, u, orders, cnst, s, h, verbose, lambda,
                   maxOrders, bootstrap, nSimul, criterion);
    if (m.m.errorExit){
        plhs[0] = armaCreateMxMatrix(0, 0);
        armaSetPr(plhs[0], {});
        return;
    }
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
    // Back to MATLAB:
    plhs[0] = armaCreateMxMatrix(m.m.par.n_rows, m.m.par.n_cols);
    armaSetPr(plhs[0], m.m.par);

    plhs[1] = armaCreateMxMatrix(m.m.yFor.n_rows, m.m.yFor.n_cols);
    armaSetPr(plhs[1], m.m.yFor);

    plhs[2] = armaCreateMxMatrix(m.m.FFor.n_rows, m.m.FFor.n_cols);
    armaSetPr(plhs[2], m.m.FFor);

    plhs[3] = armaCreateMxMatrix(m.m.ySimul.n_rows, m.m.ySimul.n_cols);
    armaSetPr(plhs[3], m.m.ySimul);

    plhs[4] = mxCreateDoubleScalar(1);
    *mxGetPr(plhs[4]) = m.m.lambda;

    plhs[5] = armaCreateMxMatrix(m.m.orders.n_rows, m.m.orders.n_cols);
    armaSetPr(plhs[5], m.m.orders);

    plhs[6] = mxCreateDoubleScalar(1);
    *mxGetPr(plhs[6]) = m.m.cnst;

    plhs[7] = armaCreateMxMatrix(m.m.u.n_rows, m.m.u.n_cols);
    armaSetPr(plhs[7], m.m.u);

    plhs[8] = mxCreateDoubleScalar(1);
    *mxGetPr(plhs[8]) = m.m.BIC;

    plhs[9] = mxCreateDoubleScalar(1);
    *mxGetPr(plhs[9]) = m.m.AIC;

    plhs[10] = mxCreateDoubleScalar(1);
    *mxGetPr(plhs[10]) = m.m.AICc;

    plhs[11] = mxCreateDoubleScalar(1);
    *mxGetPr(plhs[11]) = m.m.IC;

    vector<string> table = m.m.table;
    int elem = table.size();
    mxArray *tab = mxCreateCellMatrix(elem, 1);
    for (mwIndex i = 0; i<elem; i++) {
        mxArray *str = mxCreateString(table[i].c_str());
        mxSetCell(tab, i, mxDuplicateArray(str));
    }
    plhs[12] = tab; //table is back in cell format
    
    plhs[13] = armaCreateMxMatrix(m.m.v.n_rows, m.m.v.n_cols);
    armaSetPr(plhs[13], m.m.v);
    return;
}
