/*
 * UCompC.cpp is a bridge between MATLAB/Octave and C++ to use forecasting and estimation tools
 * developed with Armadillo library.
 *
 * Note: read the file README.txt for information about building MEX files
 *
 * Created with:
 * MATLAB R2019a
 * Platform: win64
 * MinGW64 Compiler (C++)
 *
 * MEX File function
 */

#include "cpp/defines.hpp"
#include "cpp/armaMex.hpp"
// #include "mex.hpp"
// #include "mexAdapter.hpp"
// #include <stdio.h>
#include <armadillo>
#include <string>
#include <vector>
#include <math.h>
using namespace arma;
using namespace std;
// #include "cpp/DJPTtools.h"
// #include "cpp/optim.h"
// #include "cpp/stats.h"
// #include "cpp/SSpace.h"
// #include "cpp/ARMAmodel.h"
#include "cpp/BSMmodel.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

    string command = mxArrayToString(prhs[0]);
    vec y = armaGetPr(prhs[1]);
    mat u = armaGetPr(prhs[2]);
    string model = mxArrayToString(prhs[3]);
    int h = mxGetScalar(prhs[4]);
    mat comp = armaGetPr(prhs[5]);
    mat compV = armaGetPr(prhs[6]);
    vec v = armaGetPr(prhs[7]);
    vec yFit = armaGetPr(prhs[8]);
    vec yFor = armaGetPr(prhs[9]);
    vec yFitV = armaGetPr(prhs[10]);
    vec yForV = armaGetPr(prhs[11]);
    mat a = armaGetPr(prhs[12]);   
    mat P = armaGetPr(prhs[13]); 
    mat eta = armaGetPr(prhs[14]); 
    vec eps = armaGetPr(prhs[15]);
    const mwSize *dims = mxGetDimensions(prhs[16]); 
    const mxArray *cell = prhs[16];
    const mxArray *cellElem;
    mwIndex jcell;
    vector<string> table;
    string t;
    for(jcell=0;jcell<dims[0];jcell++){
        cellElem = mxGetCell(cell,jcell);
        t = mxArrayToString(cellElem);
        table.push_back(t);
    }
    double outlier = mxGetScalar(prhs[17]); 
    bool tTest = mxIsLogicalScalarTrue(prhs[18]);
    string criterion = mxArrayToString(prhs[19]);
    vec periods = armaGetPr(prhs[20]);
    vec rhos = armaGetPr(prhs[21]);
    bool verbose = mxIsLogicalScalarTrue(prhs[22]);
    bool stepwise = mxIsLogicalScalarTrue(prhs[23]);
    vec p0 = armaGetPr(prhs[24]);
    //bool cLlik = mxIsLogicalScalarTrue(prhs[26]);
    vec criteria = armaGetPr(prhs[25]);
    bool arma = mxIsLogicalScalarTrue(prhs[26]);
    vec grad = armaGetPr(prhs[27]);
    mat covp = armaGetPr(prhs[28]);
    vec p = armaGetPr(prhs[29]);
    double lambda = mxGetScalar(prhs[30]); 
    vec TVP = armaGetPr(prhs[31]);
    string trendOptions = mxArrayToString(prhs[32]);
    string seasonalOptions = mxArrayToString(prhs[33]);
    string irregularOptions = mxArrayToString(prhs[34]);
    int d_t = mxGetScalar(mxGetFieldByNumber(prhs[35],0,0));
    string estimOk=mxArrayToString(mxGetFieldByNumber(prhs[35],0,1));
    double objFunValue = mxGetScalar(mxGetFieldByNumber(prhs[35],0,2));
    double innVariance = mxGetScalar(mxGetFieldByNumber(prhs[35],0,3));
    int nonStationaryTerms = mxGetScalar(mxGetFieldByNumber(prhs[35],0,4));
    vec ns = armaGetPr(mxGetFieldByNumber(prhs[35],0,5));
    vec nPar = armaGetPr(mxGetFieldByNumber(prhs[35],0,6));
    vec harmonics = armaGetPr(mxGetFieldByNumber(prhs[35],0,7));
    vec constPar = armaGetPr(mxGetFieldByNumber(prhs[35],0,8));
    vec typePar = armaGetPr(mxGetFieldByNumber(prhs[35],0,9));
    mat cycleLimits = armaGetPr(mxGetFieldByNumber(prhs[35],0,10));
    mat typeOutliers = armaGetPr(mxGetFieldByNumber(prhs[35],0,11));
    vec beta = armaGetPr(mxGetFieldByNumber(prhs[35],0,12));
    vec betaV = armaGetPr(mxGetFieldByNumber(prhs[35],0,13));
    vec truePar = armaGetPr(mxGetFieldByNumber(prhs[35],0,14));
    double seas = mxGetScalar(mxGetFieldByNumber(prhs[35],0,15));
    int Iter = mxGetScalar(mxGetFieldByNumber(prhs[35],0,16));
    bool MSOE = mxIsLogicalScalarTrue(mxGetFieldByNumber(prhs[35],0,17));
    bool PTSnames = mxIsLogicalScalarTrue(mxGetFieldByNumber(prhs[35],0,18));

    // Correcting dimensions of u (k x n)
    size_t k = u.n_rows;
    size_t n = u.n_cols;
    mat up(k,n); 
    mat typeOutliersp(typeOutliers.n_rows,typeOutliers.n_cols);
    up=u;
    typeOutliersp=typeOutliers;
    if (k > n){
        up = up.t();
    }
    if (k == 1 && n == 2){
        up.resize(0);
    }
    if (typeOutliers(0, 0) == -1){
        typeOutliersp.reset(); 
    }
//     if (k > n){
//         u = u.t();
//     }
//     printf("%s", "line 99\n");
//     if (k == 1 && n == 2){
//         printf("%s", "inside if 103");
//         u.reset();
//         printf("%s", "inside if 105 after reset()");
//     }
//     u.print("u 105");
//     printf("%s", "line 103\n");
//     if (typeOutliers(0, 0) == -1){
//         typeOutliers.reset();
//     }
//     printf("%s", "line 107\n");
//     periods.print("periods 108");
    //double outlier = rubbish(4);
    vec pp(2); pp(0) = periods.n_elem * 2 + 2; pp(1) = sum(ns);
    int iniObs = max(pp);
    //int iniObs;
    // Setting inputs
    SSinputs inputsSS;
    BSMmodel inputsBSM;
    // Pre-processing
    bool errorExit = preProcess(y, up, model, h, outlier, criterion, periods, p0, iniObs,
                                trendOptions, seasonalOptions, irregularOptions, TVP, lambda);
    if (errorExit){
        string model = "error";
        plhs[2] = mxCreateString(model.c_str());
        armaSetPr(plhs[2],model);
        return;
    }
    if (sum(TVP) > 0)
        outlier = 0;
    // End of pre-processing
    if (command == "estimate"){
        inputsSS.y = y.rows(iniObs, y.n_elem - 1);
    } else {
        inputsSS.y = y;
    }
    mat uIni;
    if (iniObs > 0 && up.n_rows > 0 && command == "estimate"){
        inputsSS.u = up.cols(iniObs, up.n_cols - 1);
        uIni = up.cols(0, iniObs - 1);
    } else {
        inputsSS.u= up;
    }
    inputsBSM.model = model;
    inputsBSM.periods = periods;
    inputsBSM.rhos = rhos;
    inputsSS.h = h;
    inputsBSM.tTest = tTest;
    inputsBSM.criterion = criterion;
    inputsSS.grad = grad; //rubbish2.col(0);
    inputsSS.p = p;
    inputsSS.p0 = p0;
    inputsSS.v = v;
    inputsSS.F = yFitV;
    inputsSS.d_t = d_t; //rubbish(0);
    inputsSS.innVariance = innVariance; //rubbish(1);
    inputsSS.objFunValue = objFunValue; //rubbish(2);
    inputsSS.cLlik = true; //rubbish(3);
    inputsSS.outlier = outlier;
//     vec aux(1); aux(0) = inputsSS.outlier;
//     if (aux.has_nan()){
//         inputsSS.outlier = 0;
//     }
    inputsSS.Iter = Iter; //rubbish(6);
    inputsSS.verbose = verbose;
    inputsSS.estimOk = estimOk;
    inputsSS.nonStationaryTerms = nonStationaryTerms;
    inputsSS.criteria = criteria;
    inputsSS.betaAug = beta;
    inputsSS.betaAugVar = betaV;
    
    inputsBSM.seas = seas; //rubbish(7);
    inputsBSM.stepwise = stepwise;
    //inputsBSM.ns = rubbish3.col(0);
    inputsBSM.nPar = nPar; //rubbish3.col(1);
    if (harmonics.has_nan()){
        inputsBSM.harmonics.resize(1);
        inputsBSM.harmonics(0) = 0;
    } else {
        inputsBSM.harmonics = conv_to<uvec>::from(harmonics);
    }
    inputsBSM.constPar = constPar; //rubbish2.col(1);
    inputsBSM.typePar = typePar; //rubbish2.col(2);
    inputsBSM.typeOutliers = typeOutliersp;
    inputsBSM.arma = arma; //rubbish(5);
    inputsBSM.MSOE = MSOE;
    inputsBSM.PTSnames = PTSnames;
    inputsBSM.TVP = TVP;
    inputsBSM.trendOptions = trendOptions;
    inputsBSM.seasonalOptions = seasonalOptions;
    inputsBSM.irregularOptions = irregularOptions;
    // inputsBSM.iniObs = iniObs;

    // BoxCox transformation
    if (lambda == 9999.9)
        lambda = testBoxCox(y, periods);
    inputsBSM.lambda = lambda;
    inputsSS.y = BoxCox(inputsSS.y, inputsBSM.lambda);

    // Building model
    BSMclass sysBSM = BSMclass(inputsSS, inputsBSM);
    // Commands
    SSinputs inputs;
    BSMmodel inputs2;
    
    if (command == "estimate"){
        // Estimating and Forecasting
        sysBSM.estim(inputsSS.verbose);
        sysBSM.forecast();
        
        // Values to return
        inputs = sysBSM.SSmodel::getInputs();
        inputs2 = sysBSM.getInputs();
        vec harmonicsVec = conv_to<vec>::from(inputs2.harmonics);
        inputsBSM.harmonics = conv_to<uvec>::from(harmonics);
        
        // Correcting ns
        inputs2.ns(2) = inputs2.periods.n_elem * 2 - any(inputs2.periods == 2);
        mat pars = join_horiz(inputs.p, inputs.pTransform);
        //Further corrections due to interpolation
        if (iniObs > 0){
            if (inputs.u.n_rows > up.n_rows){  // Outlier outputs
                uIni = join_vert(uIni, zeros(inputs.u.n_rows - up.n_rows, iniObs));
            }
            if (up.n_rows > 0){
                // Check outliers that add u for outliers
                up = join_horiz(uIni, inputs.u);
            }
        } else {
            up = inputs.u;
        }
        // Back to MATLAB:
        plhs[0] = armaCreateMxMatrix(inputs.p.n_rows,inputs.p.n_cols); 
        armaSetPr(plhs[0],pars);
        
        plhs[1] = armaCreateMxMatrix(inputs.p0.n_rows,inputs.p0.n_cols); 
        armaSetPr(plhs[1],inputs2.p0Return); 
        
        plhs[2] = mxCreateString(inputs2.model.c_str());
        
        plhs[3] = armaCreateMxMatrix(inputs.yFor.n_rows,inputs.yFor.n_cols); 
        armaSetPr(plhs[3],inputs.yFor);
        
        plhs[4] = armaCreateMxMatrix(inputs2.periods.n_rows,inputs2.periods.n_cols); 
        armaSetPr(plhs[4],inputs2.periods);
        
        plhs[5] = armaCreateMxMatrix(inputs2.rhos.n_rows,inputs2.rhos.n_cols); 
        armaSetPr(plhs[5],inputs2.rhos);
        
        plhs[6] = armaCreateMxMatrix(inputs.FFor.n_rows,inputs.FFor.n_cols); 
        armaSetPr(plhs[6],inputs.FFor);
        
        plhs[7] = mxCreateString(inputs.estimOk.c_str());
        
        plhs[8] = armaCreateMxMatrix(harmonicsVec.n_rows,harmonicsVec.n_cols); 
        armaSetPr(plhs[8],harmonicsVec);
        
        plhs[9] = armaCreateMxMatrix(inputs2.cycleLimits.n_rows,inputs2.cycleLimits.n_cols); 
        armaSetPr(plhs[9],inputs2.cycleLimits);
        
        nonStationaryTerms = inputs.nonStationaryTerms;
        plhs[10] = mxCreateDoubleScalar(1);
        *mxGetPr(plhs[10]) = nonStationaryTerms;
        
        plhs[11] = armaCreateMxMatrix(inputs.betaAug.n_rows,inputs.betaAug.n_cols); 
        armaSetPr(plhs[11],inputs.betaAug);
        
        plhs[12] = armaCreateMxMatrix(inputs.betaAugVar.n_rows,inputs.betaAugVar.n_cols); 
        armaSetPr(plhs[12],inputs.betaAugVar);
        
        plhs[13] = armaCreateMxMatrix(up.n_rows,up.n_cols); 
        armaSetPr(plhs[13],up);
        
        plhs[14] = armaCreateMxMatrix(inputs2.typeOutliers.n_rows,inputs2.typeOutliers.n_cols); 
        armaSetPr(plhs[14],inputs2.typeOutliers);
        
        plhs[15] = armaCreateMxMatrix(inputs.criteria.n_rows,inputs.criteria.n_cols); 
        armaSetPr(plhs[15],inputs.criteria);
        
        plhs[16] = mxCreateDoubleScalar(1);
        *mxGetPr(plhs[16]) = inputs.d_t + iniObs;
        
        plhs[17] = mxCreateDoubleScalar(1);
        *mxGetPr(plhs[17]) = inputs.innVariance;
         
        plhs[18] = mxCreateDoubleScalar(1);
        *mxGetPr(plhs[18]) = inputs.objFunValue;
        
        plhs[19] = armaCreateMxMatrix(inputs.grad.n_rows,inputs.grad.n_cols); 
        armaSetPr(plhs[19],inputs.grad);
        
        plhs[20] = armaCreateMxMatrix(inputs2.constPar.n_rows,inputs2.constPar.n_cols); 
        armaSetPr(plhs[20],inputs2.constPar);
        
        plhs[21] = armaCreateMxMatrix(inputs2.typePar.n_rows,inputs2.typePar.n_cols); 
        armaSetPr(plhs[21],inputs2.typePar);
        
        plhs[22] = armaCreateMxMatrix(inputs2.ns.n_rows,inputs2.ns.n_cols); 
        armaSetPr(plhs[22],inputs2.ns);
        
        plhs[23] = armaCreateMxMatrix(inputs2.nPar.n_rows,inputs2.nPar.n_cols); 
        armaSetPr(plhs[23],inputs2.nPar);

        plhs[24] = mxCreateDoubleScalar(1);
        *mxGetPr(plhs[24]) = inputs.h;

        plhs[25] = mxCreateDoubleScalar(1);
        *mxGetPr(plhs[25]) = inputs.outlier;

        plhs[26] = mxCreateDoubleScalar(1);
        *mxGetPr(plhs[26]) = inputs.Iter;

        plhs[27] = mxCreateDoubleScalar(1);
        *mxGetPr(plhs[27]) = inputs2.lambda;


    } else if(command == "validate"){
        sysBSM.validate(false);
        
        // Values to return
        inputs = sysBSM.SSmodel::getInputs();
        inputs2 = sysBSM.getInputs();

        //Back to MATLAB
        plhs[0] = armaCreateMxMatrix(inputs.v.n_rows,inputs.v.n_cols); 
        armaSetPr(plhs[0],inputs.v);

        table = inputs.table;
        int elem = table.size();
        mxArray *tab = mxCreateCellMatrix(elem, 1);
        for (mwIndex i = 0; i<elem; i++) {
            mxArray *str = mxCreateString(table[i].c_str());
            mxSetCell(tab, i, mxDuplicateArray(str));
        }

        plhs[1] = tab; //table is back in cell format
        
        plhs[2] = armaCreateMxMatrix(inputs.coef.n_rows,inputs.coef.n_cols); 
        armaSetPr(plhs[2],inputs.coef);

        //plhs[3] = mxCreateString(inputs2.parNames.c_str());
        
    }else if(command == "filter" || command == "smooth" || command == "disturb"){
        
        sysBSM.setSystemMatrices();
        if (command == "filter"){
            sysBSM.filter();
        } else if (command == "smooth") {
            sysBSM.smooth(false);
        } else {
            sysBSM.disturb();
        }
        // Corrections for interpolation
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
        
        // Values to return
        inputs = sysBSM.SSmodel::getInputs();
        inputs2 = sysBSM.getInputs();
        
        //Back to MATLAB
        plhs[0] = armaCreateMxMatrix(inputs.a.n_rows,inputs.a.n_cols); 
        armaSetPr(plhs[0],inputs.a);
        
        plhs[1] = armaCreateMxMatrix(inputs.P.n_rows,inputs.P.n_cols); 
        armaSetPr(plhs[1],inputs.P);
        
        plhs[2] = armaCreateMxMatrix(inputs.v.n_rows,inputs.v.n_cols); 
        armaSetPr(plhs[2],inputs.v);
        
        plhs[3] = armaCreateMxMatrix(inputs.F.n_rows,inputs.F.n_cols); 
        armaSetPr(plhs[3],inputs.F);
        
        plhs[4] = armaCreateMxMatrix(inputs.yFit.n_rows,inputs.yFit.n_cols); 
        armaSetPr(plhs[4],inputs.yFit);
        
        plhs[5] = armaCreateMxMatrix(inputs2.eps.n_rows,inputs2.eps.n_cols); 
        armaSetPr(plhs[5],inputs2.eps);
        
        plhs[6] = armaCreateMxMatrix(inputs.eta.n_rows,inputs.eta.n_cols); 
        armaSetPr(plhs[6],inputs.eta);
        
        plhs[7] = mxCreateString(statesN.c_str());

    }else if (command == "components"){
        
        sysBSM.setSystemMatrices();
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
        inputs2 = sysBSM.getInputs();

        //Back to MATLAB
        plhs[0] = armaCreateMxMatrix(inputs2.comp.n_rows,inputs2.comp.n_cols); 
        armaSetPr(plhs[0],inputs2.comp);
        
        plhs[1] = armaCreateMxMatrix(inputs2.compV.n_rows,inputs2.compV.n_cols); 
        armaSetPr(plhs[1],inputs2.compV);
        
        int m=inputs2.comp.n_rows;
        plhs[2] = mxCreateDoubleScalar(1);
        *mxGetPr(plhs[2]) = m;

        plhs[3] = mxCreateString(inputs2.compNames.c_str());
        
    }
    return;
}
