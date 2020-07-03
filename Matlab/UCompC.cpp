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

#include "armaMex.hpp"
#include <stdio.h>
#include <armadillo>
#include <string>
#include <vector>
#include <math.h>
using namespace arma;
using namespace std;
#include "DJPTtools.h"
#include "optim.h"
#include "stats.h"
#include "SSpace.h"
#include "ARMAmodel.h"
#include "BSMmodel.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

    //Converting MATLAB inputs to C++
    string commands = mxArrayToString(prhs[0]);
    vec y = armaGetPr(prhs[1]);
    mat u = armaGetPr(prhs[2]);
    string model = mxArrayToString(prhs[3]);
    int h = mxGetScalar(prhs[4]);
    mat comp = armaGetPr(prhs[5]);
    mat compV = armaGetPr(prhs[6]);
    vec p = armaGetPr(prhs[7]);
    vec v = armaGetPr(prhs[8]);
    vec yFit = armaGetPr(prhs[9]);
    vec yFor = armaGetPr(prhs[10]);
    vec yFitV = armaGetPr(prhs[11]);
    vec yForV = armaGetPr(prhs[12]);
    mat a = armaGetPr(prhs[13]);   
    mat P = armaGetPr(prhs[14]); 
    mat eta = armaGetPr(prhs[15]); 
    vec eps = armaGetPr(prhs[16]);
    const mwSize *dims = mxGetDimensions(prhs[17]); 
    const mxArray *cell = prhs[17];
    const mxArray *cellElem;
    mwIndex jcell;
    vector<string> table;
    string t;
    for(jcell=0;jcell<dims[0];jcell++){
        cellElem = mxGetCell(cell,jcell);
        t = mxArrayToString(cellElem);
        table.push_back(t);
    }
    double outlier = mxGetScalar(prhs[18]); 
    bool tTest = mxIsLogicalScalarTrue(prhs[19]);
    string criterion = mxArrayToString(prhs[20]);
    vec periods = armaGetPr(prhs[21]);
    vec rhos = armaGetPr(prhs[22]);
    bool verbose = mxIsLogicalScalarTrue(prhs[23]);
    bool stepwise = mxIsLogicalScalarTrue(prhs[24]);
    vec p0 = armaGetPr(prhs[25]);
    bool cLlik = mxIsLogicalScalarTrue(prhs[26]);
    vec criteria = armaGetPr(prhs[27]);
    bool arma = mxIsLogicalScalarTrue(prhs[28]);
    vec grad = armaGetPr(mxGetFieldByNumber(prhs[29],0,0));
    int d_t = mxGetScalar(mxGetFieldByNumber(prhs[29],0,1));
    string estimOk=mxArrayToString(mxGetFieldByNumber(prhs[29],0,2));
    double objFunValue = mxGetScalar(mxGetFieldByNumber(prhs[29],0,3));
    double innVariance = mxGetScalar(mxGetFieldByNumber(prhs[29],0,4));
    int nonStationaryTerms = mxGetScalar(mxGetFieldByNumber(prhs[29],0,5));
    vec ns = armaGetPr(mxGetFieldByNumber(prhs[29],0,6));
    vec nPar = armaGetPr(mxGetFieldByNumber(prhs[29],0,7));
    vec harmonics = armaGetPr(mxGetFieldByNumber(prhs[29],0,8));
    vec constPar = armaGetPr(mxGetFieldByNumber(prhs[29],0,9));
    vec typePar = armaGetPr(mxGetFieldByNumber(prhs[29],0,10));
    mat cycleLimits = armaGetPr(mxGetFieldByNumber(prhs[29],0,11));
    mat typeOutliers = armaGetPr(mxGetFieldByNumber(prhs[29],0,12));
    mat beta = armaGetPr(mxGetFieldByNumber(prhs[29],0,13));
    vec betaV = armaGetPr(mxGetFieldByNumber(prhs[29],0,14));
    
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
    
    // Setting inputs
    SSinputs inputsSS;
    BSMinputs inputsBSM;
    inputsSS.y = y;
    inputsSS.u = up;
    inputsBSM.model = model;
    inputsBSM.periods = periods;
    inputsBSM.rhos = rhos;
    inputsSS.h = h;
    inputsBSM.tTest = tTest;
    inputsBSM.criterion = criterion;
    inputsSS.grad = grad;
    inputsSS.p = p;
    inputsSS.p0 = p0;
    inputsSS.v = v;
    inputsSS.F = yFitV;
    inputsSS.d_t = d_t;
    inputsSS.innVariance = innVariance;
    inputsSS.objFunValue = objFunValue;
    inputsSS.cLlik = cLlik;
    inputsSS.outlier = outlier;
    vec aux(1); aux(0) = inputsSS.outlier;
    if (aux.has_nan()){
        inputsSS.outlier = 0;
    }
    inputsSS.verbose = verbose;
    inputsSS.estimOk = estimOk;
    inputsSS.nonStationaryTerms = nonStationaryTerms;
    inputsSS.criteria = criteria;
    inputsSS.betaAug = beta;
    inputsSS.betaAugVar = betaV;
    inputsBSM.stepwise = stepwise;
    inputsBSM.ns = ns;
    inputsBSM.nPar = nPar;
    if (harmonics.has_nan()){
        inputsBSM.harmonics.resize(1);
        inputsBSM.harmonics(0) = datum::nan;
    } else {
        inputsBSM.harmonics = conv_to<uvec>::from(harmonics);
    }
    inputsBSM.constPar = constPar;
    inputsBSM.typePar = typePar;
    inputsBSM.typeOutliers = typeOutliersp;
    inputsBSM.arma = arma;
    
    // Building model
    BSMmodel sysBSM = BSMmodel(inputsSS, inputsBSM); 
    
    // Commands
    SSinputs inputs;
    BSMinputs inputs2;
    
    if (commands == "estimate"){
        
        // Estimating and Forecasting
        sysBSM.estim();
        sysBSM.forecast();
        
        // Values to return
        inputs = sysBSM.SSmodel::getInputs();
        inputs2 = sysBSM.getInputs();
        vec harmonicsVec = conv_to<vec>::from(inputs2.harmonics);
        inputsBSM.harmonics = conv_to<uvec>::from(harmonics);
        
        // Back to MATLAB:
        plhs[0] = armaCreateMxMatrix(inputs.p.n_rows,inputs.p.n_cols); 
        armaSetPr(plhs[0],inputs.p);
        
        plhs[1] = armaCreateMxMatrix(inputs.p0.n_rows,inputs.p0.n_cols); 
        armaSetPr(plhs[1],inputs.p0); 
        
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
        
        plhs[13] = armaCreateMxMatrix(inputs.u.n_rows,inputs.u.n_cols); 
        armaSetPr(plhs[13],inputs.u);
        
        plhs[14] = armaCreateMxMatrix(inputs2.typeOutliers.n_rows,inputs2.typeOutliers.n_cols); 
        armaSetPr(plhs[14],inputs2.typeOutliers);
        
        plhs[15] = armaCreateMxMatrix(inputs.criteria.n_rows,inputs.criteria.n_cols); 
        armaSetPr(plhs[15],inputs.criteria);
        
        plhs[16] = mxCreateDoubleScalar(1);
        *mxGetPr(plhs[16]) = inputs.d_t;
        
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
        
    }else if(commands == "validate"){
        
        sysBSM.validate();
        
        // Values to return
        inputs = sysBSM.SSmodel::getInputs();
        
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
        
    }else if(commands == "filter" || commands == "smooth" || commands == "disturb"){
        
        sysBSM.setSystemMatrices();
        if (commands == "filter"){
            sysBSM.filter();
        } else if (commands == "smooth") {
            sysBSM.smooth(false);
        } else {
            sysBSM.disturb();
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
        
    }else if (commands == "components"){
        
        sysBSM.setSystemMatrices();
        sysBSM.components();
        
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
    }
}
