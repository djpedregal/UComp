 /*************************
 Basic Structural models
 Needs Armadillo
 Needs SSpace.h
 Needs ARMAmodel.h
 Needs BSMident.h
 Needs stats.h
 Needs DJPTtools.h
***************************/
struct BSMinputs{
  string model = "llt/none/equal/arma(0,0)", // model to fit
         criterion = "aic", // identification criterion
         trend,   // type of trend
         cycle,   // type of cycle
         seasonal,  // type of seasonal component
         irregular, // type of irregular
         cycle0;  // type of cycle without numbers
  int ar, ma;     // AR and MA orders of irregular component
  vec periods,    // vector of periods for harmonics
      rhos,       // vector indicating whether period is cyclical or seasonal
      ns,         // number of states in components (trend, cycle, seasonal, irregular)
      nPar,       // number of parameters in components (trend, cycle, seasonal, irregular)
      typePar,    // type of parameter (0: variance; -1: damped of trend; 1: cycle rhos; 
                  //                    2: cycle periods; 3: ARMA; 4: inputs)
      eps,        // observation perturbation 
      beta0ARMA,  // initial estimates of ARMA model (without variance)
      // beta0REG,   // initial estimates of regression terms
      constPar;   // constrained parameters (0: not constrained; 1: concentrated-out; 
                  //                         2: variance constrained to 0;
                  //                         3: alpha constrained to 0 or 1)
  uvec harmonics; // vector with the indices of harmonics selected
  mat comp,       // estimated components
      compV,      // variance of components
      typeOutliers, // Matrix with type of outliers and sample of each outlier
      cycleLimits; // limits for period of cycle estimation
  bool stepwise,  // stepwise identification or brute one
       tTest = false, // unit roots test or not for identification
       arma = true;   // check arma models for irregular component
};
/**************************
 * Model CLASS BSM
 ***************************/
class BSMmodel : public SSmodel{
  private:
    BSMinputs inputs;
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
    BSMmodel(SSinputs, BSMinputs);
    //Estimation
    void estim();
    void estim(vec);
    // Identification
    void ident();
    // Outlier detection
    void estimOutlier();
    // Components
    void components();
    // Validation of BSM models
    void validate();
    // Disturbance smoother (to recover just trend and epsilons)
    void disturb();
    // Get data
    BSMinputs getInputs(){
      return inputs;
    }
    // Set data
    void setInputs(BSMinputs inputs){
      this->inputs = inputs;
    }
    //Print inputs on screen
    // void showInputs();
};
/***************************************************
 * Auxiliar function declarations
 ****************************************************/
// Variance matrices in standard BSM
void bsmMatrices(vec, SSmatrix*, void*);
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
void calculateLimits(int, vec, vec, mat&);
/****************************************************
 // BSM implementations for univariate UC models
 ****************************************************/
// Constructor
BSMmodel::BSMmodel(SSinputs data, BSMinputs inputs) : SSmodel(data){
  inputs.rhos = ones(size(inputs.periods));
  lower(inputs.criterion);
  SSmodel::inputs = data;
  this->inputs = inputs;
  this->inputs.cycleLimits.resize(1, 1);
  this->inputs.cycleLimits(0, 0) = datum::nan;
  vec reserve = inputs.constPar;
  setModel(inputs.model, inputs.periods, inputs.rhos, true);
  if (!reserve.has_nan())
      this->inputs.constPar = reserve;
}
// Set model (part of constructor)
void BSMmodel::setModel(string model, vec periods, vec rhos, bool runFromConstructor){
  string trend, cycle, seasonal, irregular;
  vec ns(5), nPar(5), typePar, noVar, constPar;
  mat cycleLimits;
  splitModel(model, trend, cycle, seasonal, irregular);
  // Checking cycle model and correcting from string input
  if (cycle[0] != 'n' && cycle != "?"){
    modelCorrect(model, cycle, inputs.cycle0, periods, rhos);
  }
  if (cycle[0] != 'n' && inputs.cycleLimits.has_nan()){
    calculateLimits(SSmodel::inputs.y.n_elem, periods, rhos, cycleLimits);
    this->inputs.cycleLimits = cycleLimits;
  } else if (cycle[0] != 'n') {
    cycleLimits = inputs.cycleLimits;
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
        SSmodel::inputs.p0(0) = datum::nan;
    typePar = this->inputs.typePar;
    initParBsm();
  }
  this->inputs.trend = trend;
  this->inputs.cycle = cycle;
  this->inputs.seasonal = seasonal;
  this->inputs.irregular = irregular;
}
// Print inputs on screen
// void BSMmodel::showInputs(){
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
// Estimation: runs estim(p) or ident()
void BSMmodel::estim(){
  if (inputs.trend != "?" && inputs.cycle != "?" && inputs.seasonal != "?" && inputs.irregular != "?"){
      // Particular model
      if (SSmodel::inputs.outlier == 0){
        // Without outlier detection
        estim(SSmodel::inputs.p0);
      } else {
        // With outlier detection
        estimOutlier();
      }
  } else {
    // Some or all the components to identify
      ident();
  }
}
void BSMmodel::estim(vec p){
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
    // Revert rhos and periods to their original values
    uvec aux = find(inputs.rhos > 0);
    inputs.rhos = inputs.rhos(aux);
    inputs.periods = inputs.periods(aux);
    SSmodel::inputs.v.reset();
}
// Estimation of a family of UC models
void BSMmodel::estimUCs(vector <string> allUCModels, uvec harmonics, 
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
  BSMinputs bestBSM = inputs;
  if (isnan(oldMinCrit)){
    oldMinCrit = 1e12;
  }
  minCrit = oldMinCrit;
  for (unsigned int i = 0; i < allUCModels.size(); i++){
    setModel(allUCModels[i], inputs.periods(harmonics), inputs.rhos(harmonics), false);
    // Cleaning variables for outliers starting anew
    if (SSmodel::inputs.u.n_elem > 0){
      if (nuInit > 0){
        SSmodel::inputs.u = SSmodel::inputs.u.rows(0, nuInit - 1);
      } else {
        SSmodel::inputs.u.resize(0);
      }
    }
    // Model estimation
    estim();
    AIC = SSmodel::inputs.criteria(1);
    BIC = SSmodel::inputs.criteria(2);
    AICc = SSmodel::inputs.criteria(3);
    // Avoid selecting a model with problems
    if (AIC == -datum::inf || AIC == datum::inf){
      AIC = BIC = AICc = datum::nan;
    }
    if (VERBOSE){
      printf(" %*s: %8.4f %8.4f %8.4f\n", 30, allUCModels[i].c_str(), AIC, BIC, AICc);
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
}
// Identification
void BSMmodel::ident(){
  wall_clock timer;
  timer.tic();
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
  bool VERBOSE = SSmodel::inputs.verbose;
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
        inputTrend = "some";
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
  if (inputSeasonal[0] != 'n'){
    vec betaHR;
    selectHarmonics(SSmodel::inputs.y, SSmodel::inputs.u, periods, harmonics, betaHR, isSeasonal);
    if (harmonics.n_rows == 0){
      inputSeasonal = "none";
    }
  }
  if (season == 4 && harmonics.n_rows > 0){
      harmonics.reset();
      harmonics = regspace<uvec>(0, 1);
  }
  inputs.harmonics = harmonics;
  // UC identification
  double minCrit; // = 1e12, minCrit1;
  if (VERBOSE){
    printf("------------------------------------------------------------\n");
    if (SSmodel::inputs.outlier < 0){
      printf(" Identification started WITH outlier detection\n");
    } else {
      printf(" Identification started WITHOUT outlier detection\n");
    }
    printf("------------------------------------------------------------\n");
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
  } else if (inputTrend[0] != 's'){
    trendTypes = inputTrend;
    runAll = true;
  } else if (inputTrend[0] == 's'){
    runAll = false;
  }
  if (runAll){    // no stepwise
    if (inputTrend == "?")
      trendTypes = "none/rw/llt/dt";
    if (inputSeasonal == "?")
      seasTypes = "none/equal/different";
    else
      seasTypes = inputSeasonal;
    findUCmodels(trendTypes, cycTypes, seasTypes, irrTypes, allUCModels);
    estimUCs(allUCModels, harmonics, minCrit, VERBOSE, 1e12, nuInit);
  } else {                  // stepwise
    if (inputSeasonal[0] == 'n'){
      // Annual or non seasonal data
      if (inputTrend == "?" || inputTrend == "some")
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
  if (VERBOSE){
    double nSeconds = timer.toc();
    printf("------------------------------------------------------------\n");
    printf("  Identification time: %10.5f seconds\n", nSeconds);
    printf("------------------------------------------------------------\n");
  }
  SSmodel::inputs.verbose = VERBOSE;
  // Updating inputs
  inputs.harmonics = harmonics;
  inputs.rhos = inputs.rhos(harmonics);
  SSmodel::inputs.outlier = outlierCopy;
  if (harmonics.n_elem > 0){
    inputs.periods = inputs.periods(harmonics);
  } else {
    inputs.periods.resize(1);
    inputs.periods.fill(1);
  }
}
// Outlier detection a la Harvey and Koopman
void BSMmodel::estimOutlier(){
  // Havey, A.C. and Koopman, S.J. (1992), Diagnostic checking of unobserved
  //        components time series models , JBES, 10, 377-389.
  int n = SSmodel::inputs.y.n_elem - 1, //nNan,
    nu = SSmodel::inputs.u.n_rows,
    lu = 0;
  bool VERBOSE = SSmodel::inputs.verbose;
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
  estim(SSmodel::inputs.p0);
  inputs.periods = periodsCopy;
  inputs.rhos = rhosCopy;
  // Storing initial model clean
  SSinputs bestSS = SSmodel::inputs;
  BSMinputs bestBSM = inputs;
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
  inputs.typeOutliers = join_vert(join_vert(zeros(size(indAO)), ones(size(indLS))), 2 * ones(size(indSC)));
  uvec ind = join_vert(join_vert(indAO, indLS), indSC);
  vec  val = join_vert(join_vert(valAO, valLS), valSC);
  // Continue in case there are outliers found
  bool cLlikCopy = SSmodel::inputs.cLlik,
    augmentedCopy = SSmodel::inputs.augmented,
    exactCopy = SSmodel::inputs.exact;
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
    estim(SSmodel::inputs.p0);
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
            estim(SSmodel::inputs.p0);
            inputs.periods = periodsCopy;
            inputs.rhos = rhosCopy;
          }
        }
        count++;
      } while (count < 4 && ind.n_elem > 0 && remove.n_elem > 0);
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
}
// Components
void BSMmodel::components(){
  SSmodel::smooth(true);
  int nCycles = sum(inputs.rhos < 0), k = SSmodel::inputs.u.n_rows;
  inputs.comp.set_size(4 + nCycles + k, SSmodel::inputs.yFit.n_rows);
  inputs.comp.fill(datum::nan);
  inputs.compV = inputs.comp;
  vec nsCum = cumsum(inputs.ns);
  // Level
  if (inputs.ns(0) > 0 && SSmodel::inputs.system.T(0, 0) != 0){
      inputs.comp.row(0) = SSmodel::inputs.a.row(0);
      inputs.compV.row(0) = SSmodel::inputs.P.row(0);
  }
  // Slope
  if (inputs.ns(0) > 1){
      inputs.comp.row(1) = SSmodel::inputs.a.row(1);
      inputs.compV.row(1) = SSmodel::inputs.P.row(1);
  }
  // Seasonal
  if (inputs.ns(2) > 0){
      urowvec ind = regspace<urowvec>(nsCum(1), 2, nsCum(2) - 1);
      inputs.comp.row(2) = sum(SSmodel::inputs.a.rows(ind));
      inputs.compV.row(2) = sum(SSmodel::inputs.P.rows(ind));
  }
  // Irregular
  uword ny = SSmodel::inputs.y.n_elem - 1;
  if (inputs.ns(3) == 0){     // White noise
      inputs.comp.row(3).cols(0, ny) = SSmodel::inputs.y.t() - SSmodel::inputs.yFit.rows(0, ny).t();
      inputs.compV.row(3).cols(0, ny) = SSmodel::inputs.F.rows(0, ny).t();
  } else {                    // ARMA
      inputs.comp.row(3) = SSmodel::inputs.a.row(nsCum(2));
      inputs.compV.row(3) = SSmodel::inputs.P.row(nsCum(2));
  }
  // Cycle
  if (inputs.ns(1) > 0){
    for (int i = 0; i < nCycles; i++){
      inputs.comp.row(4 + i) = SSmodel::inputs.a.row(nsCum(0) + 2 * i);
      inputs.compV.row(4 + i) = SSmodel::inputs.P.row(nsCum(0) + 2 * i);
    }
  }
  // Inputs
  if (k > 0){
    for (int i = 0; i < k; i++){
      inputs.comp.row(4 + nCycles + i) = SSmodel::inputs.system.D(i) * 
                                         SSmodel::inputs.u.submat(i, 0, i, SSmodel::inputs.u.n_cols - 1);
    }
  }
  inputs.comp.submat(0, 0, inputs.comp.n_rows - 1, SSmodel::inputs.y.n_elem - 1).replace(datum::nan, 0);
  inputs.comp.submat(0, 0, inputs.comp.n_rows - 1, SSmodel::inputs.y.n_elem - 1).replace(datum::inf, 0);
}
// Validation of BSM models
void BSMmodel::validate(){
    SSmodel::validate(false);
    vector<string> names;
    // Parameter names
    // Trend
    if (inputs.trend[0] == 'd'){
        names.push_back("Damping");
    }
    if (inputs.trend[0] != 'n' && inputs.trend[0] != 'i')
        names.push_back("Level");
    if (inputs.trend[0] != 'n' && inputs.trend[0] != 'r'){
        names.push_back("Slope");
    }
    vec nsCum = cumsum(inputs.ns);
    vec nparCum = cumsum(inputs.nPar);
    // Cycle
    int nCycles = sum(inputs.rhos < 0);
    uvec aux;
    vec pCycle, typeParC;
    if (inputs.cycle[0] != 'n'){
      aux = regspace<uvec>(nparCum(0), nparCum(1) - 1);
      pCycle = SSmodel::inputs.p(aux);
      typeParC = inputs.typePar(aux);
      int count;
      char name[20];
      for (count = 0; count < nCycles; count++){
        sprintf(name, "Rho(%1.0i)", count + 1);
        names.push_back(name);
      }
      for (count = 0; count < nCycles; count++){
        if (inputs.periods(count) < 0){
          sprintf(name, "Period(%1.0i)", count + 1);
          names.push_back(name);
        }
      }
      for (count = 0; count < nCycles; count++){
        sprintf(name, "Var(%1.0i)", count + 1);
        names.push_back(name);
      }
    }
    // Seasonal
    if (inputs.seasonal[0] == 'e')
        names.push_back("Seas(All)");
    if (inputs.seasonal[0] == 'd'){
        char seasNames[20];
        for (unsigned int i = sum(inputs.rhos < 0); i < inputs.periods.n_elem; i++){
            sprintf(seasNames, "Seas(%1.1f)", inputs.periods(i));
            names.push_back(seasNames);
        }
    }
    // Irregular
    if (inputs.irregular[0] != 'n')
        names.push_back("Irregular");
    if (inputs.irregular != "arma(0,0)" && inputs.irregular[0] != 'n'){
        char arNames[20];
        for (int i = 0; i < inputs.ar; i++){
            sprintf(arNames, "AR(%1.0i)", i + 1);
            names.push_back(arNames);
        }
        for (int i = 0; i < inputs.ma; i++){
            sprintf(arNames, "MA(%1.0i)", i + 1);
            names.push_back(arNames);
        }
    }
    // Inputs
    int nOut = inputs.typeOutliers.n_rows;
    int nu = SSmodel::inputs.u.n_rows;
    if (nu - nOut > 0){
      char betas[20];
      for (int i = 0; i < nu - nOut; i++){
        sprintf(betas, "Beta(%1.0i)", i + 1);
        names.push_back(betas);
      }
    }
    // Outliers
    if (nOut > 0){
      char betas[20], typeO[5];
      for (int i = 0; i < nOut; i++){
        if (inputs.typeOutliers(i, 0) == 0){
          sprintf(typeO, "AO");
        } else if (inputs.typeOutliers(i, 0) == 1){
          sprintf(typeO, "LS");
        } else if (inputs.typeOutliers(i, 0) == 2){
          sprintf(typeO, "SC");
        }
        sprintf(betas, "%s%0.0f", typeO, inputs.typeOutliers(i, 1));
        names.push_back(betas);
      }
    }
    // Parameter values
    vec p, stdP;
    int ind1 = 0;
    if (nu > 0){
      int ind1 = SSmodel::inputs.betaAug.n_elem - nu;
      p = join_vert(SSmodel::inputs.p, SSmodel::inputs.betaAug.rows(ind1, ind1 + nu - 1));
      stdP = join_vert(SSmodel::inputs.stdP, sqrt(SSmodel::inputs.betaAugVar.rows(ind1, ind1 + nu - 1)));
    } else {
      p = SSmodel::inputs.p;
      stdP = SSmodel::inputs.stdP;
    }
    // Trend
    vec isVar(p.n_elem);
    isVar.fill(0);
    aux = find(inputs.typePar == 0);
    isVar(aux).fill(1);
    vec parValues(p.n_elem);
    if (inputs.nPar(0) == 2 && inputs.ns(0) == 2){        // LLT
        parValues(0) = exp(2 * p(0));
        parValues(1) = exp(2 * p(1));
    } else if (inputs.nPar(0) == 1 && inputs.ns(0) == 1){  // RW trend
        parValues(0) = exp(2 * p(0));
    } else if (inputs.nPar(0) == 3){                     // Damped trend
        double alpha = p(0);
        constrain(alpha, regspace<vec>(0, 1)); //exp(p(0)) / (1+ exp(p(0)));
        parValues(0) = alpha;
        parValues(1) = exp(2 * p(1));
        parValues(2) = exp(2 * p(2));
        isVar(0) = 0;
    } else if (inputs.nPar(0) == 1 && inputs.ns(0) == 2){    // IRW
        parValues(0) = exp(2 * p(0));
    }
    //Cycle
    if (inputs.nPar(1) > 0){
      // Rhos
      int pos;
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
      parValues(span(pos, nparCum(1) - 1)) = exp(2 * pCycle(span(nCycles + nn, 2 * nCycles + nn - 1)));
    }
    // Seasonal
    if (inputs.nPar(2) > 0){                               // With seasonal
        // Index of states for seasonal
        uvec ind = regspace<uvec>(inputs.ns(1), 1, nsCum(2) - 1);
        if (inputs.nPar(2) == 1){                         // Equal variances
            parValues(nparCum(1)) = exp(2 * p(nparCum(1)));
        } else {                                        // Different variances
            // Index of parameters for seasonal
            uvec ind1 = regspace<uvec>(nparCum(1), 1, nparCum(2) - 1);
            parValues(ind1) = exp(2 * p(ind1));
        }
    }
    // Irregular
    if (inputs.irregular[0] != 'n'){
        parValues(nparCum(2)) = exp(2 * p(nparCum(2)));   // Noise variance of ARMA
    }
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
    if (nu > 0){
      ind1 = SSmodel::inputs.betaAug.n_elem - nu;
      parValues.rows(nparCum(3), nparCum(3) + nu - 1) = SSmodel::inputs.betaAug.rows(ind1, ind1 + nu - 1);
    }
    // Calculating t stats and pValues
    char str[70];
    vec t = abs(p / stdP), pValue(p.n_elem);
    uvec nn = find_finite(SSmodel::inputs.y);
    pValue = 2 * (1- tCdf(t, nn.n_elem - p.n_elem));
    aux = find(t > 1000);
    if (aux.n_elem > 0){
        t(aux).fill(datum::inf);
        pValue(aux).fill(0);
    }
    // Creating table
    string fullModel;
    if (SSmodel::inputs.u.n_rows == 0){
      fullModel = inputs.model;
    } else {
      fullModel = inputs.model + " + inputs";
    }
    sprintf(str, " Model: %s\n", fullModel.c_str());
    auto it = SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 1, str);
    if (SSmodel::inputs.cLlik)
        SSmodel::inputs.table.insert(it, " Concentrated Maximum-Likelihood\n");
    else
        SSmodel::inputs.table.insert(it, " Maximum-Likelihood\n");
    vec col1;
    int insert = 0;
    // Periods
    vec periods1 = inputs.periods(find(inputs.rhos > 0));
    int lPer = periods1.n_elem;
    if (lPer > 0 && periods1(0) > 1 && inputs.nPar(2) > 0){
      string line;
      sprintf(str, " Periods: %5.1f", periods1(0));
      line = str;
      for (int i = 1; i < lPer; i++){
        sprintf(str, " /%5.1f", periods1(i));
        line += str;
      }
      SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 3, line + "\n");
      insert++;
    } else {
      SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 3, " Periods: \n");
      insert++;
    }
    if (any(inputs.constPar == 1)){
        sprintf(str, " (*)  concentrated out parameter\n");
        SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 4 + insert, str);
        insert++;
    }
    if (any(inputs.constPar > 1)){
        sprintf(str, " (**) constrained parameters during estimation\n");
        SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 4 + insert, str);
        insert++;
    }
    SSmodel::inputs.table.at(5 + insert) = "                     Param          |T|    P-value     |Grad| \n";
    col1 = parValues;
    if (SSmodel::inputs.cLlik){
        col1(find(isVar)) = col1(find(isVar)) * SSmodel::inputs.innVariance;
    }
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
    uvec ind = find(inputs.constPar > 0);
    t(ind).fill(datum::nan);
    pValue(ind).fill(datum::nan);
    SSmodel::inputs.grad(ind).fill(datum::nan);
    vec gradBetas(nu);
    gradBetas.fill(0);
    vec grad = join_vert(SSmodel::inputs.grad, gradBetas);
    // stars for constrained parameters
    vector<string> col2;
    string chari;
    for (uword i = 0; i < inputs.constPar.n_elem; i++){
        if (inputs.constPar(i) == 0)
            chari = "  ";
        else if (inputs.constPar(i) == 1)
            chari = "* ";
        else
            chari = "**";
        col2.push_back(chari);
    }
    // Adding spaces for betas
    for (int i = 0; i < nu; i++){
      col2.push_back("  ");
    }
    for (unsigned i = 0; i < p.n_elem; i++){
        if (abs(col1(i)) > 1e-3 || abs(col1(i)) == 0 || abs(col1(i)) == 1)
            sprintf(str, "%*s: %12.4f%2s %10.4f %10.4f %10.2e\n", 12, names.at(i).c_str(), col1(i), col2.at(i).c_str(), t(i), pValue(i), abs(grad(i)));
        else
            sprintf(str, "%*s: %12.2e%2s %10.4f %10.4f %10.2e\n", 12, names.at(i).c_str(), col1(i), col2.at(i).c_str(), t(i), pValue(i), abs(grad(i)));
        SSmodel::inputs.table.at(i + 7 + insert) = str;
    }
    // for (auto i = SSmodel::inputs.table.begin(); i != SSmodel::inputs.table.end(); i++){
    //     cout << *i << " ";
    // }
    for (unsigned int i = 0; i < SSmodel::inputs.table.size(); i++){
      printf("%s ", SSmodel::inputs.table[i].c_str());
    }
}
// Disturbance smoother (to recover just trend and epsilons)
void BSMmodel::disturb(){
    inputs.eps.zeros(SSmodel::inputs.y.n_elem);
    if (inputs.irregular[0] == 'a' && inputs.ar == 0 && inputs.ma == 0){
        // Modification adding the observation noise as a final state
        SSinputs  copiaSS  = SSmodel::getInputs();
        // Modifying system
        int nsAll = sum(inputs.ns) + 1;
        // int nu = SSmodel::inputs.u.n_rows;
        mat T(nsAll, nsAll), R(nsAll, nsAll), Q(nsAll, nsAll), Z(1, nsAll); //, D(1, nu);
        T.fill(0);
        R.eye();
        Q.fill(0);
        Z.fill(0);
        nsAll -= 2;
        T(span(0, nsAll), span(0, nsAll)) = copiaSS.system.T;
        R(span(0, nsAll), span(0, nsAll + 1)) = copiaSS.system.R;
        R(nsAll + 1, nsAll + 1) = 1;
        Q = copiaSS.system.Q;
        Q(nsAll + 1, nsAll + 1) = copiaSS.system.H(0, 0);
        Z(0, span(0, nsAll)) = copiaSS.system.Z;
        Z(0, nsAll + 1) = copiaSS.system.C(0, 0);
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
int BSMmodel::quasiNewtonBSM(std::function <double (vec& x, void* inputsFake)> objFun,
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
    gradNew = gradFun(xUncon, inputsFake, objNew, nFuns);
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
        gradNew = gradFun(xUncon, inputsFake, objNew, nFuns);
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
            gradNew = gradFun(xUncon, inputsFake, objNew, nFuns);
            nOverallFuns += nFuns;
            if (cLlik){
                gradNew(concentratedOutPar).fill(0);
            }
            iHess.eye(nx, nx);
        }
        counter++;
    } while (!flag);
    SSmodel::inputs.flag = flag;
    return flag;
}
// Count states and parameters of BSM model
void BSMmodel::countStates(vec periods, string trend, string cycle, string seasonal, string irregular){
  // string trend, string cycle, string seasonal, string irregular, 
  // int nu, vec P, vec rhos, vec& ns, vec& nPar, int& arOrder, 
  // int& maOrder, bool& exact
  inputs.ns = zeros(5);
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
  } else if (trend[0] == 'd'){  // Damped trend
    inputs.ns(0) = 2;
    inputs.nPar(0) = 3;
    SSmodel::inputs.exact = false;
    SSmodel::inputs.cLlik = true;
    SSmodel::inputs.augmented = true;
  } else if(trend[0] == 'r'){   // RW trend
    inputs.ns(0) = 1;
    inputs.nPar(0) = 1;
  } else if(trend[0] == 'i'){   // IRW trend
    inputs.ns(0) = 2;
    inputs.nPar(0) = 1;
  } else {                      // No trend
    inputs.ns(0) = 1;
    inputs.nPar(0) = 0;
  }
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
    SSmodel::inputs.augmented = true;
  }
}
// Fix matrices in standard BSM models (all except variances)
void BSMmodel::initMatricesBsm(vec periods, vec rhos, string trend, string cycle, string seasonal, string irregular){
  int nsCol;
  countStates(periods, trend, cycle, seasonal, irregular);
  // Initializing system matrices
  int nsAll = sum(inputs.ns);
  SSmodel::inputs.system.T.eye(nsAll, nsAll);
  nsCol = sum(inputs.ns(span(0, 2))) + 1;
  SSmodel::inputs.system.R.eye(nsAll, nsCol);
  SSmodel::inputs.system.Q.zeros(nsCol, nsCol);
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
    aux = find(rhos > 0);
    bsm2ss(inputs.ns(0) + inputs.ns(1), inputs.ns(2), abs(periods(aux)), 
           abs(rhos(aux)), &SSmodel::inputs.system.T, &SSmodel::inputs.system.Z);
  }
  // Irregular as ARMA
  if (inputs.ar > 0 || inputs.ma > 0){   // ARMA
    SSmodel::inputs.system.C.zeros(1, 1);
    SSmodel::inputs.system.Z.col(nsCol - 1) = 1.0;
  }
  // Inputs in case of pure regression
  if (SSmodel::inputs.u.n_elem > 0 && sum(inputs.nPar.rows(0, 3)) == 1){
    SSmodel::inputs.system.T(0, 0) = 0;
  }
}
// Initializing parameters of BSM model
void BSMmodel::initParBsm(){
  int nTrue = sum(inputs.nPar);
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
  } else if (inputs.nPar(0) == 1 && inputs.ns(0) > 1){   // IRW trend
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
    // double season = aux(0);
    vec stdBeta, e, orders(2), betaHR;
    double BIC, AIC, AICc;
    if (inputs.nPar(3) > 1){
      double ini = sum(inputs.nPar(span(0, 2))) + 1;
      aux = regspace<uvec>(ini, 1, sum(inputs.nPar) - 1);
      inputs.typePar(aux).fill(3);
      // Running from estim(), in the case of ident() inputs.beta0ARMA is already set
      if (inputs.beta0ARMA.n_elem == 0){
        orders(0) = inputs.ar;
        orders(1) = inputs.ma; //nPar(3) - inputs.ar - 1;
        if (inputs.nPar(2) > 0){  // with seasonal component and inputs or no inputs
          harmonicRegress(SSmodel::inputs.y, SSmodel::inputs.u, periods, betaHR, stdBeta, e);
        } else if (inputs.nPar(4) > 0){ // with inputs and no seasonal
          regress(SSmodel::inputs.y, SSmodel::inputs.u.cols(0, SSmodel::inputs.y.n_elem - 1).t(), betaHR, stdBeta, e, AIC, BIC, AICc);
        } else {    // just ARMA
          e = SSmodel::inputs.y;
        }
        linearARMA(e, orders, inputs.beta0ARMA, stdBeta);
      }
      // AR pars
      vec beta0aux, beta0aux1;
      uvec ind;
      if (inputs.ar > 0){
        beta0aux = inputs.beta0ARMA(span(0, inputs.ar - 1));
        beta0aux1 = -beta0aux;
        // Correction for non-stationary polynomial
        arToPacf(beta0aux1);
        ind = find(abs(beta0aux1) >= 1);
        if (ind.n_elem > 0){
          beta0aux1(ind) = sign(beta0aux1(ind)) * 0.98;
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
        beta0aux = inputs.beta0ARMA(span(inputs.ar, inputs.beta0ARMA.n_elem - 1));
        // Bringing MA polynomial to invertibility
        maInvert(beta0aux);
        // Parameterising polynomial to be invertible
        invPolyStationary(beta0aux);
        aux = regspace<uvec>(ini + inputs.ar, 1, ini + inputs.ar + inputs.ma - 1);
        SSmodel::inputs.p0(aux) = beta0aux;        
      }
      // if (inputs.ma > 0){
      //   beta0aux = inputs.beta0ARMA(span(inputs.ar, inputs.beta0ARMA.n_elem - 1));
      //   beta0aux1 = -beta0aux;
      //   // Correction for non-invertible polynomial
      //   arToPacf(beta0aux1);
      //   ind = find(abs(beta0aux1) >= 1);
      //   if (ind.n_elem > 0){
      //     beta0aux1(ind) = sign(beta0aux1(ind)) * 0.98;
      //     pacfToAr(beta0aux1);
      //     beta0aux = -beta0aux1;
      //   }
      //   invPolyStationary(beta0aux);
      //   beta0aux.elem(find_nonfinite(beta0aux)).zeros();
      //   aux = regspace<uvec>(ini + inputs.ar, 1, ini + inputs.ar + inputs.ma - 1);
      //   SSmodel::inputs.p0(aux) = beta0aux;
      // }
    }
  int nu = SSmodel::inputs.u.n_rows;
  int ns = SSmodel::inputs.betaAug.n_rows;
  if (nu > 0 && ns > 1){
    SSmodel::inputs.system.D = SSmodel::inputs.betaAug.rows(ns - nu, ns - 1);
  }
  if (isfinite(p0(0))){    // User do not supply initial values (NA)
    SSmodel::inputs.p0 = p0;
    int nUser = SSmodel::inputs.p0.n_elem;
    if (nUser < nTrue){  // Make longer user inputs.p0
      SSmodel::inputs.p0 = join_vert(SSmodel::inputs.p0, SSmodel::inputs.p0(nUser - 1) * ones<vec>(nTrue - nUser));
    } else {              // Select part of inputs.p0 supplied by user
      SSmodel::inputs.p0(span(0, nTrue - 1)) = SSmodel::inputs.p0(span(0, nTrue - 1));
    }
  }
  // Choosing initial concentrated parameters
  inputs.constPar = zeros(nTrue);
  if (SSmodel::inputs.cLlik){
    if (inputs.nPar(3) > 0){         // arma(0,0) or arma(p,q)
      inputs.constPar(inputs.nPar(0) + inputs.nPar(1) + inputs.nPar(2)) = 1;
    } else if (inputs.nPar(3) == 0){   // no irregular component
      uvec minIndex = find(inputs.typePar == 0);
      inputs.constPar(minIndex(0)) = 1;
    }
  }
}
/*************************************************************
//  * Implementation of auxiliar functions
//  ************************************************************/
// Variance matrices in standard BSM on top of fixed structure
void bsmMatrices(vec p, SSmatrix* model, void* userInputs){
  BSMinputs* inp = (BSMinputs*)userInputs;
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
    model->T(1, 1) = p(0);
    model->Q(0, 0) = exp(2 * p(1));
    model->Q(1, 1) = exp(2 * p(2));
  } else if (inp->nPar(0) == 1 && inp->ns(0) == 2){    // IRW
    model->Q(1, 1) = exp(2 * p(0));
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
    if (inp->nPar(2) == 1){                         // Equal variances
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
  (*Z).cols(aux1) = ones(1, aux1.n_elem);
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
    uvec sIndex = sort_index(abs(periods), "descend");
    periods = sign(periods(sIndex)) % sort(abs(periods), "descend");
}
// Calculate limits for cycle periods for estimation
void calculateLimits(int n, vec periods, vec rhos, mat& cycleLimits){
  double media;
  vec pCycles;
  int nCycles = sum(rhos < 0);
  cycleLimits.resize(nCycles, 2);
  pCycles = periods(span(0, nCycles - 1));
  for (int i = 1; i < nCycles; i++){
    if (pCycles(i) > 0){
      cycleLimits(i, 1) = pCycles(i);
    }
    if (pCycles(i - 1) > 0){
      cycleLimits(i - 1, 0) = pCycles(i - 1);
    }
    if (pCycles(i) > 0 && pCycles(i - 1) < 0){
      cycleLimits(i - 1, 0) = abs(pCycles(i)) + 1;
    } else if (pCycles(i) < 0 && pCycles(i - 1) > 0){
      cycleLimits(i, 1) = pCycles(i - 1) - 1;
    } else if (pCycles(i) < 0 && pCycles(i - 1) < 0){
      media = (abs(pCycles(i)) + abs(pCycles(i - 1))) / 2;
      cycleLimits(i, 1) = media;
      cycleLimits(i - 1, 0) = media + 1;
    }
  }
  // Correcting extreme limits
  cycleLimits(0, 1) = pCycles(0);
  cycleLimits(nCycles - 1, 0) = pCycles(nCycles - 1);
  if (pCycles(0) < 0){
    cycleLimits(0, 1) = 1.5 * abs(cycleLimits(0, 1));
  }
  if (pCycles(nCycles - 1) < 0){
    if (periods(nCycles) == 1){
      vec aux(2);
      aux(0) = 6;
      aux(1) = min(abs(periods(rhos < 1))) * 0.5;
      cycleLimits(nCycles - 1, 0) = max(aux);
    } else {
      cycleLimits(nCycles - 1, 0) = periods(nCycles) * 1.5;
    }
  }
}
