#include <iostream>
#include <armadillo>
#include <string>
#include <math.h>
using namespace arma;
using namespace std;
#include "DJPTtools.h"
#include "optim.h"
#include "stats.h"
#include "SSpace.h"
#include "ARMAmodel.h"
#include "BSMmodel.h"

int main(){
  // Creating BSM and SS structures
  BSMinputs inputsBSM;
  SSinputs inputsSS;
  ////////////////
  // Initialising UComp model inputs (this may be changed by the user)
  ////////////////
  inputsSS.y = {};   // Output data (vec)
  inputsSS.u = {};   // Inputs data (mat)
  inputsBSM.model = "?/none/?/?";  // Model (string)
  inputsBSM.periods = {12, 6, 4, 3, 2.4, 2}; // Harmonics periods (vec)
  inputsSS.h = 24;   // Horizon for forecasts (double)
  inputsSS.outlier = 0;   // Outlier detection (double)
  inputsBSM.criterion = "aic";  // Identification information criterion (string)
  inputsBSM.tTest = false;      // Unit roots tests (bool)
  inputsBSM.stepwise = false;   // Stepwise procedure (bool)
  inputsSS.verbose = true;      // Verbose ouptut (bool)
  inputsBSM.arma = true;        // Identification of arma irregular (bool)
  inputsSS.p0 = {-9999.9};        // Initial estimates to search for optimal (vec)
  ////////////////
  // Initialising some UComp model outputs (do not change!!)
  ////////////////
  inputsSS.p = {datum::nan};        // Estimated parameters (vec)
  inputsSS.v = {datum::nan};        // Estimated innovations (vec)
  inputsSS.yFit = {datum::nan};     // Fitted values (vec)
  inputsSS.yFor = {datum::nan};     // Point forecasts (vec)
  inputsSS.F = {datum::nan};        // Innovations variance (vec)
  inputsSS.FFor = {datum::nan};     // Variance of forecasts (vec)
  inputsBSM.comp = {datum::nan};    // Estimated components (mat)
  inputsBSM.compV = {datum::nan};   // Variance of estimated components (mat)
  inputsSS.a = {datum::nan};        // Estimated states (mat)
  inputsSS.P = {datum::nan};        // Estimated variance of states (mat)
  inputsSS.eta = {datum::nan};      // Estimated transition perturbations (mat)
  inputsBSM.eps = {datum::nan};     // Estimated observed perturbations (vec)
  inputsSS.criteria = {datum::nan}; // Likelihood and information criteria at optimum (vec)
  inputsBSM.rhos = inputsBSM.periods; inputsBSM.rhos.fill(1);

  // Building UComp system
  BSMmodel sysBSM(inputsSS, inputsBSM);
  // Estimation
  sysBSM.estim();
  // Validation
  sysBSM.validate();
  // Estimating components
  sysBSM.components();

  return 0;

}
