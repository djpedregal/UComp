/*************************************
 * Quasi-Newton estimation with BFGS inverse hessian approximation
 * Needs Armadillo
 ************************************/
// struct solution{
//   mat x, grad, iHess;
//   double obj;
//   int flag;
// };
/*************************************
 * Function declarations
 *************************************/
// Minimum searcher 
// First with numerical gradient function and second with
//                gradient supplied by user
int quasiNewton(std::function <double (vec&, void*)>, 
                 std::function <vec (vec&, void*, double, int&)>,
                 vec&, void*, double&, vec&, mat&, bool);
// Stop criteria in non-linear search 
int stopCriteria(vec, double, double, double, int, int);
// BFGS update of inverse Hessian 
void bfgs(mat&, vec, vec, int, int);
// Backtracking-Armijo line search
void lineSearch(std::function <double (vec&, void*)>, double&, vec&, double&, vec&, vec,
                int, int&, void*);
/*************************************
 * Function implementations
 *************************************/
// Minimum searcher
// First with numerical gradient function and second with
//                gradient supplied by user
int quasiNewton(std::function <double (vec& x, void* inputs)> objFun, 
                 std::function <vec (vec& x, void* inputs, double obj, int& nFuns)> gradFun,
                 vec& xNew, void* inputs, double& objNew, vec& gradNew, mat& iHess, bool verbose){
  int nx = xNew.n_elem, flag = 0, nOverallFuns, nFuns = 0, nIter = 0;
  double objOld, alpha_i;
  vec gradOld(nx), xOld = xNew, d(nx); 
  vec crit(3); crit(0) = 1e-8; crit(1) = 1e-12; crit(2) = 1000;
  
  iHess.eye(nx, nx);
  objNew = objFun(xNew, inputs);
  gradNew = gradFun(xNew, inputs, objNew, nFuns);
  nOverallFuns = nFuns + 1;
  // Head of table
  if (verbose){
      printf(" Iter FunEval  Objective       Step\n");
      printf("%5.0i %5.0i %12.5f %12.5f\n", nIter, nOverallFuns, objNew, 1.0);
  }
  // Main loop
  do{
    nIter++;
    // Search direction
    d = -iHess * gradNew;
    // Line Search
    xOld = xNew; gradOld = gradNew; objOld = objNew;
    alpha_i = 0.5;
    lineSearch(objFun, alpha_i, xNew, objNew, gradNew, d, nIter, nFuns, inputs);
    nOverallFuns = nOverallFuns + nFuns;
    gradNew = gradFun(xNew, inputs, objNew, nFuns);
    nOverallFuns = nOverallFuns + nFuns;
    // Verbose
    if (verbose){
        printf("%5.0i %5.0i %12.5f %12.5f\n", nIter, nOverallFuns, objNew, alpha_i);
    }
    // Stop Criteria
    // flag = stopCriteria(crit, gradNew, objOld - objNew, nIter, nOverallFuns);
    flag = stopCriteria(crit, mean(abs(gradNew)), abs(objOld - objNew) / abs(objOld), 
                        mean(abs(xOld - xNew) / abs(xOld)), nIter, nOverallFuns);
    // Inverse Hessian BFGS update
    if (!flag){
      bfgs(iHess, gradNew - gradOld, xNew - xOld, nx, nIter);
    }
  } while (!flag);
  return flag;
}
// Stop criteria in non-linear search 
int stopCriteria(vec crit, double gradNew, double dobj, double dPar, int nIter, int nOverallFuns){
  int flag = 0;
  // crit: Criteria to stop Gauss-Newton:
        // Gradient
        // Difference in obj function
        // Difference in parameter values
        // Maximum number of iterations
        // Maximum number of function evaluations
  // flags:
        // Gradient
        // Difference in obj function
        // Difference in parameter values
        // Maximum number of iterations
        // Maximum number of function evaluations
        // Unable to decrease objective function in gradient direction
        // objective function with nan value
  if (gradNew < crit(0))
    flag = 1;
  else if (abs(dobj) < crit(1))
    flag = 2;
  else if (dPar < crit(2))
      flag = 3;
  else if (dobj < 0)
    flag = 6;
  else if (nIter > crit(3))
    flag = 4;
  else if (nOverallFuns > crit(4))
    flag = 5;
  if (isnan(dobj))
      flag = 7;
  
  return flag;
}
// BFGS update of inverse Hessian 
void bfgs(mat& iHess, vec dGrad, vec dx, int nx, int nIter){
  vec dxdGrad(1), iHessdGrad(nx);

  dxdGrad = dx.t() * dGrad;
  if (nIter == 1)
    iHess = as_scalar(dxdGrad / (dGrad.t() * dGrad)) * eye(nx, nx);
  iHessdGrad = iHess * dGrad;
  iHess = iHess + as_scalar((1 + dGrad.t() * iHessdGrad / dxdGrad) / dxdGrad) * dx * dx.t() -
      (dx * iHessdGrad.t() + iHessdGrad * dx.t()) / dxdGrad(0, 0);
}
// Backtracking-Armijo line search
void lineSearch(std::function <double (vec& x, void* inputs)> objFun,
                double& alpha_i, vec& x, double& obj, vec& grad, vec d,
                int nIter, int& nFuns, void* inputs){
  float beta = 0.1, tau = 0.5;
  vec dir, x0 = x, aux(2); //, xNaN = x;
  double obj0;

  // Initialising variables
  nFuns = 0;
  aux(0) = 1 / norm(grad);
  aux(1) = 1;
  obj0 = obj;
  if (nIter == 1)
    alpha_i = min(aux) / tau;
  else
    alpha_i = 1 / tau;  // Normally this is 1 / tau
  nFuns = 0;
  dir = beta * grad.t() * d;
  // Main loop
  do{
    // obj0 = obj;    // to remember in case of nan
    // xNaN = x;
    alpha_i *= tau;
    x = x0 + alpha_i * d;
    if (x.is_finite()){
        obj = objFun(x, inputs);
    } else{
        obj = datum::nan;
    }
    nFuns += 1;
  } while (obj > obj0 + alpha_i * dir(0) && alpha_i > 1e-5 && nFuns < 1001);
  // } while (obj > obj0 + alpha_i * dir(0) && alpha_i > 1e-5 && !isnan(obj) && nFuns < 1001);
// In case of nan obj
  // if (isnan(obj)){
  //     obj = obj0 + 1e-6;
  //     alpha_i /= tau;
  //     x = xNaN; //x0 + alpha_i * d;
  //     nFuns -= 1;
  // }
}


