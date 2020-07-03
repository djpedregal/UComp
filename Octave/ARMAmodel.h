/*************************
 Stationary ARMA models with zero mean
Needs Armadillo
Needs SSpace.h
***************************/
struct ARMAinputs{
  int ar, ma;
};
/**************************
 * Model CLASS stationary ARMA
 ***************************/
class ARMAmodel : public SSmodel{
  private:
    ARMAinputs dataARMA;
    int ns;
  public:
    ARMAmodel(SSinputs, int, int);
};
/***************************************************
 * Auxiliar function declarations
 ****************************************************/
// Convert non-invertible ma polynomial into invertible
void maInvert(vec&);
// Returns the parameters of an AR model from the PACF
void pacfToAr(vec&);
// Returns the PACF from the parameters of an AR model
void arToPacf(vec&);
// Returns stationary polynomial from an arbitrary one
void polyStationary(vec&);
// Inverse of polyStationary
void InvPolyStationary(vec&);
// Initialising matrices
void initMatricesArma(int, int, int&, SSmatrix&);
// Filling changing matrices
void armaMatrices(vec p, SSmatrix* model, void* userInputs);
//#include ARMAmodel.cpp
/****************************************************
 // ARMA implementations for stationary ARMA models
 ****************************************************/
// Constructors
ARMAmodel::ARMAmodel(SSinputs data, int ar, int ma) : SSmodel(data){
  //int ns;
  // Initialising matrices
  initMatricesArma(ar, ma, ns, data.system);
  // Storing information
  this->inputs.system = data.system;
  this->dataARMA.ar = ar;
  this->dataARMA.ma = ma;
  if (ar == 0){
    this->inputs.exact = true;
  } else {
    this->inputs.exact = false;
  }
  this->inputs.userInputs = &this->dataARMA;
  // User function to fill the changing matrices
  this->inputs.userModel = armaMatrices;
  // Initializing parameters of ARMA model
  this->inputs.p0.zeros(ar + ma + 1);
  this->inputs.p0(0) = -1;
}
/*************************************************************
 * Implementation of auxiliar functions
 ************************************************************/
// Convert non-invertible ma polynomial into invertible
void maInvert(vec& maPoly){
  vec ma(maPoly.n_elem + 1);
  ma.row(0) = 1;
  ma.rows(1, maPoly.n_elem) = maPoly;
  unsigned int q = max(find(ma != 0));
  cx_vec maRoots;
  cx_double iRoot;
  ma = ma.rows(0, q);
  maRoots = roots(flipud(ma));
  uvec ind = find(abs(maRoots) < 1);
  cx_vec poly = zeros<cx_vec>(q + 1);
  poly.row(0) = 1;
  if (ind.n_elem > 0){
    // Some roots are not invertible
    maRoots(ind) = 1 / maRoots(ind);
    for (unsigned i = 0; i < q; i++){
      iRoot = as_scalar(maRoots.row(i));
      poly.rows(0, i + 1) = poly.rows(0, i + 1) - 
        join_vert(poly.rows(i + 1, i + 1), poly.rows(0, i) / iRoot);
    }
    maPoly = real(poly.rows(1, poly.n_elem - 1));
  }
}
// Returns the parameters of an AR model from the PACF
void pacfToAr(vec& PAR){
  // y(t) = PAR(1) * y(t - 1) + PAR(2) * y(t - 2) + ...
  // Monahan, JF (1984), A note on enforcing stationarity in ARMA models,
  // Biometrika, 71, 2, 403-404.
  vec par0 = PAR;
  for (unsigned int i = 0; i < PAR.n_elem - 1; i++){
    PAR(i + 1) = par0(i + 1);
    PAR(span(0, i)) = (PAR(span(0, i)) - PAR(i + 1) * flipud(PAR(span(0, i))));
  }
}
// Returns the PACF from the parameters of an AR model
void arToPacf(vec& PAR){
  // y(t) = PAR(1) * y(t - 1) + PAR(2) * y(t - 2) + ...
  // Monahan, JF (1984), A note on enforcing stationarity in ARMA models,
  // Biometrika, 71, 2, 403-404.
  for (int i = PAR.n_elem - 1; i > 0; i--){
    int j = i - 1;
    PAR(span(0, j)) = (PAR(span(0, j)) + PAR(i) * flipud(PAR(span(0, j))))
      / (1 - PAR(i) * PAR(i));
  }
}
// Returns stationary polynomial from an arbitrary one
void polyStationary(vec& PAR){
  // (1 + PAR(1) * B + PAR(2) *B^2 + ...) y(t) = a(t)
  vec limits(2);
  limits(0) = -0.98;
  limits(1) = 0.98;
  constrain(PAR, limits);
  pacfToAr(PAR);
  PAR = -PAR;
}
// Inverse of polyStationary
void invPolyStationary(vec& PAR){
  // (1 + PAR(1) * B + PAR(2) *B^2 + ...) y(t) = a(t)
  mat limits(PAR.n_elem, 2);
  limits.col(0).fill(-0.98);
  limits.col(1).fill(0.98);
  PAR = -PAR;
  arToPacf(PAR);
  unconstrain(PAR, limits);
}
// Initialising matrices
void initMatricesArma(int ar, int ma, int& ns, SSmatrix& model){
  ns = std::max(ar, ma + 1);
  model.T.zeros(ns, ns);
  if (ns > 1)
    model.T.diag(1) += 1;
  model.Gam = model.D = model.H = model.C = 0.0;
  model.Z.zeros(1, ns);
  model.Z(0, 0) = 1.0;
  model.R.zeros(ns, 1);
  model.R(0) = 1;
  model.Q = 0.0;
}
// Filling changing matrices
void armaMatrices(vec p, SSmatrix* model, void* userInputs){
  ARMAinputs* inp = (ARMAinputs*)userInputs;
  vec ARpoly, MApoly; //, aux, uno(1);
  // AR and MA stationary polys
  if (inp->ar > 0){
    ARpoly = p(span(1, inp->ar));
    polyStationary(ARpoly);
  }
  if (inp->ma > 0){
    MApoly = p(span(inp->ar + 1, inp->ar + inp->ma));
    polyStationary(MApoly);
  }
  // SS matrices
  model->Q(0, 0) = exp(2 * p(0));
  if (inp->ma > 0){
    model->R(span(1, inp->ma), 0) = MApoly;
  }
  if (inp->ar > 0)
    model->T(span(0, inp->ar - 1), 0) = -ARpoly;
}
