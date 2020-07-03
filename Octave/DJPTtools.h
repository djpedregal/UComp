/*************************
 DJPTtools
 Needs Armadillo
 Needs string.h
***************************/
/**********************
 Function declarations
 **********************/
// Converting string variable to lower letters
void lower(string&);
// Replace "what" by string "by" in string "input"
void strReplace(string, string, string&);
// Removing spaces in string
void deblank(string&);
// Maps vector of arbitrary parameters into a given range
template <class T>
void constrain(T&, vec);
void constrain(vec&, mat);
// Undo constrain
void unconstrain(double&, vec);
void unconstrain(vec&, mat);
// Remove Nans from vector or matrix
template <class T>
T removeNans(T, int&);
// Build a lagged matrix of a time series lagged
mat lag(vec, vec);
// Chop a string according to substring
void chopString(string, string, vector<string>&);
  
/**********************
 Function implementations
 **********************/
// Converting string variable to lower letters
void lower(string& model){
  transform(model.begin(), model.end(), model.begin(), :: tolower);
}
// Replace "what" by string "by" in string "input"
void strReplace(string what, string by, string& input){
  size_t i;
  
  i = input.find(what);
  if (i <= input.length()){
    input.replace(i, what.length(), by);
    strReplace(what, by, input);
  }
}
// Removing spaces in string
void deblank(string& input){
  size_t i;
  
  i = input.find(" ");
  if (i <= input.length()){
    input.erase(i, 1);
    deblank(input);
  }
}
// Maps vector of arbitrary parameters into a given range
template <class T>
void constrain(T& p, vec limits){
  p = limits(0) + (limits(1) - limits(0)) * (0.5 * (1 + p / sqrt(1 + pow(p, 2))));
}
// template <class Tvec, class Tmat>
void constrain(vec& p, mat limits){
  p = limits.col(0) + (limits.col(1) - limits.col(0)) % (0.5 * (1 + p / sqrt(1 + pow(p, 2))));
}
// Undo constrain
void unconstrain(double& p, vec limits){
  int signs = 1;
  if (p < mean(limits))
    signs = -1;
  double H = pow((2 * (p - limits(0)) / (limits(1) - limits(0)) - 1), 2);
  p = signs * sqrt(H / (1 - H));
}
void unconstrain(vec& p, mat limits){
  int n = p.n_elem;
  vec signs = ones(n);
  signs.elem(find(p < mean(limits, 1))).fill(-1);
  vec H = pow((2 * (p - limits.col(0)) / (limits.col(1) - limits.col(0)) - 1), 2);
  p = signs % sqrt(H / (1 - H));
}
// Remove Nans from vector or matrix
template <class T>
T removeNans(T y, int& nNan){
  uvec ind = find_finite(mean(y, 1));
  int n = y.n_rows;
  nNan = n - ind.n_elem;
  return y.rows(ind);
}
// Build a lagged matrix of a time series
mat lag(vec y, vec lags){
  int n = y.n_elem, m = lags.n_elem, maxLag = max(lags), ii;
  mat lagY(n - maxLag, m);

  for (int i = 0; i < m; i++){
    ii = as_scalar(lags.row(i));
    lagY.col(i) = y(span(maxLag - ii, n - ii - 1));
  }
  return lagY;
}
// Chop a string according to substring
void chopString(string str, string sub, vector<string>& out) {
  size_t pos = -1, pos0;
  do {
    pos0 = pos;
    pos = str.find(sub, pos + 1);
    out.push_back(str.substr(pos0 + 1, pos - pos0 - 1));
  } while(pos != string::npos);
}


