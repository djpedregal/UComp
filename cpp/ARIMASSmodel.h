// ARIMA models in SS form
struct ARIMASS{
    vec orders = {0, 1, 1, 0, 1, 1}; // ARIMA(p,d,q,ps,ds,qs)
    int s = 12;                      // seasonal order
    int ns,                          // number of states
        maxArma,                     // ??
        n;                           // ??
    vec AR,                          // AR polynomial
        MA,                          // MA polynomial
        ARS,                         // AR seasonal polynomial
        MAS;                         // MA seasonal polynomial
};
/**************************
 * Model CLASS ARIMA in SS
 ***************************/
class ARIMASSclass : public SSmodel{
    public:
        ARIMASS mSS;
        ARIMASSclass(SSinputs, ARIMASS);
        void setInputs(SSinputs input1, ARIMASS input2){
            this->SSmodel::inputs = input1;
            this->mSS = input2;
        };
        SSinputs getInputs(){
            return(SSmodel::inputs);
        };
        // Set model with new parameters
        //void setModel(SSinputs, ARIMAinputs);
};
/****************************************************
// ARIMASS functions declarations
****************************************************/
// Initialising matrices
void initMatricesArima(ARIMASS, SSmatrix&);
// Filling changing matrices with current parameters
void arimaMatrices(vec, SSmatrix*, void*);

/****************************************************
// ARIMASS functions implementations
****************************************************/
ARIMASSclass::ARIMASSclass(SSinputs data, ARIMASS inputs) : SSmodel(data){
    // Number of states
    vec arma(2);
    arma(0) = inputs.orders(0) + inputs.orders(3) * inputs.s;
    arma(1) = inputs.orders(2) + inputs.orders(5) * inputs.s + 1;
    inputs.maxArma = max(arma);
    inputs.ns = inputs.orders(1) + inputs.orders(4) * inputs.s + inputs.maxArma;
    // llik or llikAug
    if (data.u.n_rows > 0){
        data.augmented = true;
        data.llikFUN = llikAug;
    } else {
        data.augmented = false;
        data.llikFUN = llik;
    }
    // Initialising matrices
    initMatricesArima(inputs, data.system);
    // AR and MA polynomials in ARIMAinputs
    inputs.AR.set_size(inputs.orders(0) + 1);
    inputs.MA.set_size(inputs.orders(2) + 1);
    inputs.ARS.set_size(inputs.orders(3) * inputs.s + 1);
    inputs.MAS.set_size(inputs.orders(5) * inputs.s + 1);
    inputs.AR.fill(0.0);
    inputs.MA.fill(0.0);
    inputs.ARS.fill(0.0);
    inputs.MAS.fill(0.0);
    inputs.AR(0) = inputs.ARS(0) = inputs.MA(0) = inputs.MAS(0) = 1.0;
    // Storing information
    inputs.n = data.y.n_elem;
    this->mSS = inputs;
    data.userInputs = &this->mSS;
    // User function to fill the changing matrices
    data.userModel = arimaMatrices;
    arimaMatrices(data.p, &data.system, data.userInputs);
    this->SSmodel::inputs = data;
}
// Initialising matrices
void initMatricesArima(ARIMASS inputs, SSmatrix& model){
    // Set up of SS system
    // Difference polynomial
    vec polyDd = {1};
    if (inputs.orders(1) > 0){
        vec polyd1 = {1, -1};
        for (unsigned int i = 0; i < inputs.orders(1); i++){
            polyDd = conv(polyDd, polyd1);
        }
    }
    if (inputs.orders(4) > 0){
        vec polyD1(inputs.s + 1, fill::zeros);
        polyD1(0) = 1;
        polyD1(inputs.s) = -1;
        for (unsigned int i = 0; i < inputs.orders(4); i++){
            polyDd = conv(polyDd, polyD1);
        }
    }
    // System matrices
    model.T.zeros(inputs.ns, inputs.ns);
    if (inputs.ns > 1)
        model.T.diag(1) += 1;
    if (inputs.orders(1) + inputs.orders(4) > 0){
        model.T.submat(inputs.maxArma, inputs.maxArma, model.T.n_rows - 1, inputs.maxArma) = -polyDd(span(1, polyDd.n_elem - 1));
        model.T(inputs.maxArma - 1, inputs.maxArma) = 0.0;
        model.T(inputs.maxArma, 0) = 1.0;
    }
    model.Gam = model.H = model.C = model.D = 0.0;
    if (polyDd.n_elem == 1){
        model.Z.zeros(1, inputs.ns);
        model.Z(0, 0) = 1.0;
    } else {
        model.Z = model.T.row(inputs.maxArma);
        //model.Z.zeros(1, inputs.ns);
        //model.Z.col(inputs.maxArma) = 1;
    }
    model.R.zeros(inputs.ns, 1);
    model.R(0, 0) = 1.0;
    model.Q = 1.0;
}
// Filling changing matrices with current parameters
void arimaMatrices(vec p, SSmatrix* model, void* userInputs){
    // Introduce parameters in SS system matrices
    ARIMASS* inp = (ARIMASS*)userInputs;
    vec ARpoly, MApoly;
    // AR and MA polys
    uvec ind;
    unsigned int np = 0;
    if (inp->orders(0) > 0){
        ind = regspace<uvec>(np, inp->orders(0) - 1);
        inp->AR(ind + 1) = p(ind);
        np += inp->orders(0);
    }
    if (inp->orders(3) > 0){
        ind = regspace<uvec>(inp->s, inp->s, inp->orders(3) * inp->s);
        inp->ARS(ind) = p(span(np, np + inp->orders(3) - 1));
        np += inp->orders(3);
    }
    if (inp->orders(2) > 0){
        ind = regspace<uvec>(np, np + inp->orders(2) - 1);
        inp->MA(span(1, inp->orders(2))) = p(ind);
        np += inp->orders(2);
    }
    if (inp->orders(5) > 0){
        ind = regspace<uvec>(inp->s, inp->s, inp->orders(5) * inp->s);
        inp->MAS(ind) = p(span(np, np + inp->orders(5) - 1));
        np += inp->orders(5);
    }
    ARpoly = conv(inp->AR, inp->ARS);
    MApoly = conv(inp->MA, inp->MAS);
    // SS matrices
    if (MApoly.n_elem > 1){
        model->R(span(0, MApoly.n_elem - 1), 0) = MApoly;
    }
    if (ARpoly.n_elem > 1){
        model->T(span(0, ARpoly.n_elem - 2), 0) = -ARpoly(span(1, ARpoly.n_elem - 1));
    }
}
