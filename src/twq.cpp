// =============================================================================
// twq.cpp
// -----------------------------------------------------------------------------
// C++ backend (Rcpp + RcppArmadillo) for the Two-way Grouped Network Quantile
// (TGNQ) autoregression model.
//
// The TGNQ model assigns each network node two latent group memberships:
//   - G-membership (row group): captures node "susceptibility"
//   - H-membership (column group): captures node "influence"
//
// Three model variants are supported:
//   - "general"        : theta_{gh}(tau) is a free parameter for each (g,h)
//   - "additive"       : theta_{gh}(tau) = alpha_g(tau) + beta_h(tau)
//   - "multiplicative" : theta_{gh}(tau) = alpha_g(tau) * beta_h(tau)
//
// This file contains:
//   * Loss-function helpers (standard check loss + smoothed "conquer" loss)
//   * Loss evaluation routines for updating G-membership
//   * Coordinate-descent update routines for H-membership
//   * "Proposal" routines that generate candidate H-vectors used by the
//     Enhanced Algorithm (Algorithm 2 in the paper)
//   * Refinement routines used to evaluate the impact of changing a single
//     node's membership (used for membership-error diagnostics)
// =============================================================================

#include <math.h>
#include <RcppArmadillo.h>
using namespace Rcpp;
#include <RcppArmadilloExtensions/sample.h>

// [[Rcpp::depends(RcppArmadillo)]]


// -----------------------------------------------------------------------------
// ExpandH
// -----------------------------------------------------------------------------
// Generate the full Cartesian product {1,...,H}^d as an (H^d) x d matrix.
// Used to enumerate all possible H-membership assignments for a small set
// of nodes (when the search space is feasible).
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
arma::mat ExpandH(int H, int d) {
  arma::field<arma::mat> M(d);
  arma::mat tmp(H,1);
  tmp.col(0)= arma::linspace<arma::vec>(1,H,H);
  M(0)=tmp;

  // Iteratively build up dimension by dimension via column-stacking
  for (int i = 0; i < (d-1); i++) {
    int L=M(i).n_rows;
    int J=M(i).n_cols;
    arma::mat tmp3(L,J+1);
    for (int j = 0; j < H; j++) {
      arma::mat tmp(L,1);
      tmp.col(0)=tmp.col(0)+j+1;
      arma::mat tmp2=join_rows(tmp,M(i));
      if(j==0)  tmp3=tmp2; else {
        tmp3=join_cols(tmp2,tmp3);
      }
    }
    M(i+1)=tmp3;
  }
  return M(d-1);
    }

// -----------------------------------------------------------------------------
// ExpandH_sample
// -----------------------------------------------------------------------------
// Random-sample alternative to ExpandH when the full enumeration H^d is
// too large to be feasible. Returns I rows, each a random vector in {1,...,H}^d.
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
arma::mat ExpandH_sample(int H, int d, int I) {
  arma::mat M(d,I);
  arma::vec choice_set =arma::linspace<arma::vec>(1,H,H);
  for (int i = 0; i < I; i++) {
    M.col(i)=Rcpp::RcppArmadillo::sample(choice_set, d, true);
  }
  return M.t();
}

// =============================================================================
// LOSS FUNCTIONS
// =============================================================================

// Smoothed quantile loss (Gaussian kernel) - kept for compatibility
// h0 = h, h1 = 1/h, h2 = 1/h^2
// [[Rcpp::export]]
double lossGaussMat(const arma::mat& res, const double tau, const double h0, const double h1, const double h2) {
  double rst = 0.0;
  int I=res.n_rows;
  int J=res.n_cols;
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      rst += 0.3989423 * h0  * exp(-0.5 * h2 * res(i,j)* res(i,j)) + tau * res(i,j) - res(i,j)*arma::normcdf(-h1 * res(i,j));
    }
  }
  return rst;
}

// Smoothed quantile loss (triangular kernel), see He et al. (2023, conquer).
// This is the workhorse smoothed loss when conquer = TRUE.
// h0 = h, h1 = 1/h, h2 = 1/h^2
double lossTriangularMat(const arma::mat& res, const double tau, const double h0, const double h1, const double h2) {
  // h0 = h, h1 = 1/h, h2 = 1/(h*h)
  const double h = h0;
  const double inv_h = h1;
  const double half_h = 0.5 * h;
  const double tau_minus_half = tau - 0.5;

  double rst = 0.0;
  const int I = res.n_rows;
  const int J = res.n_cols;

  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      double u = res(i, j);
      double x = u * inv_h;
      double abs_x = std::abs(x);

      // Smooth part comes from convolving |u| with the triangular kernel
      double smooth_part;
      if (abs_x <= 1.0) {
        // Polynomial form when within kernel support
        double x2 = x * x;
        double x4 = x2 * x2;
        smooth_part = (0.75 * x2 - 0.125 * x4 + 0.375) * half_h;
      } else {
        // Outside support: behaves like |u|/2 (i.e. (|x|*h)/2 )
        smooth_part = abs_x * half_h;
      }

      rst += smooth_part + tau_minus_half * u;
    }
  }

  return rst;
}

// Standard (non-smoothed) check loss for matrix residuals
// [[Rcpp::export]]
double Loss_fun_mat(const arma::mat& res, const double tau) {
  double rst = 0.0;
  int I=res.n_rows;
  int J=res.n_cols;
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      rst += res(i,j) >= 0 ? tau * res(i,j) : (tau - 1) * res(i,j);
  }
  }
  return rst;
}

// Standard check loss for vector residuals
// [[Rcpp::export]]
double Loss_fun_vec(const arma::vec& res, const double tau) {
  double rst = 0.0;
  int I=res.n_elem;
  for (int i = 0; i < I; i++) {
      rst += res(i) >= 0 ? tau * res(i) : (tau - 1) * res(i);
  }
  return rst;
}


// Average check loss for an R NumericVector
// [[Rcpp::export]]
double Loss_fun(NumericVector xvec, double tau){
    double val=0;
    for (int j = 0; j < xvec.length(); j++) {
     val=val+xvec[j]*tau;
      if(xvec[j]<0) val=val-xvec[j];
      }
    val=val/xvec.length();
    return val;
}


// =============================================================================
// LOSS EVALUATION FOR UPDATING G-MEMBERSHIP
// =============================================================================
// For each node i and candidate row-group g, compute the cumulative quantile
// loss across all quantile levels in `taus`, holding H-memberships fixed.
// The vanilla algorithm (Algorithm 1, Step II) picks argmin_g of this loss.
// =============================================================================

// --- Multiplicative model, standard check loss ---
// [[Rcpp::export]]
NumericMatrix Loss_memberG_mq(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_H, NumericVector taus)
{
  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);
  int G=thetaG.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  int N=Ymat.n_rows;
  int Time=Ymat.n_cols;

  // Lagged and current Y
  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);

  NumericMatrix resi2(N,G);
  NumericVector eps_g;

  // Loop over quantile levels and accumulate loss in resi2(i, g)
  for (int k=0; k<taus.length();k++){

  arma::mat Wbeta(N,N);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
  NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
  NumericVector beta = betas(_,k);

  // Build column j of W * beta_{h_j}: scale column j of W by beta[h_j]
  for (int j = 0; j < N; j++) {
    Wbeta.col(j) = W.col(j)*beta[member_H[j]-1];
  }

  arma::mat Z1=Wbeta*Z2;

  // For each node i and candidate group g, evaluate residual & loss
  arma::mat GG=as<arma::mat>(gammaG);
  for (int i = 0; i < N; i++) {
    X1=X.row(i);
    for(int g =0;g<G;g++){
      eps_g = Y.row(i) - Z1.row(i)*thetaG(g,0) - Z2.row(i)*thetaG(g,1)-GG.row(g)*X1;
      resi2(i,g)= resi2(i,g)+ Loss_fun(eps_g, taus[k]);
    }
  }
  }

  return resi2;
}


// --- Multiplicative model, smoothed (conquer) loss ---
// [[Rcpp::export]]
NumericMatrix Loss_memberG_mq_conquer(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_H, NumericVector taus, const double h0, const double h1, const double h2)
{
  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);
  int G=thetaG.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  int N=Ymat.n_rows;
  int Time=Ymat.n_cols;

  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);

  NumericMatrix resi2(N,G);
  NumericVector eps_g;

  for (int k=0; k<taus.length();k++){

    arma::mat Wbeta(N,N);
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    NumericVector beta = betas(_,k);


    for (int j = 0; j < N; j++) {
      Wbeta.col(j) = W.col(j)*beta[member_H[j]-1];
    }

    arma::mat Z1=Wbeta*Z2;


    //Compute residuals and loss
    arma::mat GG=as<arma::mat>(gammaG);

    for (int i = 0; i < N; i++) {
      X1=X.row(i);

      for(int g =0;g<G;g++){
        eps_g = Y.row(i) - Z1.row(i)*thetaG(g,0) - Z2.row(i)*thetaG(g,1)-GG.row(g)*X1;
        // Use smoothed triangular-kernel loss
        resi2(i,g)= resi2(i,g)+ lossTriangularMat(eps_g, taus[k],h0,h1,h2);
      }
    }
  }

  return resi2;
}

// --- Additive model, standard check loss ---
// theta_{gh} = alpha_g + beta_h, so the network term decomposes as
//   alpha_g * (W y_{t-1})_i + sum_j w_{ij} * beta_{h_j} * Y_{j,t-1}
// [[Rcpp::export]]
NumericMatrix Loss_memberG_mq_additive(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_H, NumericVector taus)
{
  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);
  int G=thetaG.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  int N=Ymat.n_rows;
  int Time=Ymat.n_cols;

  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);

  NumericMatrix resi2(N,G);
  NumericVector eps_g;

  for (int k=0; k<taus.length();k++){

    arma::mat Wbeta(N,N);
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    NumericVector beta = betas(_,k);

    // Z1 column is W_{.,j} * beta_{h_j} (the H-side network contribution)
    for (int j = 0; j < N; j++) {
      Wbeta.col(j) = W.col(j)*beta[member_H[j]-1];
    }

    arma::mat Z1=Wbeta*Z2;
    // Z3 = W * Y_{t-1} (the alpha-side network contribution)
    arma::mat Z3=W*Z2;


    //Compute residuals and loss
    arma::mat GG=as<arma::mat>(gammaG);

    for (int i = 0; i < N; i++) {
      X1=X.row(i);

      for(int g =0;g<G;g++){
        // Residual under additive specification
        eps_g = Y.row(i) - Z1.row(i)-thetaG(g,0)*Z3.row(i) - Z2.row(i)*thetaG(g,1) - GG.row(g)*X1;
        resi2(i,g)= resi2(i,g)+ Loss_fun(eps_g, taus[k]);
      }
    }
  }

  return resi2;
}

//Find the loss function values for updating G memberships

// --- Additive model, smoothed (conquer) loss ---
// [[Rcpp::export]]
NumericMatrix Loss_memberG_mq_additive_conquer(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_H, NumericVector taus, const double h0, const double h1, const double h2)
{
  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);
  int G=thetaG.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  int N=Ymat.n_rows;
  int Time=Ymat.n_cols;

  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);

  NumericMatrix resi2(N,G);
  NumericVector eps_g;

  for (int k=0; k<taus.length();k++){

    arma::mat Wbeta(N,N);
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    NumericVector beta = betas(_,k);


    for (int j = 0; j < N; j++) {
      Wbeta.col(j) = W.col(j)*beta[member_H[j]-1];
    }

    arma::mat Z1=Wbeta*Z2;
    arma::mat Z3=W*Z2;


    //Compute residuals and loss
    arma::mat GG=as<arma::mat>(gammaG);

    for (int i = 0; i < N; i++) {
      X1=X.row(i);

      for(int g =0;g<G;g++){
        eps_g = Y.row(i) - Z1.row(i)-thetaG(g,0)*Z3.row(i) - Z2.row(i)*thetaG(g,1) - GG.row(g)*X1;
        resi2(i,g)= resi2(i,g)+  lossTriangularMat(eps_g, taus[k],h0,h1,h2);
      }
    }
  }

  return resi2;
}



//Find the loss function values for updating G memberships
// --- General model, standard check loss ---
// theta_{gh} is a free parameter for each (g,h). Network contribution is
//   sum_j w_{ij} * theta_{g, h_j} * Y_{j, t-1}
// [[Rcpp::export]]
NumericMatrix Loss_memberG_mq_general(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_H, NumericVector taus)
{
  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  List alphabetaGHs = as<List>(theta_GH["alphabeta_GHs"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);
  NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[0]);

  int G=thetaG.nrow();
  int H=alphabetas.ncol();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  int N=Ymat.n_rows;
  int Time=Ymat.n_cols;

  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);

  NumericMatrix resi2(N,G);
  NumericVector eps_g;

  for (int k=0; k<taus.length();k++){

    arma::mat Wbeta(N,N);
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);

    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(1,p));
    NumericMatrix alphabeta = alphabetas( Range(0,G-1) , Range(0,H-1));


    //Compute residuals and loss
    arma::mat GG=as<arma::mat>(gammaG);

    // For each (i, g), build the candidate W-row weighted by theta_{g, h_j}
    for (int i = 0; i < N; i++) {
      X1=X.row(i);


      for(int g =0;g<G;g++){

        for (int j = 0; j < N; j++) {
          Wbeta(i,j) = W(i,j)*alphabeta(g,member_H[j]-1);
        }


        arma::mat Z1=Wbeta.row(i)*Z2;


        eps_g = Y.row(i) - Z1 - Z2.row(i)*thetaG(g,0) - GG.row(g)*X1;
        resi2(i,g)= resi2(i,g)+  Loss_fun(eps_g, taus[k]);
      }
    }
  }

  return resi2;
}

// --- General model, smoothed (conquer) loss ---
// [[Rcpp::export]]
NumericMatrix Loss_memberG_mq_general_conquer(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_H, NumericVector taus, const double h0, const double h1, const double h2)
{
  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  List alphabetaGHs = as<List>(theta_GH["alphabeta_GHs"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);
  NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[0]);

  int G=thetaG.nrow();
  int H=alphabetas.ncol();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  int N=Ymat.n_rows;
  int Time=Ymat.n_cols;

  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);

  NumericMatrix resi2(N,G);
  NumericVector eps_g;

  for (int k=0; k<taus.length();k++){

    arma::mat Wbeta(N,N);
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);

    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(1,p));
    NumericMatrix alphabeta = alphabetas( Range(0,G-1) , Range(0,H-1));


    //Compute residuals and loss
    arma::mat GG=as<arma::mat>(gammaG);

    for (int i = 0; i < N; i++) {
      X1=X.row(i);


      for(int g =0;g<G;g++){

        for (int j = 0; j < N; j++) {
          Wbeta(i,j) = W(i,j)*alphabeta(g,member_H[j]-1);
        }


        arma::mat Z1=Wbeta.row(i)*Z2;


        eps_g = Y.row(i) - Z1 - Z2.row(i)*thetaG(g,0) - GG.row(g)*X1;
        resi2(i,g)= resi2(i,g)+  lossTriangularMat(eps_g, taus[k],h0,h1,h2);
      }
    }
  }

  return resi2;
}


// =============================================================================
// COORDINATE-DESCENT UPDATE OF H-MEMBERSHIP (Algorithm 1, Step III)
// =============================================================================
// Iteratively update each h_j by minimizing the multi-quantile loss over the
// rows i in the follower set of node j (i.e., {i : a_{ij} = 1} \cup chains).
// The update only re-evaluates a node when one of its dependencies changed,
// dramatically reducing redundant computation.
// =============================================================================

// --- Multiplicative model, standard loss ---
// [[Rcpp::export]]
IntegerVector Update_memberH_mq(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, IntegerVector member_H_init, List FriendW,List FriendW2, NumericVector taus)
{



  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);

  int G=thetaG.nrow();
  int H=betas.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);

  // Pre-compute per-quantile working arrays to avoid recomputation
  arma::field<arma::mat> betaYs(taus.length());
  arma::field<arma::mat> alphaWs(taus.length());
  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericVector  beta = betas(_,k);
    NumericVector  alpha = thetaG(_,0);
    NumericVector  nu = thetaG(_,1);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    arma::mat GG=as<arma::mat>(gammaG);

    betaYs(k)= arma::zeros<arma::mat>(N,Time-1);
    alphaWs(k)= arma::zeros<arma::mat>(N,N);
    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      X1=X.row(j);
      betaYs(k).row(j) = Z2.row(j)*betas(member_H_init[j]-1,k);
      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
      alphaWs(k).row(j) =  W.row(j)*alpha[member_G[j]-1];
    }
  }


  //update data membership H one at a time
  int change=1;
    int iter=1;
    IntegerVector member_H = clone(member_H_init);

  arma::uvec Update_idx = arma::linspace<arma::uvec>(0,N-1,N);

  while ((change>0) & (iter <100)){

    arma::uvec Update_idx_new(1);
    change=0;
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];

      arma::uvec IC=arma::intersect(fri_j,Update_idx);
      // Only update node j if any of its followers were recently updated
      if(!IC.is_empty()){
        arma::uvec fri_j2=FriendW2[j];
        arma::vec loss(H);

        int h_min=0;

  // Evaluate loss for each candidate H-membership of node j
  for (int k=0; k<taus.length();k++){

        for(int h =0; h<H; h++){
          betaYs(k).row(j) = betas(h,k)*Z2.row(j);
          loss(h) = loss(h)+ Loss_fun_mat(eps_mats(k).rows(fri_j-1)-alphaWs(k).submat(fri_j-1,fri_j2-1)*betaYs(k).rows(fri_j2-1),taus[k]);

        }
  }

        h_min=loss.index_min()+1;
  for (int k=0; k<taus.length();k++){
    betaYs(k).row(j) = betas(h_min-1,k)*Z2.row(j);
  }
        if(member_H[j]!=h_min){
          change=change+1;
          member_H[j]=h_min;
          // Mark followers of j as dirty
          Update_idx_new=arma::unique(arma::join_vert(Update_idx_new,fri_j));
        }

      }

    }


    Update_idx=arma::unique(Update_idx_new);


    iter=iter+1;
  }


  return member_H;
}


//Find the loss function values for updating H memberships
// --- Additive model, standard loss ---
// [[Rcpp::export]]
IntegerVector Update_memberH_mq_additive(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, IntegerVector member_H_init, List FriendW,List FriendW2, NumericVector taus)
{



  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);

  int G=thetaG.nrow();
  int H=betas.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;// alpha-side network term

  arma::field<arma::mat> betaYs(taus.length());
  arma::field<arma::mat> alphaWs(taus.length());
  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericVector  beta = betas(_,k);
    NumericVector  alpha = thetaG(_,0);
    NumericVector  nu = thetaG(_,1);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    arma::mat GG=as<arma::mat>(gammaG);

    betaYs(k)= arma::zeros<arma::mat>(N,Time-1);
    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      X1=X.row(j);
      betaYs(k).row(j) = Z2.row(j)*betas(member_H_init[j]-1,k);
      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1-Z3.row(j)*alpha[member_G[j]-1];
    }
  }


  //update data membership H one at a time
  int change=1;
  int iter=1;
  IntegerVector member_H = clone(member_H_init);

  arma::uvec Update_idx = arma::linspace<arma::uvec>(0,N-1,N);

  while ((change>0) & (iter <100)){

    arma::uvec Update_idx_new(1);
    change=0;
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];

      arma::uvec IC=arma::intersect(fri_j,Update_idx);
      if(!IC.is_empty()){
        arma::uvec fri_j2=FriendW2[j];
        arma::vec loss(H);

        int h_min=0;

        for (int k=0; k<taus.length();k++){

          for(int h =0; h<H; h++){
            betaYs(k).row(j) = betas(h,k)*Z2.row(j);
            // For additive model, the network factor is just W (no alpha multiplier)
            loss(h) = loss(h)+ Loss_fun_mat(eps_mats(k).rows(fri_j-1)-W.submat(fri_j-1,fri_j2-1)*betaYs(k).rows(fri_j2-1),taus[k]);

          }
        }

        h_min=loss.index_min()+1;
        for (int k=0; k<taus.length();k++){
          betaYs(k).row(j) = betas(h_min-1,k)*Z2.row(j);
        }
        if(member_H[j]!=h_min){
          change=change+1;
          member_H[j]=h_min;

          Update_idx_new=arma::unique(arma::join_vert(Update_idx_new,fri_j));
        }

      }

    }


    Update_idx=arma::unique(Update_idx_new);


    iter=iter+1;
  }

  return member_H;
}

//Find the loss function values for updating H memberships
// --- Additive model, smoothed (conquer) loss ---
// [[Rcpp::export]]
IntegerVector Update_memberH_mq_additive_conquer(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, IntegerVector member_H_init, List FriendW,List FriendW2, NumericVector taus,const double h0, const double h1,const double h2)
{



  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);

  int G=thetaG.nrow();
  int H=betas.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;

  arma::field<arma::mat> betaYs(taus.length());
  arma::field<arma::mat> alphaWs(taus.length());
  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericVector  beta = betas(_,k);
    NumericVector  alpha = thetaG(_,0);
    NumericVector  nu = thetaG(_,1);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    arma::mat GG=as<arma::mat>(gammaG);

    betaYs(k)= arma::zeros<arma::mat>(N,Time-1);
    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      X1=X.row(j);
      betaYs(k).row(j) = Z2.row(j)*betas(member_H_init[j]-1,k);
      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1-Z3.row(j)*alpha[member_G[j]-1];
    }
  }


  //update data membership H one at a time
  int change=1;
  int iter=1;
  IntegerVector member_H = clone(member_H_init);

  arma::uvec Update_idx = arma::linspace<arma::uvec>(0,N-1,N);

  while ((change>0) & (iter <100)){

    arma::uvec Update_idx_new(1);

    change=0;
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];

      arma::uvec IC=arma::intersect(fri_j,Update_idx);
      if(!IC.is_empty()){
        arma::uvec fri_j2=FriendW2[j];
        arma::vec loss(H);

        int h_min=0;

        for (int k=0; k<taus.length();k++){

          for(int h =0; h<H; h++){
            betaYs(k).row(j) = betas(h,k)*Z2.row(j);
            arma::mat res=eps_mats(k).rows(fri_j-1)-W.submat(fri_j-1,fri_j2-1)*betaYs(k).rows(fri_j2-1);
            loss(h) = loss(h)+ lossTriangularMat(arma::vectorise(res,0),taus[k],h0,h1,h2);

          }
        }


        h_min=loss.index_min()+1;
        for (int k=0; k<taus.length();k++){
          betaYs(k).row(j) = betas(h_min-1,k)*Z2.row(j);
        }
        if(member_H[j]!=h_min){
          change=change+1;
          member_H[j]=h_min;

          Update_idx_new=arma::unique(arma::join_vert(Update_idx_new,fri_j));
        }

      }

    }


    Update_idx=arma::unique(Update_idx_new);


    iter=iter+1;
  }


  return member_H;
}

//Find the loss function values for updating H memberships
// --- Multiplicative model, smoothed (conquer) loss ---
// [[Rcpp::export]]
IntegerVector Update_memberH_mq_conquer(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, IntegerVector member_H_init, List FriendW,List FriendW2, NumericVector taus,const double h0, const double h1,const double h2)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);

  int G=thetaG.nrow();
  int H=betas.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);

  arma::field<arma::mat> betaYs(taus.length());
  arma::field<arma::mat> alphaWs(taus.length());
  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericVector  beta = betas(_,k);
    NumericVector  alpha = thetaG(_,0);
    NumericVector  nu = thetaG(_,1);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    arma::mat GG=as<arma::mat>(gammaG);

    betaYs(k)= arma::zeros<arma::mat>(N,Time-1);
    alphaWs(k)= arma::zeros<arma::mat>(N,N);
    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      X1=X.row(j);
      betaYs(k).row(j) = Z2.row(j)*betas(member_H_init[j]-1,k);
      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
      alphaWs(k).row(j) =  W.row(j)*alpha[member_G[j]-1];
    }
  }


  //update data membership H one at a time
  int change=1;
  int iter=1;
  IntegerVector member_H = clone(member_H_init);

  arma::uvec Update_idx = arma::linspace<arma::uvec>(0,N-1,N);

  while ((change>0) & (iter <100)){

    arma::uvec Update_idx_new(1);

    change=0;
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];

      arma::uvec IC=arma::intersect(fri_j,Update_idx);
      if(!IC.is_empty()){
        arma::uvec fri_j2=FriendW2[j];
        arma::vec loss(H);

        int h_min=0;

        for (int k=0; k<taus.length();k++){

          for(int h =0; h<H; h++){
            betaYs(k).row(j) = betas(h,k)*Z2.row(j);
            arma::mat res=eps_mats(k).rows(fri_j-1)-alphaWs(k).submat(fri_j-1,fri_j2-1)*betaYs(k).rows(fri_j2-1);
            loss(h) = loss(h)+ lossTriangularMat(arma::vectorise(res,0),taus[k],h0,h1,h2);
          }
        }

        h_min=loss.index_min()+1;
        for (int k=0; k<taus.length();k++){
          betaYs(k).row(j) = betas(h_min-1,k)*Z2.row(j);
        }
        if(member_H[j]!=h_min){
          change=change+1;
          member_H[j]=h_min;

          Update_idx_new=arma::unique(arma::join_vert(Update_idx_new,fri_j));
        }

      }

    }


    Update_idx=arma::unique(Update_idx_new);


    iter=iter+1;
  }


  return member_H;
}


//Find the loss function values for updating H memberships
// --- General model, standard loss ---
// [[Rcpp::export]]
IntegerVector Update_memberH_mq_general(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, IntegerVector member_H_init, List FriendW,List FriendW2, NumericVector taus)
{



  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  List alphabetaGHs = as<List>(theta_GH["alphabeta_GHs"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);
  NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[0]);

  int G=thetaG.nrow();
  int H=alphabetas.ncol();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;

  arma::field<arma::mat> alphabetaWs(taus.length());
  arma::field<arma::mat> eps_mats(taus.length());


  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);

    NumericVector  nu = thetaG(_,0);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(1,p));
    arma::mat GG=as<arma::mat>(gammaG);

    alphabetaWs(k)= arma::zeros<arma::mat>(N,N);
    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;
      X1=X.row(j);

      // Sparse construction: only fill non-zero (follower) entries
      for (int i = 0; i < I; i++) {
        alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alphabetas(member_G[fri_j(i)-1]-1,member_H_init[j]-1);
      }
      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
    }
  }


  //update data membership H one at a time
  int change=1;
  int iter=1;
  IntegerVector member_H = clone(member_H_init);

  arma::uvec Update_idx = arma::linspace<arma::uvec>(0,N-1,N);

  while ((change>0) & (iter <100)){

    arma::uvec Update_idx_new(1);

    change=0;
    for (int j = 0; j < N; j++) {

      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;

      arma::uvec IC=arma::intersect(fri_j,Update_idx);
      if(!IC.is_empty()){
        arma::uvec fri_j2=FriendW2[j];
        arma::vec loss(H);

        int h_min=0;

        for (int k=0; k<taus.length();k++){
          NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);

          for(int h =0; h<H; h++){
            // Tentatively set h_j = h and update sparse entries of alphabetaWs
            for(int i =0; i<I; i++){
              alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alphabetas(member_G[fri_j(i)-1]-1,h);
            }

            arma::mat res=eps_mats(k).rows(fri_j-1)-alphabetaWs(k).submat(fri_j-1,fri_j2-1)*Z2.rows(fri_j2-1);

            loss(h) = loss(h)+ Loss_fun_mat(res,taus[k]);


          }
        }

        h_min=loss.index_min()+1;
        // Commit chosen h_min to the sparse matrix
        for (int k=0; k<taus.length();k++){
          NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);
          for(int i =0; i<I; i++){
            alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alphabetas(member_G[fri_j(i)-1]-1,h_min-1);
          }
        }
        if(member_H[j]!=h_min){
          change=change+1;
          member_H[j]=h_min;

          Update_idx_new=arma::unique(arma::join_vert(Update_idx_new,fri_j));
        }

      }

    }


    Update_idx=arma::unique(Update_idx_new);


    iter=iter+1;
  }


  return member_H;
}

//Find the loss function values for updating H memberships
// --- General model, smoothed (conquer) loss ---
// [[Rcpp::export]]
IntegerVector Update_memberH_mq_general_conquer(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, IntegerVector member_H_init, List FriendW,List FriendW2, NumericVector taus,const double h0, const double h1,const double h2)
{



  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  List alphabetaGHs = as<List>(theta_GH["alphabeta_GHs"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);
  NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[0]);

  int G=thetaG.nrow();
  int H=alphabetas.ncol();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;

  arma::field<arma::mat> alphabetaWs(taus.length());
  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);

    NumericVector  nu = thetaG(_,0);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(1,p));
    arma::mat GG=as<arma::mat>(gammaG);

    alphabetaWs(k)= arma::zeros<arma::mat>(N,N);
    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      X1=X.row(j);
      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;

      for (int i = 0; i < I; i++) {
        alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alphabetas(member_G[fri_j(i)-1]-1,member_H_init[j]-1);
      }
      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
    }
  }


  //update data membership H one at a time
  int change=1;
  int iter=1;
  IntegerVector member_H = clone(member_H_init);

  arma::uvec Update_idx = arma::linspace<arma::uvec>(0,N-1,N);

  while ((change>0) & (iter <100)){

    arma::uvec Update_idx_new(1);

    change=0;
    for (int j = 0; j < N; j++) {

      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;

      arma::uvec IC=arma::intersect(fri_j,Update_idx);
      if(!IC.is_empty()){
        arma::uvec fri_j2=FriendW2[j];
        arma::vec loss(H);

        int h_min=0;

        for (int k=0; k<taus.length();k++){
          NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);

          for(int h =0; h<H; h++){
            for(int i =0; i<I; i++){
              alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alphabetas(member_G[fri_j(i)-1]-1,h);
            }

            arma::mat res=eps_mats(k).rows(fri_j-1)-alphabetaWs(k).submat(fri_j-1,fri_j2-1)*Z2.rows(fri_j2-1);

            loss(h) = loss(h)+ lossTriangularMat(arma::vectorise(res,0),taus[k],h0,h1,h2);

          }
        }

        h_min=loss.index_min()+1;
        for (int k=0; k<taus.length();k++){
          NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);
          for(int i =0; i<I; i++){
            alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alphabetas(member_G[fri_j(i)-1]-1,h_min-1);
          }
        }
        if(member_H[j]!=h_min){
          change=change+1;
          member_H[j]=h_min;

          Update_idx_new=arma::unique(arma::join_vert(Update_idx_new,fri_j));
        }

      }

    }


    Update_idx=arma::unique(Update_idx_new);


    iter=iter+1;
  }

  return member_H;
}

// =============================================================================
// PROPOSAL ROUTINES (used by the Enhanced Algorithm 2 to escape local minima)
// =============================================================================
// For each node j, generate a candidate H-membership for its followers
// (active set) by enumeration (small d) or random sampling (large d).
// The output is an N x N matrix where row i lists the proposed memberships
// for nodes within node i's neighborhood.
// =============================================================================

// --- Multiplicative model, standard loss ---
// [[Rcpp::export]]
NumericMatrix Proposal_memberH_mq(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, arma::vec& member_H_init, List FriendW, arma::uvec& idx_update, NumericVector taus,int nsample)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);

  int G=thetaG.nrow();
  int H=betas.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;


  arma::field<arma::mat> alphabetaWs(taus.length());

  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericVector  beta = betas(_,k);
    NumericVector  alpha = thetaG(_,0);
    NumericVector  nu = thetaG(_,1);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    arma::mat GG=as<arma::mat>(gammaG);

    alphabetaWs(k)= arma::zeros<arma::mat>(N,N);

    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;
      X1=X.row(j);

      for (int i = 0; i < I; i++) {
        alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alpha[member_G[fri_j(i)-1]-1]*beta[member_H_init[j]-1];
      }

      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
    }
  }


  //update data memberships H one at a time
  NumericMatrix Proposal(N,N);

  for (int j = 0; j < N; j++) {
    arma::uvec fri_j=FriendW[j];
    int I=fri_j.n_elem;
    for (int i=0;i<I;i++){
      Proposal(j,fri_j(i)-1)=member_H_init(fri_j(i)-1);
    }
  }


  // For each node j, search for the best H assignment among its followers
  for (int j = 0; j < N; j++) {
    arma::uvec fri_j=FriendW[j];
    arma::uvec IC=arma::intersect(fri_j,idx_update);

    if(!IC.is_empty()){

      int D =IC.n_elem;

      if((D*log(H))<log(nsample)){  // only consider smaller D, remove this line to use random samples

        arma::mat Hpool(nsample,H);

        if((D*log(H))<log(nsample)){
          Hpool= ExpandH(H,D);
        }else{
          Hpool= ExpandH_sample(H,D,nsample);
          Hpool=join_cols(Hpool,member_H_init(IC-1).t());
        }

        int I= Hpool.n_rows;
        arma::vec loss(I);

        for (int i = 0; i <I ; i++) {

          for (int k=0; k<taus.length();k++){
            NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
            NumericVector  beta = betas(_,k);
            NumericVector  alpha = thetaG(_,0);
            arma::mat Wj=alphabetaWs(k).row(j);

            for(int d =0; d<D; d++){
              int l=Hpool(i,d)-1;

              Wj(0,IC(d)-1) =  W(j,IC(d)-1)*alpha[member_G[j]-1]*beta[l];
            }

            arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
            loss(i) = loss(i)+ Loss_fun_mat(tmp_vec,taus[k]);
          }


        }

        int h_min=loss.index_min();

        for(int d =0; d<D; d++){
          Proposal(j,IC(d)-1)=Hpool(h_min,d);
        }
      }

    }
  }
  return Proposal;
}

// --- Additive model, standard loss ---
// [[Rcpp::export]]
NumericMatrix Proposal_memberH_mq_additive(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, arma::vec& member_H_init, List FriendW, arma::uvec& idx_update, NumericVector taus,int nsample)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);

  int G=thetaG.nrow();
  int H=betas.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;

  arma::field<arma::mat> betaWs(taus.length());
  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericVector  alpha = thetaG(_,0);
    NumericVector  nu = thetaG(_,1);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    arma::mat GG=as<arma::mat>(gammaG);

      betaWs(k)= arma::zeros<arma::mat>(N,N);
    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      X1=X.row(j);
      int I=fri_j.n_elem;

      for (int i = 0; i < I; i++) {
        betaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*betas(member_H_init[j]-1,k);
      }

      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1-Z3.row(j)*alpha[member_G[j]-1];
    }
  }


  //update data memberships H one at a time
  NumericMatrix Proposal(N,N);

  for (int j = 0; j < N; j++) {
    arma::uvec fri_j=FriendW[j];
    int I=fri_j.n_elem;
    for (int i=0;i<I;i++){
      Proposal(j,fri_j(i)-1)=member_H_init(fri_j(i)-1);
    }
  }



  for (int j = 0; j < N; j++) {
    arma::uvec fri_j=FriendW[j];
    arma::uvec IC=arma::intersect(fri_j,idx_update);

    if(!IC.is_empty()){

      int D =IC.n_elem;

      if((D*log(H))<log(nsample)){  // only consider smaller D, remove this line to use random samples

      arma::mat Hpool(nsample,H);

      if((D*log(H))<log(nsample)){
        Hpool= ExpandH(H,D);
      }else{
        Hpool= ExpandH_sample(H,D,nsample);
        Hpool=join_cols(Hpool,member_H_init(IC-1).t());
      }

      int I= Hpool.n_rows;
      arma::vec loss(I);

      for (int i = 0; i <I ; i++) {

        for (int k=0; k<taus.length();k++){

          arma::mat Wj=betaWs(k).row(j);

          for(int d =0; d<D; d++){
            int l=Hpool(i,d)-1;

            Wj(0,IC(d)-1) =  W(j,IC(d)-1)*betas(l,k);
          }

          arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
          loss(i) = loss(i)+ Loss_fun_mat(tmp_vec,taus[k]);
          }


      }

      int h_min=loss.index_min();

      for(int d =0; d<D; d++){
        Proposal(j,IC(d)-1)=Hpool(h_min,d);
      }
      }


    }
  }

  return Proposal;
}

// --- Multiplicative model, smoothed (conquer) loss ---
// [[Rcpp::export]]
NumericMatrix Proposal_memberH_mq_conquer(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, arma::vec& member_H_init, List FriendW, arma::uvec& idx_update, NumericVector taus,int nsample, const double h0, const double h1, const double h2)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);

  int G=thetaG.nrow();
  int H=betas.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;


  arma::field<arma::mat> alphabetaWs(taus.length());

  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericVector  beta = betas(_,k);
    NumericVector  alpha = thetaG(_,0);
    NumericVector  nu = thetaG(_,1);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    arma::mat GG=as<arma::mat>(gammaG);

    alphabetaWs(k)= arma::zeros<arma::mat>(N,N);

    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      X1=X.row(j);
      int I=fri_j.n_elem;

      for (int i = 0; i < I; i++) {
        alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alpha[member_G[fri_j(i)-1]-1]*beta[member_H_init[j]-1];
      }

      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
    }
  }


  //update data memberships H one at a time
  NumericMatrix Proposal(N,N);

  for (int j = 0; j < N; j++) {
    arma::uvec fri_j=FriendW[j];
    int I=fri_j.n_elem;
    for (int i=0;i<I;i++){
      Proposal(j,fri_j(i)-1)=member_H_init(fri_j(i)-1);
    }
  }



  for (int j = 0; j < N; j++) {
    arma::uvec fri_j=FriendW[j];
    arma::uvec IC=arma::intersect(fri_j,idx_update);

    if(!IC.is_empty()){

      int D =IC.n_elem;

      if((D*log(H))<log(nsample)){  // only consider smaller D, remove this line to use random samples

      arma::mat Hpool(nsample,H);

      if((D*log(H))<log(nsample)){
        Hpool= ExpandH(H,D);
      }else{
        Hpool= ExpandH_sample(H,D,nsample);
        Hpool=join_cols(Hpool,member_H_init(IC-1).t());
      }

      int I= Hpool.n_rows;
      arma::vec loss(I);

      for (int i = 0; i <I ; i++) {

        for (int k=0; k<taus.length();k++){
          NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
          NumericVector  beta = betas(_,k);
          NumericVector  alpha = thetaG(_,0);
          arma::mat Wj=alphabetaWs(k).row(j);

          for(int d =0; d<D; d++){
            int l=Hpool(i,d)-1;

            Wj(0,IC(d)-1) =  W(j,IC(d)-1)*alpha[member_G[j]-1]*beta[l];
          }

          arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
          loss(i) = loss(i)+ lossTriangularMat(tmp_vec,taus[k],h0,h1,h2);
        }


      }

      int h_min=loss.index_min();

      for(int d =0; d<D; d++){
        Proposal(j,IC(d)-1)=Hpool(h_min,d);
      }

      }

    }
  }

  return Proposal;
}

// --- Additive model, smoothed (conquer) loss ---
// [[Rcpp::export]]
NumericMatrix Proposal_memberH_mq_additive_conquer(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, arma::vec& member_H_init, List FriendW, arma::uvec& idx_update, NumericVector taus,int nsample, const double h0, const double h1, const double h2)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);

  int G=thetaG.nrow();
  int H=betas.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;

  arma::field<arma::mat> betaWs(taus.length());
  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericVector  alpha = thetaG(_,0);
    NumericVector  nu = thetaG(_,1);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    arma::mat GG=as<arma::mat>(gammaG);

    betaWs(k)= arma::zeros<arma::mat>(N,N);
    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      X1=X.row(j);
      int I=fri_j.n_elem;

      for (int i = 0; i < I; i++) {
        betaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*betas(member_H_init[j]-1,k);
      }

      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1-Z3.row(j)*alpha[member_G[j]-1];
    }
  }


  //update data memberships H one at a time
  NumericMatrix Proposal(N,N);

  for (int j = 0; j < N; j++) {
    arma::uvec fri_j=FriendW[j];
    int I=fri_j.n_elem;
    for (int i=0;i<I;i++){
      Proposal(j,fri_j(i)-1)=member_H_init(fri_j(i)-1);
    }
  }



  for (int j = 0; j < N; j++) {
    arma::uvec fri_j=FriendW[j];
    arma::uvec IC=arma::intersect(fri_j,idx_update);

    if(!IC.is_empty()){

      int D =IC.n_elem;

      if((D*log(H))<log(nsample)){  // only consider smaller D, remove this line to use random samples

        arma::mat Hpool(nsample,H);

        if((D*log(H))<log(nsample)){
          Hpool= ExpandH(H,D);
        }else{
          Hpool= ExpandH_sample(H,D,nsample);
          Hpool=join_cols(Hpool,member_H_init(IC-1).t());
        }

        int I= Hpool.n_rows;
        arma::vec loss(I);

        for (int i = 0; i <I ; i++) {

          for (int k=0; k<taus.length();k++){

            arma::mat Wj=betaWs(k).row(j);

            for(int d =0; d<D; d++){
              int l=Hpool(i,d)-1;

              Wj(0,IC(d)-1) =  W(j,IC(d)-1)*betas(l,k);
            }

            arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
            loss(i) = loss(i)+ lossTriangularMat(tmp_vec,taus[k],h0,h1,h2);
          }


        }

        int h_min=loss.index_min();

        for(int d =0; d<D; d++){
          Proposal(j,IC(d)-1)=Hpool(h_min,d);
        }
      }


    }
  }

  return Proposal;
}

// --- General model, smoothed (conquer) loss ---
// [[Rcpp::export]]
NumericMatrix Proposal_memberH_mq_general_conquer(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, arma::vec& member_H_init, List FriendW, arma::uvec& idx_update, NumericVector taus,int nsample, const double h0, const double h1, const double h2)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  List alphabetaGHs = as<List>(theta_GH["alphabeta_GHs"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);
  NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[0]);

  int G=thetaG.nrow();
  int H=alphabetas.ncol();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;



  arma::field<arma::mat> alphabetaWs(taus.length());
  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);

    NumericVector  nu = thetaG(_,0);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(1,p));
    arma::mat GG=as<arma::mat>(gammaG);

    alphabetaWs(k)= arma::zeros<arma::mat>(N,N);
    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;
      X1=X.row(j);
      for (int i = 0; i < I; i++) {
        alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alphabetas(member_G[fri_j(i)-1]-1,member_H_init[j]-1);
      }
      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
    }
  }


  //update data memberships H one at a time
  NumericMatrix Proposal(N,N);

  for (int j = 0; j < N; j++) {
    arma::uvec fri_j=FriendW[j];
    int I=fri_j.n_elem;
    for (int i=0;i<I;i++){
      Proposal(j,fri_j(i)-1)=member_H_init(fri_j(i)-1);
    }
  }



  for (int j = 0; j < N; j++) {
    arma::uvec fri_j=FriendW[j];
    arma::uvec IC=arma::intersect(fri_j,idx_update);

    if(!IC.is_empty()){



      int D =IC.n_elem;

      if((D*log(H))<log(nsample)){  // only consider smaller D, remove this line to use random samples

        arma::mat Hpool(nsample,H);

        if((D*log(H))<log(nsample)){
          Hpool= ExpandH(H,D);
        }else{
          Hpool= ExpandH_sample(H,D,nsample);
          Hpool=join_cols(Hpool,member_H_init(IC-1).t());
        }

        int I= Hpool.n_rows;
        arma::vec loss(I);

        for (int i = 0; i <I ; i++) {

          for (int k=0; k<taus.length();k++){
            NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);
            arma::mat Wj=alphabetaWs(k).row(j);

            for(int d =0; d<D; d++){
              int l=Hpool(i,d)-1;
              Wj(0,IC(d)-1) =  W(j,IC(d)-1)*alphabetas(member_G[j]-1,l);
            }
            arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
            loss(i) = loss(i)+ lossTriangularMat(tmp_vec,taus[k],h0,h1,h2);
          }


        }

        int h_min=loss.index_min();

        for(int d =0; d<D; d++){
          Proposal(j,IC(d)-1)=Hpool(h_min,d);
        }


      }
    }
  }

  return Proposal;
}

// --- General model, standard loss ---
// [[Rcpp::export]]
NumericMatrix Proposal_memberH_mq_general(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, arma::vec& member_H_init, List FriendW, arma::uvec& idx_update, NumericVector taus,int nsample)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  List alphabetaGHs = as<List>(theta_GH["alphabeta_GHs"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);
  NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[0]);

  int G=thetaG.nrow();
  int H=alphabetas.ncol();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;



  arma::field<arma::mat> alphabetaWs(taus.length());
  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);

    NumericVector  nu = thetaG(_,0);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(1,p));
    arma::mat GG=as<arma::mat>(gammaG);

    alphabetaWs(k)= arma::zeros<arma::mat>(N,N);
    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;
      X1=X.row(j);

      for (int i = 0; i < I; i++) {
        alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alphabetas(member_G[fri_j(i)-1]-1,member_H_init[j]-1);
      }
      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
    }
  }


  //update data memberships H one at a time
  NumericMatrix Proposal(N,N);

  for (int j = 0; j < N; j++) {
    arma::uvec fri_j=FriendW[j];
    int I=fri_j.n_elem;
    for (int i=0;i<I;i++){
      Proposal(j,fri_j(i)-1)=member_H_init(fri_j(i)-1);
    }
  }



  for (int j = 0; j < N; j++) {
    arma::uvec fri_j=FriendW[j];
    arma::uvec IC=arma::intersect(fri_j,idx_update);

    if(!IC.is_empty()){



      int D =IC.n_elem;

      if((D*log(H))<log(nsample)){  // only consider smaller D, remove this line to use random samples

        arma::mat Hpool(nsample,H);

        if((D*log(H))<log(nsample)){
          Hpool= ExpandH(H,D);
        }else{
          Hpool= ExpandH_sample(H,D,nsample);
          Hpool=join_cols(Hpool,member_H_init(IC-1).t());
        }

        int I= Hpool.n_rows;
        arma::vec loss(I);

        for (int i = 0; i <I ; i++) {

          for (int k=0; k<taus.length();k++){
            NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);
            arma::mat Wj=alphabetaWs(k).row(j);

            for(int d =0; d<D; d++){
              int l=Hpool(i,d)-1;
              Wj(0,IC(d)-1) =  W(j,IC(d)-1)*alphabetas(member_G[j]-1,l);
            }
            arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
            loss(i) = loss(i)+ Loss_fun_mat(tmp_vec,taus[k]);
          }


        }

        int h_min=loss.index_min();

        for(int d =0; d<D; d++){
          Proposal(j,IC(d)-1)=Hpool(h_min,d);
        }


      }
    }
  }

  return Proposal;
}

// =============================================================================
// REFINEMENT ROUTINES (for membership diagnostics / refinement)
// =============================================================================
// Refine_G_* : For each node j, evaluate the loss the node would attain if
//              its G-membership were changed to each candidate g in 1,...,G,
//              while best-optimizing H-memberships of j's followers.
// Refine_H_* : Analogous diagnostic for H-membership.
// =============================================================================

// --- G-refinement, multiplicative model, standard loss ---
// [[Rcpp::export]]
NumericMatrix Refine_G_mq(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, arma::vec& member_H_init, List FriendW, NumericVector taus,int nsample)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);

  int G=thetaG.nrow();
  int H=betas.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;


  arma::field<arma::mat> alphabetaWs(taus.length());

  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericVector  beta = betas(_,k);
    NumericVector  alpha = thetaG(_,0);
    NumericVector  nu = thetaG(_,1);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    arma::mat GG=as<arma::mat>(gammaG);

    alphabetaWs(k)= arma::zeros<arma::mat>(N,N);

    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;
      X1=X.row(j);

      for (int i = 0; i < I; i++) {
        alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alpha[member_G[fri_j(i)-1]-1]*beta[member_H_init[j]-1];
      }

      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
    }
  }


//Find the original loss given the estimated memberships
  NumericMatrix Result(N,2+G);
  // Column 1: baseline loss with current G-membership
  for (int j = 0; j < N; j++) {
    Result(j,0) = member_G[j];
    arma::uvec fri_j=FriendW[j];

    for (int k=0; k<taus.length();k++){
      NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
      NumericVector  beta = betas(_,k);
      NumericVector  alpha = thetaG(_,0);
      arma::mat Wj=alphabetaWs(k).row(j);

      arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
      Result(j,1) = Result(j,1)+ Loss_fun_mat(tmp_vec,taus[k]);
    }
  }


// For each candidate group g, evaluate per-node loss after best H-update
  for(int g=0;g<G;g++){

//Update the residuals
for(int k=0;k<taus.length();k++){
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
  NumericVector  beta = betas(_,k);
  NumericVector  alpha = thetaG(_,0);
  NumericVector  nu = thetaG(_,1);
  NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
  arma::mat GG=as<arma::mat>(gammaG);

  eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
  for (int j = 0; j < N; j++) {
    X1=X.row(j);
      //Find residual part corresponding to memberG part
    eps_mats(k).row(j) = Y.row(j)-nu[g]*Z2.row(j)-GG.row(g)*X1;
  }
}


  for (int j = 0; j < N; j++) {
    arma::uvec fri_j=FriendW[j];
    arma::uvec IC=fri_j;

    if(!IC.is_empty()){

      int D =IC.n_elem;

      if((D*log(H))<log(nsample)){  // only consider smaller D, remove this line to use random samples

        arma::mat Hpool(nsample,H);

        if((D*log(H))<log(nsample)){
          Hpool= ExpandH(H,D);
        }else{
          Hpool= ExpandH_sample(H,D,nsample);
          Hpool=join_cols(Hpool,member_H_init(IC-1).t());
        }

        int I= Hpool.n_rows;
        arma::vec loss(I);

        for (int i = 0; i <I ; i++) {

          for (int k=0; k<taus.length();k++){
            NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
            NumericVector  beta = betas(_,k);
            NumericVector  alpha = thetaG(_,0);
            arma::mat Wj=alphabetaWs(k).row(j);

            for(int d =0; d<D; d++){
              int l=Hpool(i,d)-1;

              Wj(0,IC(d)-1) =  W(j,IC(d)-1)*alpha[g]*beta[l];
            }

            arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
            loss(i) = loss(i)+ Loss_fun_mat(tmp_vec,taus[k]);
          }


        }

        int h_min=loss.index_min();


        Result(j,g+2)=loss(h_min);

      }


    }
  }
  }

  return Result;
}


// --- G-refinement, multiplicative model, smoothed loss ---
// [[Rcpp::export]]
NumericMatrix Refine_G_mq_conquer(const arma::mat& Ymat, const arma::mat& W,  arma::cube X, List theta_GH, IntegerVector member_G, arma::vec& member_H_init, List FriendW, NumericVector taus,int nsample, const double h0, const double h1, const double h2)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);

  int G=thetaG.nrow();
  int H=betas.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;


  arma::field<arma::mat> alphabetaWs(taus.length());

  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericVector  beta = betas(_,k);
    NumericVector  alpha = thetaG(_,0);
    NumericVector  nu = thetaG(_,1);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    arma::mat GG=as<arma::mat>(gammaG);

    alphabetaWs(k)= arma::zeros<arma::mat>(N,N);

    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;
      X1=X.row(j);

      for (int i = 0; i < I; i++) {
        alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alpha[member_G[fri_j(i)-1]-1]*beta[member_H_init[j]-1];
      }

      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
    }
  }


  //Find the original loss given the estimated memberships
  NumericMatrix Result(N,2+G);

  for (int j = 0; j < N; j++) {
    Result(j,0) = member_G[j];
    arma::uvec fri_j=FriendW[j];

    for (int k=0; k<taus.length();k++){
      NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
      NumericVector  beta = betas(_,k);
      NumericVector  alpha = thetaG(_,0);
      arma::mat Wj=alphabetaWs(k).row(j);

      arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
      Result(j,1) = Result(j,1)+lossTriangularMat(tmp_vec,taus[k],h0,h1,h2);
    }
  }



  for(int g=0;g<G;g++){

    //Update the residuals
    for(int k=0;k<taus.length();k++){
      NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
      NumericVector  beta = betas(_,k);
      NumericVector  alpha = thetaG(_,0);
      NumericVector  nu = thetaG(_,1);
      NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
      arma::mat GG=as<arma::mat>(gammaG);

      eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
      for (int j = 0; j < N; j++) {
        X1=X.row(j);
        //Find residual part corresponding to memberG part
        eps_mats(k).row(j) = Y.row(j)-nu[g]*Z2.row(j)-GG.row(g)*X1;
      }
    }


    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      arma::uvec IC=fri_j;

      if(!IC.is_empty()){

        int D =IC.n_elem;

        if((D*log(H))<log(nsample)){  // only consider smaller D, remove this line to use random samples

          arma::mat Hpool(nsample,H);

          if((D*log(H))<log(nsample)){
            Hpool= ExpandH(H,D);
          }else{
            Hpool= ExpandH_sample(H,D,nsample);
            Hpool=join_cols(Hpool,member_H_init(IC-1).t());
          }

          int I= Hpool.n_rows;
          arma::vec loss(I);

          for (int i = 0; i <I ; i++) {

            for (int k=0; k<taus.length();k++){
              NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
              NumericVector  beta = betas(_,k);
              NumericVector  alpha = thetaG(_,0);
              arma::mat Wj=alphabetaWs(k).row(j);

              for(int d =0; d<D; d++){
                int l=Hpool(i,d)-1;

                Wj(0,IC(d)-1) =  W(j,IC(d)-1)*alpha[g]*beta[l];
              }

              arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
              loss(i) = loss(i)+ lossTriangularMat(tmp_vec,taus[k],h0,h1,h2);
            }


          }

          int h_min=loss.index_min();


          Result(j,g+2)=loss(h_min);
        }


      }
    }
  }

  return Result;
}


// --- G-refinement, additive model, standard loss ---
// [[Rcpp::export]]
NumericMatrix Refine_G_mq_additive(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, arma::vec& member_H_init, List FriendW, NumericVector taus,int nsample)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);

  int G=thetaG.nrow();
  int H=betas.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;


  arma::field<arma::mat> alphabetaWs(taus.length());

  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericVector  beta = betas(_,k);
    NumericVector  alpha = thetaG(_,0);
    NumericVector  nu = thetaG(_,1);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    arma::mat GG=as<arma::mat>(gammaG);

    alphabetaWs(k)= arma::zeros<arma::mat>(N,N);

    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;
      X1=X.row(j);

      for (int i = 0; i < I; i++) {
        alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*(alpha[member_G[fri_j(i)-1]-1]+beta[member_H_init[j]-1]);
      }

      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
    }
  }


  //Find the original loss given the estimated memberships
  NumericMatrix Result(N,2+G);

  for (int j = 0; j < N; j++) {
    Result(j,0) = member_G[j];
    arma::uvec fri_j=FriendW[j];

    for (int k=0; k<taus.length();k++){
      NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
      NumericVector  beta = betas(_,k);
      NumericVector  alpha = thetaG(_,0);
      arma::mat Wj=alphabetaWs(k).row(j);

      arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
      Result(j,1) = Result(j,1)+ Loss_fun_mat(tmp_vec,taus[k]);
    }
  }



  for(int g=0;g<G;g++){

    //Update the residuals
    for(int k=0;k<taus.length();k++){
      NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
      NumericVector  beta = betas(_,k);
      NumericVector  alpha = thetaG(_,0);
      NumericVector  nu = thetaG(_,1);
      NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
      arma::mat GG=as<arma::mat>(gammaG);

      eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
      for (int j = 0; j < N; j++) {
        X1=X.row(j);
        //Find residual part corresponding to memberG part
        eps_mats(k).row(j) = Y.row(j)-nu[g]*Z2.row(j)-GG.row(g)*X1;
      }
    }


    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      arma::uvec IC=fri_j;

      if(!IC.is_empty()){

        int D =IC.n_elem;

        if((D*log(H))<log(nsample)){  // only consider smaller D, remove this line to use random samples

          arma::mat Hpool(nsample,H);

          if((D*log(H))<log(nsample)){
            Hpool= ExpandH(H,D);
          }else{
            Hpool= ExpandH_sample(H,D,nsample);
            Hpool=join_cols(Hpool,member_H_init(IC-1).t());
          }

          int I= Hpool.n_rows;
          arma::vec loss(I);

          for (int i = 0; i <I ; i++) {

            for (int k=0; k<taus.length();k++){
              NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
              NumericVector  beta = betas(_,k);
              NumericVector  alpha = thetaG(_,0);
              arma::mat Wj=alphabetaWs(k).row(j);

              for(int d =0; d<D; d++){
                int l=Hpool(i,d)-1;

                Wj(0,IC(d)-1) =  W(j,IC(d)-1)*(alpha[g]+beta[l]);
              }

              arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
              loss(i) = loss(i)+ Loss_fun_mat(tmp_vec,taus[k]);
            }


          }

          int h_min=loss.index_min();


          Result(j,g+2)=loss(h_min);

        }

      }
    }
  }

  return Result;
}



// --- G-refinement, additive model, smoothed loss ---
// [[Rcpp::export]]
NumericMatrix Refine_G_mq_additive_conquer(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, arma::vec& member_H_init, List FriendW, NumericVector taus,int nsample, const double h0, const double h1, const double h2)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  NumericMatrix  betas = as<NumericMatrix >(theta_GH["beta"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);

  int G=thetaG.nrow();
  int H=betas.nrow();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;


  arma::field<arma::mat> alphabetaWs(taus.length());

  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericVector  beta = betas(_,k);
    NumericVector  alpha = thetaG(_,0);
    NumericVector  nu = thetaG(_,1);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
    arma::mat GG=as<arma::mat>(gammaG);

    alphabetaWs(k)= arma::zeros<arma::mat>(N,N);

    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;
      X1=X.row(j);

      for (int i = 0; i < I; i++) {
        alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*(alpha[member_G[fri_j(i)-1]-1]+beta[member_H_init[j]-1]);
      }

      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
    }
  }


  //Find the original loss given the estimated memberships
  NumericMatrix Result(N,2+G);

  for (int j = 0; j < N; j++) {
    Result(j,0) = member_G[j];
    arma::uvec fri_j=FriendW[j];

    for (int k=0; k<taus.length();k++){
      NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
      NumericVector  beta = betas(_,k);
      NumericVector  alpha = thetaG(_,0);
      arma::mat Wj=alphabetaWs(k).row(j);

      arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
      Result(j,1) = Result(j,1)+lossTriangularMat(tmp_vec,taus[k],h0,h1,h2);
    }
  }



  for(int g=0;g<G;g++){

    //Update the residuals
    for(int k=0;k<taus.length();k++){
      NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
      NumericVector  beta = betas(_,k);
      NumericVector  alpha = thetaG(_,0);
      NumericVector  nu = thetaG(_,1);
      NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(2,p+1));
      arma::mat GG=as<arma::mat>(gammaG);

      eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
      for (int j = 0; j < N; j++) {
        X1=X.row(j);
        //Find residual part corresponding to memberG part
        eps_mats(k).row(j) = Y.row(j)-nu[g]*Z2.row(j)-GG.row(g)*X1;
      }
    }


    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      arma::uvec IC=fri_j;

      if(!IC.is_empty()){

        int D =IC.n_elem;

        if((D*log(H))<log(nsample)){  // only consider smaller D, remove this line to use random samples

          arma::mat Hpool(nsample,H);

          if((D*log(H))<log(nsample)){
            Hpool= ExpandH(H,D);
          }else{
            Hpool= ExpandH_sample(H,D,nsample);
            Hpool=join_cols(Hpool,member_H_init(IC-1).t());
          }

          int I= Hpool.n_rows;
          arma::vec loss(I);

          for (int i = 0; i <I ; i++) {

            for (int k=0; k<taus.length();k++){
              NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
              NumericVector  beta = betas(_,k);
              NumericVector  alpha = thetaG(_,0);
              arma::mat Wj=alphabetaWs(k).row(j);

              for(int d =0; d<D; d++){
                int l=Hpool(i,d)-1;

                Wj(0,IC(d)-1) =  W(j,IC(d)-1)*(alpha[g]+beta[l]);
              }

              arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
              loss(i) = loss(i)+ lossTriangularMat(tmp_vec,taus[k],h0,h1,h2);
            }


          }

          int h_min=loss.index_min();


          Result(j,g+2)=loss(h_min);

        }


      }
    }
  }

  return Result;
}



// --- G-refinement, general model, standard loss ---
// [[Rcpp::export]]
NumericMatrix Refine_G_mq_general(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, arma::vec& member_H_init, List FriendW, NumericVector taus,int nsample)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  List alphabetaGHs = as<List>(theta_GH["alphabeta_GHs"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);
  NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[0]);

  int G=thetaG.nrow();
  int H=alphabetas.ncol();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);
  arma::mat Z3=W*Z2;


  arma::field<arma::mat> alphabetaWs(taus.length());

  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);

    NumericVector  nu = thetaG(_,0);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(1,p));
    arma::mat GG=as<arma::mat>(gammaG);

    alphabetaWs(k)= arma::zeros<arma::mat>(N,N);

    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      X1=X.row(j);
      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;

      for (int i = 0; i < I; i++) {
        alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alphabetas(member_G[fri_j(i)-1]-1,member_H_init[j]-1);
      }

      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
    }
  }


  //Find the original loss given the estimated memberships
  NumericMatrix Result(N,2+G);

  for (int j = 0; j < N; j++) {
    Result(j,0) = member_G[j];
    arma::uvec fri_j=FriendW[j];

    for (int k=0; k<taus.length();k++){
      arma::mat Wj=alphabetaWs(k).row(j);
      arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
      Result(j,1) = Result(j,1)+ Loss_fun_mat(tmp_vec,taus[k]);
    }
  }



  for(int g=0;g<G;g++){

    //Update the residuals
    for(int k=0;k<taus.length();k++){
      NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
      NumericVector  nu = thetaG(_,0);
      NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(1,p));
      arma::mat GG=as<arma::mat>(gammaG);

      eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
      for (int j = 0; j < N; j++) {
        X1=X.row(j);
        //Find residual part corresponding to memberG part
        eps_mats(k).row(j) = Y.row(j)-nu[g]*Z2.row(j)-GG.row(g)*X1;
      }
    }


    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      arma::uvec IC=fri_j;

      if(!IC.is_empty()){

        int D =IC.n_elem;

        if((D*log(H))<log(nsample)){  // only consider smaller D, remove this line to use random samples

          arma::mat Hpool(nsample,H);

          if((D*log(H))<log(nsample)){
            Hpool= ExpandH(H,D);
          }else{
            Hpool= ExpandH_sample(H,D,nsample);
            Hpool=join_cols(Hpool,member_H_init(IC-1).t());
          }

          int I= Hpool.n_rows;
          arma::vec loss(I);

          for (int i = 0; i <I ; i++) {

            for (int k=0; k<taus.length();k++){
              NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);
              arma::mat Wj=alphabetaWs(k).row(j);

              for(int d =0; d<D; d++){
                int l=Hpool(i,d)-1;
                Wj(0,IC(d)-1) =  W(j,IC(d)-1)*alphabetas(g,l);
              }

              arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
              loss(i) = loss(i)+ Loss_fun_mat(tmp_vec,taus[k]);
            }


          }

          int h_min=loss.index_min();


          Result(j,g+2)=loss(h_min);

        }


      }
    }
  }

  return Result;
}


// --- G-refinement, general model, smoothed loss ---
// [[Rcpp::export]]
NumericMatrix Refine_G_mq_general_conquer(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, arma::vec& member_H_init, List FriendW, NumericVector taus,int nsample, const double h0, const double h1, const double h2)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  List alphabetaGHs = as<List>(theta_GH["alphabeta_GHs"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);
  NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[0]);

  int G=thetaG.nrow();
  int H=alphabetas.ncol();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);


  arma::field<arma::mat> alphabetaWs(taus.length());

  arma::field<arma::mat> eps_mats(taus.length());

  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);
    NumericVector  nu = thetaG(_,0);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(1,p));
    arma::mat GG=as<arma::mat>(gammaG);

    alphabetaWs(k)= arma::zeros<arma::mat>(N,N);

    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;
      X1=X.row(j);

      for (int i = 0; i < I; i++) {
        alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alphabetas(member_G[fri_j(i)-1]-1,member_H_init[j]-1);
      }

      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
    }
  }


  //Find the original loss given the estimated memberships
  NumericMatrix Result(N,2+G);

  for (int j = 0; j < N; j++) {
    Result(j,0) = member_G[j];
    arma::uvec fri_j=FriendW[j];

    for (int k=0; k<taus.length();k++){
      NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);
      arma::mat Wj=alphabetaWs(k).row(j);
      arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
      Result(j,1) = Result(j,1)+lossTriangularMat(tmp_vec,taus[k],h0,h1,h2);
    }
  }



  for(int g=0;g<G;g++){

    //Update the residuals
    for(int k=0;k<taus.length();k++){
      NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
      NumericVector  nu = thetaG(_,0);
      NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(1,p));
      arma::mat GG=as<arma::mat>(gammaG);

      eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
      for (int j = 0; j < N; j++) {
        X1=X.row(j);

        //Find residual part corresponding to memberG part
        eps_mats(k).row(j) = Y.row(j)-nu[g]*Z2.row(j)-GG.row(g)*X1;
      }
    }


    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      arma::uvec IC=fri_j;

      if(!IC.is_empty()){

        int D =IC.n_elem;

        if((D*log(H))<log(nsample)){  // only consider smaller D, remove this line to use random samples

          arma::mat Hpool(nsample,H);

          if((D*log(H))<log(nsample)){
            Hpool= ExpandH(H,D);
          }else{
            Hpool= ExpandH_sample(H,D,nsample);
            Hpool=join_cols(Hpool,member_H_init(IC-1).t());
          }

          int I= Hpool.n_rows;
          arma::vec loss(I);

          for (int i = 0; i <I ; i++) {

            for (int k=0; k<taus.length();k++){
              NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);

              arma::mat Wj=alphabetaWs(k).row(j);

              for(int d =0; d<D; d++){
                int l=Hpool(i,d)-1;

                Wj(0,IC(d)-1) =  W(j,IC(d)-1)*alphabetas(g,l);
              }

              arma::vec tmp_vec=(eps_mats(k).row(j)-Wj.cols(fri_j-1)*Z2.rows(fri_j-1)).t();
              loss(i) = loss(i)+ lossTriangularMat(tmp_vec,taus[k],h0,h1,h2);
            }



          }

          int h_min=loss.index_min();


          Result(j,g+2)=loss(h_min);

        }


      }
    }
  }

  return Result;
}


// --- H-refinement, general model, smoothed loss ---
// [[Rcpp::export]]
NumericMatrix Refine_H_mq_general_conquer(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, arma::vec& member_H_init, List FriendW,List FriendW2, NumericVector taus,int nsample,const double h0, const double h1,const double h2)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  List alphabetaGHs = as<List>(theta_GH["alphabeta_GHs"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);
  NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[0]);

  int G=thetaG.nrow();
  int H=alphabetas.ncol();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);

  arma::field<arma::mat> alphabetaWs(taus.length());
  arma::field<arma::mat> eps_mats(taus.length());


  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);

    NumericVector  nu = thetaG(_,0);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(1,p));
    arma::mat GG=as<arma::mat>(gammaG);

    alphabetaWs(k)= arma::zeros<arma::mat>(N,N);
    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;
      X1=X.row(j);

      for (int i = 0; i < I; i++) {
        alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alphabetas(member_G[fri_j(i)-1]-1,member_H_init[j]-1);
      }
      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
    }
  }


//Compute the loss with the original H-memberships
NumericMatrix Result(N,2+H);

for (int j = 0; j < N; j++) {

  Result(j,0) = member_H_init[j];

  arma::uvec fri_j=FriendW[j];
  arma::uvec fri_j2=FriendW2[j];

    for (int k=0; k<taus.length();k++){
      NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);
      arma::mat res=eps_mats(k).rows(fri_j-1)-alphabetaWs(k).submat(fri_j-1,fri_j2-1)*Z2.rows(fri_j2-1);

      Result(j,1) = Result(j,1)+ lossTriangularMat(arma::vectorise(res,0),taus[k],h0,h1,h2);
    }

}




  for (int j = 0; j < N; j++) {

    arma::uvec fri_j=FriendW[j];
    arma::uvec fri_j2=FriendW2[j];
    arma::uvec IC=fri_j2;

    if(!IC.is_empty()){

      int D =IC.n_elem;
      if((D*log(H))<log(nsample)){  // only consider smaller D, remove this line to use random samples

        arma::mat Hpool(nsample,D);

        if((D*log(H))<log(nsample)){
          Hpool= ExpandH(H,D);
        }else{
          Hpool= ExpandH_sample(H,D,nsample);
          Hpool=join_cols(Hpool,member_H_init(IC.col(0)-1).t());
        }

        int I= Hpool.n_rows;
        int L=fri_j.n_elem;

        // For each candidate h, optimize the rest of the affected H-vector
        for(int h=0;h<H;h++){

        arma::vec loss(I);


          for (int k=0; k<taus.length();k++){
            NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);

              arma::mat Wj(N,N);



          for (int i = 0; i <I ; i++) {

            for(int l =0; l<L; l++){
              for(int d =0; d<D; d++){
                int q=Hpool(i,d)-1;
              Wj(fri_j(l)-1,IC(d,0)-1) =  W(fri_j(l)-1,IC(d,0)-1)*alphabetas(member_G[fri_j(l)-1]-1,q);
            }
            }


            for(int l =0; l<L; l++){
              Wj(fri_j(l)-1,j) =  W(fri_j(l)-1,j)*alphabetas(member_G[fri_j(l)-1]-1,h);
            }

            arma::mat res=eps_mats(k).rows(fri_j-1)-Wj.submat(fri_j-1,fri_j2-1)*Z2.rows(fri_j2-1);
            loss(i) = loss(i)+ lossTriangularMat(arma::vectorise(res,0),taus[k],h0,h1,h2);
          }


        }

        int h_min=loss.index_min();


        Result(j,h+2)=loss(h_min);

      }
      }
    }
    Rcout << "The value of v : " << j << "\n";
}

  return Result;
}


// --- H-refinement, general model, standard loss ---
// [[Rcpp::export]]
NumericMatrix Refine_H_mq_general(const arma::mat& Ymat, const arma::mat& W, arma::cube X, List theta_GH, IntegerVector member_G, arma::vec& member_H_init, List FriendW,List FriendW2, NumericVector taus,int nsample)
{

  // import estimated values
  List thetaGs = as<List>(theta_GH["theta_Gs"]);
  List alphabetaGHs = as<List>(theta_GH["alphabeta_GHs"]);
  NumericMatrix thetaG = as<NumericMatrix>(thetaGs[0]);
  NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[0]);

  int G=thetaG.nrow();
  int H=alphabetas.ncol();
  arma::mat X1=X.row(0);
  int p=X1.n_rows;
  const int N=Ymat.n_rows;
  const int Time=Ymat.n_cols;

  if(member_G.length()!=N){
    Rcout << "The dimension of member_G and N does not match!";
  }


  arma::mat Z2 = Ymat.cols(0,Time-2);
  arma::mat Y = Ymat.cols(1,Time-1);

  arma::field<arma::mat> alphabetaWs(taus.length());
  arma::field<arma::mat> eps_mats(taus.length());


  for(int k=0;k<taus.length();k++){
    NumericMatrix thetaG = as<NumericMatrix>(thetaGs[k]);
    NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);

    NumericVector  nu = thetaG(_,0);
    NumericMatrix gammaG = thetaG( Range(0,G-1) , Range(1,p));
    arma::mat GG=as<arma::mat>(gammaG);

    alphabetaWs(k)= arma::zeros<arma::mat>(N,N);
    eps_mats(k)= arma::zeros<arma::mat>(N,Time-1);
    for (int j = 0; j < N; j++) {
      X1=X.row(j);

      arma::uvec fri_j=FriendW[j];
      int I=fri_j.n_elem;

      for (int i = 0; i < I; i++) {
        alphabetaWs(k)(fri_j(i)-1,j) = W(fri_j(i)-1,j)*alphabetas(member_G[fri_j(i)-1]-1,member_H_init[j]-1);
      }
      //Find residual part corresponding to memberG part
      eps_mats(k).row(j) = Y.row(j)-nu[member_G[j]-1]*Z2.row(j)-GG.row(member_G[j]-1)*X1;
    }
  }


  //Compute the loss with the original H-memberships
  NumericMatrix Result(N,2+H);

  for (int j = 0; j < N; j++) {

    Result(j,0) = member_H_init[j];

    arma::uvec fri_j=FriendW[j];
    arma::uvec fri_j2=FriendW2[j];

    for (int k=0; k<taus.length();k++){
      NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);
      arma::mat res=eps_mats(k).rows(fri_j-1)-alphabetaWs(k).submat(fri_j-1,fri_j2-1)*Z2.rows(fri_j2-1);
      Result(j,1) = Result(j,1)+ Loss_fun_mat(arma::vectorise(res,0),taus[k]);
    }

  }




  for (int j = 0; j < N; j++) {

    arma::uvec fri_j=FriendW[j];
    arma::uvec fri_j2=FriendW2[j];
    arma::uvec IC=fri_j2;

    if(!IC.is_empty()){

      int D =IC.n_elem;

      if((D*log(H))<log(nsample)){  // only consider smaller D, remove this line to use random samples

      arma::mat Hpool(nsample,D);

      if((D*log(H))<log(nsample)){
        Hpool= ExpandH(H,D);
      }else{
        Hpool= ExpandH_sample(H,D,nsample);
        Hpool=join_cols(Hpool,member_H_init(IC.col(0)-1).t());
      }

      int I= Hpool.n_rows;
      int L=fri_j.n_elem;


      for(int h=0;h<H;h++){

        arma::vec loss(I);


        for (int k=0; k<taus.length();k++){
          NumericMatrix alphabetas = as<NumericMatrix>(alphabetaGHs[k]);

          arma::mat Wj(N,N);



          for (int i = 0; i <I ; i++) {

            for(int l =0; l<L; l++){
              for(int d =0; d<D; d++){
                int q=Hpool(i,d)-1;
                Wj(fri_j(l)-1,IC(d,0)-1) =  W(fri_j(l)-1,IC(d,0)-1)*alphabetas(member_G[fri_j(l)-1]-1,q);
              }
            }


            for(int l =0; l<L; l++){
              Wj(fri_j(l)-1,j) =  W(fri_j(l)-1,j)*alphabetas(member_G[fri_j(l)-1]-1,h);
            }

            arma::mat res=eps_mats(k).rows(fri_j-1)-Wj.submat(fri_j-1,fri_j2-1)*Z2.rows(fri_j2-1);
            loss(i) = loss(i)+ Loss_fun_mat(arma::vectorise(res,0),taus[k]);
          }


        }

        int h_min=loss.index_min();


        Result(j,h+2)=loss(h_min);

      }
      }
    }
    Rcout << "The value of v : " << j << "\n";
  }

  return Result;
}
