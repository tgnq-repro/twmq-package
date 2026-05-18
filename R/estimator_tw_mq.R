# =============================================================================
# estimator_tw_mq.R
# -----------------------------------------------------------------------------
# Top-level R interface for the Two-way Grouped Network Quantile (TGNQ)
# autoregression model. This file orchestrates:
#
#   1. Initial G-/H-membership generation (via individual-level LASSO + K-means)
#   2. Parameter estimation given memberships (general / additive / multiplicative)
#   3. The Vanilla Algorithm (Algorithm 1)            -> twmq.estimate
#   4. The Enhanced Algorithm (Algorithm 2)           -> update_NARG_twmq*
#   5. Multi-start automatic estimation               -> twmq.estimate.auto*
#   6. Confidence intervals                           -> twmq_ci
#   7. Diagnostic refinement (per-node loss probing)  -> Refine_G/Refine_H
#
# Most heavy inner loops are delegated to C++ in twq.cpp.
# =============================================================================

#' @useDynLib twmq, .registration = TRUE
#' @importFrom Rcpp evalCpp
NULL

library(quantreg)
library(conquer)
library(Rcpp)
library(CEoptim)


# -----------------------------------------------------------------------------
# Utility: root-mean-squared error scalars
# -----------------------------------------------------------------------------
RMSE<-function(a,b)
{
  sqrt(mean((a-b)^2))
}

# Average relative RMSE across a list of matrices (used to assess parameter
# convergence between iterates).
RMSE_L<-function(a,b, ratio = T)
{
  l = length(a)
  rr = rep(0, l)
  for (k in 1:l){
    rr[k] = RMSE((a[[k]]-b[[k]])/(10^(-3)+abs(b[[k]])), 0)
  }
  return(mean(rr))
}

# Standard quantile check loss
check.func<-function(u, tau)
{
  u*(tau-(u<0)*1)
}

# Smoothed quantile loss with Gaussian kernel (kept for compatibility)
loss.ConquerTri <- function(u,h,tau){
  x <- u/h
  (sqrt(2/pi)*exp(-x^2/2)+x*(1-2*pnorm(-x)))*h/2+(tau-1/2)*u
}

# Smoothed quantile loss with triangular kernel (this overload wins; matches
# the C++ lossTriangularMat implementation)
loss.ConquerTri <- function(u,h,tau){
  x <- u/h
  ((3*x^2/4-x^4/8+3/8)*(abs(x)<=1)+abs(x)*(abs(x)>1))*h/2+(tau-1/2)*u
}

# Logistic-kernel smoothed loss
loss.ConquerLogit <- function(u,h,tau){
  x <- u/h
  tau*u+h*log(1+exp(-x))
}


# given membership update the parameter for one round
# =============================================================================
# Single-pass parameter update for the MULTIPLICATIVE model
# theta_{gh} = alpha_g * beta_h
# Iterates between (alpha, theta_G) and beta, holding memberships fixed.
# =============================================================================
twmq.estimate_thetaGH.member <-function(Ymat, X_tensor, W, member_G, member_H,
                                       beta, taus, conquer, h_conquer)
{
  G = max(member_G); H = max(member_H)
  Time = ncol(Ymat); N = nrow(Ymat)
  p = dim(X_tensor)[2]
  # if G = H = 1
  # --- Trivial case: only one row group AND one column group ---
  if (G == 1 & H == 1){
    Ymat1 = (W%*%Ymat)
    Z_all = cbind(as.vector(Ymat1[,-Time]),
                  as.vector(Ymat[,-Time]),
                  matrix(aperm(X_tensor,c(1,3,2)),nrow = N*(Time-1)))
    Y = as.vector(Ymat[,-1])

    if (conquer)
    {
      fit.conquer = conquer.process(Z_all, Y, tauSeq = taus)
      theta = fit.conquer$coeff
      theta[1:3,] = rbind(fit.conquer$coeff[2:3,], fit.conquer$coeff[1,])
      Loss.fun <- function(u,tau){
        if (length(dim(u))==2){
          u = as.matrix(u)
        }
        loss.ConquerTri(u,h=h_conquer,tau)
      }
    }else{
      Loss.fun <- check.func
      dat = data.frame(Y = Y, Z_all)
      resrq = quantreg::rq(Y~.-1, taus, data = dat)
      theta = resrq$coefficients
    }


    return(list(theta, Loss.fun=Loss.fun))
  }


  # --- Step (a): if beta is missing, get a rough beta using the largest G-group ---
  if (is.null(beta))
  {
    tmp <- table(member_G)
    g_max <- names(tmp)[which.max(tmp)]
    ind_g = which(member_G == g_max)

    N_g1 = length(ind_g)
    Ymat_g1 = Ymat[ind_g,,drop=F]
    W_g1 = W[ind_g,,drop=F]

    # Each Zh_g1 column corresponds to one column-group's contribution
    Zh_g1 = matrix(0, N_g1*(Time - 1), ncol = H)
    for (h in 1:H)
    {
      ind_h = which(member_H==h)
      Zh_g1[,h] = as.vector(W_g1[,ind_h,drop=F]%*%Ymat[ind_h,-Time,drop=F])
    }
    Z2_g1 = as.vector(Ymat[ind_g,-Time,drop=F])
    Z_all_g1 = cbind(Zh_g1, Z2_g1,
                     matrix(aperm(X_tensor[ind_g,,],c(1,3,2)),nrow = length(ind_g)*(Time-1)))
    Y_g1 = as.vector(Ymat[ind_g,-1,drop=F])


    if (conquer){
      fit.conquer = conquer.process(Z_all_g1, Y_g1, tauSeq = taus)
      theta_init = fit.conquer$coeff
    }else{
      dat = data.frame(Y = Y_g1, Z_all_g1)
      resrq = quantreg::rq(Y~.-1, taus , data = dat)
      theta_init = resrq$coefficients
    }


    if(length(taus)==1) beta=as.matrix(theta_init) else
      beta = theta_init[,,drop=F][1:H,]
  }

  # --- Step (b): given beta, estimate alpha + (nu, gamma) per row-group ---
  alpha = matrix(0, G, length(taus))
  theta_Gs = list()
  eps_mats = list()

  for (k in 1:length(taus)){
    tau = taus[k]
    beta_all = beta[member_H,k]
    Wbeta = W%*%diag(beta_all)

    Z1 = Wbeta%*%Ymat[,-Time]
    Z2 = Ymat[,-Time]
    thetaG = matrix(0, nrow = G, ncol = p+2)
    eps_mat = matrix(0, N, Time - 1)

    for (g in 1:G)
    {
      ind_g = which(member_G == g)
      Z_all = cbind(as.vector(Z1[ind_g,,drop=F]),
                    as.vector(Z2[ind_g,,drop=F]),
                    matrix(aperm(X_tensor[ind_g,,],c(1,3,2)),nrow = length(ind_g)*(Time-1)))
      Y_vec = as.vector(Ymat[ind_g, -1,drop=F])

      if (conquer){
        fit.conquer = conquer(Z_all, Y_vec, tau)
        thetaG[g,1:2] = fit.conquer$coeff[2:3]
        thetaG[g,-(1:2)] = c(fit.conquer$coeff[1],
                             fit.conquer$coeff[-(1:3)])
        Loss.fun <- function(u,tau){
          loss.ConquerTri(u,h=h_conquer,tau)
        }

      }else{
        dat = data.frame(Y = Y_vec, Z_all)
        resrq = quantreg::rq(Y~.-1, tau , data = dat)

        thetaG[g,] = resrq$coefficients
        Loss.fun <- check.func
      }

      # Residual after subtracting non-network components
      eps_mat[ind_g,] = matrix(Y_vec - Z_all[,-1]%*%thetaG[g,-1],
                               nrow = length(ind_g))
    }
    eps_mats[[k]] = eps_mat
    theta_Gs[[k]] = thetaG
    alpha[,k] = thetaG[,1]
  }


  # --- Step (c): given alpha, re-estimate beta using residuals ---
  beta_new = matrix(0, H, length(taus))
  loss = rep(0, length(taus))

  for (k in 1:length(taus)){
    tau = taus[k]
    eps_mat = eps_mats[[k]]

    eps_vec = as.vector(eps_mat)
    alpha_all = alpha[member_G, k]
    alphaW = diag(alpha_all)%*%W

    Zh = matrix(0, N*(Time - 1), ncol = H)
    for (h in 1:H)
    {
      ind_h = which(member_H==h)
      Zh[,h] = as.vector(alphaW[,ind_h,drop=F]%*%Ymat[ind_h,-Time,drop=F])
    }
    if (conquer){
      fit.conquer = conquer(Zh, eps_vec, tau = tau, h=h_conquer)
      beta_new[,k] = fit.conquer$coeff[-1]
      resi = fit.conquer$residual
    }else{
      dat = data.frame(Y = eps_vec, Zh)
      resrq = quantreg::rq(Y~.-1, tau , data = dat)
      beta_new[,k] = resrq$coefficients
      resi = resrq$residuals
    }

    loss[k] = sum(Loss.fun(resi, tau))
  }





  # Identifiability: rescale so that alpha_1(tau) = 1 for each tau
  for (k in 1:length(taus)){
    alpha1 = alpha[1,k]
    alpha[,k] = alpha[,k]/alpha1
    theta_Gs[[k]][,1] = theta_Gs[[k]][,1]/alpha1
    beta_new[,k] = beta_new[,k]*alpha1
  }



  return(list(theta_Gs = theta_Gs, alpha = alpha,
              beta = beta_new, loss = loss,
              Loss.fun=Loss.fun))
}



# =============================================================================
# Iterated parameter estimation given memberships
# Dispatches to the appropriate single-pass routine and iterates if needed
# (only the multiplicative model requires extra outer iterations).
# =============================================================================
#' @export
twmq.estimate_thetaGH.member.iterate <- function(Ymat, X_tensor, W,
                                               member_G, member_H,
                                               taus, conquer,h_conquer,
                                               theta=NULL,method, n_iter.max = 2, verbose = F, rq_lambda=0.01)
{

  if(method=="general"){theta = twmq.estimate_thetaGH.member.general(Ymat, X_tensor, W, member_G, member_H, taus,
                                             conquer,h_conquer,rq_lambda=rq_lambda)}

  if(method=="additive"){theta = twmq.estimate_thetaGH.member.additive(Ymat, X_tensor, W, member_G, member_H, taus,
                                                  conquer,h_conquer)}

  if(method=="multiplicative"){
    # Multiplicative requires alternating between alpha and beta
    theta = twmq.estimate_thetaGH.member(Ymat, X_tensor, W,
                                         member_G, member_H,
                                         beta = theta$beta, taus,conquer,h_conquer)

    del = 1; n_iter = 1; converge = T
    while(del>10^{-3} & n_iter<n_iter.max)
    {
      if (converge){
        theta_new = twmq.estimate_thetaGH.member(Ymat, X_tensor, W,
                                                 member_G, member_H,
                                                 beta = theta$beta, taus,
                                                 conquer,h_conquer)
      }else{
        # Fall back to standard rq if the conquer step regressed loss
        theta_new = twmq.estimate_thetaGH.member(Ymat, X_tensor, W,
                                                 member_G, member_H,
                                                 beta = theta$beta, taus,
                                                 conquer = F,h_conquer)
      }


      del = max(RMSE_L(theta_new$theta_Gs, theta$theta_Gs, ratio = T),
                RMSE((theta_new$beta - theta$beta)/(10^(-3)+abs(theta$beta)),
                     0))

      if (verbose){
        cat(n_iter, del, "\n")
      }
      # Roll back any quantile where the new fit is materially worse
      idx <- which(theta$loss<(theta_new$loss*0.95))

      if(length(idx)>0){
        converge = F
        for(k in idx){
          theta_new$theta_Gs[[k]]=  theta$theta_Gs[[k]]
          theta_new$alpha[,k] =theta$alpha[,k]
          theta_new$beta[,k] =theta$beta[,k]
          theta_new$loss[k] <- theta$loss[k]
        }

      }

      theta = theta_new
      n_iter = n_iter + 1
      if (verbose)
        cat(n_iter, del, "\n")
    }
    theta$converge = converge
  }

  return(theta)
}


# =============================================================================
# Update G-membership given theta and H (Algorithm 1, Step II)
# =============================================================================
twmq.estimate_memberG <-function(Ymat, X_tensor, W, member_H,
                                 theta_GH, taus, verbose = F,Loss.fun,conquer,h_conquer,method)
{
  p <- dim(X_tensor)[2]
  if (method=="general") {
    G = nrow(theta_GH$alphabeta_GHs[[1]])

    if (conquer)
      resi2 <- Loss_memberG_mq_general_conquer(Ymat, as.matrix(W),
                                               X_tensor, theta_GH, member_H, taus, h_conquer, 1/h_conquer,
                                               1/h_conquer^2)
    else resi2 <- Loss_memberG_mq_general(Ymat, as.matrix(W),
                                          X_tensor, theta_GH, member_H, taus)
  }

  if (method=="additive") {
    G = nrow(theta_GH$theta_Gs[[1]])
    if (conquer)
      resi2 <- Loss_memberG_mq_additive_conquer(Ymat, as.matrix(W),
                                                X_tensor, theta_GH, member_H, taus, h_conquer, 1/h_conquer,
                                                1/h_conquer^2)
    else resi2 <- Loss_memberG_mq_additive(Ymat, as.matrix(W),
                                           X_tensor, theta_GH, member_H, taus)
  }

  if(method=="multiplicative"){
    G = nrow(theta_GH$theta_Gs[[1]])
    if (conquer)
      resi2 <- Loss_memberG_mq_conquer(Ymat, as.matrix(W),
                                       X_tensor, theta_GH, member_H, taus, h_conquer, 1/h_conquer,
                                       1/h_conquer^2)
    else resi2 <- Loss_memberG_mq(Ymat, as.matrix(W), X_tensor,
                                  theta_GH, member_H, taus)
  }
  member_G = apply(resi2, 1, which.min)

  # Avoid degenerate (too-small) groups: redistribute from largest group
  ID <- unique(names(table(member_G)))
  ID <- ID[which(table(member_G) >= p)]
  while (length(ID) < G) {
    g_max <- (names(table(member_G)))[which.max(table(member_G))]
    g_new <- which(!((1:G) %in% ID))[1]
    idx <- which(member_G == g_max)
    member_G[sample(idx, size = length(idx)/2)] <- g_new
    ID <- unique(member_G)
  }
  return(member_G)
}


# =============================================================================
# Update H-membership given theta and G (Algorithm 1, Step III)
# =============================================================================
twmq.estimate_memberH <-function(Ymat, X_tensor, W, member_G, member_H_init,
                                 theta_GH, taus, verbose = F,
                                 Loss.fun,FriendW,FriendW2,conquer,h_conquer,method)
{
  if (method=="general") {
    H = ncol(theta_GH$alphabeta_GHs[[1]])

    if (conquer)
      member_H = Update_memberH_mq_general_conquer(Ymat,
                                                   as.matrix(W), X_tensor, theta_GH, member_G, member_H_init,
                                                   FriendW, FriendW2, taus, h_conquer, 1/h_conquer,
                                                   1/h_conquer^2)
    else member_H = Update_memberH_mq_general(Ymat, as.matrix(W),
                                              X_tensor, theta_GH, member_G, member_H_init, FriendW, FriendW2,
                                              taus)
  }

  if (method=="additive") {
    H = nrow(theta_GH$beta)
    if (conquer)
      member_H = Update_memberH_mq_additive_conquer(Ymat,
                                                    as.matrix(W), X_tensor, theta_GH, member_G, member_H_init,
                                                    FriendW, FriendW2, taus, h_conquer, 1/h_conquer,
                                                    1/h_conquer^2)
    else member_H = Update_memberH_mq_additive(Ymat, as.matrix(W),
                                               X_tensor, theta_GH, member_G, member_H_init, FriendW, FriendW2,
                                               taus)
  }

  if(method=="multiplicative"){
    H = nrow(theta_GH$beta)
    if (conquer)
      member_H = Update_memberH_mq_conquer(Ymat, as.matrix(W),
                                           X_tensor, theta_GH, member_G, member_H_init, FriendW,
                                           FriendW2, taus, h_conquer, 1/h_conquer, 1/h_conquer^2)
    else member_H = Update_memberH_mq(Ymat, as.matrix(W),
                                      X_tensor, theta_GH, member_G, member_H_init, FriendW, FriendW2,
                                      taus)
  }

  # Guard against tiny H-groups
  ID <- unique(names(table(member_H)))
  ID <- ID[which(table(member_H) >= 2)]
  while (length(ID) < H) {
    h_max <- (names(table(member_H)))[which.max(table(member_H))]
    h_new <- which(!((1:H) %in% ID))[1]
    idx <- which(member_H == h_max)
    member_H[sample(idx, size = length(idx)/2)] <- h_new
    ID <- unique(member_H)
  }
  return(member_H)
}

# =============================================================================
# Vanilla Algorithm (Algorithm 1 in the paper):
# Coordinate descent over (theta, G, H) given a starting point.
# =============================================================================
twmq.estimate <-function(Ymat, X_tensor, W, member_G_init, member_H_init, taus, verbose = F,
                         conquer = F,h_conquer=NA,method,theta_GH=NULL,Maxit=100,rq_lambda=0.01)
{
  N <- nrow(Ymat)
  member_G = member_G_init
  member_H = member_H_init

  # Build follower lists used in the C++ H-update
  #   FriendW[[i]]  = direct followers of node i
  #   FriendW2[[i]] = followers-of-followers (used in cross terms)
  FriendW <-FriendW2 <- vector("list", N)
  for(i in 1:N){
    FriendW[[i]] <- as.vector(which(W[,i]!=0))
    for (j in  FriendW[[i]]) {
      FriendW2[[i]] <- unique(c(FriendW2[[i]],as.vector(which(W[j,]!=0))))
    }
  }

  # Step I: parameter estimation given initial memberships
  theta_GH = twmq.estimate_thetaGH.member.iterate(Ymat, X_tensor, W,
                                                  member_G_init,
                                                  member_H_init, taus,
                                                  verbose = F,
                                                  conquer,h_conquer,theta=theta_GH,method,rq_lambda=rq_lambda)
  converge <- FALSE
  del = 1
  changeGH = 1
  iter = 1
  while ((del > 10^{-3} | changeGH>0)& iter < Maxit)
  {
    # Step II: update G
    member_G_new = twmq.estimate_memberG(Ymat, X_tensor, W,
                                         member_H = member_H,
                                         theta_GH, taus, verbose = F,
                                         theta_GH$Loss.fun,conquer,h_conquer,method)

    # Step III: update H (coordinate-descent)
    member_H_new = twmq.estimate_memberH(Ymat, X_tensor, W, member_G_new,
                                         member_H_init = member_H,
                                         theta_GH, taus, verbose = F,
                                         theta_GH$Loss.fun,FriendW,FriendW2,conquer,h_conquer,method)

    # Step I (re-estimate parameters given new memberships)
    theta_GH_new = twmq.estimate_thetaGH.member.iterate(Ymat, X_tensor, W,
                                                        member_G_new,
                                                        member_H_new, taus,
                                                        verbose = F,
                                                        conquer,h_conquer,theta=theta_GH,method,rq_lambda=rq_lambda)

    if(method=="general"){

      del =  max(RMSE_L(theta_GH_new$theta_Gs, theta_GH$theta_Gs, ratio = T),
                 RMSE_L(theta_GH_new$alphabeta_GHs, theta_GH$alphabeta_GHs, ratio = T))
    }else{

      del =  max(RMSE_L(theta_GH_new$theta_Gs, theta_GH$theta_Gs, ratio = T),
                 RMSE((theta_GH_new$beta-theta_GH$beta)/(10^(-3)+abs(theta_GH$beta)), 0))
    }


    changeGH = sum(member_G_new!=member_G)+sum(member_H_new!=member_H)


    if (verbose){
      cat(iter, del, sum(member_G_new!=member_G), sum(member_H_new!=member_H), sum(theta_GH_new$loss),"\n")
    }

    member_G = member_G_new
    member_H = member_H_new
    theta_GH = theta_GH_new
    iter = iter + 1
  }

  if(iter<Maxit) converge <- TRUE

  return(list(theta_GH = theta_GH,
              member_G = member_G,
              member_H = member_H,converge=converge))
}


# =============================================================================
# An older / alternative initialization based on per-node LASSO
# (kept for reference; the main entry uses twq.init_xgg below)
# =============================================================================
twq.member_GH.init2<-function(Ymat, W, X_tensor, G, H, tau, lambda = 10^{-3},
                              conquer = F, verbose = F,h_conquer)
{
  N = nrow(Ymat); Time = ncol(Ymat)
  WYmat = W%*%Ymat[, -Time]
  para_G = matrix(0, N, 2)

  # Per-node LASSO quantile regression to get crude individual coefficients
  for (i in 1:N)
  {
    if (verbose){
      cat("G_mem", i, "\r")
    }

    X_i = cbind(Ymat[i,-Time], WYmat[i,-Time])
    Y_i = Ymat[i, -1]

    dat = data.frame(Y = Y_i, X_i)
    resrq = quantreg::rq(Y~., tau , data = dat,
                         method="lasso", lambda = lambda)
    para_vec = resrq$coefficients

    para_G[i,] = para_vec[-1]
  }

  km_G = kmeans(para_G, G)

  # Initialize H by clustering "average network coefficient towards group g"
  para_H = matrix(0, N, G)
  for (g in 1:G)
  {
    Ymat_g = Ymat[km_G$cluster==g,]
    ind_g = which(km_G$cluster==g)
    Y_g = as.vector(Ymat_g[,-1])
    for (j in 1:N)
    {
      if (verbose){
        cat("H_mem", g, j, "\r")
      }
      X_j = cbind(as.vector(W[ind_g,j,drop=F]%*%t(Ymat[j,-Time])),
                  as.vector(Ymat_g[,-Time]),
                  X[rep(ind_g, Time-1),-1])


      if (conquer){
        if (all(X_j[,1]==0)){
          fit.conquer = conquer(X_j[,-1], Y_g, tau = tau,h=h_conquer)
          para_H[j, g] = fit.conquer$coeff[1]
        }else{
          fit.conquer = conquer(X_j, Y_g, tau = tau,h=h_conquer)
          para_H[j, g] = fit.conquer$coeff[2]
        }

      }else{
        dat = data.frame(Y = Y_g, X_j)
        resrq = quantreg::rq(Y~., tau , data = dat,
                             method="lasso", lambda = lambda)

        para_H[j, g] = resrq$coefficients[2]

      }
    }
  }

  km_H = kmeans(para_H, H)
  return(list(member_G = km_G$cluster,
              member_H = km_H$cluster))

}


# =============================================================================
# Label switching: align estimated memberships with reference memberships
# (purely cosmetic; reorders rows / columns of theta accordingly)
# =============================================================================
#' @export
twmq.label.switch<-function(res, member_G0,member_H0, method)
{

  member_G = res$member_G
  member_H = res$member_H
  G <- length(unique(member_G))
  H <- length(unique(member_H))
  theta_Gs = res$theta_GH$theta_Gs
  n_taus = length(theta_Gs)

  # Permutation that maps estimated G-labels to ground-truth labels
  neworder <- numeric(G)
  for (g in 1:G) {
    idx <- which(member_G==g)
    tmp <- table(member_G0[idx])
    neworder[g] <-  names(tmp)[which.max(tmp)]
  }

  indG = match(1:G, neworder)
  theta_Gs_new = list()
  for (k in 1:n_taus){
    theta_Gs_new[[k]] = theta_Gs[[k]][indG,]
  }

  indG1 = match(neworder, 1:G)
  member_G_new = indG1[member_G]

  # Permutation for H-labels
  neworder <- numeric(H)
  for (h in 1:H) {
    idx <- which(member_H==h)
    tmp <- table(member_H0[idx])
    neworder[h] <-  names(tmp)[which.max(tmp)]
  }

  indH = match(1:H, neworder)
  indH1 = match(neworder, 1:H)
  member_H_new = indH1[member_H]

  if(method=="general"){
    alphabeta_GHs <- res$theta_GH$alphabeta_GHs
    alphabeta_GHs_new <- list()

    for (k in 1:n_taus){
      alphabeta_GHs_new[[k]] <- (alphabeta_GHs[[k]])[indG,indH]
    }
    out <- list(theta_Gs = theta_Gs_new, alphabeta_GHs = alphabeta_GHs_new,loss = res$theta_GH$loss,Loss_fun=res$theta_GH$Loss.fun)
  }else{
  alpha = res$theta_GH$alpha
  beta = res$theta_GH$beta
  alpha = alpha[indG,,drop = F]

  # Re-apply identifiability constraint after permutation
  if (!additive){
    for (k in 1:n_taus){
      alpha1 = alpha[1,k]
      theta_Gs_new[[k]][,1] = theta_Gs_new[[k]][,1]/alpha1
      beta[,k] = beta[,k]*alpha1
      alpha[,k] = alpha[,k]/alpha1
    }
  }else{
    for (k in 1:n_taus){
      alpha1 = alpha[1,k]
      theta_Gs_new[[k]][,1] = theta_Gs_new[[k]][,1]-alpha1
      beta[,k] = beta[,k]+alpha1
    }
  }


  beta_new = beta[indH,]

 out <- list(theta_Gs = theta_Gs_new, alpha = alpha,beta = beta_new,loss = res$theta_GH$loss,Loss_fun=res$theta_GH$Loss.fun)
  }

  return(list(theta_GH=out, member_G = member_G_new, member_H = member_H_new))
}





# =============================================================================
# Single-pass parameter update for the ADDITIVE model
# theta_{gh} = alpha_g + beta_h
# Solves all parameters jointly via a single block quantile regression
# (faster than the iterative scheme for the multiplicative model).
# =============================================================================
twmq.estimate_thetaGH.member.additive <- function(Ymat, X_tensor, W, member_G, member_H, taus, conquer, h_conquer)
{
  G = max(member_G); H = max(member_H)
  Time = ncol(Ymat); N = nrow(Ymat)
  p = dim(X_tensor)[2]
  # --- Trivial G = H = 1 case ---
  if (G == 1 & H == 1){
    Ymat1 = (W%*%Ymat)
      Z_all = cbind(as.vector(Ymat1[,-Time]),
                    as.vector(Ymat[,-Time]),
                    matrix(aperm(X_tensor,c(1,3,2)),nrow = N*(Time-1)))
    Y = as.vector(Ymat[,-1])

    if (conquer)
    {
      fit.conquer = conquer.process(Z_all, Y, tauSeq = taus)
      theta = fit.conquer$coeff
      theta[1:3,] = rbind(fit.conquer$coeff[2:3,], fit.conquer$coeff[1,])
      Loss.fun <- function(u,tau){
        if (length(dim(u))==2){
          u = as.matrix(u)
        }
        loss.ConquerTri(u,h=h_conquer,tau)
      }
    }else{
      Loss.fun <- check.func
      dat = data.frame(Y = Y, Z_all)
      resrq = quantreg::rq(Y~.-1, taus, data = dat)
      theta = resrq$coefficients
    }

    return(list(theta, Loss.fun=Loss.fun))
  }

  # --- General case: build block design matrix ---
  Y1 <- Ymat[,-Time]
  Y <- Ymat[,-1]
  WY1 <- W%*%Y1

  # Per-G block columns:
  Z1 <- matrix(0,N*(Time-1),G) # alpha-side network covariate
  Z2 <- matrix(0,N*(Time-1),G) # autoregressive covariate
  X1 <- matrix(0,N*(Time-1),G*p) # exogenous covariates
  Zh <- matrix(0,N*(Time-1),H) # beta-side network covariate (per H-group)

  # Pre-compute W * (Y1 indicator-of H-group) for each h
  Yh <- list()
  for (h in 1:H) {
    Yh[[h]] = W%*%(Y1*(member_H==h))
  }

  Y_all <-NULL

  count.row <- 0
  for (g in 1:G)
  {
    ind_g = which(member_G == g)
    Y_g=Y[ind_g,,drop=F]
    Y1_g=Y1[ind_g,,drop=F]
    tmp0 <- as.vector(Y1_g)
    n_g <- length(tmp0)
    Y_all <- c(Y_all,as.vector(Y[ind_g,,drop=F]))
    Z1[1:n_g+count.row,g] <- tmp0
    Z2[1:n_g+count.row,g] <- as.vector(WY1[ind_g,,drop=F])
    for (h in 1:H) {
      Zh[1:n_g+count.row,h] <- as.vector((Yh[[h]])[ind_g,,drop=F])
    }


    X1[1:n_g+count.row,(g-1)*p+(1:p)] = matrix(aperm(X_tensor[ind_g,,],c(1,3,2)),nrow = length(ind_g)*(Time-1))
    count.row <- count.row+n_g
  }

  # Drop first column of Z2 to enforce alpha_1 = 0 (identifiability)
  Z_all <- cbind(Z2[,-1,drop=T],Z1,X1[,,drop=T],Zh) ##Set alpha[1,1]=0

  if (conquer){
    fit.conquer = conquer.process(Z_all, Y_all, tauSeq = taus)
    theta = fit.conquer$coeff
    Loss.fun <- function(u,tau){
      loss.ConquerTri(u,h=h_conquer,tau)
    }
  }else{
    dat = data.frame(Y = Y_all, Z_all)
    resrq = quantreg::rq(Y~.-1, taus , data = dat)
    theta = resrq$coefficients
    Loss.fun <- check.func
  }

  theta <-as.matrix(theta)

  resi<- Y_all-Z_all%*%theta

  # --- Reshape solution back into (theta_Gs, alpha, beta) ---
  alpha = matrix(0, G, length(taus))
  theta_Gs = list()
  beta_new = matrix(0, H, length(taus))

  for (k in 1:length(taus)){
    thetaG = matrix(0, nrow = G, ncol = p+2)
    thetaG[-1,1] <- theta[1:(G-1),k]# alpha_g (g >= 2)
    thetaG[,2] <- theta[G:(2*G - 1),k] # nu_g
    thetaG[,1:p+2] <-  matrix(theta[(2*G):(2*G+G*p-1),k],G,p,byrow = TRUE)# gamma_g
    theta_Gs[[k]] = thetaG
    alpha[,k] = thetaG[,1]
    beta_new[,k] =theta[-(1:((p+2)*G-1)),k]
  }


  # --- Per-quantile loss ---
  loss = rep(0, length(taus))

  for (k in 1:length(taus)){
    tau = taus[k]
    loss[k] = sum(Loss.fun(resi[,k], tau))
  }



  return(list(theta_Gs = theta_Gs, alpha = alpha,
              beta = beta_new, loss = loss,
              Loss.fun=Loss.fun))
}




# =============================================================================
# Initialization: a richer multi-start strategy
# Returns a set of candidate (G-, H-) memberships obtained from per-node
# LASSO quantile regression followed by k-means.
# =============================================================================
twmq.init <-function(Ymat, W, G, H, taus, lambda=0.01,ntrial=100)
{
  n_taus = length(taus)
  N = nrow(Ymat)
  Time = ncol(Ymat)
  Ymat_lag = Ymat[,-Time]
  Ymat1 = Ymat[,-1]

  # Per-node coefficients (lag-self, lag-mean-friend) and per-friend coefs
  beta_lag=beta_fix = matrix(0, N, n_taus)
  fri_lag = array(0, dim = c(N, N, n_taus))

  for (i in 1:N)
  {
    fri_i = which(W[i,]!=0)
    di = length(fri_i)

    if (di>1)
      Xmean = cbind(Ymat_lag[i,], t(Ymat_lag[fri_i,])/di)  else
        Xmean = cbind(Ymat_lag[i,], (Ymat_lag[fri_i,])/di)


    dat = data.frame(Y = Ymat1[i,], X = Xmean)
    resrq = quantreg::rq(Y~., taus , data = dat,
                         method="lasso", lambda = lambda)

    beta_lag[i,] <-  resrq$coefficients[2,]
    beta_fix[i,] <- resrq$coefficients[1,]
    fri_lag[i, fri_i,] = resrq$coefficients[-(1:2),]
  }

  # G-membership candidates: cluster on (lag) or (intercept) coefficients
  Member.int <- NULL
  totss <-NULL
  for(type in c(1,2)){
    for(i in 1:ntrial){
      if(type==1)  km = kmeans(beta_lag, centers = G,iter.max = 100,nstart = 10) else
        if(type==2)  km = kmeans(beta_fix, centers = G,iter.max = 100,nstart = 10)
        Member.int <- rbind(Member.int,km$cluster)
        totss <- rbind(totss,c(km$totss,km$tot.withinss))
    }
  }
  totss <- round(totss,6)
  memberG.int_set <- Member.int[!duplicated(totss),,drop=F]

  # For each candidate G-member, build candidate H-members from per-(g,j) coefs
  memberGH.int_set <- NULL

  for (m in 1:dim(memberG.int_set)[1]) {
    member_G_tmp <-   memberG.int_set[m,]
    tmp <- table(member_G_tmp)
    g_max <- names(tmp)[which.max(tmp)]

    Member.int <- NULL

    # member H initialize
    Member.int <- NULL
    totss <-NULL
    for (i in 1:ntrial){
      fri_H <- 0
      for (g in 1:G) {
        idx_g <- which(member_G_tmp==g)
        fri_lag_g <- fri_lag[idx_g,,]
        non0_ind = apply(fri_lag_g, c(1,2), function(x) any(x!=0))
        fri_mat_g = sapply(1:n_taus, function(k) fri_lag_g[,,k][non0_ind])


        km_g = kmeans(fri_mat_g, centers = H,nstart=10)
        fri_cl_g = matrix(0, length(idx_g), N)
        fri_cl_g[non0_ind] = km_g$cluster

        fri_H2 = sapply(1:N, function(i){
          tmp = sapply(1:H, function(h) {
            indh = which(fri_cl_g[,i]==h)
            if (length(indh)==0)
              return(matrix(0, nrow = 1, ncol = n_taus))
            else{
              return(colMeans(fri_lag_g[indh,i,, drop = F]))
            }
          })
          as.vector(tmp)
        })



        fri_H <- fri_H+fri_H2
      }
      fri_H = t(fri_H)

      km = kmeans(fri_H, centers = H, nstart=10)
      Member.int <- rbind(Member.int,km$cluster)
      totss <- rbind(totss,c(km$totss,km$tot.withinss))
    }

    totss <- round(totss,6)

    memberH.int_set <- Member.int[!duplicated(totss),,drop=F]

    memberGH.int_set <- rbind(memberGH.int_set,
                              cbind(memberG.int_set[rep(m,dim(memberH.int_set)[1]),,drop=F],
                                    memberH.int_set))
  }

  return(list(memberG.int_set = memberGH.int_set[,1:N,drop=F],
              memberH.int_set = memberGH.int_set[,1:N+N,drop=F]))
}



# =============================================================================
# Initialize H-membership given a fixed G-membership
# =============================================================================
twq.init_memberH <-function (Ymat, X_tensor, W, G, H, tau, lambda = 0.01, ntrial = 100,member_G)
{
  N = nrow(Ymat)
  p <- ncol(X)
  Time = ncol(Ymat)
  Ymat_lag = Ymat[, -Time]
  Ymat1 = Ymat[, -1]
  beta_lag = beta_fix = matrix(0, N, 1)
  fri_lag = matrix(0, N, N)
  for (i in 1:N) {
    fri_i = which(W[i, ] != 0)
    di = length(fri_i)
    if (di > 1)
      Xmean = cbind(Ymat_lag[i, ], t(Ymat_lag[fri_i, ])/di)
    else Xmean = cbind(Ymat_lag[i, ], (Ymat_lag[fri_i, ])/di)
    dat = data.frame(Y = Ymat1[i, ], X = Xmean)
    resrq = quantreg::rq(Y ~ ., tau, data = dat, method = "lasso",
                         lambda = lambda)
    beta_lag[i, 1] <- resrq$coefficients[2]
    beta_fix[i, 1] <- resrq$coefficients[1]
    fri_lag[i, fri_i] = resrq$coefficients[-(1:2)]
  }

  memberGH.int_set <- NULL

  member_G_tmp <- member_G

  # Avoid empty G-groups
  ID <- unique(names(table(member_G_tmp)))
  ID <- ID[which(table(member_G_tmp) >= p)]
  while (length(ID) < G) {
    g_max <- (names(table(member_G_tmp)))[which.max(table(member_G_tmp))]
    g_new <- which(!((1:G) %in% ID))[1]
    idx <- which(member_G_tmp == g_max)
    member_G_tmp[sample(idx, size = length(idx)/2)] <- g_new
    ID <- unique(member_G_tmp)
  }

  Member.int <- NULL
  totss <- NULL
  fri_H <- NULL
  for (g in 1:G) {
    idx_g <- which(member_G_tmp == g)
    fri_lag_g <- fri_lag[idx_g, ]
    fri_H <- cbind(fri_H, colSums(fri_lag_g)/ (colSums(W[idx_g,]!=0)+0.0001))
  }

  fri_H <- rowSums(fri_H)

  for (i in 1:ntrial) {
    km = kmeans(fri_H, centers = H, nstart = 10)
    Member.int <- rbind(Member.int, km$cluster)
    totss <- rbind(totss, c(km$totss, km$tot.withinss))
  }
  totss <- round(totss, 6)
  memberH.int_set <- Member.int[!duplicated(totss), , drop = F]
  memberG.int_set <- NULL
  for (m1 in 1:dim(memberH.int_set)[1]) {
    member_H_tmp <- memberH.int_set[m1, ]
    ID <- unique(names(table(member_H_tmp)))
    ID <- ID[which(table(member_H_tmp) >= 2)]
    while (length(ID) < H) {
      h_max <- (names(table(member_H_tmp)))[which.max(table(member_H_tmp))]
      h_new <- which(!((1:H) %in% ID))[1]
      idx <- which(member_H_tmp == h_max)
      member_H_tmp[sample(idx, size = length(idx)/2)] <- h_new
      ID <- unique(member_H_tmp)
    }
    memberH.int_set[m1, ] <- member_H_tmp
    memberG.int_set <- rbind(memberG.int_set,member_G)
  }

  return(list(memberG.int_set = memberG.int_set,
              memberH.int_set = memberH.int_set))
}

# =============================================================================
# Initialization variant scaling lambda by feature variance
# (current default used by twmq.estimate.auto)
# =============================================================================
twq.init_xgg <-function (Ymat, X_tensor, W, G, H, tau, lambda = 0.01, ntrial = 100)
{
  N = nrow(Ymat)
  p = dim(X_tensor)[2]
  Time = ncol(Ymat)
  Ymat_lag = Ymat[, -Time]
  Ymat1 = Ymat[, -1]
  beta_lag = beta_fix = matrix(0, N, 1)
  fri_lag = matrix(0, N, N)

  # Per-node LASSO quantile regression; lambda scaled by sqrt(mean feature variance)
  for (i in 1:N) {
    fri_i = which(W[i, ] != 0)
    di = length(fri_i)
    if (di > 1)
      X = cbind(Ymat_lag[i, ], t(Ymat_lag[fri_i, ])/di)
    else X = cbind(Ymat_lag[i, ], (Ymat_lag[fri_i, ])/di)
    dat = data.frame(Y = Ymat1[i, ], X = X)
    resrq = quantreg::rq(Y ~ ., tau, data = dat, method = "lasso",
                         lambda = lambda*sqrt(mean(diag(t(X)%*%X))))
    beta_lag[i, 1] <- resrq$coefficients[2]
    beta_fix[i, 1] <- resrq$coefficients[1]
    fri_lag[i, fri_i] = resrq$coefficients[-(1:2)]
  }

  # Build candidate G-memberships from k-means on per-node coefficients
  Member.int <- NULL
  totss <- NULL
  for (type in c(1, 2)) {
    for (i in 1:ntrial) {
      if (type == 1)
        km = kmeans(beta_lag, centers = G, iter.max = 100,
                    nstart = 10)
      else if (type == 2)
        km = kmeans(beta_fix, centers = G, iter.max = 100,
                    nstart = 10)
      Member.int <- rbind(Member.int, km$cluster)
      totss <- rbind(totss, c(km$totss, km$tot.withinss))
    }
  }
  totss <- round(totss, 6)
  memberG.int_set <- Member.int[!duplicated(totss), , drop = F]
  memberGH.int_set <- NULL
  for (m in 1:dim(memberG.int_set)[1]) {
    member_G_tmp <- memberG.int_set[m, ]

    ID <- unique(names(table(member_G_tmp)))
    ID <- ID[which(table(member_G_tmp) >= p)]
    while (length(ID) < G) {
      g_max <- (names(table(member_G_tmp)))[which.max(table(member_G_tmp))]
      g_new <- which(!((1:G) %in% ID))[1]
      idx <- which(member_G_tmp == g_max)
      member_G_tmp[sample(idx, size = length(idx)/2)] <- g_new
      ID <- unique(member_G_tmp)
    }
    memberG.int_set[m, ] <- member_G_tmp
    tmp <- table(member_G_tmp)
    g_max <- names(tmp)[which.max(tmp)]
    Member.int <- NULL
    totss <- NULL
    fri_H <- NULL
    for (g in 1:G) {
      idx_g <- which(member_G_tmp == g)
      fri_lag_g <- fri_lag[idx_g, ]
      fri_H <- cbind(fri_H, colSums(fri_lag_g)/ (colSums(W[idx_g,]!=0)+0.0001))
    }

    fri_H <- rowSums(fri_H)

    for (i in 1:ntrial) {
      km = kmeans(fri_H, centers = H, nstart = 10)
      Member.int <- rbind(Member.int, km$cluster)
      totss <- rbind(totss, c(km$totss, km$tot.withinss))
    }
    totss <- round(totss, 6)
    memberH.int_set <- Member.int[!duplicated(totss), , drop = F]
    for (m1 in 1:dim(memberH.int_set)[1]) {
      member_H_tmp <- memberH.int_set[m1, ]
      ID <- unique(names(table(member_H_tmp)))
      ID <- ID[which(table(member_H_tmp) >= 2)]
      while (length(ID) < H) {
        h_max <- (names(table(member_H_tmp)))[which.max(table(member_H_tmp))]
        h_new <- which(!((1:H) %in% ID))[1]
        idx <- which(member_H_tmp == h_max)
        member_H_tmp[sample(idx, size = length(idx)/2)] <- h_new
        ID <- unique(member_H_tmp)
      }
      memberH.int_set[m1, ] <- member_H_tmp
    }
    memberGH.int_set <- rbind(memberGH.int_set, cbind(memberG.int_set[rep(m,
                                                                          dim(memberH.int_set)[1]), , drop = F], memberH.int_set))
  }

  # Cap to at most 10 candidates to keep multi-start cost bounded
  n_init <- dim(memberGH.int_set)[1]
  if(n_init >10) memberGH.int_set <- memberGH.int_set[sample(1:n_init,size=10),,drop=F]
  return(list(memberG.int_set = memberGH.int_set[, 1:N, drop = F],
              memberH.int_set = memberGH.int_set[, 1:N + N, drop = F]))
}

# =============================================================================
# Multi-start vanilla estimation (sequential)
# =============================================================================
#' Two-way network quantile estimation (automatic initialization)
#'
#' @param Ymat N x T matrix of responses
#' @param X_tensor N x p x (T-1) tensor of covariates
#' @param W N x N weight matrix
#' @param G number of G-groups
#' @param H number of H-groups
#' @param taus vector of quantile levels
#' @param method "general", "additive", or "multiplicative"
#' @param verbose logical; print progress or not
#' @param conquer logical; use conquer-based loss or standard check loss
#' @param h_conquer bandwidth for conquer loss
#' @param ntrial number of random initializations
#' @param member_init0 optional list of initial memberships
#' @param Maxit maximum iterations in inner estimation
#'
#' @return A list with elements res_min, res_all, loss, member_init
#' @export
twmq.estimate.auto <- function(Ymat, X_tensor, W,G,H, taus, method, verbose = T,
                               conquer = F, h_conquer=0.05, ntrial = 100,member_init0=NULL,Maxit=100,rq_lambda=0.01)
{
  # Build initial pool (use median quantile tau = 0.5 for init)
  member_init = twq.init_xgg(Ymat,X_tensor, W, G, H, 0.5, lambda=0.01,ntrial=ntrial) # use tau = 0.5 for init
  if(!is.null(member_init0)){
    member_init$memberG.int_set <- rbind(member_init$memberG.int_set,member_init0$memberG.int_set)
    member_init$memberH.int_set <- rbind(member_init$memberH.int_set,member_init0$memberH.int_set)
  }
  n_init = nrow(member_init$memberG.int_set)
  print(n_init)
  res_all = list()
  loss = rep(0, n_init)

  for (k in 1:n_init)
  {
    res = twmq.estimate(Ymat, X_tensor, W,
                        member_init$memberG.int_set[k,],
                        member_init$memberH.int_set[k,], taus, verbose = F,
                        conquer = conquer, h_conquer=h_conquer, method = method,Maxit=Maxit,rq_lambda=rq_lambda)
    res_all[[k]] = res
    loss[k] = sum(res$theta_GH$loss)
    if (verbose)
    {
      cat(k, loss[k], "\n")
    }
  }
  res_min <- res_all[[which.min(loss)]]
  list(res_min=res_min, res_all=res_all,loss=loss,member_init=member_init)
}


# =============================================================================
# Multi-start vanilla estimation (parallel via doSNOW)
# =============================================================================
#' @export
twmq.estimate.auto.parallel <- function(Ymat, X_tensor, W,
                                        G, H, taus, method,
                                        verbose = TRUE,
                                        conquer = FALSE,
                                        h_conquer = 0.05,
                                        ntrial = 100,
                                        member_init0 = NULL,
                                        Maxit = 100,
                                        numCores = 10,rq_lambda=0.01) {

  # 1. Build initial pool
  member_init <- twq.init_xgg(
    Ymat, X_tensor, W,
    G, H, 0.5,          # tau = 0.5 for init
    lambda = 0.01,
    ntrial = ntrial
  )

  if (!is.null(member_init0)) {
    member_init$memberG.int_set <- rbind(
      member_init$memberG.int_set,
      member_init0$memberG.int_set
    )
    member_init$memberH.int_set <- rbind(
      member_init$memberH.int_set,
      member_init0$memberH.int_set
    )
  }

  n_init <- nrow(member_init$memberG.int_set)
  print(n_init)

  numCores <- min(numCores, n_init)

  # 2. Set up cluster and load required packages on each worker
  cl <- parallel::makeCluster(numCores)
  doSNOW::registerDoSNOW(cl)

  parallel::clusterEvalQ(cl, {
    library(twmq)          # so workers can call twmq.estimate directly
    library(RhpcBLASctl)   # to control BLAS threads on workers
    NULL
  })

  # 3. Progress bar
  pb <- utils::txtProgressBar(max = n_init, style = 3)
  progress <- function(count) utils::setTxtProgressBar(pb, count)
  opts <- list(progress = progress)

  # 4. Run each initialization in parallel
  res_all <- foreach::foreach(
    k = 1:n_init,
    .options.snow = opts
  ) %dopar% {
    RhpcBLASctl::blas_set_num_threads(1)

    twmq.estimate(
      Ymat, X_tensor, W,
      member_init$memberG.int_set[k, ],
      member_init$memberH.int_set[k, ],
      taus,
      verbose = FALSE,
      conquer = conquer,
      h_conquer = h_conquer,
      method = method,
      Maxit = Maxit,rq_lambda=rq_lambda
    )
  }

  close(pb)
  parallel::stopCluster(cl)

  # 5. Compute losses and pick the best initialization
  loss <- numeric(n_init)

  for (k in 1:n_init) {
    res <- res_all[[k]]
    loss[k] <- sum(res$theta_GH$loss)
    if (verbose) {
      cat(k, loss[k], "\n")
    }
  }

  res_min <- res_all[[which.min(loss)]]

  list(
    res_min = res_min,
    res_all = res_all,
    loss = loss,
    member_init = member_init
  )
}


# =============================================================================
# Single-pass parameter update for the GENERAL model
# theta_{gh}(tau) is a free parameter for each (g,h) pair
# Each row group's parameters are estimated independently via quantile regression
# =============================================================================
#' @export
twmq.estimate_thetaGH.member.general <- function(Ymat, X_tensor, W, member_G, member_H, taus, conquer, h_conquer, rq_lambda=0.01)
{
  G = max(member_G); H = max(member_H)
  Time = ncol(Ymat); N = nrow(Ymat)
  p = dim(X_tensor)[2]
  # --- Trivial G = H = 1 case ---
  if (G == 1 & H == 1){
    Ymat1 = (W%*%Ymat)
    Z_all = cbind(as.vector(Ymat1[,-Time]),
                  as.vector(Ymat[,-Time]),
                  matrix(aperm(X_tensor,c(1,3,2)),nrow = N*(Time-1)))
    Y = as.vector(Ymat[,-1])

    if (conquer)
    {
      fit.conquer = myconquer(Z_all, Y, tauSeq = taus)
      theta = fit.conquer$coeff
      Loss.fun <- function(u,tau){
        if (length(dim(u))==2){
          u = as.matrix(u)
        }
        loss.ConquerTri(u,h=h_conquer,tau)
      }
    }else{

      # Count constant columns in Z_all (excluding the all-ones intercept column)
      const_col_num <- sum(apply(Z_all, 2, function(x) {
        x_no_na <- x[!is.na(x)]
        length(unique(x_no_na)) <= 1
      }))

      Loss.fun <- check.func
      dat <- data.frame(Y = Y, Z_all)

      if (const_col_num > 1) {
        # More than one constant column => use lasso to ensure stable solution
        resrq <- quantreg::rq(
          Y ~ . - 1,
          taus,
          data = dat,
          method = "lasso",
          lambda = rq_lambda
        )
      } else {
        # Otherwise use ordinary rq
        resrq <- quantreg::rq(
          Y ~ . - 1,
          taus,
          data = dat
        )
      }

      theta <- resrq$coefficients
      resi = resrq$residuals

    }

    return(list(theta, Loss.fun=Loss.fun,resi = resi))
  }



  # --- General case ---
  Y1 <- Ymat[,-Time]
  Y <- Ymat[,-1]
  WY1 <- W%*%Y1

  # H-side network covariate: column h is sum_{j in C_h} w_{ij} Y_{j,t-1}
  Yh <- list()
  for (h in 1:H) {
    Yh[[h]] = W%*%(Y1*(member_H==h))
  }

  # Estimate parameters separately for each row group g
  alphabeta_GHs <-   theta_Gs  <-  list()
  resi <- NULL
  for (k in 1:length(taus)){
    theta_Gs[[k]] = matrix(0, nrow = G, ncol = p+1)
    alphabeta_GHs[[k]] = matrix(0, nrow = G, ncol =H)
  }


  for (g in 1:G)
  {
    ind_g = which(member_G == g)
    Y_g=Y[ind_g,,drop=F]
    Y1_g=Y1[ind_g,,drop=F]
    Z1 <- as.vector(Y1_g)
    Y_all <- as.vector(Y[ind_g,,drop=F])
    # H-side covariate, one column per H-group
    Zh <- NULL
    for (h in 1:H) {
      Zh <- cbind(Zh,as.vector((Yh[[h]])[ind_g,,drop=F]))
    }

    X1 = matrix(aperm(X_tensor[ind_g,,],c(1,3,2)),nrow = length(ind_g)*(Time-1))

    Z_all <- cbind(Z1,X1[,,drop=T],Zh)

    if (conquer){
      fit.conquer = myconquer(Z_all, Y_all, tauSeq = taus)
      theta = fit.conquer$coeff
      Loss.fun <- function(u,tau){
        loss.ConquerTri(u,h=h_conquer,tau)
      }
    }else{
      const_col_num <- sum(apply(Z_all, 2, function(x) {
        x_no_na <- x[!is.na(x)]
        length(unique(x_no_na)) <= 1
      }))

      Loss.fun <- check.func
      dat <- data.frame(Y = Y_all, Z_all)

      if (const_col_num > 1) {
        # More than one constant column => use lasso
        resrq <- quantreg::rq(
          Y ~ . - 1,
          taus,
          data = dat,
          method = "lasso",
          lambda = rq_lambda
        )
      } else {
        # Otherwise use ordinary rq
        resrq <- quantreg::rq(
          Y ~ . - 1,
          taus,
          data = dat
        )
      }

      theta <- resrq$coefficients



    }



    theta <-as.matrix(theta)


    resi<- rbind(resi,Y_all-Z_all%*%theta)
    # given beta estimate alpha

    for (k in 1:length(taus)){
      theta_Gs[[k]][g,1:(p+1)] <- theta[1:(p+1),k]
      alphabeta_GHs[[k]][g,] <- theta[-(1:(1+p)),k]
    }
  }




  # compute the loss
  loss = rep(0, length(taus))

  for (k in 1:length(taus)){
    tau = taus[k]
    loss[k] = sum(Loss.fun(resi[,k], tau))
  }



  return(list(theta_Gs = theta_Gs, alphabeta_GHs = alphabeta_GHs,loss = loss,
              Loss.fun=Loss.fun))
}



# =============================================================================
# General-model specific G-/H-membership update wrappers (used by some scripts;
# main code uses the dispatcher functions twmq.estimate_member* above)
# =============================================================================
twmq.estimate.general_memberG <-function(Ymat, X_tensor, W, member_H,
                                         theta_GH, taus, verbose = F,Loss.fun,conquer,h_conquer)
{
  G = nrow(theta_GH$alphabeta_GHs[[1]])
  p <- dim(X)[2]
  if (conquer)
    resi2 <- Loss_memberG_mq_general_conquer(Ymat, as.matrix(W),
                                             X_tensor, theta_GH, member_H, taus, h_conquer, 1/h_conquer,
                                             1/h_conquer^2)
  else resi2 <- Loss_memberG_mq_general(Ymat, as.matrix(W),
                                        X_tensor, theta_GH, member_H, taus)
  member_G = apply(resi2, 1, which.min)
  ID <- unique(names(table(member_G)))
  ID <- ID[which(table(member_G) >= p)]
  while (length(ID) < G) {
    g_max <- (names(table(member_G)))[which.max(table(member_G))]
    g_new <- which(!((1:G) %in% ID))[1]
    idx <- which(member_G == g_max)
    member_G[sample(idx, size = length(idx)/2)] <- g_new
    ID <- unique(member_G)
  }
  return(member_G)
}

# update member H
twmq.estimate.general_memberH <-function(Ymat, X_tensor, W, member_G, member_H_init,
                                         theta_GH, taus, verbose = F,
                                         Loss.fun,FriendW,FriendW2,conquer,h_conquer)
{
  H = ncol(theta_GH$alphabeta_GHs[[1]])

  if (conquer)
    member_H = Update_memberH_mq_general_conquer(Ymat,
                                                 as.matrix(W), X_tensor, theta_GH, member_G, member_H_init,
                                                 FriendW, FriendW2, taus, h_conquer, 1/h_conquer,
                                                 1/h_conquer^2)
  else member_H = Update_memberH_mq_general(Ymat, as.matrix(W),
                                            X_tensor, theta_GH, member_G, member_H_init, FriendW, FriendW2,
                                            taus)

  ID <- unique(names(table(member_H)))
  ID <- ID[which(table(member_H) >= 2)]
  while (length(ID) < H) {
    h_max <- (names(table(member_H)))[which.max(table(member_H))]
    h_new <- which(!((1:H) %in% ID))[1]
    idx <- which(member_H == h_max)
    member_H[sample(idx, size = length(idx)/2)] <- h_new
    ID <- unique(member_H)
  }
  return(member_H)
}





# Convert a list of equally-sized matrices into a 3D array
list2array <- function(ll){
  array(unlist(ll), dim = c(dim(ll[[1]]), length(ll)))
}




# =============================================================================
# Per-node H-optimization fallback (used when enumeration is infeasible)
# Solves a discrete optimization for the H-membership of node i's neighbors
# using either exhaustive enumeration (if feasible) or CEoptim.
# =============================================================================
optmize_qi <- function(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,additive,H,idx.fix=NULL){
  Time = ncol(Ymat)
  N = nrow(Ymat)
  Ymat_lag = Ymat[,-Time]
  Ymat1 = Ymat[,-1]

  gi <- member_G[i]
  fri_i = which(W[i,]!=0)

  theta_Gs = theta$theta_Gs
  beta = theta$beta
  alpha = theta$alpha
  Loss.fun <- theta$Loss.fun

  updatei <- setdiff(fri_i,intersect(fri_i, idx.fix))
  di = length(updatei)

  alphaWs <- eps_mats <- list()

  fri_lagi <- rep(NA,N)
  fri_lagi[fri_i] <- member_H[fri_i]

  if(di>0){

    if (length(fri_i)>1)
      Ymat_lag_fri =  t(Ymat_lag[fri_i,]) else
        Ymat_lag_fri = matrix(Ymat_lag[fri_i,], ncol = 1)

      # Pre-compute residuals (excluding H-side network terms)
      for (k in 1:length(taus)){
        alpha_i = alpha[gi, k]
        thetaG = theta_Gs[[k]]
        alphaWs[[k]] = alpha_i*W[i,fri_i]
        nu_i = thetaG[,2][gi]
        gamma_i = thetaG[gi,-(1:2)]
        eps_i = Ymat[i,-1] - nu_i*Ymat[i,-Time] - colSums(X_tensor[i,,]*gamma_i)
        eps_mats[[k]] = eps_i
      }

      # Define the loss as a function of the trial H-vector
      if(additive){
        update_Hi <- function(member_H_i,Ymat_lag_fri,eps_mats,alphaWs,beta,H,member_H,updatei,fri_i,taus) {
          member_Hi <- member_H
          member_Hi[updatei] <- c(member_H_i+1)
          member_Hi <- member_Hi[fri_i]
          fval <- 0
          for (k in 1:length(taus)){
            # Additive: prediction uses (beta + alpha) effects separately
            Resi_pred = Ymat_lag_fri%*%(beta[member_Hi,k]/length(fri_i)+alphaWs[[k]])
            fval = fval + sum(Loss.fun(eps_mats[[k]]-Resi_pred, taus[k]))
          }
          fval
        }
      }else{

        update_Hi <- function(member_H_i,Ymat_lag_fri,eps_mats,alphaWs,beta,H,member_H,updatei,fri_i,taus) {
          member_Hi <- member_H
          member_Hi[updatei] <- c(member_H_i+1)
          member_Hi <- member_Hi[fri_i]
          fval <- 0
          for (k in 1:length(taus)){
            # Multiplicative: prediction uses alpha * beta
            Resi_pred = Ymat_lag_fri%*%(beta[member_Hi,k]*alphaWs[[k]])
            fval = fval + sum(Loss.fun(eps_mats[[k]]-Resi_pred, taus[k]))
          }
          fval
        }
      }

      # Enumerate if feasible; otherwise use CEoptim
      if(H^di<=10000){
        l <- rep(list(1:H), di)
        member_H_i <- as.matrix(expand.grid(l))-1
        ft_new <- function(member_Hi){
          update_Hi(member_Hi,Ymat_lag_fri,eps_mats,alphaWs,beta,H,member_H,updatei,fri_i,taus)
        }
        fval_new <-  apply(member_H_i, 1, ft_new)
        fri_lagi[updatei] <- member_H_i[which.min(fval_new),]+1
        print(min(fval_new))
      }else{
        obj_i <- CEoptim(f = update_Hi, f.arg = list(Ymat_lag_fri=Ymat_lag_fri,eps_mats=eps_mats,alphaWs=alphaWs,beta=beta,H=H,member_H=member_H,updatei=updatei,fri_i=fri_i,taus=taus),discrete = list(categories = as.integer(rep(H,length(updatei))),smoothProb=0.5),N = 1000, rho = 0.001, verbose = F,noImproveThr=2)
        fri_lagi[updatei] <- obj_i$optimizer$discrete+1
      }

  }
  if(i%%50==0) print(i)
  fri_lagi
}

# Per-node H-optimization for the GENERAL model
optmize_qi_general <- function(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,H,idx.fix=NULL){
  Time = ncol(Ymat)
  N = nrow(Ymat)
  Ymat_lag = Ymat[,-Time]
  Ymat1 = Ymat[,-1]

  gi <- member_G[i]
  fri_i = which(W[i,]!=0)

  theta_Gs = theta$theta_Gs
  alphabeta_GHs <- theta$alphabeta_GHs

  Loss.fun <- theta$Loss.fun

  updatei <- setdiff(fri_i,intersect(fri_i, idx.fix))
  di = length(updatei)

  eps_mats <- list()

  fri_lagi <- rep(NA,N)
  fri_lagi[fri_i] <- member_H[fri_i]

  if(di>0){

    if (length(fri_i)>1)
      Ymat_lag_fri =  t(Ymat_lag[fri_i,]) else
        Ymat_lag_fri = matrix(Ymat_lag[fri_i,], ncol = 1)

      for (k in 1:length(taus)){
        thetaG = theta_Gs[[k]]
        nu_i = thetaG[,1][gi]
        gamma_i = thetaG[gi,-1]
        eps_i = Ymat[i,-1] - nu_i*Ymat[i,-Time] - colSums(X_tensor[i,,]*gamma_i)
        eps_mats[[k]] = eps_i
      }

      update_Hi <- function(member_H_i,Ymat_lag_fri,eps_mats,alphabeta_GHs,gi,H,member_H,updatei,fri_i,taus) {
        member_Hi <- member_H
        member_Hi[updatei] <- c(member_H_i+1)
        member_Hi <- member_Hi[fri_i]
        fval <- 0
        for (k in 1:length(taus)){
          alphabetas = alphabeta_GHs[[k]][gi,]
          Resi_pred = Ymat_lag_fri%*%(alphabetas[member_Hi]/length(fri_i))
          fval = fval + sum(Loss.fun(eps_mats[[k]]-Resi_pred, taus[k]))
        }
        fval
      }

      ###enumerate methods
      if(H^di<=10000){
        l <- rep(list(1:H), di)
        member_H_i <- as.matrix(expand.grid(l))-1
        ft_new <- function(member_Hi){
          update_Hi(member_Hi,Ymat_lag_fri,eps_mats,alphabeta_GHs,gi,H,member_H,updatei,fri_i,taus)
        }
        fval_new <-  apply(member_H_i, 1, ft_new)
        print(min(fval_new))

        fri_lagi[updatei] <- member_H_i[which.min(fval_new),]+1
      }else{
        obj_i <- CEoptim(f = update_Hi, f.arg = list(Ymat_lag_fri=Ymat_lag_fri,eps_mats=eps_mats,alphabeta_GHs=alphabeta_GHs,gi=gi,H=H,member_H=member_H,updatei=updatei,fri_i=fri_i,taus=taus),discrete = list(categories = as.integer(rep(H,length(updatei))),smoothProb=0.5),N = 1000, rho = 0.001, verbose = F,noImproveThr=2)
        fri_lagi[updatei] <- obj_i$optimizer$discrete+1
      }

  }
  if(i%%50==0) print(i)
  fri_lagi
}


# Create the function to find the mode.
getmode <- function(v) {
  v<-v[!is.na(v)]
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Create the function to find the mode.
getsample <- function(v) {
  v<-v[!is.na(v)]
  d.v <- length(v)
  if(d.v==0) return(NA) else return(sample(v,1))
}


# =============================================================================
# ENHANCED ALGORITHM (Algorithm 2 in the paper, sequential version)
# Iteratively builds proposal H-vectors and accepts those that strictly
# decrease the loss, helping to escape local minima found by the Vanilla.
# =============================================================================
#' @export
update_NARG_twmq <-function(Ymat, W,X_tensor, member_G,member_H,taus,theta,method,G,H,conquer,h_conquer,Iter,frac=0.5,MaxOutIter=100,Maxit=5,rq_lambda=0.01)
{
  N = nrow(Ymat)
  Time = ncol(Ymat)

  # Step I: refit the model under current memberships
  theta = twmq.estimate_thetaGH.member.iterate(Ymat, X_tensor, W,member_G, member_H,taus, conquer,h_conquer,theta=theta, method, n_iter.max = 100, verbose = F,rq_lambda=rq_lambda)
  ###Now start update memberships and parameters

  idx.fix <- NULL
  converge <- FALSE
  out.iter <- 1

  idx_update <- setdiff(1:N,idx.fix)

  # Pre-build follower lists
  FriendW <- vector("list", N)
  for(i in 1:N){
    FriendW[[i]] <- as.vector(which(W[i,]!=0))
  }

  # Build initial proposal matrix using vectorized C++ routines when possible
  if(method=="general"){
    if (conquer)
      fri_lag0 <-  Proposal_memberH_mq_general_conquer(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000, h_conquer, 1/h_conquer,1/h_conquer^2) else
        fri_lag0 <-  Proposal_memberH_mq_general(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000)
  }

  if(method=="additive"){
    if (conquer)
      fri_lag0 <-  Proposal_memberH_mq_additive_conquer(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000, h_conquer, 1/h_conquer,1/h_conquer^2) else
        fri_lag0 <-  Proposal_memberH_mq_additive(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000)
  }

  if(method=="multiplicative"){
    if (conquer)    fri_lag0 <-  Proposal_memberH_mq_conquer(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000, h_conquer, 1/h_conquer,1/h_conquer^2) else
      fri_lag0 <-  Proposal_memberH_mq(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000)
  }
  fri_lag0[fri_lag0==0]<- NA

  ###Use integer programming when di is too large
  for(i in 1:N){
    fri_i = which(W[i,]!=0)
    updatei <- setdiff(fri_i,intersect(fri_i, idx.fix))
    di = length(updatei)
    if(H^di >= 10000){
      if(method=="general")   fri_lag0[i,]<- optmize_qi_general(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,H,idx.fix=idx.fix)
      if(method=="additive")    fri_lag0[i,]<- optmize_qi(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,additive=T,H,idx.fix=idx.fix)
      if(method=="multiplicative")   fri_lag0[i,]<- optmize_qi(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,additive=F,H,idx.fix=idx.fix)
    }
  }



  dfval <-NULL
  fval0 <- sum(theta$loss)

  count_halve <- 0


  while((out.iter <= MaxOutIter)&(!converge)){

    change <- 0

    n.dif <- 0
    iter <- 1

    while(iter <= Iter&(count_halve<=2)) {

      fri_lag <- fri_lag0
      # If we've halved the active set, regenerate proposals using
      # only currently-active nodes
      if(count_halve!=0){

        for (j in 1:count_halve) {
          tmp.H <-  apply(fri_lag, 2, getsample)

          idx <- which(!is.na(tmp.H))
          member_H_set <- member_H
          member_H_set[idx] <- tmp.H[idx]

          idx.fix <- which(member_H_set==member_H)
          print(length(idx.fix))


          idx_update <- setdiff(1:N,idx.fix)

          if(method=="general"){
            if (conquer)
              fri_lag <-  Proposal_memberH_mq_general_conquer(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000, h_conquer, 1/h_conquer,1/h_conquer^2) else
                fri_lag <-  Proposal_memberH_mq_general(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000)
          }

          if(method=="additive"){
            if (conquer)
              fri_lag <-  Proposal_memberH_mq_additive_conquer(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000, h_conquer, 1/h_conquer,1/h_conquer^2) else
                fri_lag <-  Proposal_memberH_mq_additive(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000)
          }


          if(method=="multiplicative") {
            if (conquer)    fri_lag <-  Proposal_memberH_mq_conquer(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000, h_conquer, 1/h_conquer,1/h_conquer^2) else
              fri_lag <-  Proposal_memberH_mq(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000)
          }

          fri_lag[fri_lag==0]<- NA

          ###Use integer programming when di is too large
          for(i in 1:N){
            fri_i = which(W[i,]!=0)
            updatei <- setdiff(fri_i,intersect(fri_i, idx.fix))
            di = length(updatei)

            if(H^di >= 10000){
              if(method=="general")   fri_lag[i,]<- optmize_qi_general(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,H,idx.fix=idx.fix)
              if(method=="additive")    fri_lag[i,]<- optmize_qi(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,additive=T,H,idx.fix=idx.fix)
              if(method=="multiplicative")   fri_lag[i,]<- optmize_qi(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,additive=F,H,idx.fix=idx.fix)
            }

          }


        }

      }

      # Sample one proposed H-membership per column from the proposal pool
      tmp.H <-  apply(fri_lag, 2, getsample)

      idx <- which(!is.na(tmp.H))
      member_H_set <- member_H
      member_H_set[idx] <- tmp.H[idx]

      idx.fix <- which(member_H_set==member_H)
      print(length(idx.fix))

      idx.dif <- which(member_H!=member_H_set)
      n.dif <- length(idx.dif)

      # Try to find at least one differing proposal
      try=0;
      while(n.dif==0&try<=500){
        tmp.H <-  apply(fri_lag, 2, getsample)

        idx <- which(!is.na(tmp.H))
        member_H_set <- member_H
        member_H_set[idx] <- tmp.H[idx]

        idx.fix <- which(member_H_set==member_H)
        idx.dif <- which(member_H!=member_H_set)
        n.dif <- length(idx.dif)
        try<- try+1
      }

      # Take only a fraction of the differing positions
      member_H_new <- member_H
      if(n.dif!=0){
        idx_new <- sample(idx.dif,ceiling(n.dif*frac))
        member_H_new[idx_new] <-   member_H_set[idx_new]
      }

      # Robust threshold against pathological loss jumps
      if(length(dfval)<=10) dfval.sd <- 10^(10) else dfval.sd <- IQR(dfval)*1.5

      # Try a quick refit to gauge the loss change
      obj.new = twmq.estimate(Ymat, X_tensor, W, member_G, member_H_new, taus, verbose = F, conquer, h_conquer, method,theta_GH=theta,Maxit=2,rq_lambda=rq_lambda)
      ftry1 <- fval_new <- sum(obj.new$theta_GH$loss)

      if((fval_new-fval0)<dfval.sd){
        # Promising; refit with full Maxit
        obj.new = twmq.estimate(Ymat, X_tensor, W, member_G, member_H_new, taus, verbose = F, conquer, h_conquer, method,theta_GH=theta,Maxit=Maxit,rq_lambda=rq_lambda)
        fval_new <- sum(obj.new$theta_GH$loss)
        dfval <- c(dfval,abs(fval_new-ftry1))
        print("haha")
      }


      if(fval_new>=fval0) {
        # Try the complementary subset of differing positions
        if(iter < Iter)  iter <- iter +1
        if(H==1){idx_new}
        idx_new <- setdiff(idx.dif,idx_new)
        member_H_new[idx_new] <-   member_H_set[idx_new]
        obj.new = twmq.estimate(Ymat, X_tensor, W, member_G, member_H_new, taus, verbose = F, conquer, h_conquer, method,theta_GH=theta,Maxit=2,rq_lambda=rq_lambda)
        ftry1 <- fval_new <- sum(obj.new$theta_GH$loss)
        if((fval_new-fval0)<dfval.sd){
          obj.new = twmq.estimate(Ymat, X_tensor, W, member_G, member_H_new, taus, verbose = F, conquer, h_conquer, method,theta_GH=theta,Maxit=Maxit,rq_lambda=rq_lambda)
          fval_new <- sum(obj.new$theta_GH$loss)
          dfval <- c(dfval,abs(fval_new-ftry1))
          print("haha")
        }
      }


      if(fval_new< fval0) {

        ###If update memberships successful, do more iterations
        member_H_new <- obj.new$member_H
        member_G <- obj.new$member_G
        theta <- obj.new$theta_GH
        if(!obj.new$converge)  obj.new = twmq.estimate(Ymat, X_tensor, W, member_G, member_H_new, taus, verbose = F, conquer, h_conquer, method,theta_GH=theta,Maxit=100,rq_lambda=rq_lambda)

        fval_new <- sum(obj.new$theta_GH$loss)
        member_H_new <- obj.new$member_H
        id.change <- which(member_H_new!= member_H)

        member_G <- obj.new$member_G
        member_H <- obj.new$member_H
        theta <- obj.new$theta_GH
        fval0 <- fval_new


        if(length(id.change)>0) {

          idx.fix=NULL
          idx_update <- setdiff(1:N,idx.fix)
          # Regenerate the proposal pool given the updated theta and members
          if(method=="additive"){
            if (conquer)
              fri_lag0 <-  Proposal_memberH_mq_additive_conquer(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000, h_conquer, 1/h_conquer,1/h_conquer^2) else
                fri_lag0 <-  Proposal_memberH_mq_additive(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000)
          }

          if(method=="general"){
            if (conquer)
              fri_lag0 <-  Proposal_memberH_mq_general_conquer(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000, h_conquer, 1/h_conquer,1/h_conquer^2) else
                fri_lag0 <-  Proposal_memberH_mq_general(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000)
          }

          if(method=="multiplicative") {
            if (conquer)    fri_lag0 <-  Proposal_memberH_mq_conquer(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000, h_conquer, 1/h_conquer,1/h_conquer^2) else
              fri_lag0 <-  Proposal_memberH_mq(Ymat, as.matrix(W), X_tensor, theta, member_G, member_H, FriendW, idx_update,taus,nsample=10000)
          }

          fri_lag0[fri_lag0==0]<- NA

          ###Use integer programming when di is too large
          for(i in 1:N){
            fri_i = which(W[i,]!=0)
            updatei <- setdiff(fri_i,intersect(fri_i, idx.fix))
            di = length(updatei)

            if(H^di >= 10000){
              if(method=="general")   fri_lag0[i,]<- optmize_qi_general(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,H,idx.fix=idx.fix)
              if(method=="additive")    fri_lag0[i,]<- optmize_qi(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,additive=T,H,idx.fix=idx.fix)
              if(method=="multiplicative")   fri_lag0[i,]<- optmize_qi(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,additive=F,H,idx.fix=idx.fix)
            }

          }


          change <- change+1
          iter <- 0
        }

      }

      if(iter==Iter&(count_halve<=2)){
        change <- 0
        count_halve <- count_halve+1
        iter <- 0
      }

      print(c(iter,fval0,fval_new,n.dif,change,Maxit,count_halve))

      iter <- iter+1
    }
    # If no improvement found this round, declare convergence
    if(change==0){ ###Try second set of iterations and failed again-> converged!
      converge <- TRUE
    }
    out.iter <- out.iter+1

  }

  list(member_G=member_G,member_H=member_H,theta_GH=theta,converge=converge,frac=frac)
}


# =============================================================================
# ENHANCED ALGORITHM (parallel version)
# Same logic as update_NARG_twmq but parallelizes the per-node H optimization
# across cores using doSNOW.
# =============================================================================
#' @export
update_NARG_twmq_parallel <- function(Ymat, W, X_tensor,
                                      member_G, member_H,
                                      taus, theta, method, G, H,
                                      conquer, h_conquer, Iter,
                                      frac = 0.1, numCores,
                                      MaxOutIter = 100, Maxit = 5,rq_lambda=0.01) {
  N <- nrow(Ymat)
  Time <- ncol(Ymat)

  ### First fit the model with given memberships
  theta <- twmq.estimate_thetaGH.member.iterate(
    Ymat, X_tensor, W,
    member_G, member_H, taus,
    conquer, h_conquer,
    theta = theta,
    method = method,
    n_iter.max = 100,
    verbose = FALSE,
    rq_lambda=rq_lambda
  )

  ### Now start update memberships and parameters
  idx.fix <- NULL
  converge <- FALSE
  out.iter <- 1

  ## ---- Parallel setup: create cluster and load required packages on workers ----
  cl <- parallel::makeCluster(numCores)
  doSNOW::registerDoSNOW(cl)

  parallel::clusterEvalQ(cl, {
    library(twmq)
    library(CEoptim)
    library(RhpcBLASctl)
    NULL
  })

  ### Create progress report
  pb <- utils::txtProgressBar(max = N, style = 3)
  progress <- function(count) utils::setTxtProgressBar(pb, count)
  opts <- list(progress = progress)

  fri_lag0 <- foreach::foreach(
    i = 1:N,
    .options.snow = opts,
    .combine = rbind
  ) %dopar% {
    RhpcBLASctl::blas_set_num_threads(1)

    if (method == "general") {
      tmp <- optmize_qi_general(
        i, Ymat, X_tensor, W,
        member_G, member_H, taus,
        theta, H, idx.fix = NULL
      )
    } else {
      if (method == "additive") {
        tmp <- optmize_qi(
          i, Ymat, X_tensor, W,
          member_G, member_H, taus,
          theta, additive = TRUE,
          H, idx.fix = NULL
        )
      }
      if (method == "multiplicative") {
        tmp <- optmize_qi(
          i, Ymat, X_tensor, W,
          member_G, member_H, taus,
          theta, additive = FALSE,
          H, idx.fix = NULL
        )
      }
    }
    tmp
  }
  close(pb)

  dfval <- NULL
  fval0 <- sum(theta$loss)

  count_halve <- 0


  while ((out.iter <= MaxOutIter) && (!converge)) {

    change <- 0
    n.dif <- 0
    iter <- 1

    while (iter <= Iter && (count_halve <= 2)) {

      fri_lag <- fri_lag0

      if (count_halve != 0) {

        for (j in 1:count_halve) {
          tmp.H <- apply(fri_lag, 2, getsample)

          idx <- which(!is.na(tmp.H))
          member_H_set <- member_H
          member_H_set[idx] <- tmp.H[idx]

          idx.fix <- which(member_H_set == member_H)
          print(length(idx.fix))

          ### Create progress report
          pb <- utils::txtProgressBar(max = N, style = 3)
          progress <- function(count) utils::setTxtProgressBar(pb, count)
          opts <- list(progress = progress)

          fri_lag <- foreach::foreach(
            i = 1:N,
            .options.snow = opts,
            .combine = rbind
          ) %dopar% {
            if (method == "general") {
              tmp <- optmize_qi_general(
                i, Ymat, X_tensor, W,
                member_G, member_H, taus,
                theta, H, idx.fix = idx.fix
              )
            } else {
              if (method == "additive") {
                tmp <- optmize_qi(
                  i, Ymat, X_tensor, W,
                  member_G, member_H, taus,
                  theta, additive = TRUE,
                  H, idx.fix = idx.fix
                )
              }
              if (method == "multiplicative") {
                tmp <- optmize_qi(
                  i, Ymat, X_tensor, W,
                  member_G, member_H, taus,
                  theta, additive = FALSE,
                  H, idx.fix = idx.fix
                )
              }
            }
            tmp
          }
          close(pb)
        }
      }


      tmp.H <- apply(fri_lag, 2, getsample)

      idx <- which(!is.na(tmp.H))
      member_H_set <- member_H
      member_H_set[idx] <- tmp.H[idx]

      idx.fix <- which(member_H_set == member_H)
      print(length(idx.fix))

      idx.dif <- which(member_H != member_H_set)
      n.dif <- length(idx.dif)

      try <- 0

      while (n.dif == 0 && try <= 500) {
        tmp.H <- apply(fri_lag, 2, getsample)

        idx <- which(!is.na(tmp.H))
        member_H_set <- member_H
        member_H_set[idx] <- tmp.H[idx]

        idx.fix <- which(member_H_set == member_H)
        idx.dif <- which(member_H != member_H_set)
        n.dif <- length(idx.dif)
        try <- try + 1
      }



      member_H_new <- member_H
      if (n.dif != 0) {
        idx_new <- sample(idx.dif, ceiling(n.dif * frac))
        member_H_new[idx_new] <- member_H_set[idx_new]
      }

      if (length(dfval) <= 10) {
        dfval.sd <- 10^10
      } else {
        dfval.sd <- IQR(dfval) * 1.5
      }


      obj.new <- twmq.estimate(
        Ymat, X_tensor, W,
        member_G, member_H_new, taus,
        verbose = FALSE,
        conquer = conquer,
        h_conquer = h_conquer,
        method = method,
        theta_GH = theta,
        Maxit = 2,
        rq_lambda=rq_lambda
      )
      ftry1 <- fval_new <- sum(obj.new$theta_GH$loss)

      if ((fval_new - fval0) < dfval.sd) {
        obj.new <- twmq.estimate(
          Ymat, X_tensor, W,
          member_G, member_H_new, taus,
          verbose = FALSE,
          conquer = conquer,
          h_conquer = h_conquer,
          method = method,
          theta_GH = theta,
          Maxit = Maxit,
          rq_lambda=rq_lambda
        )
        fval_new <- sum(obj.new$theta_GH$loss)
        dfval <- c(dfval, abs(fval_new - ftry1))
        print("haha")
      }


      if (fval_new >= fval0) {
        if (iter < Iter) iter <- iter + 1
        if (H == 1) idx_new <- c()
        idx_new <- setdiff(idx.dif, idx_new)
        member_H_new[idx_new] <- member_H_set[idx_new]

        obj.new <- twmq.estimate(
          Ymat, X_tensor, W,
          member_G, member_H_new, taus,
          verbose = FALSE,
          conquer = conquer,
          h_conquer = h_conquer,
          method = method,
          theta_GH = theta,
          Maxit = 2,
          rq_lambda=rq_lambda
        )
        ftry1 <- fval_new <- sum(obj.new$theta_GH$loss)

        if ((fval_new - fval0) < dfval.sd) {
          obj.new <- twmq.estimate(
            Ymat, X_tensor, W,
            member_G, member_H_new, taus,
            verbose = FALSE,
            conquer = conquer,
            h_conquer = h_conquer,
            method = method,
            theta_GH = theta,
            Maxit = Maxit,
            rq_lambda=rq_lambda
          )
          fval_new <- sum(obj.new$theta_GH$loss)
          dfval <- c(dfval, abs(fval_new - ftry1))
          print("haha")
        }
      }


      if (fval_new < fval0) {

        ### If update memberships successful, do more iterations
        member_H_new <- obj.new$member_H
        member_G <- obj.new$member_G
        theta <- obj.new$theta_GH

        if (!obj.new$converge) {
          obj.new <- twmq.estimate(
            Ymat, X_tensor, W,
            member_G, member_H_new, taus,
            verbose = FALSE,
            conquer = conquer,
            h_conquer = h_conquer,
            method = method,
            theta_GH = theta,
            Maxit = 100,
            rq_lambda=rq_lambda
          )
        }

        fval_new <- sum(obj.new$theta_GH$loss)
        member_H_new <- obj.new$member_H
        id.change <- which(member_H_new != member_H)

        member_G <- obj.new$member_G
        member_H <- obj.new$member_H
        theta <- obj.new$theta_GH
        fval0 <- fval_new

        if (length(id.change) > 0) {
          ### Create progress report
          pb <- utils::txtProgressBar(max = N, style = 3)
          progress <- function(count) utils::setTxtProgressBar(pb, count)
          opts <- list(progress = progress)

          fri_lag0 <- foreach::foreach(
            i = 1:N,
            .options.snow = opts,
            .combine = rbind
          ) %dopar% {
            if (method == "general") {
              tmp <- optmize_qi_general(
                i, Ymat, X_tensor, W,
                member_G, member_H, taus,
                theta, H, idx.fix = NULL
              )
            } else {
              if (method == "additive") {
                tmp <- optmize_qi(
                  i, Ymat, X_tensor, W,
                  member_G, member_H, taus,
                  theta, additive = TRUE,
                  H, idx.fix = NULL
                )
              }
              if (method == "multiplicative") {
                tmp <- optmize_qi(
                  i, Ymat, X_tensor, W,
                  member_G, member_H, taus,
                  theta, additive = FALSE,
                  H, idx.fix = NULL
                )
              }
            }
            tmp
          }
          close(pb)

          change <- change + 1
          iter <- 0
        }
      }


      if (iter == Iter && (count_halve <= 2)) {
        change <- 0
        count_halve <- count_halve + 1
        iter <- 0
      }

      print(c(iter, fval0, fval_new, n.dif, change, Maxit, count_halve))

      iter <- iter + 1
    }

    if (change == 0) {
      ### Try second set of iterations and failed again -> converged!
      converge <- TRUE
    }
    out.iter <- out.iter + 1
  }

  # Stop the cluster
  parallel::stopCluster(cl)

  list(
    member_G = member_G,
    member_H = member_H,
    theta_GH = theta,
    converge = converge,
    frac = frac
  )
}


# =============================================================================
# Clustering error metrics
# =============================================================================
#' Mismatch error rate after best label-permutation
#' @export
err_rate_mapping<-function(member, member0)
{
  G = length(unique(member))
  member10 = table(member, member0)
  chi = apply(member10, 1, which.max)
  member10[cbind(1:G, chi)] = 0
  sum(member10)/length(member)
}

# Returns indices of mismatched nodes (after relabeling) for diagnostic plots
err_members<-function(member, member0)
{
  G = length(unique(member))
  member10 = table(member, member0)
  chi = apply(member10, 1, which.max)
  member_new <- member
  for (i in 1:G) {
    member_new[member==i] <- chi[i]
  }
  idx <- which(member_new!=member0)
  list(idx=idx,member=member_new[idx])
}

# =============================================================================
# Per-node refinement: how would total loss change if node i's G-/H-
# membership were forced to each candidate value?
# =============================================================================
#' @export
Refine_G <- function(Ymat,X_tensor,W,member_G, member_H, taus,theta,method,G,H){
  N = nrow(Ymat)
  loss_old <- loss_new <- g_r <- numeric()
  for (i in 1:N) {
    if(method=="additive")
      tmp <- Refine_G_i(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,additive=T,G,H) else
        if(method=="multiplicative")
          tmp <- Refine_G_i(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,additive=F,G,H) else
            if(method=="general")
              tmp <- Refine_G_i_general(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,G,H)

            loss_old[i] <- tmp$loss_old
            loss_new[i] <- tmp$loss_new
            g_r[i] <- tmp$g_r
            print(i)
  }
  list(loss_old = loss_old, g_r = g_r, loss_new = loss_new)
}


#' @export
Refine_H <- function(Ymat,X_tensor,W,member_G, member_H, taus,theta,method,G,H, FriendW, FriendW2){
  N = nrow(Ymat)
  loss_old <- loss_new <- h_r <- numeric()
  for (i in 1:N) {
    if(method=="additive")
      tmp <- Refine_H_i(i, Ymat, X_tensor, W, member_G, member_H, taus, theta, G,
                        H, additive=T, FriendW, FriendW2)  else
                          if(method=="multiplicative")
                            tmp <- Refine_H_i(i, Ymat, X_tensor, W, member_G, member_H, taus, theta, G,
                                              H, additive=F, FriendW, FriendW2) else
                                                if(method=="general")
                                                  tmp <- Refine_H_i_general(i, Ymat, X_tensor, W, member_G, member_H, taus, theta, G,
                                                                            H, FriendW, FriendW2)

                                                loss_old[i] <- tmp$loss_old
                                                loss_new[i] <- tmp$loss_new
                                                h_r[i] <- tmp$h_r
                                                print(i)
  }
  list(loss_old = loss_old, h_r = h_r, loss_new = loss_new)
}


# Parallel variants of the per-node G/H refinement
#' @export
Refine_G_parallel <- function(Ymat, X_tensor, W, member_G, member_H, taus, theta, method, G, H, numCores) {
  N <- nrow(Ymat)
  loss_old <- loss_new <- g_r <- numeric()

  cl <- parallel::makeCluster(numCores)
  doSNOW::registerDoSNOW(cl)

  parallel::clusterEvalQ(cl, {
    library(twmq)
    library(CEoptim)
    library(RhpcBLASctl)
    NULL
  })


  pb <- utils::txtProgressBar(max = N, style = 3)
  progress <- function(count) utils::setTxtProgressBar(pb, count)
  opts <- list(progress = progress)

  res_all <- foreach::foreach(i = 1:N, .options.snow = opts) %dopar% {
    RhpcBLASctl::blas_set_num_threads(1)

    if (method == "additive") {
      tmp <- Refine_G_i(i, Ymat, X_tensor, W, member_G, member_H, taus, theta, additive = TRUE, G, H)
    } else if (method == "multiplicative") {
      tmp <- Refine_G_i(i, Ymat, X_tensor, W, member_G, member_H, taus, theta, additive = FALSE, G, H)
    } else if (method == "general") {
      tmp <- Refine_G_i_general(i, Ymat, X_tensor, W, member_G, member_H, taus, theta, G, H)
    }
    tmp
  }

  close(pb)
  parallel::stopCluster(cl)

  for (i in 1:N) {
    tmp <- res_all[[i]]
    loss_old[i] <- tmp$loss_old
    loss_new[i] <- tmp$loss_new
    g_r[i] <- tmp$g_r
  }

  list(loss_old = loss_old, g_r = g_r, loss_new = loss_new)
}


#' @export
Refine_H_parallel <- function(Ymat, X_tensor, W, member_G, member_H, taus, theta, method, G, H, FriendW, FriendW2, numCores) {
  N <- nrow(Ymat)
  loss_old <- loss_new <- h_r <- numeric()

  cl <- parallel::makeCluster(numCores)
  doSNOW::registerDoSNOW(cl)

  parallel::clusterEvalQ(cl, {
    library(twmq)
    library(CEoptim)
    library(RhpcBLASctl)
    NULL
  })

  pb <- utils::txtProgressBar(max = N, style = 3)
  progress <- function(count) utils::setTxtProgressBar(pb, count)
  opts <- list(progress = progress)

  res_all <- foreach::foreach(i = 1:N, .options.snow = opts) %dopar% {
    RhpcBLASctl::blas_set_num_threads(1)

    if (method == "additive") {
      tmp <- Refine_H_i(i, Ymat, X_tensor, W, member_G, member_H, taus, theta, G, H, additive = TRUE, FriendW, FriendW2)
    } else if (method == "multiplicative") {
      tmp <- Refine_H_i(i, Ymat, X_tensor, W, member_G, member_H, taus, theta, G, H, additive = FALSE, FriendW, FriendW2)
    } else if (method == "general") {
      tmp <- Refine_H_i_general(i, Ymat, X_tensor, W, member_G, member_H, taus, theta, G, H, FriendW, FriendW2)
    }
    tmp
  }

  close(pb)
  parallel::stopCluster(cl)

  for (i in 1:N) {
    tmp <- res_all[[i]]
    loss_old[i] <- tmp$loss_old
    loss_new[i] <- tmp$loss_new
    h_r[i] <- tmp$h_r
  }

  list(loss_old = loss_old, h_r = h_r, loss_new = loss_new)
}


# =============================================================================
# Per-node refinement helpers (additive & multiplicative)
# =============================================================================
Refine_G_i <- function(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,additive,G,H){
  Time = ncol(Ymat)
  N = nrow(Ymat)
  Ymat_lag = Ymat[,-Time]
  Ymat1 = Ymat[,-1]
  Gloss <- numeric()

  fri_i = which(W[i,]!=0)

  theta_Gs = theta$theta_Gs
  beta = theta$beta
  alpha = theta$alpha
  Loss.fun <- theta$Loss.fun

  updatei <- fri_i
  di = length(updatei)

  alphaWs <- eps_mats <- list()



  for (j in 1:G) {
    gi <- j
    fri_lagi <- rep(NA,N)
    fri_lagi[fri_i] <- member_H[fri_i]

    if(di>0){

      if (length(fri_i)>1)
        Ymat_lag_fri =  t(Ymat_lag[fri_i,]) else
          Ymat_lag_fri = matrix(Ymat_lag[fri_i,], ncol = 1)

        # Pre-compute residuals using candidate group gi parameters
        for (k in 1:length(taus)){
          alpha_i = alpha[gi, k]
          thetaG = theta_Gs[[k]]
          alphaWs[[k]] = alpha_i*W[i,fri_i]
          nu_i = thetaG[,2][gi]
          gamma_i = thetaG[gi,-(1:2)]
          eps_i = Ymat[i,-1] - nu_i*Ymat[i,-Time] - colSums(X_tensor[i,,]*gamma_i)
          eps_mats[[k]] = eps_i
        }

        if(additive){
          update_Hi <- function(member_H_i,Ymat_lag_fri,eps_mats,alphaWs,beta,H,member_H,updatei,fri_i,taus) {
            member_Hi <- member_H
            member_Hi[updatei] <- c(member_H_i+1)
            member_Hi <- member_Hi[fri_i]
            fval <- 0
            for (k in 1:length(taus)){
              Resi_pred = Ymat_lag_fri%*%(beta[member_Hi,k]/length(fri_i)+alphaWs[[k]])
              fval = fval + sum(Loss.fun(eps_mats[[k]]-Resi_pred, taus[k]))
            }
            fval
          }
        }else{

          update_Hi <- function(member_H_i,Ymat_lag_fri,eps_mats,alphaWs,beta,H,member_H,updatei,fri_i,taus) {
            member_Hi <- member_H
            member_Hi[updatei] <- c(member_H_i+1)
            member_Hi <- member_Hi[fri_i]
            fval <- 0
            for (k in 1:length(taus)){
              Resi_pred = Ymat_lag_fri%*%(beta[member_Hi,k]*alphaWs[[k]])
              fval = fval + sum(Loss.fun(eps_mats[[k]]-Resi_pred, taus[k]))
            }
            fval
          }
        }

        ft_new <- function(member_Hi){
          update_Hi(member_Hi,Ymat_lag_fri,eps_mats,alphaWs,beta,H,member_H,updatei,fri_i,taus)
        }

        ###enumerate methods
        if(H^di<=10000){
          l <- rep(list(1:H), di)
          member_H_i <- as.matrix(expand.grid(l))-1
          fval_new <-  apply(member_H_i, 1, ft_new)
          fri_lagi[updatei] <- member_H_i[which.min(fval_new),]+1
          Gloss[j] <- min(fval_new)

        }else{
          obj_i <- CEoptim(f = update_Hi, f.arg = list(Ymat_lag_fri=Ymat_lag_fri,eps_mats=eps_mats,alphaWs=alphaWs,beta=beta,H=H,member_H=member_H,updatei=updatei,fri_i=fri_i,taus=taus),discrete = list(categories = as.integer(rep(H,length(updatei))),smoothProb=0.5),N = 1000, rho = 0.001, verbose = F,noImproveThr=2)
          fri_lagi[updatei] <- obj_i$optimizer$discrete+1
          Gloss[j] <- obj_i$optimum
        }



    }
    ##Compute the loss using un-refined memberships
    if(gi==member_G[i]) {
      loss_old = ft_new(member_H[updatei]-1)
    }
  }
  list(Gloss=Gloss,g_r=which.min(Gloss),loss_old=loss_old,loss_new=min(Gloss))
}


###Refine G-membership for general models
Refine_G_i_general <- function(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,G,H){
  Time = ncol(Ymat)
  N = nrow(Ymat)
  Ymat_lag = Ymat[,-Time]
  Ymat1 = Ymat[,-1]
  Gloss <- numeric()

  fri_i = which(W[i,]!=0)

  theta_Gs = theta$theta_Gs
  alphabeta_GHs <- theta$alphabeta_GHs
  Loss.fun <- theta$Loss.fun

  updatei <- fri_i
  di = length(updatei)

  alphaWs <- eps_mats <- list()



  for (j in 1:G) {
    gi <- j
    fri_lagi <- rep(NA,N)
    fri_lagi[fri_i] <- member_H[fri_i]

    if(di>0){

      if (length(fri_i)>1)
        Ymat_lag_fri =  t(Ymat_lag[fri_i,]) else
          Ymat_lag_fri = matrix(Ymat_lag[fri_i,], ncol = 1)

        for (k in 1:length(taus)){
          thetaG = theta_Gs[[k]]
          nu_i = thetaG[,1][gi]
          gamma_i = thetaG[gi,-1]
          eps_i = Ymat[i,-1] - nu_i*Ymat[i,-Time] - colSums(X_tensor[i,,]*gamma_i)
          eps_mats[[k]] = eps_i
        }

        update_Hi <- function(member_H_i,Ymat_lag_fri,eps_mats,alphabeta_GHs,gi,H,member_H,updatei,fri_i,taus) {
          member_Hi <- member_H
          member_Hi[updatei] <- c(member_H_i+1)
          member_Hi <- member_Hi[fri_i]
          fval <- 0
          for (k in 1:length(taus)){
            alphabetas = alphabeta_GHs[[k]][gi,]
            Resi_pred = Ymat_lag_fri%*%(alphabetas[member_Hi]/length(fri_i))
            fval = fval + sum(Loss.fun(eps_mats[[k]]-Resi_pred, taus[k]))
          }
          fval
        }

        ft_new <- function(member_Hi){
          update_Hi(member_Hi,Ymat_lag_fri,eps_mats,alphabeta_GHs,gi,H,member_H,updatei,fri_i,taus)
        }

        ###enumerate methods
        if(H^di<=10000){
          l <- rep(list(1:H), di)
          member_H_i <- as.matrix(expand.grid(l))-1
          fval_new <-  apply(member_H_i, 1, ft_new)
          fri_lagi[updatei] <- member_H_i[which.min(fval_new),]+1
          Gloss[j] <- min(fval_new)

        }else{
          obj_i <- CEoptim(f = update_Hi, f.arg = list(Ymat_lag_fri=Ymat_lag_fri,eps_mats=eps_mats,alphabeta_GHs=alphabeta_GHs,gi=gi,H=H,member_H=member_H,updatei=updatei,fri_i=fri_i,taus=taus),discrete = list(categories = as.integer(rep(H,length(updatei))),smoothProb=0.5),N = 1000, rho = 0.001, verbose = F,noImproveThr=2)
          fri_lagi[updatei] <- obj_i$optimizer$discrete+1
          Gloss[j] <- obj_i$optimum
        }



    }
    ##Compute the loss using un-refined memberships
    if(gi==member_G[i]) {
      loss_old = ft_new(member_H[updatei]-1)
    }
  }

  list(Gloss=Gloss,g_r=which.min(Gloss),loss_old=loss_old,loss_new=min(Gloss))
}


###Refine H-membership for additive and multiplicative models
Refine_H_i <- function(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,G,H,additive,FriendW,FriendW2){

  ###Import estimated parameters
  N = nrow(Ymat);
  Time = ncol(Ymat)
  Ymat_lag = Ymat[,-Time]

  theta_Gs = theta$theta_Gs
  beta = theta$beta
  alpha = theta$alpha
  Loss.fun <- theta$Loss.fun


  ##Extract all friend memberships
  fri_i <- FriendW[[i]]
  fri_i2 <- FriendW2[[i]]
  updatei <- setdiff(fri_i2,i)
  di <- length(updatei)
  member_Gi <- member_G[fri_i]
  Wi <- W[fri_i,fri_i2]


  if(di>0){

    Ymat_lag_fri2 =  Ymat_lag[fri_i2,,drop=F]

    eps_mats = list()

    for (k in 1:length(taus)){
      thetaG = theta_Gs[[k]]
      nu_all = thetaG[,2][member_G]
      gamma_all = thetaG[member_G,-(1:2),drop=F]
      eps_mat = Ymat[,-1,drop=F] - nu_all*Ymat[,-Time,drop=F] -
        matrix(rowSums(matrix(aperm(X_tensor,c(1,3,2)),nrow = N*(Time-1)) * gamma_all[rep(1:N, Time-1),]), nrow = N)
      eps_mats[[k]] = eps_mat
    }

    Hloss <- numeric(H)
    for(j in 1:H){
      hi <- j


      if(additive){
        update_Hi <- function(member_H_i,hi,Ymat_lag_fri2,Wi,eps_mats,alpha,beta,member_Gi,member_H,updatei,fri_i,fri_i2,taus) {
          member_Hi <- member_H
          member_Hi[updatei] <- c(member_H_i+1)
          member_Hi[i] <- hi
          member_Hi <- member_Hi[fri_i2]

          fval <- 0
          for (k in 1:length(taus)){
            alphabetai = outer(alpha[member_Gi,k], beta[member_Hi,k],"+")

            Resi_pred =(alphabetai*Wi)%*%Ymat_lag_fri2
            fval = fval + sum(Loss.fun((eps_mats[[k]])[fri_i,]-Resi_pred, taus[k]))
          }
          fval
        }
      }else{

        update_Hi <- function(member_H_i,hi,Ymat_lag_fri2,Wi,eps_mats,alpha,beta,member_Gi,member_H,updatei,fri_i,fri_i2,taus) {
          member_Hi <- member_H
          member_Hi[updatei] <- c(member_H_i+1)
          member_Hi[i] <- hi
          member_Hi <- member_Hi[fri_i2]

          fval <- 0
          for (k in 1:length(taus)){
            alphabetai = outer(alpha[member_Gi,k], beta[member_Hi,k],"*")

            Resi_pred =(alphabetai*Wi)%*%Ymat_lag_fri2
            fval = fval + sum(Loss.fun((eps_mats[[k]])[fri_i,]-Resi_pred, taus[k]))
          }
          fval
        }

      }


      ft_new <- function(member_Hi){
        update_Hi(member_Hi,hi,Ymat_lag_fri2,Wi,eps_mats,alpha,beta,member_Gi,member_H,updatei,fri_i,fri_i2,taus)
      }

      ###enumerate methods
      if(H^di<=10000){
        l <- rep(list(1:H), di)
        member_H_i <- as.matrix(expand.grid(l))-1
        fval_new <-  apply(member_H_i, 1, ft_new)
        Hloss[j] <- min(fval_new)

      }else{
        obj_i <- CEoptim(f = update_Hi, f.arg = list(hi=hi,Ymat_lag_fri2=Ymat_lag_fri2,Wi=Wi,eps_mats=eps_mats,alpha=alpha,beta=beta,member_Gi=member_Gi,member_H=member_H,updatei=updatei,fri_i=fri_i,fri_i2=fri_i2,taus=taus),discrete = list(categories = as.integer(rep(H,length(updatei))),smoothProb=0.5),N = 1000, rho = 0.001, verbose = F,noImproveThr=2)
        Hloss[j] <- obj_i$optimum
      }

      ##Compute the loss using un-refined memberships

      if(hi==member_H[i]) {
        loss_old = ft_new(member_H[updatei]-1)
      }
    }

    h_r=which.min(Hloss)
  }else{

    Hloss=numeric(H)+NA
    loss_old=NA
    h_r=NA
  }
  list(Hloss=Hloss,h_r=h_r,loss_old=loss_old,loss_new=min(Hloss))
}

###Refine H-membership for general models
Refine_H_i_general <- function(i,Ymat,X_tensor,W,member_G, member_H, taus,theta,G,H,FriendW,FriendW2){

  ###Import estimated parameters
  N = nrow(Ymat);
  Time = ncol(Ymat)
  Ymat_lag = Ymat[,-Time]

  theta_Gs = theta$theta_Gs
  alphabetas = theta$alphabeta_GHs
  Loss.fun <- theta$Loss.fun


  ##Extract all friend memberships
  fri_i <- FriendW[[i]]
  fri_i2 <- FriendW2[[i]]
  updatei <- setdiff(fri_i2,i)
  di <- length(updatei)
  member_Gi <- member_G[fri_i]
  Wi <- W[fri_i,fri_i2]


  if(di>0){

    Ymat_lag_fri2 =  Ymat_lag[fri_i2,,drop=F]

    eps_mats = list()

    for (k in 1:length(taus)){
      thetaG = theta_Gs[[k]]
      nu_all = thetaG[,1][member_G]
      gamma_all = thetaG[member_G,-1,drop=F]
      eps_mat = Ymat[,-1,drop=F] - nu_all*Ymat[,-Time,drop=F] -
        matrix(rowSums(matrix(aperm(X_tensor,c(1,3,2)),nrow = N*(Time-1)) * gamma_all[rep(1:N, Time-1),]), nrow = N)
      eps_mats[[k]] = eps_mat
    }

    Hloss <- numeric(H)
    for(j in 1:H){
      hi <- j

      update_Hi <- function(member_H_i,hi,Ymat_lag_fri2,Wi,eps_mats,alphabetas,member_Gi,member_H,updatei,fri_i,fri_i2,taus) {
        member_Hi <- member_H
        member_Hi[updatei] <- c(member_H_i+1)
        member_Hi[i] <- hi
        member_Hi <- member_Hi[fri_i2]

        fval <- 0
        for (k in 1:length(taus)){
          alphabetai = alphabetas[[k]][member_Gi,member_Hi]

          Resi_pred =(alphabetai*Wi)%*%Ymat_lag_fri2
          fval = fval + sum(Loss.fun((eps_mats[[k]])[fri_i,]-Resi_pred, taus[k]))
        }
        fval
      }


      ft_new <- function(member_Hi){
        update_Hi(member_Hi,hi,Ymat_lag_fri2,Wi,eps_mats,alphabetas,member_Gi,member_H,updatei,fri_i,fri_i2,taus)
      }

      ###enumerate methods
      if(H^di<=10000){
        l <- rep(list(1:H), di)
        member_H_i <- as.matrix(expand.grid(l))-1
        fval_new <-  apply(member_H_i, 1, ft_new)
        Hloss[j] <- min(fval_new)

      }else{
        obj_i <- CEoptim(f = update_Hi, f.arg = list(hi=hi,Ymat_lag_fri2=Ymat_lag_fri2,Wi=Wi,eps_mats=eps_mats,alphabetas=alphabetas,member_Gi=member_Gi,member_H=member_H,updatei=updatei,fri_i=fri_i,fri_i2=fri_i2,taus=taus),discrete = list(categories = as.integer(rep(H,length(updatei))),smoothProb=0.5),N = 1000, rho = 0.001, verbose = F,noImproveThr=2)
        Hloss[j] <- obj_i$optimum
      }

      ##Compute the loss using un-refined memberships

      if(hi==member_H[i]) {
        loss_old = ft_new(member_H[updatei]-1)
      }
    }

    h_r=which.min(Hloss)

  }
  else{

    Hloss=numeric(H)+NA
    loss_old=NA
    h_r=NA
  }
  list(Hloss=Hloss,h_r=h_r,loss_old=loss_old,loss_new=min(Hloss))


}







# =============================================================================
# CONFIDENCE-INTERVAL VARIANTS
# These mirror the main parameter-update functions, but additionally extract
# 95% normal-based CIs from quantreg::summary(... se="nid").
# =============================================================================

# CI version: multiplicative model
twmq.estimate_thetaGH.member.ci <-function(Ymat, X_tensor, W, member_G, member_H,
                                           beta, taus, conquer, h_conquer)
{
  G = max(member_G); H = max(member_H)
  Time = ncol(Ymat); N = nrow(Ymat)# ; p = ncol(X)
  p = dim(X_tensor)[2]

  # if beta = NULL, then we need to estimate beta first (using the first group G = 1)
  if (is.null(beta))
  {
    tmp <- table(member_G)
    g_max <- names(tmp)[which.max(tmp)]
    ind_g = which(member_G == g_max)

    N_g1 = length(ind_g)
    Ymat_g1 = Ymat[ind_g,,drop=F]
    W_g1 = W[ind_g,,drop=F]

    Zh_g1 = matrix(0, N_g1*(Time - 1), ncol = H)
    for (h in 1:H)
    {
      ind_h = which(member_H==h)
      Zh_g1[,h] = as.vector(W_g1[,ind_h,drop=F]%*%Ymat[ind_h,-Time,drop=F])
    }
    Z2_g1 = as.vector(Ymat[ind_g,-Time,drop=F])
    Z_all_g1 = cbind(Zh_g1, Z2_g1,
                     matrix(aperm(X_tensor[ind_g,,],c(1,3,2)),nrow = length(ind_g)*(Time-1)))
    Y_g1 = as.vector(Ymat[ind_g,-1,drop=F])


    if (conquer){
      fit.conquer = conquer.process(Z_all_g1, Y_g1, tauSeq = taus)
      theta_init = fit.conquer$coeff
    }else{
      dat = data.frame(Y = Y_g1, Z_all_g1)
      resrq = quantreg::rq(Y~.-1, taus , data = dat)
      theta_init = resrq$coefficients
    }

    if(length(taus)==1) beta=as.matrix(theta_init) else
      beta = theta_init[,,drop=F][1:H,]
  }


  # given beta estimate alpha
  alpha = matrix(0, G, length(taus))
  alpha_up = matrix(0, G, length(taus))
  alpha_low = matrix(0, G, length(taus))
  theta_Gs = list()
  theta_Gs_up = list()
  theta_Gs_low = list()
  eps_mats = list()

  for (k in 1:length(taus)){
    tau = taus[k]
    beta_all = beta[member_H,k]
    Wbeta = W%*%diag(beta_all)

    Z1 = Wbeta%*%Ymat[,-Time]
    Z2 = Ymat[,-Time]
    thetaG = matrix(0, nrow = G, ncol = p+2)
    thetaG_up = matrix(0, nrow = G, ncol = p+2)
    thetaG_low = matrix(0, nrow = G, ncol = p+2)
    eps_mat = matrix(0, N, Time - 1)

    for (g in 1:G)
    {
      ind_g = which(member_G == g)
      Z_all = cbind(as.vector(Z1[ind_g,,drop=F]),
                    as.vector(Z2[ind_g,,drop=F]),
                    matrix(aperm(X_tensor[ind_g,,],c(1,3,2)),nrow = length(ind_g)*(Time-1)))
      Y_vec = as.vector(Ymat[ind_g, -1,drop=F])

      if (conquer){
      }else{
        dat = data.frame(Y = Y_vec, Z_all)
        resrq = quantreg::rq(Y~.-1, tau,data = dat)
        thetaG[g,] = resrq$coefficients
        summ = summary(resrq, se="nid")$coefficients[,2]
        thetaG_up[g,] =  thetaG[g,] + 1.96 * summ
        thetaG_low[g,] = thetaG[g,] - 1.96 * summ
        Loss.fun <- check.func
      }

      eps_mat[ind_g,] = matrix(Y_vec - Z_all[,-1]%*%thetaG[g,-1],
                               nrow = length(ind_g))
    }
    eps_mats[[k]] = eps_mat
    theta_Gs[[k]] = thetaG
    theta_Gs_up[[k]] = thetaG_up
    theta_Gs_low[[k]] = thetaG_low
    alpha[,k] = thetaG[,1]
    alpha_up[,k] = thetaG_up[,1]
    alpha_low[,k] = thetaG_low[,1]
  }

  # given alpha estimate beta
  beta_new = matrix(0, H, length(taus))
  beta_new_up = matrix(0, H, length(taus))
  beta_new_low = matrix(0, H, length(taus))
  loss = rep(0, length(taus))

  for (k in 1:length(taus)){
    tau = taus[k]
    eps_mat = eps_mats[[k]]

    eps_vec = as.vector(eps_mat)
    alpha_all = alpha[member_G, k]
    alphaW = diag(alpha_all)%*%W

    Zh = matrix(0, N*(Time - 1), ncol = H)
    for (h in 1:H)
    {
      ind_h = which(member_H==h)
      Zh[,h] = as.vector(alphaW[,ind_h,drop=F]%*%Ymat[ind_h,-Time,drop=F])
    }
    if (conquer){
    }else{
      dat = data.frame(Y = eps_vec, Zh)
      resrq = quantreg::rq(Y~.-1, tau , data = dat)
      beta_new[,k] = resrq$coefficients
      summ = summary(resrq, se="nid")$coefficients[,2]
      beta_new_up[,k] = beta_new[,k] + 1.96 * summ
      beta_new_low[,k] = beta_new[,k]- 1.96 * summ
      resi = resrq$residuals
    }

    loss[k] = sum(Loss.fun(resi, tau))
  }

  # adjust the values of alpha and beta to make alpha[1] = 1
  for (k in 1:length(taus)){
    alpha1 = alpha[1,k]
    alpha[,k] = alpha[,k]/alpha1
    alpha_up[,k] = alpha_up[,k]/alpha1
    alpha_low[,k] = alpha_low[,k]/alpha1
    theta_Gs[[k]][,1] = theta_Gs[[k]][,1]/alpha1
    theta_Gs_up[[k]][,1] = theta_Gs_up[[k]][,1]/alpha1
    theta_Gs_low[[k]][,1] = theta_Gs_low[[k]][,1]/alpha1
    beta_new[,k] = beta_new[,k]*alpha1
    beta_new_up[,k] = beta_new_up[,k]*alpha1
    beta_new_low[,k] = beta_new_low[,k]*alpha1
  }

  ci_list = list()
  ci_list$point = list(theta_Gs = theta_Gs, alpha = alpha,
                       beta = beta_new, loss = loss,
                       Loss.fun=Loss.fun)
  ci_list$up = list(theta_Gs = theta_Gs_up, alpha = alpha_up,
                    beta = beta_new_up)
  ci_list$low = list(theta_Gs = theta_Gs_low, alpha = alpha_low,
                     beta = beta_new_low)
  return(ci_list)
}

# CI version: additive model
twmq.estimate_thetaGH.member.additive.ci <- function(Ymat, X_tensor, W, member_G, member_H, taus, conquer, h_conquer)
{
  G = max(member_G); H = max(member_H)
  Time = ncol(Ymat); N = nrow(Ymat)
  p = dim(X_tensor)[2]

  Y1 <- Ymat[,-Time]
  Y <- Ymat[,-1]
  WY1 <- W%*%Y1
  Z1 <- matrix(0,N*(Time-1),G)
  Z2 <- matrix(0,N*(Time-1),G)
  X1 <- matrix(0,N*(Time-1),G*p)
  Zh <- matrix(0,N*(Time-1),H)

  Yh <- list()
  for (h in 1:H) {
    Yh[[h]] = W%*%(Y1*(member_H==h))
  }

  Y_all <-NULL

  count.row <- 0
  for (g in 1:G)
  {
    ind_g = which(member_G == g)
    Y_g=Y[ind_g,,drop=F]
    Y1_g=Y1[ind_g,,drop=F]
    tmp0 <- as.vector(Y1_g)
    n_g <- length(tmp0)
    Y_all <- c(Y_all,as.vector(Y[ind_g,,drop=F]))
    Z1[1:n_g+count.row,g] <- tmp0
    Z2[1:n_g+count.row,g] <- as.vector(WY1[ind_g,,drop=F])
    for (h in 1:H) {
      Zh[1:n_g+count.row,h] <- as.vector((Yh[[h]])[ind_g,,drop=F])
    }

    X1[1:n_g+count.row,(g-1)*p+(1:p)] = matrix(aperm(X_tensor[ind_g,,],c(1,3,2)),nrow = length(ind_g)*(Time-1))
    count.row <- count.row+n_g
  }

  Z_all <- cbind(Z2[,-1,drop=T],Z1,X1[,,drop=T],Zh) ##Set alpha[1,1]=0

  if (conquer){
  }else{
    dat = data.frame(Y = Y_all, Z_all)
    resrq = quantreg::rq(Y~.-1, taus , data = dat)
    theta = resrq$coefficients
    summ =summary(resrq, se="nid")
    theta_up = cbind(sapply(summ, function(lis){lis$coefficients[,1] + 1.96 * lis$coefficients[,2]}))
    theta_low = cbind(sapply(summ, function(lis){lis$coefficients[,1]-  1.96 * lis$coefficients[,2]}))

    Loss.fun <- check.func
  }

  theta <-as.matrix(theta)
  theta_up <-as.matrix(theta_up)
  theta_low <-as.matrix(theta_low)

  resi<- Y_all-Z_all%*%theta
  alpha = matrix(0, G, length(taus))
  alpha_up = matrix(0, G, length(taus))
  alpha_low = matrix(0, G, length(taus))
  theta_Gs = list()
  theta_Gs_up = list()
  theta_Gs_low = list()
  beta_new = matrix(0, H, length(taus))
  beta_new_up = matrix(0, H, length(taus))
  beta_new_low = matrix(0, H, length(taus))

  for (k in 1:length(taus)){
    thetaG = matrix(0, nrow = G, ncol = p+2)
    thetaG_up = matrix(0, nrow = G, ncol = p+2)
    thetaG_low = matrix(0, nrow = G, ncol = p+2)
    thetaG[-1,1] <- theta[1:(G-1),k]  ##
    thetaG[,2] <- theta[G:(2*G - 1),k]  ##
    thetaG[,1:p+2] <-  matrix(theta[(2*G):(2*G+G*p-1),k],G,p,byrow = TRUE)  ##
    theta_Gs[[k]] = thetaG
    alpha[,k] = thetaG[,1]
    beta_new[,k] =theta[-(1:((p+2)*G-1)),k]

    thetaG_up[-1,1] <- theta_up[1:(G-1),k]  ##
    thetaG_up[,2] <- theta_up[G:(2*G - 1),k]  ##
    thetaG_up[,1:p+2] <-  matrix(theta_up[(2*G):(2*G+G*p-1),k],G,p,byrow = TRUE)  ##
    theta_Gs_up[[k]] = thetaG_up
    alpha_up[,k] = thetaG_up[,1]
    beta_new_up[,k] =theta_up[-(1:((p+2)*G-1)),k]

    thetaG_low[-1,1] <- theta_low[1:(G-1),k]  ##
    thetaG_low[,2] <- theta_low[G:(2*G - 1),k]  ##
    thetaG_low[,1:p+2] <-  matrix(theta_low[(2*G):(2*G+G*p-1),k],G,p,byrow = TRUE)  ##
    theta_Gs_low[[k]] = thetaG_low
    alpha_low[,k] = thetaG_low[,1]
    beta_new_low[,k] =theta_low[-(1:((p+2)*G-1)),k]
  }


  # compute the loss
  loss = rep(0, length(taus))

  for (k in 1:length(taus)){
    tau = taus[k]
    loss[k] = sum(Loss.fun(resi[,k], tau))
  }

  ci_list = list()
  ci_list$point = list(theta_Gs = theta_Gs, alpha = alpha,
                       beta = beta_new, loss = loss,
                       Loss.fun=Loss.fun)
  ci_list$up = list(theta_Gs = theta_Gs_up, alpha = alpha_up,
                    beta = beta_new_up)
  ci_list$low = list(theta_Gs = theta_Gs_low, alpha = alpha_low,
                     beta = beta_new_low)
  return(ci_list)
}

# CI version: general model
twmq.estimate_thetaGH.member.general.ci <- function(Ymat, X_tensor, W, member_G, member_H, taus, conquer, h_conquer)
{
  G = max(member_G); H = max(member_H)
  Time = ncol(Ymat); N = nrow(Ymat)#; p = ncol(X)
  p = dim(X_tensor)[2]

  Y1 <- Ymat[,-Time]
  Y <- Ymat[,-1]
  WY1 <- W%*%Y1


  Yh <- list()
  for (h in 1:H) {
    Yh[[h]] = W%*%(Y1*(member_H==h))
  }

  ##Set up the parameter matrix
  alphabeta_GHs <-   theta_Gs  <-  list()
  alphabeta_GHs_up <-   theta_Gs_up  <-  list()
  alphabeta_GHs_low <-   theta_Gs_low  <-  list()
  resi <- NULL
  for (k in 1:length(taus)){
    theta_Gs[[k]] = matrix(0, nrow = G, ncol = p+1)
    theta_Gs_up[[k]] = matrix(0, nrow = G, ncol = p+1)
    theta_Gs_low[[k]] = matrix(0, nrow = G, ncol = p+1)
    alphabeta_GHs[[k]] = matrix(0, nrow = G, ncol =H)
    alphabeta_GHs_up[[k]] = matrix(0, nrow = G, ncol =H)
    alphabeta_GHs_low[[k]] = matrix(0, nrow = G, ncol =H)
  }


  for (g in 1:G)
  {
    ind_g = which(member_G == g)
    Y_g=Y[ind_g,,drop=F]
    Y1_g=Y1[ind_g,,drop=F]
    Z1 <- as.vector(Y1_g)
    Y_all <- as.vector(Y[ind_g,,drop=F])
    Zh <- NULL
    for (h in 1:H) {
      Zh <- cbind(Zh,as.vector((Yh[[h]])[ind_g,,drop=F]))
    }

    X1 = matrix(aperm(X_tensor[ind_g,,],c(1,3,2)),nrow = length(ind_g)*(Time-1))
    Z_all <- cbind(Z1,X1[,,drop=T],Zh)

    if (conquer){

    }else{
      dat = data.frame(Y = Y_all, Z_all)
      resrq = quantreg::rq(Y~.-1, taus , data = dat)
      theta = resrq$coefficients

      summ =summary(resrq, se="nid")
      theta_up = cbind(sapply(summ, function(lis){lis$coefficients[,1] + 1.96 * lis$coefficients[,2]}))
      theta_low = cbind(sapply(summ, function(lis){lis$coefficients[,1]-  1.96 * lis$coefficients[,2]}))

      Loss.fun <- check.func
    }



    theta <-as.matrix(theta)
    theta_up <-as.matrix(theta_up)
    theta_low <-as.matrix(theta_low)


    resi<- rbind(resi,Y_all-Z_all%*%theta)
    # given beta estimate alpha

    for (k in 1:length(taus)){
      theta_Gs[[k]][g,1:(p+1)] <- theta[1:(p+1),k]
      theta_Gs_up[[k]][g,1:(p+1)] <- theta_up[1:(p+1),k]
      theta_Gs_low[[k]][g,1:(p+1)] <- theta_low[1:(p+1),k]
      alphabeta_GHs[[k]][g,] <- theta[-(1:(1+p)),k]
      alphabeta_GHs_up[[k]][g,] <- theta_up[-(1:(1+p)),k]
      alphabeta_GHs_low[[k]][g,] <- theta_low[-(1:(1+p)),k]
    }
  }




  # compute the loss
  loss = rep(0, length(taus))

  for (k in 1:length(taus)){
    tau = taus[k]
    loss[k] = sum(Loss.fun(resi[,k], tau))
  }


  ci_list = list()
  ci_list$point = list(theta_Gs = theta_Gs, alphabeta_GHs = alphabeta_GHs,loss = loss,
                       Loss.fun=Loss.fun)
  ci_list$up=list(theta_Gs = theta_Gs_up, alphabeta_GHs = alphabeta_GHs_up)
  ci_list$low=list(theta_Gs = theta_Gs_low, alphabeta_GHs = alphabeta_GHs_low)
  return(ci_list)
}

# Top-level CI dispatcher
#' @export
twmq_ci <- function(Ymat, X_tensor, W,
                                                    member_G, member_H,
                                                    taus, conquer,h_conquer,
                                                    theta=NULL,method, n_iter.max = 2, verbose = F)
{

  if(method=="general"){
    theta_ci = twmq.estimate_thetaGH.member.general.ci(Ymat, X_tensor, W, member_G, member_H, taus,
                                                       conquer,h_conquer)
  }

  if(method=="additive"){
    theta_ci = twmq.estimate_thetaGH.member.additive.ci(Ymat, X_tensor, W, member_G, member_H, taus,
                                                        conquer,h_conquer)
  }

  if(method=="multiplicative"){
    theta_ci = twmq.estimate_thetaGH.member.ci(Ymat, X_tensor, W,
                                               member_G, member_H,
                                               beta = theta$beta, taus,conquer,h_conquer)
    del = 1; n_iter = 1; converge = T
    while(del>10^{-3} & n_iter<n_iter.max)
    {
      if (converge){
        theta_new_ci = twmq.estimate_thetaGH.member.ci(Ymat, X_tensor, W,
                                                       member_G, member_H,
                                                       beta = theta_ci$point$beta, taus,
                                                       conquer,h_conquer)
      }else{
        theta_new_ci = twmq.estimate_thetaGH.member.ci(Ymat, X_tensor, W,
                                                       member_G, member_H,
                                                       beta = theta_ci$point$beta, taus,
                                                       conquer = F,h_conquer)
      }
      del = max(RMSE_L(theta_new_ci$point$theta_Gs, theta_ci$point$theta_Gs, ratio = T),
                RMSE((theta_new_ci$point$beta - theta_ci$point$beta)/(10^(-3)+abs(theta_ci$point$beta)),
                     0))
      if (verbose){
        cat(n_iter, del, "\n")
      }
      idx <- which(theta_ci$point$loss<(theta_new_ci$point$loss*0.95))

      if(length(idx)>0){
        # Roll back any quantile that worsened
        converge = F
        for(k in idx){
          theta_new_ci$point$theta_Gs[[k]]=  theta_ci$point$theta_Gs[[k]]
          theta_new_ci$point$alpha[,k] =theta_ci$point$alpha[,k]
          theta_new_ci$point$beta[,k] =theta_ci$point$beta[,k]
          theta_new_ci$point$loss[k] <- theta_ci$point$loss[k]

          theta_new_ci$up$theta_Gs[[k]]=  theta_ci$up$theta_Gs[[k]]
          theta_new_ci$up$alpha[,k] =theta_ci$up$alpha[,k]
          theta_new_ci$up$beta[,k] =theta_ci$up$beta[,k]

          theta_new_ci$low$theta_Gs[[k]]=  theta_ci$low$theta_Gs[[k]]
          theta_new_ci$low$alpha[,k] =theta_ci$low$alpha[,k]
          theta_new_ci$low$beta[,k] =theta_ci$low$beta[,k]
        }

      }
      theta_ci = theta_new_ci
      n_iter = n_iter + 1
      if (verbose)
        cat(n_iter, del, "\n")
    }
    theta_ci$point$converge = converge
  }

  return(theta_ci)
}

# =============================================================================
# myconquer
# Wrapper around conquer.process that:
#   1) auto-detects an intercept (all-ones) column in the design matrix,
#   2) removes it before calling conquer.process (which adds its own intercept),
#   3) reorders the returned coefficient matrix so it lines up with the
#      original column order of X.
# =============================================================================
myconquer <- function(X,y,tauSeq) {
  # Convert to matrix for safe indexing
  X <- as.matrix(X)

  # Detect columns whose entries are all 1 (with tolerance for rounding)
  is_intercept_col <- function(col, tol = 1e-8) {
    all(abs(col - 1) < tol)
  }
  intercept_idx <- which(apply(X, 2, is_intercept_col))

  # If multiple all-ones columns were found, only use the first
  if (length(intercept_idx) > 1) {
    warning("Multiple all-ones columns detected; using only the first as intercept.")
    intercept_idx <- intercept_idx[1]
  }

  # No intercept column: pass X through unchanged
  if (length(intercept_idx) == 0) {
    fit <- conquer::conquer.process(Y = y, X = X,tauSeq = tauSeq, kernel = "triangular")
    return(fit)
  }

  # Has intercept column: drop it, since conquer.process supplies its own
  X_no_int <- X[, -intercept_idx, drop = FALSE]

  fit <- conquer::conquer.process(Y = y, X = X_no_int,tauSeq = tauSeq, kernel = "triangular")

  # ---- Reorder coef so it matches the original column order of X ----
  # conquer.process layout:
  #   row 1            : intercept
  #   rows 2..(p0+1)   : columns of X_no_int
  beta_proc <- fit$coef
  if (is.null(beta_proc)) {
    warning("conquer.process did not return $coef; cannot reorder coefficients.")
    return(fit)
  }

  p_full <- ncol(X)
  p_proc <- nrow(beta_proc)

  if (p_proc != p_full) {
    warning(sprintf(
      "coef has %d rows but original X has %d columns; check model spec",
      p_proc, p_full
    ))
  }

  # Build the reordered coefficient matrix matching the original X column order
  coef_full <- matrix(NA_real_, nrow = p_full, ncol = ncol(beta_proc))

  # Place conquer.process's intercept at the original intercept column
  rownames(coef_full) <- colnames(X)
  colnames(coef_full) <- colnames(beta_proc)
  coef_full[intercept_idx, ] <- beta_proc[1, , drop = FALSE]

  # Fill remaining rows in original order
  coef_full[-intercept_idx, ] <- beta_proc[-1, , drop = FALSE]

  fit$coef_original <- beta_proc
  fit$coeff <- coef_full

  return(fit)
}
