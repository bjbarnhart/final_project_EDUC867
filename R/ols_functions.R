# OLS regression using matrix algebra with fitting and prediction functions
# Author: Brendan Barnhart
# Created for EDUC867 Assignment 4 (10/11/2026)

#' This function will calculate all OLS regression output with matrix algebra.
#'  
#' @param df          data.frame containing the response variable and all 
#'                    predictor variables
#' @param predictors  character vector of predictor column names 
#'                    c("parent", "peer")
#' @param response    single string that is the name of the numeric outcome 
#'                    column
#' 
#' @return returns a list with:
#'  coefficients      data.frame; p rows: Estimate, Std. Error, 
#'                    t value, Pr(>|t|)
#'  model_stats       1-row data.frame: R2, Adj_R2, Sigma2, Sigma, F_stat, P_Val_F,
#'                    df_model, df_resid
#'  fitted.values     numeric vector, length n
#'  residuals         numeric vector, length n
#'  sum_of_squares    numeric vector: SSR, SSE, SST
#'  
#' @details Error Failures and Assumptions:
#'  1. no NA in the response or predictors (no listwise deletion is performed)
#'  2. X has full column rank
#'  3. The intercept is always included and predictors are numeric
#'
#'  All OLS assumptions (e.g., homoscedasticity, normal errors) are the 
#'  responsibility of the user 
#'  
#' @examples 
#' fit <- fit_ols(motivation, c("parent","sibling","peer"), "motivation")
#' fit$coefficients

fit_ols = function(df, predictors, response){
  
  # Loud failure 1: no missing data allowed
  if(anyNA(df[, c(response, predictors)])){
    stop("Missing values found in the response or predictors. ", 
         "Handle them before refitting the model.")
  }
  
  # Construct right-hand side formula for X
  formula = as.formula(paste("~", paste(predictors, collapse = " + ")))
  
  # Defining X and y and computing n and p
  X = model.matrix(formula, data = df)
  
  # Loud failure 2: refusing a rank-deficient predictor matrix
  if(qr(X)$rank < ncol(X)){
    stop("Predictor matrix is rank-deficient where at least one predictor ",
         "is a linear combination of the others. The inverse does not exist.")
  }
  
  y = as.matrix(df[[response]])
  n = nrow(X)
  p = ncol(X)
  modeldf = p - 1
  residualdf = n - p
  
  # Computing the betas
  XtX = t(X) %*% X
  Xty = t(X) %*% y
  beta = solve(XtX, Xty)
  
  # Predicted values, resids, and sum of squares
  y_hat = X %*% beta
  e = y - y_hat
  SSE = as.numeric(t(e) %*% e)
  SSR = as.numeric(t(beta) %*% Xty - sum(y)^2 / n)
  SST = SSR + SSE
  
  # Residual variance and standard errors
  sigma2_hat = as.numeric(SSE / residualdf)
  var_beta = sigma2_hat * solve(XtX)
  se_beta = sqrt(diag(var_beta))
  
  # T-stats and two-sided p-values for the betas
  t_stat = beta / se_beta
  p_val_t = 2 * (1-pt(abs(t_stat),df = residualdf))
  
  # R-squared statistics
  r_squared = SSR / SST
  adj_r_squared = 1 - ((SSE / residualdf) / (SST / (n - 1)))
  
  # Model F test
  f_stat = (SSR / modeldf) / (SSE / residualdf)
  p_val_f = 1 - pf(f_stat, df1 = modeldf, df2 = residualdf)
  
  # Cleaning up results
  model_stats = data.frame(
    R2 = r_squared,
    Adj_R2 = adj_r_squared,
    Sigma2 = sigma2_hat,
    Sigma = sqrt(sigma2_hat),
    F_stat = f_stat,
    P_Val_F = p_val_f,
    df_model = modeldf,
    df_resid = residualdf,
    row.names = ""
  )
  
  # This is assembling the coefficient table like lm()
  coefs = data.frame(
    Estimate = as.vector(beta),
    "Std. Error" = se_beta,
    "t value" = t_stat,
    "Pr(>|t|)" = p_val_t,
    row.names = colnames(X),
    check.names = FALSE
  )
  
  return(list(
    coefficients = coefs,
    model_stats = model_stats,
    fitted.values = as.vector(y_hat),
    residuals = as.vector(e),
    sum_of_squares = c(SSR = SSR, SSE = SSE, SST = SST)
  )) 
}

#' This function will take a fitted result and new predictor values and 
#' calculate predicted outcomes.
#'  
#' @param coef_table  data.frame of the coefficients from fit_ols() 
#'                    (e.g., fit$coefficients)
#' @param preds       data.frame that contains the stored predictor names and
#'                    predictor values as rows
#'                    (e.g., new_vals = data.frame(
#'                              parent = c(3.4, 2.4, 3.8),
#'                              sibling = c(1.5, 3.2, 5.7),
#'                              peer = c(3.1, 4.5, 2.5)
#'                              ))
#' 
#' @return returns a numeric vector of length nrow(preds)
#' 
#' @details Error Failures and Assumptions:
#'  1. the coef_table comes from the fit_ols() function
#'  2. preds contains columns with the same names and predictors are numeric
#'  3. no NA's are in the preds
#'
#' @examples 
#' scored_data = predict_ols(coef_table = results$coefficients, 
#'                           preds = new_vals)

predict_ols = function(coef_table, preds){
  
  # One more check to ensure no missing predictor values
  if (anyNA(preds)){
    stop("Missing values found in preds. Remove them before attempting again.")
  } 
  
  # Grab predictor names from the coef_table
  predictors = rownames(coef_table)[-1]
  
  # Get the predictor matrix 
  formula = as.formula(paste("~", paste(predictors, collapse = " + ")))
  X = model.matrix(formula, data = preds)  
  
  betas = as.matrix(coef_table[, "Estimate", drop = FALSE])
  
  y_hat_new = X %*% betas
  
  return(as.vector(y_hat_new))
}