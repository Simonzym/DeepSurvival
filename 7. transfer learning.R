library(survival)
library(glmnet)
library(dplyr)
library(caret)
library(survivalROC)
library(randomForestSRC)
library(c060)
library(peperr)
library(survAUC)
set.seed(123)
ft_total <- read.csv("ft_total.csv", 
                     header = TRUE, sep = ",", stringsAsFactors = FALSE)

ft_total <- ft_total %>%
  filter(!is.na(Clinical_T_Stage)) %>%
  filter(!is.na(age)) 

surv.time <- ft_total$Survival_time
status <- ft_total$Status
surv_obj = Surv(ft_total$Survival_time, ft_total$Status)

ft_total_mat <- ft_total%>%
  dplyr::select(-one_of("Survival_time", "Status", 'male')) %>%
  as.data.frame() %>%
  select_if(~!any(is.na(.))) 


ft_con = ft_total_mat[,-c(1:15)]
ft_con_scale = ft_con%>%scale(., scale = TRUE, center = TRUE)
zero_index = which(is.na(ft_con_scale[1,]))

#get rid of the covariates(features) which are the same for every observation
ft_con_scale = ft_con_scale[, -zero_index]
ft_total_mat = as.matrix(cbind(ft_total_mat[, c(1:15)], ft_con_scale))

cutoff <- 365*3
ts <- seq(1, cutoff, 5)
# create folds of index for CV

nk <- 10
index.kfold <- createFolds(seq(1, nrow(ft_total_mat)), k = nk, list = TRUE, returnTrain = FALSE)

# create empty vectors to store results
auc_cox <- rep(NA, nk)
auc_lasso <- rep(NA, nk)
auc_ridge <- rep(NA, nk)
auc_en <- rep(NA, nk)
auc_en2 <- rep(NA, nk)

bs_cox <- rep(NA, nk)
bs_lasso <- rep(NA, nk)
bs_ridge <- rep(NA, nk)
bs_en <- rep(NA, nk)
bs_en2 <- rep(NA, nk)


for (k in 1:nk){
  # extract index for testing
  te.ind <- index.kfold[[k]]
  n.te <- length(te.ind)
  
  # for getting Brier scores
  surv.tr <- Surv(surv.time, status)[-te.ind]
  surv.te <- Surv(surv.time, status)[te.ind]
  
  # lasso
  cv.fit.lasso <- cv.glmnet(ft_total_mat[-te.ind,], surv_obj[-te.ind,], family = "cox", alpha = 1)
  lasso.fit <- glmnet(ft_total_mat[-te.ind,], surv_obj[-te.ind,], family = "cox", maxit = 1000000, 
                      alpha = 1, lambda = cv.fit.lasso$lambda.min, standardize = TRUE)
  lasso.temp.tr <- exp(predict(lasso.fit,  newx = ft_total_mat[-te.ind,], type = "link", s = cv.fit.lasso$lambda.min))
  lasso.temp.te <- exp(predict(lasso.fit,  newx = ft_total_mat[te.ind,], type = "link", s = cv.fit.lasso$lambda.min))
  lasso.pred.tr = log(lasso.temp.tr)
  lasso.pred.te = log(lasso.temp.te)
  # lasso.pred.tr <- lasso.temp.tr/(lasso.temp.tr + 1)
  # lasso.pred.te <- lasso.temp.te/(lasso.temp.te + 1)
  
  lasso.res <- survivalROC(Stime = surv.time[te.ind],
                           status  = status[te.ind],
                           marker       = lasso.pred.te,
                           predict.time = cutoff,
                           span = 0.25*n.te^(-0.20))
  auc_lasso[k] <- lasso.res$AUC
  
  lasso.ERR <- predErr(surv.tr, surv.te, lasso.pred.tr, lasso.pred.te, type = "brier", times = ts, int.type = "weighted")
  bs_lasso[k] <- lasso.ERR$ierror
  
  # ridge
  cv.fit.ridge <- cv.glmnet(ft_total_mat[-te.ind,], surv_obj[-te.ind,], family = "cox", alpha = 0)
  ridge.fit <- glmnet(ft_total_mat[-te.ind,], surv_obj[-te.ind,], family = "cox", maxit = 1000000, 
                      alpha = 0, lambda = cv.fit.ridge$lambda.min, standardize = TRUE)
  ridge.temp.tr <- exp(predict(ridge.fit,  newx = ft_total_mat[-te.ind,], type = "link", s = cv.fit.ridge$lambda.min))
  ridge.temp.te <- exp(predict(ridge.fit,  newx = ft_total_mat[te.ind,], type = "link", s = cv.fit.ridge$lambda.min))
  ridge.pred.tr = log(ridge.temp.tr)
  ridge.pred.te = log(ridge.temp.te)
    # ridge.pred.tr <- ridge.temp.tr/(ridge.temp.tr + 1)
  # ridge.pred.te <- ridge.temp.te/(ridge.temp.te + 1)
  
  ridge.res <- survivalROC(Stime = surv.time[te.ind],
                           status  = status[te.ind],
                           marker = ridge.pred.te,
                           predict.time = cutoff,
                           span = 0.25*n.te^(-0.20))
  auc_ridge[k] <- ridge.res$AUC
  
  ridge.ERR <- predErr(surv.tr, surv.te, ridge.pred.tr, ridge.pred.te, type = "brier", times = ts, int.type = "weighted")
  bs_ridge[k] <- ridge.ERR$ierror
  
  
  # # elastic-net 
  # a <- seq(0.01, 0.99, 0.02)
  # search <- foreach(i = a, .combine = rbind) %dopar% {
  #   cv <- cv.glmnet(ft_total_mat[-te.ind,], surv_obj[-te.ind,], family = "cox", alpha = i)
  #   data.frame(cvm = cv$cvm[cv$lambda == cv$lambda.min], lambda.min = cv$lambda.min, alpha = i)
  # }
  # param.en <- search[search$cvm == min(search$cvm),]
  # 
  # en.fit <- glmnet(ft_total_mat[-te.ind,], surv_obj[-te.ind,], family = "cox", lambda = param.en$lambda.min, alpha = param.en$alpha, standardize = TRUE)
  # 
  # en.temp.tr <- exp(predict(en.fit,  newx = ft_total_mat[-te.ind,], type = "link", s = param.en$lambda.min))
  # en.temp.te <- exp(predict(en.fit,  newx = ft_total_mat[te.ind,], type = "link", s = param.en$lambda.min))
  # en.pred.tr <- en.temp.tr/(en.temp.tr + 1)
  # en.pred.te <- en.temp.te/(en.temp.te + 1)
  # 
  # en.res <- survivalROC(Stime = surv.time[te.ind],
  #                       status  = status[te.ind],
  #                       marker       = en.pred.te,
  #                       predict.time = cutoff,
  #                       span = 0.25*n.te^(-0.20))
  # auc_en[k] <- en.res$AUC
  # 
  # en.ERR <- predErr(surv.tr, surv.te, en.pred.tr, en.pred.te, type = "brier", times = ts, int.type = "weighted")
  # bs_en[k] <- mean(en.ERR$error, na.rm = T)
  
  # elastic-net 2
  cv.fit.en2 <- cv.glmnet(ft_total_mat[-te.ind,], surv_obj[-te.ind,], family = "cox", alpha = 0.5)
  en.fit2 <- glmnet(ft_total_mat[-te.ind,], surv_obj[-te.ind,], family = "cox", lambda = cv.fit.en2$lambda.min, alpha = 0.5, standardize = TRUE)
  
  en.temp.tr2 <- exp(predict(en.fit2,  newx = ft_total_mat[-te.ind,], type = "link", s = cv.fit.en2$lambda.min))
  en.temp.te2 <- exp(predict(en.fit2,  newx = ft_total_mat[te.ind,], type = "link", s = cv.fit.en2$lambda.min))
  en.pred.tr2 = log(en.temp.tr2)
  en.pred.te2 = log(en.temp.te2)
  # en.pred.tr2 <- en.temp.tr2/(en.temp.tr2 + 1)
  # en.pred.te2 <- en.temp.te2/(en.temp.te2 + 1)
  
  en.res2 <- survivalROC(Stime = surv.time[te.ind],
                         status  = status[te.ind],
                         marker       = en.pred.te2,
                         predict.time = cutoff,
                         span = 0.25*n.te^(-0.20))
  auc_en2[k] <- en.res2$AUC
  
  en.ERR <- predErr(surv.tr, surv.te, en.pred.tr2, en.pred.te2, type = "brier", times = ts, int.type = "weighted")
  bs_en2[k] <- en.ERR$ierror
  
}

results = rbind(auc_lasso, auc_ridge, auc_en, auc_en2,  bs_lasso, bs_ridge, bs_en, bs_en2)
write.csv(results, 'transfer_result.csv')