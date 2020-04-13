# load libraries
library(survival)
library(survivalROC)
library(survminer)
library(glmnet)
library(dplyr)
library(pec)
library(caret)
library(randomForestSRC)
library(c060)
library(peperr)
library(survAUC)
library(ggplot2)
# read data
set.seed(123)
features <- read.csv("data431_surv.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
dim(features)
cutoff <- 365*3
ts <- seq(1, cutoff, 5)

features <- features %>% 
  filter(!is.na(F1)) %>%
  filter(!is.na(Histology)) %>%
  filter(!is.na(age)) %>%
  filter(!is.na(Overall.Stage)) %>%
  filter(!is.na(clinical.T.Stage))

id <- features$PatientID
features <- features %>% dplyr::select(-c(PatientID))

# create folders
nk <- 10
index.kfold <- createFolds(seq(1, nrow(features)), k = nk, 
                           list = TRUE, returnTrain = FALSE)

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

# categorical variables
# one-hot-encoding
features_cat <- features %>%
  dplyr::select(Overall.Stage, Histology, gender)
dmy <-  dummyVars("~.", data =features_cat) 
features_cat_ohe <- data.frame(predict(dmy, newdata = features_cat))
# other clinical variables
features_clc <- features %>%
  dplyr::select(clinical.T.Stage, Clinical.N.Stage, Clinical.M.Stage, age) %>%
  mutate(Tstage = as.numeric(clinical.T.Stage), Nstage = as.numeric(Clinical.N.Stage), Mstage = as.numeric(Clinical.M.Stage)) %>%
  dplyr::select(age, Tstage, Nstage, Mstage)

# radiomics features
features_rad <- features %>%
  dplyr::select(-c(1:9))

surv.time <- features$Survival.time
status <- features$deadstatus.event
surv_obj <- Surv(surv.time, status)
# pca
pca_res <- prcomp(features_rad, center = TRUE, scale. = TRUE)
features_pca <- cbind(t(pca_res$rotation[1:20,]), surv.time, status, features_clc, features_cat)
features_rad_scale = features_rad%>%scale(., center = T, scale = T)

# matrix for regularized cox
features_mat <- as.matrix(cbind(features_rad_scale, features_clc, features_cat_ohe))
#features_mat <- scale(features_mat, center = TRUE, scale = TRUE)

# start cross validation
for (k in 1:nk){
  # extract index for testing
  te.ind <- index.kfold[[k]]
  n.te <- length(te.ind)
  
  # for getting Brier scores
  surv.tr <- Surv(surv.time, status)[-te.ind]
  surv.te <- Surv(surv.time, status)[te.ind]
  
  # cox
  coxph.fit <- coxph(Surv(surv.time, status) ~ ., data = features_pca[-te.ind,], x = TRUE, y = TRUE,
                     method = 'breslow')
  tmp = survfit(coxph.fit, features_pca[te.ind,])
  cox.pred.tr <- predict(coxph.fit, newdata = features_pca[-te.ind,])
  cox.pred.te <- predict(coxph.fit, newdata = features_pca[te.ind,])
  
  cox.res <- survivalROC(Stime = surv.time[te.ind],
              status  = status[te.ind],
              marker       = cox.pred.te,
              predict.time = cutoff,
              span = 0.25*n.te^(-0.20))
  auc_cox[k] <- cox.res$AUC
  
  cox.ERR <- predErr(surv.tr, surv.te, cox.pred.tr, cox.pred.te, type = "brier", times = ts, int.type = "weighted")
  bs_cox[k] <- cox.ERR$ierror
  
  # lasso
  cv.fit.lasso <- cv.glmnet(features_mat[-te.ind,], surv_obj[-te.ind,], family = "cox", alpha = 1)
  lasso.fit <- glmnet(features_mat[-te.ind,], surv_obj[-te.ind,], family = "cox", maxit = 1000000, 
                      alpha = 1, lambda = cv.fit.lasso$lambda.min, standardize = TRUE)
  lasso.temp.tr <- exp(predict(lasso.fit,  newx = features_mat[-te.ind,], type = "link", s = cv.fit.lasso$lambda.min))
  lasso.temp.te <- exp(predict(lasso.fit,  newx = features_mat[te.ind,], type = "link", s = cv.fit.lasso$lambda.min))
  lasso.pred.tr = log(lasso.temp.tr)
  lasso.pred.te = log(lasso.temp.te)
  
  lasso.res <- survivalROC(Stime = surv.time[te.ind],
                           status  = status[te.ind],
                           marker       = lasso.pred.te,
                           predict.time = cutoff,
                           span = 0.25*n.te^(-0.20))
  auc_lasso[k] <- lasso.res$AUC
  
  lasso.ERR <- predErr(surv.tr, surv.te, lasso.pred.tr, lasso.pred.te, type = "brier", times = ts, int.type = "weighted")
  bs_lasso[k] <- lasso.ERR$ierror
  
  # ridge
  cv.fit.ridge <- cv.glmnet(features_mat[-te.ind,], surv_obj[-te.ind,], family = "cox", alpha = 0)
  ridge.fit <- glmnet(features_mat[-te.ind,], surv_obj[-te.ind,], family = "cox", maxit = 1000000, 
                      alpha = 0, lambda = cv.fit.ridge$lambda.min, standardize = TRUE)
  ridge.temp.tr <- exp(predict(ridge.fit,  newx = features_mat[-te.ind,], type = "link", s = cv.fit.ridge$lambda.min))
  ridge.temp.te <- exp(predict(ridge.fit,  newx = features_mat[te.ind,], type = "link", s = cv.fit.ridge$lambda.min))
  ridge.pred.tr = log(ridge.temp.tr)
  ridge.pred.te = log(ridge.temp.te)
  
  ridge.res <- survivalROC(Stime = surv.time[te.ind],
                           status  = status[te.ind],
                           marker = ridge.pred.te,
                           predict.time = cutoff,
                           span = 0.25*n.te^(-0.20))
  auc_ridge[k] <- ridge.res$AUC
  
  ridge.ERR <- predErr(surv.tr, surv.te, ridge.pred.tr, ridge.pred.te, type = "brier", times = ts, int.type = "weighted")
  bs_ridge[k] <- ridge.ERR$ierror

  
  # elastic-net 2

  cv.fit.en2 <- cv.glmnet(features_mat[-te.ind,], surv_obj[-te.ind,], family = "cox", alpha = 0.5)
  en.fit2 <- glmnet(features_mat[-te.ind,], surv_obj[-te.ind,], family = "cox", lambda = cv.fit.en2$lambda.min, alpha = 0.5, standardize = TRUE)
  
  en.temp.tr2 <- exp(predict(en.fit2,  newx = features_mat[-te.ind,], type = "link", s = cv.fit.en2$lambda.min))
  en.temp.te2 <- exp(predict(en.fit2,  newx = features_mat[te.ind,], type = "link", s = cv.fit.en2$lambda.min))
  en.pred.tr2 = log(en.temp.tr2)
  en.pred.te2 = log(en.temp.te2)
  en.res2 <- survivalROC(Stime = surv.time[te.ind],
                        status  = status[te.ind],
                        marker       = en.pred.te2,
                        predict.time = cutoff,
                        span = 0.25*n.te^(-0.20))
  auc_en2[k] <- en.res2$AUC
  
  en.ERR <- predErr(surv.tr, surv.te, en.pred.tr2, en.pred.te2, type = "brier", times = ts, int.type = "weighted")
  bs_en2[k] <- en.ERR$ierror
}

results = rbind(auc_cox, auc_lasso, auc_ridge, auc_en, auc_en2, bs_cox, bs_lasso, bs_ridge, bs_en, bs_en2)
write.csv(results, 'cox_result.csv')
