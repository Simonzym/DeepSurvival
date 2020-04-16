library(survival)
library(survivalROC)
library(survminer)
library(glmnet)
library(dplyr)
library(caret)
library(randomForestSRC)
library(c060)
library(peperr)
library(pec)
library(survAUC)
# read data
set.seed(123)
features <- read.csv("data431_surv.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
dim(features)
cutoff <- 365*3
ts <- seq(1, cutoff, 5)


features <- features %>% 
  mutate(PatientID = as.character(PatientID)) %>%
  #mutate(shortID = substr(PatientID, 7,9)) %>%
  #filter(!shortID %in% exclude.char) %>%
  filter(!is.na(F1)) %>%
  filter(!is.na(Histology)) %>%
  filter(!is.na(age)) %>%
  filter(!is.na(Overall.Stage)) %>%
  filter(!is.na(clinical.T.Stage)) 

features_cat <- features %>%
  dplyr::select(Overall.Stage, Histology, gender)
dmy <-  dummyVars("~.", data =features_cat) 
features_cat_ohe <- data.frame(predict(dmy, newdata = features_cat))

features_con = features[, -c(1:10)]
features_con_scale = features_con%>%scale(., scale = TRUE, center = TRUE)
features_mat <- as.matrix(cbind(features_cat_ohe, features[,c(2:5)], features_con_scale))

#features_mat <- scale(features_mat, center = T, scale = T)

features_mat <- as.data.frame(features_mat)
features_mat$survivaltime <- as.numeric(features$Survival.time)
features_mat$status <- features$deadstatus.event

nk <- 10
index.kfold <- createFolds(seq(1, nrow(features)), k = nk, 
                           list = TRUE, returnTrain = FALSE)
c_index = rep(NA, nk)
for(k in 1:nk){
  te.ind <- index.kfold[[k]]
  n.te <- length(te.ind)
  
  rf.fit <- rfsrc(Surv(survivaltime, status) ~., data = features_mat[-te.ind,])
  rf.pred = predict(rf.fit, features_mat[te.ind,])
  #cox.fit = coxph(Surv(survivaltime, status) ~., data = features_mat, x = TRUE)
  prederror = pec(list(rf.fit), data = features_mat[te.ind,], formula = as.formula(Hist(survivaltime, status) ~ 1), splitMethod = 'bootcv',
                  B = 50, times = ts, maxtime = cutoff)
  print(prederror)
  c_index[k] = 1 - rf.pred$err.rate[rf.pred$ntree]
}
bs = c(0.23,0.224,0.211,0.225,0.21,0.232,0.226,0.19,0.23,0.215)
