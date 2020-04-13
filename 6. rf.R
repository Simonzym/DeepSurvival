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

features_mat <- as.data.frame(features_mat)
features_mat$survivaltime <- as.numeric(features$Survival.time)
features_mat$status <- features$deadstatus.event
rf.fit <- rfsrc(Surv(survivaltime, status) ~., data = features_mat)

cox.fit = coxph(Surv(survivaltime, status) ~., data = features_mat, x = TRUE)
prederror = pec(list(rf.fit), data = features_mat, formula = as.formula(Hist(survivaltime, status) ~ 1), splitMethod = 'bootcv',
                B = 50, times = ts, maxtime = cutoff)
c_index = 1 - rf.fit$err.rate[rf.fit$ntree]
