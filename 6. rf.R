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
# # exclude sujects with no/bad mask
# exclude = c(3, 7, 34, 36, 40, 44, 50, 58, 61, 67, 68, 69, 74, 83, 84, 94, 96,
#             110, 128, 135, 137, 141, 143, 144, 146, 148, 164, 166, 167, 176,
#             180, 191, 200, 202, 207, 208, 212, 214, 222, 227, 230, 238, 250,
#             251, 254, 265, 279, 301, 302, 312, 314, 316, 319, 322, 324, 325,
#             327, 329, 330, 333, 336, 338, 344, 346, 348, 351, 352, 354, 355,
#             356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368,
#             369, 370, 371, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382,
#             383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395,
#             396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408,
#             409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422)
# 
# exclude.char <- sprintf("%03d", exclude)

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
rf.fit <- rfsrc(Surv(survivaltime, status) ~., data = features_mat)

cox.fit = coxph(Surv(survivaltime, status) ~., data = features_mat, x = TRUE)
prederror = pec(list(rf.fit), data = features_mat, formula = as.formula(Hist(survivaltime, status) ~ 1), splitMethod = 'bootcv',
                B = 50, times = ts, maxtime = cutoff)
c_index = 1 - rf.fit$err.rate[rf.fit$ntree]
# nt <- seq(50,150,5)
# ns <- seq(3, 20, 1)
# error_rate <- rep(NA, length(nt)*length(ns))
# bs = list()
# index = 1
# for (i in 1:length(nt)){
#   for (j in 1:length(ns)){
#     index = index + 1
#     tree <- rfsrc(Surv(survivaltime, status) ~., data = features_mat, ntree = nt[i], nsplit = ns[j])
#     prederror = pec(list(tree), data = features_mat, formula = as.formula(Hist(survivaltime, status) ~ 1), splitMethod = 'bootcv',
#                     B = 50)
#     bs[[index]] = prederror
#     error_rate[index] <- tree$err.rate[nt[i]]
#   }
# }