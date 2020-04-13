# DeepSurvival
Codes implemented by the paper "Comparative Study of Deep Learning and Other Predictive Models for Survival in Lung Cancer".

## File
The Lung cancer dataset we used in this analysis is a open source. For replicating the results, you may download the data and images from [\url], and run the following codes on them.

## R
--cox.R: survival prediction by cox model, cox model with ridge penalty, cox model with lasso penalty, cox model with Elastic-Net penalty;

--rf.R: survival prediction by random survival forest

--transfer learning.R: survival prediction by cox model with ridge, lasso, Elastic-Net penalty with extra features obtained by transfer learning.

## Python
--Image_Surviva_3D.py: survival prediction by CNN

--feature_extract.py: extract features by transfer learning
