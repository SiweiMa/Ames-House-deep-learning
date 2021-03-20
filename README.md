# Revisit Ames House Prices with Deep Learning Model
*The repository is a inspired by the assignment in online course courses.d2l.ai* 

#### by [Siwei Ma](https://www.linkedin.com/in/siwei-ma-28345856/)

# Executive Summary

Given [Ames Housing dataset](http://jse.amstat.org/v19n3/decock.pdf), the project started with an exploratory data analysis (EDA) to identify the missing values, suspicious data, and redundant variables. Then I performed a mixed stepwise selection to reduce the set of variables and select the best model based on AIC, BIC, and adjust R-squared. With the best model selected, the model assumptions were checked regarding normality, homoscedasticity, collinearity, and linearity between response and predictors. Several solutions were proposed to solve the assumption violation. The model was then tested on unseen data and scored on Root-Mean-Squared-Error (RMSE).
