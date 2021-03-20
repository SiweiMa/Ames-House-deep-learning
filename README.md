# Revisit Ames House Prices with Deep Learning Model
*The repository is my solution to the assignment in online course courses.d2l.ai* 

#### by [Siwei Ma](https://www.linkedin.com/in/siwei-ma-28345856/)

# Executive Summary

I have done the [Ames House Prices project](https://github.com/SiweiMa/Ames-House-Prices-Multiple-Linear-Regression-Project-in-Python) before with multiple linear regression using statsmodels api. Here I revisited the project with deep learning model using Pytorch. The neural network with one hidden layer includes ReLu as activation function and dropout aiming for a simpler model. The training process relies on the Adam optimizer with mini batching. The performance of the model was evaluated by log RMSE through k-fold cross validation. The final RMSE of the deep learning model is quite similar with one of the multiple linear regression model with feature engineering/selection. 
