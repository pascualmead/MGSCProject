#install.packages("glmnet")
#install.packages("glmnetUtils")
#install.packages("tree")
#install.packages("randomForest")
#install.packages("ggplot2")
#install.packages("forecast")

library(glmnet)
library(glmnetUtils)
library(ggplot2)

#options(scipen = 999)
setwd("~/Desktop/MGSC310-Project-master")
nycAB <-  read.csv("AB_NYC_2019.csv")

set.seed(310)

#data cleaning and feature transformation
nycAB <-  nycAB[ , !(names(nycAB) %in% c("name","id", "host_id", "host_name", "latitude", "longitude", "last_review", "calculated_host_listings_count"))]
nycAB <- na.omit(nycAB)
sum(is.na(nycAB))
nycAB$log_price = log1p(nycAB$price)
nycAB <-  nycAB[!(nycAB$log_price < 3),]
nycAB <-  nycAB[!(nycAB$log_price > 7),]
nycAB$log_num_reviews <- log1p(nycAB$number_of_reviews)
nycAB$reviews_per_month_log <-  log1p(nycAB$reviews_per_month)
nycAB$minimum_nights_log <-  log1p(nycAB$minimum_nights)

#for efficiency, take a sample of 30 percent 
sample_idx <- sample(1:nrow(nycAB), size = floor(0.3*nrow(nycAB)))
nycAB <- nycAB[sample_idx,]


#convert "neighbourhood_group" & "room_type" to numerical factors
#NOTE: Bronx == 1, Brookyln == 2, Manhattan == 3, Queens == 4, Staten Island == 5
#NOTE: Entire home/apt == 1, Private room == 2, Shared room == 3
nycAB$neighbourhood_group <- as.factor(as.numeric(as.factor(nycAB$neighbourhood_group)))
nycAB$room_type <- as.factor(as.numeric(as.factor(nycAB$room_type)))

#create train and test from sample
set.seed(310)
train_idx <- sample(1:nrow(nycAB), size = floor(0.75*nrow(nycAB)))
AB_train <- nycAB[train_idx,]
AB_test <- nycAB[-train_idx,]

#run linear model using price (ARIANA)
#AB_lm1 <- lm(price ~ ., AB_train)
#summary(AB_lm1)

#lm1_trainpreds <- predict(AB_lm1)
#lm1_testpreds <- predict(AB_lm1, newdata = AB_test)

#MSE <- function(p,t){
  #mean((t-p)^2)
#}

#MSE_lm1train <- MSE(lm1_trainpreds, AB_train$price)
#MSE_lm1test <- MSE(lm1_testpreds, AB_test$price)

#df.1 <- data.frame(MSE_train_LM1 = as.matrix(MSE_lm1train),
                   #MSE_test_LM1 = as.matrix(MSE_lm1test)) 
#df.1

#run linear model using log_price (ARIANA)
#AB_lm2 <- lm(log_price ~ ., AB_train)
#summary(AB_lm2)

#lm2_trainpreds <- predict(AB_lm2)
#lm2_testpreds <- predict(AB_lm2, newdata = AB_test)

#MSE <- function(p,t){
  #mean((t-p)^2)
#}

#MSE_lm2train <- MSE(lm2_trainpreds, AB_train$log_price)
#MSE_lm2test <- MSE(lm2_testpreds, AB_test$log_price)

#df.2 <- data.frame(MSE_train_LM2 = as.matrix(MSE_lm2train),
                   #MSE_test_LM2 = as.matrix(MSE_lm2test)) 
#df.2

#run lasso model using price
lasso1_train_subset <- subset(AB_train, select = -c(log_price, neighbourhood))
lasso1_test_subset <- subset(AB_test, select = -c(log_price, neighbourhood))
AB_lasso1 <- cv.glmnet(price ~ ., data = lasso1_train_subset, alpha = 1)

#finding the best lambda
best_lambda1 <- AB_lasso1$lambda.min
best_lambda1

#retrain lasso model & make predictions
lasso_best1 <- glmnet(price ~ ., data = lasso1_train_subset, alpha = 1, lambda = best_lambda1)
lasso1_trainpreds <- predict(lasso_best1, newdata = lasso1_train_subset)
lasso1_testpreds <- predict(lasso_best1, s = best_lambda1, newdata = lasso1_test_subset)

RMSE <- function(p,t){
  sqrt(mean((t-p)^2))
}

RMSE_lasso1train <- RMSE(lasso1_trainpreds,AB_train$price)
RMSE_lasso1test <- RMSE(lasso1_testpreds, AB_test$price)

df.3 <- data.frame(RMSE_train_LASSO1 = as.matrix(RMSE_lasso1train),
                   RMSE_test_LASSO1 = as.matrix(RMSE_lasso1test))
df.3

#finding the important coefficients
coef(lasso_best1)

#plotting
colors <- c("black", "red")
plot(AB_test$price, lasso1_testpreds, main = "Lasso Model (Price)",
     xlab = "Price", ylab = "Test predictions", col = colors)
legend("topright", legend = c("Price", "Test predictions"),
       fill = colors, cex = 0.38)
lasso1_resids = lasso1_test_subset$price - lasso1_testpreds
plot(lasso1_testpreds, lasso1_resids, main = "Lasso Model (Price)",
     xlab = "Test predictions", ylab = "Residuals", col = colors)
legend("topright", legend = c("Test predictions", "Residuals"),
       fill = colors, cex = 0.38)

#run lasso model using log_price
lasso2_train_subset <- subset(AB_train, select = -c(price, neighbourhood))
lasso2_test_subset <- subset(AB_test, select = -c(price, neighbourhood))
AB_lasso2 <- cv.glmnet(log_price ~ ., data = lasso2_train_subset, alpha = 1)

#finding the best lambda
best_lambda2 <- AB_lasso2$lambda.min
best_lambda2

#retrain lasso model & make predictions
lasso_best2 <- glmnet(log_price ~ ., data = lasso2_train_subset, alpha = 1, lambda = best_lambda2)
lasso2_trainpreds <- predict(lasso_best2, newdata = lasso2_train_subset)
lasso2_testpreds <- predict(lasso_best2, s = best_lambda1, newdata = lasso2_test_subset)

RMSE_lasso2train <- exp(RMSE(lasso2_trainpreds,AB_train$log_price))
RMSE_lasso2test <- exp(RMSE(lasso2_testpreds, AB_test$log_price))

df.4 <- data.frame(RMSE_train_LASSO2 = as.matrix(RMSE_lasso2train),
                   RMSE_test_LASSO2 = as.matrix(RMSE_lasso2test))
df.4

#finding the important coefficients
coef(lasso_best2)

#plotting
plot(AB_test$log_price, lasso2_testpreds, main = "Lasso Model (Log Price)",
     xlab = "Log Price", ylab = "Test predictions", col = colors)
legend("topright", legend = c("Log Price", "Test predictions"),
       fill = colors, cex = 0.38)
lasso2_resids = lasso2_test_subset$log_price - lasso2_testpreds
plot(lasso2_testpreds, lasso2_resids, main = "Lasso Model (Log Price)",
     xlab = "Test predictions", ylab = "Residuals", col = colors)
legend("topright", legend = c("Test predictions", "Residuals"),
       fill = colors, cex = 0.38)

#compare results of preds against true values for price & log_price
#LogPricePreds = X1 & PricePreds = X1.1
compare.df <- data.frame(LogPricePreds = as.matrix(lasso2_testpreds),
                         LogPrice = as.matrix(AB_test$log_price),
                         PricePreds = as.matrix(lasso1_testpreds),
                         Price = as.matrix(AB_test$price))
                          
head(compare.df) 

#run a decision tree using price 
library(tree)
regMod = tree(price ~ neighbourhood_group + room_type 
              + minimum_nights + number_of_reviews + reviews_per_month
              + availability_365 + log_num_reviews + reviews_per_month_log 
              + minimum_nights_log,
              data = AB_train)

#run a decision tree using log_price
logMod = tree(log_price ~ neighbourhood_group + room_type 
              + minimum_nights + number_of_reviews + reviews_per_month
              + availability_365 + log_num_reviews + reviews_per_month_log 
              + minimum_nights_log,
              data = AB_train)

#we'll plot both
plot(regMod)
text(regMod, pretty=0)

plot(logMod)
text(logMod, pretty = 0)

#using cross validation to find the best tree size for pruning
cvTreeR = cv.tree(regMod)
bestIdx = which.min(cvTreeR$dev)
cvTreeR$size[bestIdx]
#best size is 6

cvTreeL = cv.tree(logMod)
bestIdx = which.min(cvTreeL$dev)
cvTreeL$size[bestIdx]
#best size is 4

#since we have our best tree sizes, we can prune each tree & generate predictions
prunedTreeR = prune.tree(regMod, best = 5)
predsTrainR = predict(prunedTreeR)
predsTestR = predict(prunedTreeR, newdata = AB_test)

prunedTreeL = prune.tree(logMod, best = 4)
predsTrainL = predict(prunedTreeL)
predsTestL = predict(prunedTreeL, newdata = AB_test)

#we'll output the RMSE for each model

RMSE(predsTrainR, AB_train$price) #97.16214
RMSE(predsTestR, AB_test$price) #94.24316

exp(RMSE(predsTrainL, AB_train$log_price)) #1.596223
exp(RMSE(predsTestL, AB_test$log_price)) #1.5792

#the RMSE is much lower when we use the log transformation
#we do get a lower RMSE in the test set which may be an indication that 
#the model is overfitting the data

#run a RF model using log_price only
library(randomForest)
set.seed(2019) #run the model with the set.seed
#setting mtry to 5 as cross-validated best number
#maxnodes == 160  and ntree == 500 to minimize RMSE while still optimizing efficiency
bag_nycAB <- randomForest(log_price ~ neighbourhood_group + room_type 
                          + availability_365 + log_num_reviews + reviews_per_month_log 
                          + minimum_nights_log,
                          data = AB_train,
                          mtry = 5,
                          maxnodes = 160,
                          ntree = 500,
                          importance = TRUE)

library(forecast)
#generating residuals for test and train
residual_train <- AB_train$log_price - preds_bag_nycAB
residual_test <- AB_test$log_price - preds_bag_nycAB_test

#plotting residuals against predicted values from training set 
#this plot is heteroskedatic, as the variance in error term is non-constant throughout 
ggplot(mod3_train_df,aes(x=pred_train,y=resids_train)) + geom_point(alpha=0.5) + geom_smooth(color="red")
ggplot(mod3_test_df,aes(x=pred_test,y=resids_test)) + geom_point(alpha=0.5) + geom_smooth(color="red")

#generating importance plot
bag_nycAB
importance(bag_nycAB)
varImpPlot(bag_nycAB)

#prediction of train data
preds_bag_nycAB <- predict(bag_nycAB, newdata = AB_train)
#prediction of test data
preds_bag_nycAB_test<- predict(bag_nycAB, newdata = AB_test)

#creating data frame of residuals and predicted values
mod3_train_df <- data.frame(
  resids_train <- residual_train,
  pred_train <- preds_bag_nycAB)

mod3_test_df <- data.frame(
  resids_test<- residual_test,
  pred_test <- preds_bag_nycAB_test)

plot(preds_bag_nycAB, AB_train$log_price)
abline(0,1,col="red")
plot(preds_bag_nycAB_test, AB_test$log_price)
abline(0,1,col="red")

#RMSE test (log_price)
RMSE_test <- exp(RMSE(preds_bag_nycAB_test, AB_test$log_price))
RMSE_test
#1.544338

#RMSE train (log_price)
RMSE_train <- exp(RMSE(preds_bag_nycAB, AB_train$log_price))
RMSE_train
#1.498392

#random forest model has lowest error, so we will use this one
