setwd("/Users/Kianamon/R/titanic")
rm(list=ls())
#####################################################################################
#libraries in use:
library(knitr)
library(httr)
library(readr)
library(dplyr)
library(tidyr)
library(XML)
library(ggplot2)
library(stringr)
library(lubridate)
library(grid)
library(caret)
library(glmnet)
library(ranger)
library(e1071)
library(Metrics)
library(rpart)  
library(rpart.plot)
library(ModelMetrics)   
library(ipred)  
library(randomForest)
library(gbm)  
library(ROCR)
library(mlr)
library(xgboost)
library(tidyverse)
library(magrittr)
library(data.table)
library(mosaic)
library(Ckmeans.1d.dp)
library(archdata)
library(plyr)
library(foreign)
#####################################################################################
#check for missing packages and install them:
list.of.packages <- c("knitr", "httr", "readr", "dplyr", "tidyr", "XML",
                      "ggplot2", "stringr", "lubridate", "grid", "caret", 
                      "rpart", "Metrics", "e1071", "ranger", "glmnet", 
                      "randomForest", "ROCR", "gbm", "ipred", "ModelMetrics", 
                      "rpart.plot", "xgboost", "tidyverse", "magrittr", "mosaic",
                      "Ckmeans.1d.dp", "archdata", "plyr", "foreign")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
#####################################################################################
#downloading the two main data sets:
train <- read_csv("train.csv")
test <- read_csv("test.csv")

test$Survived <- 0

# Convert catagorical variables to factors
train$Survived <- factor(train$Survived)
train$Sex <- factor(train$Sex)
train$Pclass <- factor(train$Pclass)
test$Survived <- factor(test$Survived)
test$Sex <- factor(test$Sex)
test$Pclass <- factor(test$Pclass)
test$Embarked <- factor(test$Embarked)

#train$sex.name <- 0
#test$sex.name <- 0
#train$sex.name[!is.na(str_extract(train$Name, "Mr"))] <- "Mr"
#train$sex.name[!is.na(str_extract(train$Name, "Mrs"))] <- "Mrs"
#train$sex.name[!is.na(str_extract(train$Name, "Mme"))] <- "Mrs"
#train$sex.name[!is.na(str_extract(train$Name, "Miss"))] <- "Miss"
#train$sex.name[!is.na(str_extract(train$Name, "Ms"))] <- "Miss"
#train$sex.name[!is.na(str_extract(train$Name, "Mlle"))] <- "Miss"
#train$sex.name[!is.na(str_extract(train$Name, "Capt"))] <- "Officer"
#train$sex.name[!is.na(str_extract(train$Name, "Major"))] <- "Officer"
#train$sex.name[!is.na(str_extract(train$Name, "Col"))] <- "Officer"
#train$sex.name[!is.na(str_extract(train$Name, "Master"))] <- "Master"
#train$sex.name[!is.na(str_extract(train$Name, "Rev"))] <- "Officer"
#train$sex.name[!is.na(str_extract(train$Name, "Dr"))] <- "Officer"
#train$sex.name[!is.na(str_extract(train$Name, "Don"))] <- "Royalty"
#train$sex.name[!is.na(str_extract(train$Name, "Countess"))] <- "Royalty"
#train$sex.name[!is.na(str_extract(train$Name, "Jonkheer"))] <- "Royalty"

#test$sex.name[!is.na(str_extract(test$Name, "Mr"))] <- "Mr"
#test$sex.name[!is.na(str_extract(test$Name, "Mrs"))] <- "Mrs"
#test$sex.name[!is.na(str_extract(test$Name, "Mme"))] <- "Mrs"
#test$sex.name[!is.na(str_extract(test$Name, "Miss"))] <- "Miss"
#test$sex.name[!is.na(str_extract(test$Name, "Ms"))] <- "Miss"
#test$sex.name[!is.na(str_extract(test$Name, "Mlle"))] <- "Miss"
#test$sex.name[!is.na(str_extract(test$Name, "Capt"))] <- "Officer"
#test$sex.name[!is.na(str_extract(test$Name, "Major"))] <- "Officer"
#test$sex.name[!is.na(str_extract(test$Name, "Col"))] <- "Officer"
#test$sex.name[!is.na(str_extract(test$Name, "Master"))] <- "Master"
#test$sex.name[!is.na(str_extract(test$Name, "Rev"))] <- "Officer"
#test$sex.name[!is.na(str_extract(test$Name, "Dr"))] <- "Officer"
#test$sex.name[!is.na(str_extract(test$Name, "Don"))] <- "Royalty"
#test$sex.name[!is.na(str_extract(test$Name, "Countess"))] <- "Royalty"
#test$sex.name[!is.na(str_extract(test$Name, "Jonkheer"))] <- "Royalty"

#test$Name[test$sex.name == 0]
#train$Name[train$sex.name == 0]

#train$sex.name <- factor(train$sex.name)
#test$sex.name <- factor(test$sex.name)
#unique(test$sex.name)
colSums(is.na(test))
colSums(is.na(train))
fulldata <- join(test, train, type = "full")
data <- fulldata %>%
  select(Pclass, Sex, Age, SibSp, Parch, Fare)

data1 <- data %>%
  na.omit()
colSums(is.na(data1))
dim(data1)
names(data1)

age_missing_model <- lm(Age ~ Pclass + Sex + SibSp + Parch +  Fare, data = data1)


fare_missing_model<- lm(Fare ~ Pclass + Sex + SibSp + Parch + Age, data = data1)

#set.seed(1234)
#trControl <- trainControl(method = "cv", number = 5)
#grid <- expand.grid(alpha = 1, lambda = seq(0, 2, length = 101))
#fare_missing_model <- caret::train(Fare ~ Pclass + Sex + SibSp + Parch + Age + sex.name, 
#                                  data1, method = "glmnet", tuneGrid = grid, 
#                                  trControl = trControl, metric = "RMSE", preProcess = c("center", 
#                                                                                         "scale"))
#par(mar = c(4, 4, 0, 0))
#plot(fare_missing_model)
#Beta <- coef(fare_missing_model$finalModel, 1)
#R2 <- fare_missing_model$results$Rsquared[which(grid$lambda == 1)]
#adjRR <- 1 - (1 - R2) * (nrow(data1) - 1)/(nrow(data1) - sum(Beta != 0) - 1)
#adjRR
train$Age[is.na(train$Age)] <- predict(age_missing_model, train)[is.na(train$Age)]
test$Age[is.na(test$Age)] <- predict(age_missing_model, test)[is.na(test$Age)]
test$Fare[is.na(test$Fare)] <- predict(fare_missing_model, test)[is.na(test$Fare)]
train$Embarked[is.na(train$Embarked)] <- "S"
train$Embarked <- factor(train$Embarked)
colSums(is.na(test))
colSums(is.na(train))
#####################################################################################
smp_size <- floor(0.80 * nrow(train))
# set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(train)), size = smp_size)
df_train <- train[train_ind, ]
df_test <- train[-train_ind, ]
#####################################################################################
df_train1 <- df_train %>%
  select(-PassengerId, -Cabin, -Name, -Embarked, -Ticket)
head(df_train1)
#####################################################################################
set.seed(1234)  
grid <-  expand.grid(mtry = c(2,3,4,5,6), splitrule = "gini", min.node.size = 10)
fitControl <- trainControl(method = "CV",
                           number = 10,
                           verboseIter = TRUE)
modmod <- caret::train(y = df_train1$Survived, 
                       x = df_train1[, colnames(df_train1) != "Survived"], 
                       trControl = fitControl, method = "ranger",                          
                       num.trees = 2000,
                       tuneGrid = grid)
print(modmod)
pred2 <- predict(modmod, test)
#####################################################################################
#submission
submit <- data.frame(PassengerId=test$PassengerId, Survived=pred2, 
                     stringsAsFactors = TRUE)
head(submit)
write.csv(submit, file = "submission.csv", row.names=F)