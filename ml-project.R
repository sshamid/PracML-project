
library(Hmisc)
library(caret)
library(ggplot2)
library(randomForest)
library(foreach)
library(doParallel)
set.seed(137)
options(warn=-1)

training_data <- read.csv("pml-training.csv", na.strings=c("#DIV/0!") )
eval_data <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!") )
dim(training_data)
head(eval_data)
training_data[1:6, 150:160]

nav <- sapply(colnames(training_data), function(x) if(sum(is.na(training_data[, x])) > 0.8*nrow(training_data)){return(T)}else{return(F)})
training_data <- training_data[, !nav]
dim(training_data)

I also remove near zero covariates

nsv <- nearZeroVar(training_data, saveMetrics = T)
training_data <- training_data[, !nsv$nzv]
dim(training_data)
names(training_data)

I also remove user name, timestamps and windows.

model_data <- training_data[,7:ncol(training_data)]

head(model_data)
dim(model_data)

idx <- createDataPartition(y=model_data$classe, p=0.75, list=FALSE )
training <- model_data[idx,]
testing <- model_data[-idx,]


registerDoParallel()
x <- training[-ncol(training)]
y <- training$classe

rf <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
  randomForest(x, y, ntree=ntree) 
}
predictions1 <- predict(rf, newdata=training)
confusionMatrix(predictions1,training$classe)


predictions2 <- predict(rf, newdata=testing)
confusionMatrix(predictions2,testing$classe)