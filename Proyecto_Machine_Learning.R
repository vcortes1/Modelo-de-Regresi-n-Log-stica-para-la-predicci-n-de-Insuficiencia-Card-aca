library(ISLR) 
library(knitr) 
library(ggplot2) 
library(gridExtra)
library(boot)
library(dplyr)
library(caret)
library(MASS)
library(tidyr)
library(psych)
library(glmnet)
library(leaps)
library(pROC)
library(mlbench)

##Cardar la BD de Heart Failure Prediction Dataset a R-Studio
#file.choose()
BDdatos <-  read.csv("C:\\Users\\Rodrigo Araya\\Documents\\PUCV\\PUCV 2023-1\\Machine Learning\\Proyecto\\Heart Failure Prediction Dataset.csv")

##Analisis de la BD original
summary(BDdatos)
str(BDdatos)

##Preprocesamiento de los datos para poder aplicar la Regresi?n Logistica

##vARIABLE SEX
datos_modificados <- BDdatos
datos_modificados$Sex[datos_modificados$Sex=="M"]<-0
datos_modificados$Sex[datos_modificados$Sex=="F"]<-1
datos_modificados$Sex=as.integer(datos_modificados$Sex)
str(datos_modificados)

#VARIABLE CHESTPAINTYPE
datos_modificados$ChestPainType[datos_modificados$ChestPainType=="ASY"]<-1
datos_modificados$ChestPainType[datos_modificados$ChestPainType=="NAP"]<-2
datos_modificados$ChestPainType[datos_modificados$ChestPainType=="ATA"]<-3
datos_modificados$ChestPainType[datos_modificados$ChestPainType=="TA"]<-4
datos_modificados$ChestPainType=as.integer(datos_modificados$ChestPainType)
str(datos_modificados)

#VARIABLE RESTINGECG
datos_modificados$RestingECG[datos_modificados$RestingECG=="Normal"]<-1
datos_modificados$RestingECG[datos_modificados$RestingECG=="LVH"]<-2
datos_modificados$RestingECG[datos_modificados$RestingECG=="ST"]<-3
datos_modificados$RestingECG=as.integer(datos_modificados$RestingECG)
str(datos_modificados)


#VARIABLE EXCERCISEANGINA
datos_modificados$ExerciseAngina[datos_modificados$ExerciseAngina=="Y"]<-0
datos_modificados$ExerciseAngina[datos_modificados$ExerciseAngina=="N"]<-1
datos_modificados$ExerciseAngina=as.integer(datos_modificados$ExerciseAngina)
str(datos_modificados)

#VARIABLE ST_SLOPE
datos_modificados$ST_Slope[datos_modificados$ST_Slope=="Flat"]<-1
datos_modificados$ST_Slope[datos_modificados$ST_Slope=="Up"]<-2
datos_modificados$ST_Slope[datos_modificados$ST_Slope=="Down"]<-3
datos_modificados$ST_Slope=as.integer(datos_modificados$ST_Slope)
str(datos_modificados)

#VARIABLE DE INTERES HEART DISEASE
datos_modificados$HeartDisease[datos_modificados$HeartDisease == 0]<-"neg"
datos_modificados$HeartDisease[datos_modificados$HeartDisease == 1]<-"pos"
datos_modificados$HeartDisease=as.factor(datos_modificados$HeartDisease)
str(datos_modificados)

#ELIMINAR Missing Values
data_ok <- na.omit(datos_modificados)
str(data_ok)

#Creaci?n Particion de Test y Training
set.seed(123)
training.samples <- data_ok$HeartDisease %>% 
  createDataPartition(p = 0.8, list = FALSE)

train  <- data_ok[training.samples, ]
test <- data_ok[-training.samples, ]
summary(train)
summary(test)

#######################################

#Crear modelos de regresi?n Logistica vac?o y completo
modelo_completo <- glm(HeartDisease ~ ., data = train, family = binomial)
modelo_vacio <- glm(HeartDisease ~ 1, data = train, family = binomial)

#Selecci?n de Variables Backward, Fordward y Bothways
backwards = step(modelo_completo,trace = 1)
summary(backwards)
summary(modelo_completo)

forward = step(modelo_vacio,
               scope=list(lower=formula(modelo_vacio),
                          upper=formula(modelo_completo)),
               direction="forward")
summary(forward)

bothways = step(modelo_vacio, 
                list(lower=formula(modelo_vacio),
                     upper=formula(modelo_completo)), 
                direction="both",
                trace=0)

formula(backwards)
formula(forward)
formula(bothways)

###############################
#Entrenamos el Modelo con las variables que nos entrega la formula 
#de Backward ya que son iguales a las de Fordward y Bothways

#Modelo inicial con las variables segun backward (prediccion binaria).
ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProb=FALSE,
                     savePrediction=FALSE
                     #,
                     #summaryFunction = twoClassSummary 
)

LR_prb = train(
  form = formula(backwards),
  data = train,
  trControl = ctrl,
  metric = "Accuracy",
  method = "glm",
  family = "binomial"
)

summary(LR_prb)
confusionMatrix(LR_prb)
confusionMatrix.train(LR_prb)

# Que tan bueno es el modelo con el TRAINING set?
result.predicted <- predict(LR_prb, train) # Prediction
summary(result.predicted)
confusionMatrix(result.predicted,train$HeartDisease, positive = "pos")

# Que tan bueno es el modelo con el TESTING set?
test.predicted <- predict(LR_prb, test) # Prediction
summary(test.predicted)
confusionMatrix(test.predicted,test$HeartDisease, positive = "pos")

################################################################################################################
# Que tan bueno es el modelo con el TESTING set - Probabilistico ROC ?
test.predicted.prob <- predict(LR_prb, test, type="prob") # Prediction
test.roc <- roc(test$HeartDisease, test.predicted.prob$pos) # Draw ROC curve.
plot(test.roc, print.thres = "best")

result.coords <- coords(test.roc, "best", best.method="closest.topleft", ret=c("threshold", "accuracy"))
print(result.coords)#to get threshold and accuracy


##########################################################################################
### DOWNSAMPLING y UPSAMPLING

set.seed(9560)
down_train <- downSample(x = train[,-ncol(train)], #caret
                         y = train$HeartDisease)
table(down_train$Class)   

up_train <- upSample(x = train[, -ncol(train)],
                     y = train$HeartDisease)                         
table(up_train$Class)

ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProb=TRUE,
                     savePrediction=TRUE,
                     summaryFunction = twoClassSummary )

set.seed(5627)
orig_fit <- train(
  form = formula(backwards),
  data = train,
  trControl = ctrl,
  metric = "ROC",
  method = "glm",
  family = "binomial"
)

down <- train(
  form = Class~ Age + Sex + ChestPainType + Cholesterol + FastingBS + MaxHR + ExerciseAngina + Oldpeak + ST_Slope,
  data = down_train,
  trControl = ctrl,
  metric = "ROC",
  method = "glm",
  family = "binomial"
)

up <- train(
  form = Class~ Age + Sex + ChestPainType + Cholesterol + FastingBS + MaxHR + ExerciseAngina + Oldpeak + ST_Slope,
  data = up_train,
  trControl = ctrl,
  metric = "ROC",
  method = "glm",
  family = "binomial"
)


#Original
result.predicted <- predict(orig_fit, train) 
summary(result.predicted)
Matrix_0 <- confusionMatrix(result.predicted,train$HeartDisease,positive = "pos")

test_orig_fit <- predict(orig_fit, test, type="prob") # Prediction
test.roc_orig_fit <- roc(test$HeartDisease, test_orig_fit$pos) # Draw ROC curve.

roc_plot<-plot(test.roc_orig_fit, print.thres = "local maximas",print.auc = TRUE, auc.polygon=TRUE)
plot(smooth(roc_plot), add=TRUE, col="blue")
legend("bottomright", legend=c("Empirical", "Smoothed"),
       col=c(par("fg"), "blue"), lwd=2)

result.coords_Origin <- coords(test.roc_orig_fit, "best",
                        best.method="closest.topleft",
                        ret=c("threshold", "accuracy","specificity", "sensitivity" ))
print(result.coords_Origin)

#Down
result.predicted <- predict(down, train) 
summary(result.predicted)
Matrix_1 <- confusionMatrix(result.predicted,train$HeartDisease,positive = "pos")

test_down <- predict(down, test, type="prob") # Prediction
test.roc_down <- roc(test$HeartDisease, test_down$pos) # Draw ROC curve.

roc_plot<-plot(test.roc_down, print.thres = "local maximas",print.auc = TRUE, auc.polygon=TRUE)
plot(smooth(roc_plot), add=TRUE, col="blue")
legend("bottomright", legend=c("Empirical", "Smoothed"),
       col=c(par("fg"), "blue"), lwd=2)

result.coords_Down <- coords(test.roc_down, "best",
                        best.method="closest.topleft",
                        ret=c("threshold", "accuracy","specificity", "sensitivity" ))
print(result.coords_Down)

#UP
result.predicted <- predict(up, train) 
summary(result.predicted)
Matrix_2 <- confusionMatrix(result.predicted,train$HeartDisease,positive = "pos")

test_up <- predict(up, test, type="prob") # Prediction
test.roc_up <- roc(test$HeartDisease, test_up$pos) # Draw ROC curve.
test.roc_up$auc

roc_plot<-plot(test.roc_up, print.thres = "local maximas",print.auc = TRUE, auc.polygon=TRUE)
plot(smooth(roc_plot), add=TRUE, col="blue")
legend("bottomright", legend=c("Empirical", "Smoothed"),
       col=c(par("fg"), "blue"), lwd=2)

result.coords_Up <- coords(test.roc_up, "best",
                        best.method="closest.topleft",
                        ret=c("threshold", "accuracy","specificity", "sensitivity" ))
print(result.coords_Up)


##Comparaci?n
################

##COMPARAMOS RESULTADOS DE NUESTRO MODELO EN CONJUNTO DE TRAINING
print(result.coords_Origin)
print(result.coords_Down)
print(result.coords_Up)

##COMPARAMOS RESULTADOS DE NUESTRO MODELO EN CONJUUNTO DE TESTING
print(Matrix_0)
print(Matrix_1)
print(Matrix_2)
