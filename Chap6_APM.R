library(AppliedPredictiveModeling)
data(solubility)
##데이터 객체는"sol"로 시작한다.
ls(pattern = "^solT")

str(solTestX)
str(solTestXtrans) #박스-콕스 변형 처리 된 형태
str(solTrainY) #용해도를 수치형 벡터로 저장

set.seed(2)
sample(names(solTestX), 8)

#####일반 선형 회귀#####
trainingData <- solTrainXtrans 
head(trainingData)
trainingData$Solubiliy <- solTrainY #용해도 결과값을 데이터에 추가한다.

lmFitAllPredictors <- lm(Solubiliy ~ ., data = trainingData)
summary(lmFitAllPredictors)

lmPred1 <- predict(lmFitAllPredictors, solTestXtrans)
head(lmPred1)

#모델 성능 추정하기
lmValues1 <- data.frame(obs = solTestY, pred = lmPred1)
defaultSummary(lmValues1)

##로버스트 선형 회귀 모델_후버방식인 rlm function
library(MASS)
rlmFitAllPredictors <- rlm(Solubiliy ~ ., data = trainingData)
ctrl <- trainControl(method = "cv", number = 10) #trainContral함수는 리샘플링 유형을 정의한다.

set.seed(100)
#train함수는 리샘플의 성능 추정값을 구한다. 
lmFit1 <- train(x = solTrainXtrans, y = solTrainY, method = "lm", trControl = ctrl)
lmFit1

##모델의 잔차 대비 예측값 그래프
#type = "p"는 산점도, "g"는 격자선
xyplot(solTrainY ~ predict(lmFit1), type = c("p", "g"), xlab = "Predicted", ylab = "Observed")
xyplot(resid(lmFit1) ~ predict(lmFit1), type = c("p","g"), xlab = "Predicted", ylab = "Residuals")

##절대값이 0.9가 넘는 상관관계를 갖는 변수 상이 생기지 않도록 예측 변수의 수를 조절한다. 
corThresh <- 0.9
tooHigh <- findCorrelation(cor(solTrainXtrans), corThresh)
head(tooHigh)
corrPred <- names(solTrainXtrans)[tooHigh]
head(corrPred)
trainXfiltered <- solTrainXtrans[, -tooHigh]
testXfiltered <- solTrainXtrans[, -tooHigh]
    #lm함수를 사용해 
set.seed(100)
lmFiltered <- train(trainXfiltered, solTrainY, method = "lm", trControl = ctrl) #filtered변수이용.  
lmFiltered

    #rlm함수를 사용해 로버스트 선형회귀 수행 
set.seed(100)
rlmPCA<-train(trainXfiltered, solTrainY, method = "rlm", preProcess = "pca", trControl = ctrl)
rlmPCA


#####부분최소제곱#####
install.packages("pls")
library(pls)
plsFit <- plsr(Solubiliy ~ ., data = trainingData) #plsr함수는 lm함수처럼 모델 식을 사용한다.
predict(plsFit, solTestXtrans[1:5, ], ncomp = 1:2)

set.seed(10)
plsTune <- train(solTrainXtrans, solTrainY, method = "pls", tuneLength = 20, trContral = ctrl, preProc = c("center", "scale"))

#####벌점회귀모델#####
ridgeModel <- enet(x = as.matrix(solTrainXtrans), y = solTrainY, lambda = 0.001)
ridgePred <- predict(ridgeModel, newx = as.matrix(solTestXtrans), s = 1, mode = "fraction", type = "fit")
head(ridgePred)

ridgeGrid <- data.frame(.lambda = seq(0, 0.1, length = 15))
head(ridgeGrid)
set.seed(100)
ridgeRegFit <- train(solTrainXtrans, solTrainY, method = "ridge", tuneGrid = ridgeGrid, trControl = ctrl, preProc = c("center", "scale"))
ridgeRegFit

    ##lars함수를 이용한 라소모델
enetModel <- enet(x = as.matrix(solTrainXtrans), y = solTrainY, lambda = 0.01, normalize = TRUE)
enetPred <- predict(enetModel, newx = as.matrix(solTrainXtrans), s = 0.1, mode = "fraction", type = "fit")
names(enetPred)
head(enetPred$fit)
enetCoef <- predict(enetModel, newx = as.matrix(solTestXtrans), s= 0.1, mode = "fraction", type = "coefficients")
tail(enetCoef$coefficients)

    ##biglars(데이터 세트가 큰 경우), FLLat(혼합라소모델), grplasso(그룹라소모델), penalized. relaxo(완화라소모델)
enetGrid <- expand.grid(.lambda = c(0, 0.01, 0.1), .fraction = seq(0.05, 1, length = 20))
set.seed(100)
enetTune <- train(solTrainXtrans, solTrainY, method = "enet", tuneGrid = enetGrid, trControl = ctrl, preProc = c("center","scale"))
plot(enetTune)
