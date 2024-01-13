dataPath <- "C:/Users/Administrator/Downloads/MScA 31007 Statistical Analysis/Assignment 5"
dat <- read.table(paste(dataPath,'Week5_Test_Sample.csv',sep = '/'), header=TRUE)
#dat<-read.csv(file=paste(dataPath,"Res_Test.csv",sep="/"),header=TRUE,sep=",")

dat$Output  #output Y values;
dat$Input  #predictor X values.

plot(dat$Input,dat$Output, type="p",pch=19) #Project Data
nSample<-length(dat$Input)

GeneralModel <- lm(Output ~ Input,data=dat)
GeneralModel$coefficients

matplot(dat$Input,cbind(dat$Output,GeneralModel$fitted.values),type="p",pch=16,ylab="Sample and Fitted Values")

summary(GeneralModel)

#Plot residuals
estimatedResiduals<-GeneralModel$residuals
plot(dat$Input,estimatedResiduals)

Probability.Density.Residuals<-density(estimatedResiduals)
plot(Probability.Density.Residuals,ylim=c(0,.5))
lines(Probability.Density.Residuals$x,
      dnorm(Probability.Density.Residuals$x,mean=mean(estimatedResiduals),sd=sd(estimatedResiduals)))


# Create NA vectors
Train.Sample<-data.frame(trainInput=dat$Input,trainOutput=rep(NA,nSample))
Train.Sample.Steeper<-data.frame(trainSteepInput=dat$Input,
                                 trainSteepOutput=rep(NA,nSample))  
Train.Sample.Flatter<-data.frame(trainFlatInput=dat$Input,
                                 trainFlatOutput=rep(NA,nSample)) 

head(cbind(dat,
           Train.Sample,
           Train.Sample.Steeper,
           Train.Sample.Flatter))

# Create selectors
Train.Sample.Selector<- dat$Input>=-0.2
Train.Sample.Steeper.Selector<-Train.Sample.Selector&
  (dat$Output>GeneralModel$fitted.values)
Train.Sample.Flatter.Selector<-Train.Sample.Selector&
  (dat$Output<=GeneralModel$fitted.values)

# Select sub samples

Train.Sample[Train.Sample.Selector,2]<-dat[Train.Sample.Selector,1]
Train.Sample.Steeper[Train.Sample.Steeper.Selector,2]<-dat[Train.Sample.Steeper.Selector,1]
Train.Sample.Flatter[Train.Sample.Flatter.Selector,2]<-dat[Train.Sample.Flatter.Selector,1]
head(Train.Sample)

head(cbind(dat,
           Train.Sample,
           Train.Sample.Steeper,
           Train.Sample.Flatter),10)


plot(Train.Sample$trainInput,Train.Sample$trainOutput,pch=16,ylab="Training Sample Output",
     xlab="Training Sample Input")
points(Train.Sample.Steeper$trainSteepInput,Train.Sample.Steeper$trainSteepOutput,pch=20,col="green")
points(Train.Sample.Flatter$trainFlatInput,Train.Sample.Flatter$trainFlatOutput,pch=20,col="blue")

#Fit Training Samples
Train.Sample.Steep.lm <- lm(trainSteepOutput ~ trainSteepInput, data = Train.Sample.Steeper)
summary(Train.Sample.Steep.lm)$coefficients

Train.Sample.Flat.lm <- lm(trainFlatOutput ~ trainFlatInput, data = Train.Sample.Flatter)
summary(Train.Sample.Flat.lm)$coefficients

rbind(Steeper.Coefficients=Train.Sample.Steep.lm$coefficients,
      Flatter.Coefficients=Train.Sample.Flat.lm$coefficients)


plot(dat$Input,dat$Output, type="p",pch=19)
lines(dat$Input,predict(Train.Sample.Steep.lm,
                        data.frame(trainSteepInput=dat$Input),
                        interval="prediction")[,1],col="red",lwd=3)
lines(dat$Input,predict(Train.Sample.Flat.lm,data.frame(trainFlatInput=dat$Input),
                        interval="prediction")[,1],col="green",lwd=3)


# Define the distances from each Output point to both estimated training lines
Distances.to.Steeper<-abs(dat$Output-
                            dat$Input*Train.Sample.Steep.lm$coefficients[2]-
                            Train.Sample.Steep.lm$coefficients[1])
Distances.to.Flatter<-abs(dat$Output-
                            dat$Input*Train.Sample.Flat.lm$coefficients[2]-
                            Train.Sample.Flat.lm$coefficients[1])


# Define the unscramble sequence
# Define separating sequence which equals TRUE if observation belongs to model with steeper slope and FALSE otherwise.
Unscrambling.Sequence.Steeper<-Distances.to.Steeper<Distances.to.Flatter

# Define  two sub-samples with NAs in the Output columns
Subsample.Steeper<-data.frame(steeperInput=dat$Input,steeperOutput=rep(NA,nSample))
Subsample.Flatter<-data.frame(flatterInput=dat$Input,flatterOutput=rep(NA,nSample))

# Fill in the unscrambled outputs instead of NAs where necessary
Subsample.Steeper[Unscrambling.Sequence.Steeper,2]<-dat[Unscrambling.Sequence.Steeper,1]
Subsample.Flatter[!Unscrambling.Sequence.Steeper,2]<-dat[!Unscrambling.Sequence.Steeper,1]

# Check the first rows
head(cbind(dat,Subsample.Steeper,Subsample.Flatter))

# Plot the unscrambled sub-samples, include the original entire sample as a check
matplot(dat$Input,cbind(dat$Output,
                        Subsample.Steeper$steeperOutput,
                        Subsample.Flatter$flatterOutput),
        type="p",col=c("black","green","blue"),
        pch=16,ylab="Separated Subsamples")

# Mixing Probability Of Steeper Slope
(Mixing.Probability.Of.Steeper.Slope<-sum(Unscrambling.Sequence.Steeper)/length(Unscrambling.Sequence.Steeper))


binom.test(sum(Unscrambling.Sequence.Steeper), nSample, p=0.5)


#Fitting models to separated samples
Linear.Model.Steeper.Recovered <- lm(steeperOutput ~ steeperInput, data = Subsample.Steeper)
Linear.Model.Flatter.Recovered <- lm(flatterOutput ~ flatterInput, data = Subsample.Flatter)

rbind(Steeper.Coefficients=Linear.Model.Steeper.Recovered$coefficients,
      Flatter.Coefficients=Linear.Model.Flatter.Recovered$coefficients)
summary(Linear.Model.Steeper.Recovered)$r.sq
summary(Linear.Model.Flatter.Recovered)$r.sq

# Plot residuals
matplot(dat$Input,cbind(c(summary(Linear.Model.Steeper.Recovered)$residuals,
                          summary(Linear.Model.Flatter.Recovered)$residuals),
                        estimatedResiduals),type="p",pch=c(19,16),ylab="Residuals before and after unscrambling")
legend("bottomleft",legend=c("Before","After"),col=c("red","black"),pch=16)

unmixedResiduals<-c(summary(Linear.Model.Steeper.Recovered)$residuals,
                    summary(Linear.Model.Flatter.Recovered)$residuals)
apply(cbind(ResidualsAfter=unmixedResiduals,
            ResidualsBefore=estimatedResiduals),2,sd)


suppressWarnings(library(fitdistrplus))
hist(unmixedResiduals)

(residualsParam<-fitdistr(unmixedResiduals,"normal"))

ks.test(unmixedResiduals,"pnorm",residualsParam$estimate[1],residualsParam$estimate[2])

qqnorm(unmixedResiduals)
qqline(unmixedResiduals)

# Slopes
c(Steeper.SLope=Linear.Model.Steeper.Recovered$coefficients[2],Flatter.Slope=Linear.Model.Flatter.Recovered$coefficients[2])

# Intercepts
c(Steeper.Intercept=Linear.Model.Steeper.Recovered$coefficients[1],Flatter.Intercept=Linear.Model.Flatter.Recovered$coefficients[1])


#Assignment submission
mSteep <- Linear.Model.Steeper.Recovered
mFlat <- Linear.Model.Flatter.Recovered

res <- list( GeneralModel = GeneralModel,mSteep = mSteep,mFlat = mFlat)
saveRDS(res, file = paste(dataPath,'result.rds',sep = '/'))
######## Method 2 #########
plot(dat$Input,(dat$Output-mean(dat$Output))^2, type="p",pch=19,
     ylab="Squared Deviations")



plot(dat$Input,(dat$Output-mean(dat$Output))^2, type="p",pch=19,
     ylab="Squared Deviations")
points(dat$Input,clusteringParabola,pch=19,col="red")
