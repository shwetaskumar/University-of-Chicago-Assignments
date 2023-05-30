dataPath <- "C:/Users/Administrator/Downloads/MScA 31007 Statistical Analysis/Assignment 7"
test_dat <- read.table(paste(dataPath,'Week7_Test_Sample.csv',sep = '/'), header=TRUE)

#Regression.ANOVA.Data<-
#  read.csv(file=paste(dataPath,"DataForRegressionANOVA.csv",sep="/"),
#           header=TRUE,sep=",")
#head(Regression.ANOVA.Data)

#test_dat$Output #output Y values;
#test_dat$Input1 #first predictor values;
#test_dat$Input2 #second predictor values.

Regression.ANOVA.Data <- test_dat
head(Regression.ANOVA.Data)

fit.1<-lm(Output~1,data=Regression.ANOVA.Data)
fit.1.2<-lm(Output~1+Input1,data=Regression.ANOVA.Data)
fit.1.3<-lm(Output~1+Input2,data=Regression.ANOVA.Data)
fit.1.2.3<-lm(Output~.,data=Regression.ANOVA.Data)

anova(fit.1.2)

anova(fit.1.2)$Df

round(anova(fit.1.2)$"Sum Sq",4)

anova(fit.1.2)$"F value"[1]

anova(fit.1.2)$"Pr(>F)"[1]

summary(fit.1)

anova(fit.1)

c(anova(fit.1)$"Sum Sq",sum(fit.1$residuals^2))

c(anova(fit.1)$Df,fit.1$df.residual,summary(fit.1)$df[2])

summary(fit.1.2)

anova(fit.1.2)

summary(fit.1.2)$fstatistic

c(F.value=anova(fit.1.2)$"F value"[1],Df=anova(fit.1.2)$Df,P.value=anova(fit.1.2)$"Pr(>F)"[1])

summary(fit.1.2)$r.squared

summary(fit.1.3)

anova(fit.1.3)

c(F.value=anova(fit.1.3)$"F value"[1],Df=anova(fit.1.3)$Df,P.value=anova(fit.1.3)$"Pr(>F)"[1])

summary(fit.1.3)$fstatistic

summary(fit.1.2.3)

anova(fit.1.2.3)

anova(fit.1.2,fit.1.2.3)

summary(fit.1.2.3)

round(anova(fit.1,fit.1.2.3)$"Pr(>F)"[2], 4)

summary(fit.1.2.3)

c(anova(fit.1,fit.1.2.3)$F[2],summary(fit.1.2.3)$fstatistic[1])


round(anova(fit.1.3)$"Sum Sq",4)
