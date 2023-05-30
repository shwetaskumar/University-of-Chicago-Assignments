library(faraway)

dataPath <- "C:/Users/Administrator/Downloads/MScA 31007 Statistical Analysis/Assignment 8"
coagulation <- read.table(paste(dataPath,'Week8_Test_Sample.csv',sep = '/'), header=TRUE)
options(scipen = 999)
#coagulation <- read.table(paste(dataPath,'coagulationdata.csv',sep = '/'), header=TRUE, sep=',')
coagulation

summaryByGroup<-aggregate(Treatment~Output,data=coagulation,FUN=summary)
means<-cbind(Means=summaryByGroup$coag[,4],Sizes=aggregate(coag~diet,data=coagulation,FUN=length)$coag)
rownames(means)<-as.character(summaryByGroup$diet)
means

Group1.dietA<-subset(coagulation,coagulation$diet=="A")
Group1.dietA

summary(Group1.dietA)

mean(Group1.dietA[,1])

coag.model<-lm(Output~Treatment,data=coagulation)
modelSummary<-summary(coag.model)
modelANOVA<-anova(coag.model)
modelANOVA$`Pr(>F)`
modelSummary$coefficients

modelSummary$df

c(Sigma=modelSummary$sigma,Rsquared=modelSummary$r.squared)

modelSummary$fstatistic

modelANOVA$`Sum Sq`
modelANOVA$`F value`

coag.model1<-lm(Output~Treatment-1,data=coagulation)
modelSummary1<-summary(coag.model1)
modelANOVA1<-anova(coag.model1)
modelANOVA1$`Pr(>F)`
modelSummary1$coefficients

modelSummary1$df

anova(lm(Output~Treatment,data=coagulation), lm(Output~Treatment-1,data=coagulation))

c(Sigma=modelSummary1$sigma,Rsquared=modelSummary1$r.squared)

modelSummary1$fstatistic

modelANOVA1$`Sum Sq`


plot(coag.model$residuals)

hist(coag.model$residuals)

qqnorm(coag.model$residuals)
qqline(coag.model$residuals)

coag<-coagulation
coag$x1<-coag$diet=="B"
coag$x2<-coag$diet=="C"
coag$x3<-coag$diet=="D"
coag

coag.model.full<-lm(coag~x1+x2+x3, data=coag)
coag.model.null<-lm(coag~1,data=coag)
anova(coag.model.null,coag.model.full)

summary(coag.model)

anova(coag.model)

grand.mean<-mean(coagulation$coag)
create.vector.of.means<-function(my.group.data) {
  rep(my.group.data[1],my.group.data[2])
}
group.mean<-unlist(apply(means,1,create.vector.of.means))

grand.mean

group.mean

SST<-sum((coagulation$coag-grand.mean)^2)
SSE<-sum((coagulation$coag-group.mean)^2)
SSM<-sum((group.mean-grand.mean)^2)

c(SST=SST,SSE=SSE,SSM=SSM)

anova(coag.model)

anova(coag.model.null,coag.model.full)

