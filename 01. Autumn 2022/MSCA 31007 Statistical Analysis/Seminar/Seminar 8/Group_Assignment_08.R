#Seminar Project
PlantGrowth
levels(PlantGrowth$group)
plot(weight~group,PlantGrowth)
aggregate(weight~group,PlantGrowth,length)
mo<-lm(weight~group,PlantGrowth)
qqnorm(mo$residuals)
qqline(mo$residuals)
ks.test(mo$residuals,"pnorm")
plot(mo$fitted.values,mo$residuals)
aggregate(weight~group,PlantGrowth,sd)

#--Treatment Significance
summary(mo)
anova(mo)

mo$coefficients

confint(mo)

confint(mo, level=.8)

confint(mo, level=.8)

A<-PlantGrowth$group
contrasts(A)

(contrasts(A)<-cbind("Ctrlvst1"=c(1,-1,0),"Ctrlvst2"=c(1,0,-1)))

summary.lm(aov(PlantGrowth$weight~A))

TukeyHSD(aov(weight~group,PlantGrowth))


#Workshop
dataPath <- "C:/Users/Administrator/Downloads/MScA 31007 Statistical Analysis/Seminar 8"
data1 <- read.csv(paste(dataPath, "test_sample_1.csv", sep="/"))
#levels(data1$t)
#plot(x~t,data1)
aggregate(x~t,data1,length)
mo<-lm(x~t,data1)
TukeyHSD(aov(x~t,data1), conf.level=0.01)
