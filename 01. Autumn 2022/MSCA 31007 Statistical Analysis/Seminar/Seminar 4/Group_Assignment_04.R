library(fitdistrplus)

dataPath <- "C:/Users/Administrator/Downloads/MScA 31007 Statistical Analysis/Seminar 4"
data<-read.csv(file=paste(dataPath,'Method_Moments_Data.csv',sep="/"))

data$A #- sample $A$ from one-dimensional distribution;   
data$B #- #sample $B$ from one-dimensional distribution;   
data$Y #- sample $Y$, response of a linear model;    
data$X #- sample $X$, predictor of a linear model. 

################### SAMPLE A ########################
#Exponential Distribution

Exponential.fit <- fitdistr(data$A, "exponential") #Hypothesis rejected


#Uniform Distribution

Uniform.fit <- fitdist(data$A, "unif")
ks.test(data$A, "punif", min = Uniform.fit$estimate['min'], max = Uniform.fit$estimate['max']) #Hypothesis rejected


#Log Normal Distribution

Log_Normal.fit <- fitdist(data$A, "lnorm") #Hypothesis rejected

#Gamma
Gamma_Fit.fit <- fitdist(data$A, "gamma") #Hypothesis rejected

#Normal
Normal_Fit <- fitdist(data$A, "norm")
ks.test(data$A, "pnorm", mean=Normal_Fit$estimate['mean'], sd = Normal_Fit$estimate['sd'])
c(round(Normal_Fit$estimate['mean'], 3), round(Normal_Fit$estimate['sd'],3))


################### SAMPLE B ########################
#Exponential Distribution

Exponential.fit <- fitdistr(data$B, "exponential") #Hypothesis rejected
ks.test(data$B, "pexp", Exponential.fit$estimate, Exponential.fit$estimate['sd']) #Hypothesis rejected

#Uniform Distribution

Uniform.fit <- fitdist(data$B, "unif")
ks.test(data$B, "punif", min = Uniform.fit$estimate['min'], max = Uniform.fit$estimate['max']) #Hypothesis not rejected
c(round(Uniform.fit$estimate['min'], 3), round(Uniform.fit$estimate['max'],3))

#Log Normal Distribution

Log_Normal.fit <- fitdist(data$B, "lnorm") #Hypothesis rejected
ks.test(data$B, "plnorm", meanlog = Log_Normal.fit$estimate['meanlog'], sdlog = Log_Normal.fit$estimate['sdlog']) #Hypothesis rejected

#Gamma
Gamma_Fit.fit <- fitdistr(data$B, "gamma") #Hypothesis rejected
ks.test(data$B, "pgamma", Gamma_Fit.fit$estimate['shape'], Gamma_Fit.fit$estimate['rate']) #Hypothesis rejected

#Normal
Normal_Fit <- fitdist(data$B, "norm")
ks.test(data$B, "pnorm", mean=Normal_Fit$estimate['mean'], sd = Normal_Fit$estimate['sd']) #Hypothesis rejected
c(round(Normal_Fit$estimate['mean'], 3), round(Normal_Fit$estimate['sd'],3))

#Chi-squared
Chi.fit <- fitdist(data$B, "pchi")
ks.test(data$B, "pchisq")




######################### SAMPLE X and Y ###########################

Estimated.LinearModel <- lm(Y ~ X,data=data)
summary(Estimated.LinearModel)

## Long method using moments (Least squares method) ###
sum_x <- sum(data$X)
sum_y <- sum(data$Y)
sum_xy <- sum(data$X * data$Y)
sum_x2 <- sum(data$X ** 2)
sum_y2 <- sum(data$Y ** 2)

m <- ((500 * sum_xy) - (sum_x * sum_y))/(500 * sum_x2 - ((sum_x) ** 2))

intercept_y <- (sum_y - (m*(sum_x)))/500
res <- data$Y - ((m*data$X) + intercept_y)
res_2 <- res ** 2 
residual <- (sum(res_2)/(length(data$Y) - 2)) ** 0.5
                  