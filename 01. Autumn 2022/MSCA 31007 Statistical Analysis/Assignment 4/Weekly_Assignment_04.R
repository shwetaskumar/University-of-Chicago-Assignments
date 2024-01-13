dataPath <- "C:/Users/Administrator/Downloads/MScA 31007 Statistical Analysis/Assignment 4"
dat <- read.table(paste(dataPath,'Week4_Test_Sample.csv',sep = '/'), header=TRUE)

dat$X #predictor
dat$Y #Output


plot(dat$X,dat$Y)

Estimated.LinearModel <- lm(Y ~ X,data=dat) #Fit the linear model
names(Estimated.LinearModel)

summary(Estimated.LinearModel)

Estimated.Residuals <- Estimated.LinearModel$residuals
plot(dat$X, Estimated.Residuals)  #Analyze the residuals


Probability.Density.Residuals <- density(Estimated.Residuals)
plot(Probability.Density.Residuals, ylim = c(0, 0.5))
lines(Probability.Density.Residuals$x, dnorm(Probability.Density.Residuals$x, 
                                             mean = mean(Estimated.Residuals), sd = sd(Estimated.Residuals)))

c(Left.Mean = mean(Estimated.Residuals[Estimated.Residuals < 0]), 
  Right.Mean = mean(Estimated.Residuals[Estimated.Residuals > 0]))

x <- c(Estimated.Residuals[Estimated.Residuals < 0])
y <- c(Estimated.Residuals[Estimated.Residuals > 0])

Estimated.Residuals[Estimated.Residuals < 0] <- 0
Estimated.Residuals[Estimated.Residuals > 0] <- 1

Unscrambled.Selection.Sequence <- Estimated.Residuals

res <- list(Unscrambled.Selection.Sequence =  Unscrambled.Selection.Sequence)

write.table(res, file = paste(dataPath,'result.csv',sep = '/'), row.names = F)
