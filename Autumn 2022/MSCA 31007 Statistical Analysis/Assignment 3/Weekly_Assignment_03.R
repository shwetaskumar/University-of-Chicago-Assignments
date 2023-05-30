dataPath <- "C:/Users/Administrator/Downloads/MScA 31007 Statistical Analysis/Assignment 3"
dat <- read.table(paste(dataPath,'Week3_Test_Sample.csv',sep = '/'), header=TRUE)

dat$x[1] #mean of normal distribution
dat$x[2] #stddev of normal deviation
dat$x[3] #intensity of exponential distribution
dat$x[4]:dat$x[503] #sample from uniform distribution o [0,1]

datNorm <- qnorm(dat$x[4:503], dat$x[1], dat$x[2])
datExp <- qexp(dat$x[4:503], dat$x[3])

res<-cbind(datNorm=datNorm,datExp=datExp)

write.table(res, file = paste(dataPath,'result.csv',sep = '/'), row.names = F)
