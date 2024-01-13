datapath <- "C:/Users/Administrator/Downloads/MScA 31007 Statistical Analysis/Seminar 7"
test_data <-read.csv(file=paste(datapath,"test_sample.csv",sep="/"),
                     row.names=1,header=TRUE,sep=",")
options(scipen = 999)
AssignmentData <- test_data

head(AssignmentData)

#Plot input variables.
matplot(AssignmentData[,c(1:7)],type='l',ylab="Interest Rates",
        main="History of Interest Rates",xlab="Index")


#Plot input variables together with the output variable.
matplot(AssignmentData[,],type='l',ylab="Interest Rates",
        main="History of Interest Rates and Output",xlab="Index")


library(xts)
library(zoo)

Window.width<-20; 
Window.shift<-5

x <- rollapplyr(AssignmentData, width=Window.width, by=Window.shift, sd)
x[x$Output1 >= 0.30]


x<-rollapply(AssignmentData$Output1, width=Window.width, by=Window.shift, FUN = function(x) sd(x))

x[1]

all.means <- rollapply(zoo(AssignmentData, as.Date(rownames(AssignmentData))), width=Window.width,
          by=Window.shift, FUN = function(x) sd(x))
high.volatility.periods <- all.means[all.means$Output1 >= 0.30]

high.volatility.periods <- index(high.volatility.periods)

res <- list(high.volatility.periods=high.volatility.periods,
            high.slopeSpread.periods=0,
            high.slope5Y=0,
            low.r.squared = 0,
            USGG3M_insignificant=0,
            USGG5Y_insignificant=0,
            USGG30Y_insignificant=0)

saveRDS(res, file = paste(datapath,'result.rds',sep = '/'))

index(all.means$Output1 > 0.30)
all.means<-rollapply(AssignmentData,width=Window.width,
                     by=Window.shift, sd)
x <- as.data.frame(all.means)
y <- x[x$Output1 >= 0.30, ]




head(all.means)
all.means[,7] > 0.30
summary(all.means)


xts_object <- as.xts(AssignmentData, as.Date(rownames(AssignmentData)))

apply(AssignmentData[5:25,],2,sd)
apply(AssignmentData[6:25,],2,sd)
AssignmentData[,0]

Count<-1:dim(AssignmentData)[1]
Rolling.window.matrix<-rollapply(Count,
                                 width=Window.width,
                                 by=Window.shift,
                                 by.column=FALSE,FUN=function(z) stdDev(z))
Rolling.window.matrix[1:10,]    # sequence of index vectors of rolling windows

# Find middle of each window
Points.of.calculation<-Rolling.window.matrix[,10]    
Points.of.calculation[1:10]

length(Points.of.calculation)

Means.forPlot<-rep(NA,dim(AssignmentData)[1])
Means.forPlot[Points.of.calculation]<-all.means[,1]
cbind(1:25,Means.forPlot[1:25])

cbind(originalData=AssignmentData[,1],
      rollingMeans=Means.forPlot)[1:25,]

plot(AssignmentData[,1],type="l",col="blue",lwd=2,
     ylab="Interest Rate & Rolling Mean", 
     main="Rolling Mean of USGG3M")
points(Means.forPlot,col="orange",pch=1)
legend("topright",
       legend=c("USGG3M","Rolling Mean"),
       col=c("blue","orange"),lwd=2)



plot(AssignmentData[,1],type="l",xaxt="n",col="blue",lwd=2,
     ylab="Interest Rate & Rolling Mean", 
     main="Rolling Mean of USGG3M")
axis(side=1,at=1:dim(AssignmentData)[1],rownames(AssignmentData))
points(Means.forPlot,col="orange",pch=1)
legend("topright",
       legend=c("USGG3M","Rolling Mean"),
       col=c("blue","orange"),lwd=2)

