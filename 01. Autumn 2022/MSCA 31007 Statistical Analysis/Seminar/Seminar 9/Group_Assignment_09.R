dataPath <- "C:/Users/Administrator/Downloads/MScA 31007 Statistical Analysis/Seminar 9/glass.csv"
glass<-read.csv(dataPath)
head(glass)

dim(glass)

features<-glass[,-10]
head(features)

featNorm<-as.data.frame(apply(features,2,function(z) z/sd(z)))  

pairs(featNorm)
cor(featNorm)

princomp(featNorm)


idx2<-which(glass$Type==2)
idx3<-which(glass$Type==3)
idx5<-which(glass$Type==5)
idx6<-which(glass$Type==6)
idx7<-which(glass$Type==7)
plot(factors[,1],factors[,2],pch=16,main="Classes by F1 and F2")
points(factors[idx2,1],factors[idx2,2],pch=16,col="blue")
points(factors[idx3,1],factors[idx3,2],pch=16,col="orange")
points(factors[idx5,1],factors[idx5,2],pch=16,col="magenta")
points(factors[idx6,1],factors[idx6,2],pch=16,col="cyan")
points(factors[idx7,1],factors[idx7,2],pch=16,col="gold")
legend("bottomright",legend=paste0("C",c(1:3,5:7)),pch=16,
       col=c("black","blue","orange","magenta","cyan","gold"))

