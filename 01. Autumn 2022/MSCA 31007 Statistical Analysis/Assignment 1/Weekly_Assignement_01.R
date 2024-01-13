dataPath <- "C:/Users/Administrator/Downloads/MScA 31007 Statistical Analysis/statistics_01_data/"
data <- read.table(paste(dataPath,'Week1_Test_Sample.csv',sep = '/'), header=TRUE)

data$u
data$v

joint_dist <- prop.table(table(data$u, data$v))
joint_distribution <- matrix(c(joint_dist), nrow=3, ncol=4)

u_Marginal <- prop.table(table(data$u))
v_Marginal <- prop.table(table(data$v))

u_Conditional_v <- joint_distribution[,4]/v_Marginal[4]
v_Conditional_u <- joint_distribution[3,]/u_Marginal[3]

res <-list(Joint_distribution=joint_distribution,
           u_Marginal = u_Marginal,
           v_Marginal = v_Marginal,
           u_Conditional_v = u_Conditional_v,
           v_Conditional_u = v_Conditional_u          )

saveRDS(res, file = paste(dataPath,'result.rds',sep = '/'))
