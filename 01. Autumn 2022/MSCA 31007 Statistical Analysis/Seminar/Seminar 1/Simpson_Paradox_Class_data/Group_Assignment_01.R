dataPath <- "C:/Users/Administrator/Downloads/MScA 31007 Statistical Analysis/Simpson_Paradox_Class_data/"
test_data <- read.table(paste(dataPath,'test_sample.csv',sep = '/'), header=TRUE)

#test_data_df <- as.data.frame.matrix(test_data) 
#test_data_df[(test_data_df$A > test_data_df$B) & (test_data_df$A > test_data_df$C)]
#test_data_df$A


# Load dplyr package
library("dplyr")

# Using filter()
quest_1 <- filter(test_data, ((B==5) & (B>A) & (B>C)))
quest_1_prob <- length(quest_1$B)/length(filter(test_data, B==5)$B)

quest_2 <- filter(test_data, ((C==4) & (C>A) & (C>B)))
quest_2_prob <- length(quest_2$C)/length(filter(test_data, C==4)$C)

a_greater <- filter(test_data, (A > B) & (A > C))
a_greater_prob <- length(a_greater$A)/length(test_data$A)


b_greater <- filter(test_data, (B > A) & (B > C))
b_greater_prob <- length(b_greater$B)/length(test_data$B)

c_greater <- filter(test_data, (C > A) & (C > B))
c_greater_prob <- length(c_greater$C)/length(test_data$C)

quest_6_A <- length(filter(test_data, (A>B))$A)/length(test_data$A)
quest_6_B <- length(filter(test_data, (B>A))$B)/length(test_data$A)

quest_7_A <- length(filter(test_data, ((A>B) & (A>C)))$A)/length(test_data$A)
quest_7_B <- length(filter(test_data, ((B>A) & (B>C)))$B)/length(test_data$B)
quest_7_C <- length(filter(test_data, ((C>A) & (C>B)))$C)/length(test_data$C)

res <-list(quest_1_prob=quest_1_prob,
           quest_2_prob=quest_2_prob,
           a_greater_prob = a_greater_prob,
           b_greater_prob = b_greater_prob,
           c_greater_prob = c_greater_prob,
           quest_6_A=quest_6_A,
           quest_6_B=quest_6_B,
           quest_7_A=quest_7_A,
           quest_7_B=quest_7_B,
           quest_7_C=quest_7_C )

res
