library(dplyr)
library(ggplot2)
library(corrplot)

train <- read.csv("train_no_missing.csv")

anyNA(train)

# colnames(train)

# str(train)

plot(density(train$SalePrice))

hist(train$SalePrice,breaks=25)

# boxplot for SalePrice vs Neighborhood, fill by Overall Condition


# boxplot(SalePrice ~ Neighborhood , train)
train %>% ggplot(aes(Neighborhood,SalePrice, fill=OverallQual)) + geom_boxplot() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    scale_y_continuous(trans = "log2")

# boxplot for SalePrice vs Neighborhood, fill by Overall Condition, facet by year


# boxplot(SalePrice ~ Neighborhood , train)
train %>% ggplot(aes(Neighborhood,SalePrice, fill=OverallQual)) + geom_boxplot() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    scale_y_continuous(trans = "log2") +
    facet_grid(OverallCond~.)

# boxplot for SalePrice vs Neighborhood, fill by Overall Quality, facet by year

train %>% ggplot(aes(Neighborhood,SalePrice, fill=OverallQual)) + geom_boxplot() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    scale_y_continuous(trans = "log2") +
    facet_grid(YrSold~.)

# Overall Quality vs Sale Price, color by External Quality, label by Neighborhood , facet by Year sold

train %>% ggplot(aes(OverallQual,SalePrice,color=ExterQual,label=Neighborhood)) + 
geom_text() + 
scale_x_continuous(trans="log2") +
facet_grid(YrSold~.)

#how sale price changed over the years by neighborhood
train %>% filter(.,Neighborhood=="StoneBr") %>% ggplot(aes(YrSold,SalePrice,group=Neighborhood)) + 
geom_line()




