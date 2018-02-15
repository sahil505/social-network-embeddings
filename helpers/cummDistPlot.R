# Cumulative distribution plot

library(data.table)
library(plyr)
library(ggplot2)

dat<-fread("G:/socialnetworks_project_log/degree_distribution/numFriendsPerUser.csv")
dat<-as.data.frame(dat)

plot(ecdf(dat$V2)) 

ggplot(dat,aes(x = V2)) + stat_ecdf() +
  scale_x_log10() +
  scale_y_continuous(expand = c(0,0))  + ylab("Cumulative distribution") + xlab("Value") + theme_bw(16)

dat<-fread("G:/socialnetworks_project_log/degree_distribution/numTweetsPerAuthor.csv")
dat<-as.data.frame(dat)
plot(ecdf(dat$V2))

dat<-fread("G:/socialnetworks_project_log/degree_distribution/featuresUserSubset.csv")
dat<-as.data.frame(dat)
ggplot(dat,aes(x = V2)) + stat_ecdf() +
  scale_x_log10() +
  scale_y_continuous(expand = c(0,0))  + ylab("Cumulative distribution") + xlab("Value") + theme_bw(16)
ggplot(dat,aes(x = V3)) + stat_ecdf() +
  scale_x_log10() +
  scale_y_continuous(expand = c(0,0))  + ylab("Cumulative distribution") + xlab("Value") + theme_bw(16)
