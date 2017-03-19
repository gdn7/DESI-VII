install.packages("devtools")
require(devtools)
install_url("http://www.omegahat.org/Rstem/Rstem_0.4-1.tar.gz")
install_url("http://cran.r-project.org/src/contrib/Archive/sentiment/sentiment_0.1.tar.gz")
install_url("http://cran.r-project.org/src/contrib/Archive/sentiment/sentiment_0.2.tar.gz")


install.packages("plyr")
install.packages("ggplot2")
install.packages("wordcloud")
install.packages("RColorBrewer")
install.packages("tm")
install.packages("SnowballC")

library(plyr)
library(ggplot2)
library(wordcloud)
library(RColorBrewer)
library(tm)
library(SnowballC)
library(syuzhet)



workingdirectory = "C:\\Users\\gdnir\\Desktop\\mit and me\\e-discovery " #set path
setwd(workingdirectory)


Hillary=get_text_as_string("C:\\Users\\gdnir\\Desktop\\mit and me\\e-discovery\\test.txt")

token_hillary=get_tokens(Hillary, pattern = "\\W")


textdata <- get_nrc_sentiment(token_hillary)


class_emo = classify_emotion(textdata, algorithm="bayes", prior=1.0)
emotion = class_emo[,7]
emotion[is.na(emotion)] = "unknown"
class_pol = classify_polarity(textdata, algorithm="bayes")
polarity = class_pol[,4]



sent_df = data.frame(text=textdata, emotion=emotion,
                     polarity=polarity, stringsAsFactors=FALSE)
sent_df = within(sent_df,
                 emotion <- factor(emotion, levels=names(sort(table(emotion), decreasing=TRUE))))