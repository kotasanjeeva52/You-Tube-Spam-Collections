rm(list = ls(all=T))

library(readr)
library(caTools)
library(e1071)
library(rpart)
library(rpart.plot)
library(wordcloud)
library(caret)



getwd()
setwd("G:/INTERNSHIP DATA")
list.files()
read.csv("Youtube01-Psy.csv",header = T,sep = ",",stringsAsFactors = F)->psy
read.csv("Youtube02-KatyPerry.csv",header = T,sep = ",",stringsAsFactors = F)->katyperry
read.csv("Youtube03-LMFAO.csv",header = T,sep = ",",stringsAsFactors = F)->LMFAO
read.csv("Youtube04-Eminem.csv",header = T,sep = ",",stringsAsFactors = F)->Eminem
read.csv("Youtube05-Shakira.csv",header = T,sep = ",",stringsAsFactors = F)->Shakira
realdata <- rbind(Eminem,katyperry,LMFAO,psy,Shakira)
realdata
names(realdata)
str(realdata)
#remove meaningless attributes in data.
realdata[-c(1:3)]->san_data
san_data
names(san_data)
sum(is.na(san_data))
length(which(!complete.cases(san_data)))
table(san_data$CLASS)

prop.table(table(san_data$CLASS))

#no.of words in particular Content.
san_data$CONTENT_length=nchar(san_data$CONTENT)
summary(san_data$CONTENT_length)

library(tm)
getSources()
getReaders()

youtube_spam=Corpus(VectorSource(san_data$CONTENT))
print(youtube_spam)
inspect(youtube_spam[1:5])

##########################################################################################

##clean the corpus

#translate all letters to lower case
clean_corpus <- tm_map(youtube_spam, tolower)
# remove numbers
clean_corpus <- tm_map(clean_corpus, removeNumbers)
# remove punctuation
clean_corpus <- tm_map(clean_corpus, removePunctuation)
inspect(clean_corpus[1:3])
length(stopwords("english"))
stopwords()[1:10]
clean_corpus <- tm_map(clean_corpus, removeWords, stopwords())

clean_corpus <- tm_map(clean_corpus, stripWhitespace)

###########################################################################################
## Convert bag of words to data frame.

yt_dtm <- DocumentTermMatrix(clean_corpus)
inspect(yt_dtm[1:4, 30:35])
# look at words that appear atleast 200 times
findFreqTerms(yt_dtm, lowfreq = 90)
sparseWords <- removeSparseTerms(yt_dtm, 0.795)
# convert the matrix of sparse words to data frame
sparseWords <- as.data.frame(as.matrix(sparseWords))
# rename column names to proper format in order to be used by R
colnames(sparseWords) <- make.names(colnames(sparseWords))
str(sparseWords)
sparseWords$CLASS<-san_data$CLASS


##Predicting whether SMS is spam/non-spam
##split data into 75:25 and assign to train and test.
set.seed(987)
split <- sample.split(sparseWords$CLASS, SplitRatio = 0.75)
train <- subset(sparseWords, split == T)

test <- subset(sparseWords, split == F)


##Baseline Model(predicting every message as non-spam)
table(train$CLASS)

table(test$CLASS)

svm.model <- svm(CLASS ~ ., data = train, kernel = "radial", cost = 0.1, gamma = 0.1)

summary(svm.model)
head(train$CLASS)
head(test$CLASS)
predict(svm.model,train)->pr
table(pr,train$CLASS)->t
as.data.frame(t)->ss
#svm.accuracy.table <- as.data.frame(table(test$CLASS, svm.model))
print(paste("SVM accuracy:",
            100*round(((ss$Freq[1]+ss$Freq[4])/nrow(train)), 4),
            "%"))
svm.predict <- predict(svm.model, test)
table(test$CLASS, svm.predict)->tab

library(caret)
#confusionMatrix(tab)

svm.accuracy.table <- as.data.frame(table(test$CLASS, svm.predict))
print(paste("SVM accuracy:",
            100*round(((svm.accuracy.table$Freq[1]+svm.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))
############################################################################################

##Decision Tree

library(rpart)


tree.model <- rpart(CLASS ~ ., data = train, method = "class", minbucket = 35)

# visualize the decision tree. It tells us about significant words.
summary(tree.model)
prp(tree.model)

tree.predict <- predict(tree.model, test, type = "class")
table(test$CLASS, tree.predict)->t1
library(caret)
#confusionMatrix(t1)

######################################################################################
## Neural networks
library(caret)
library(nnet)
#M1

train.nnet<-nnet(CLASS~.,data=train[,10],size=3,rang=0.05,Hess=T,decay=15e-4,maxit=450)
#import the function from Github
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')


plot.nnet(train.nnet)


test.nnet<-predict(train.nnet,test)

round(test.nnet,digits = 0)->p
p
confusionMatrix(p,test$CLASS)

test.nnet

table(test$CLASS,test.nnet)->tb
which.is.max(test.nnet)

confusionMatrix(test$CLASS,test.nnet)

#confusionMatrix(tb)
#M2

nn = nnet(CLASS ~ ., data = train, size = 8, rang = 0.1, decay = 5e-4, maxit = 50)
#plot each model
plot.nnet(nn)


pr<-predict(nn,test)
round(pr,digits = 0)->pr1
pr1
#confusionMatrix(pr1,test$CLASS)
#seedsANN = nnet(irisdata[seedstrain,-8], ideal[seedstrain,], size=10, softmax=TRUE)
class(test.nnet)
head(test.nnet)
head(test$CLASS)
test.nnet2<-predict(object = train.nnet,newdata = test)
test.nnet2<-round(test.nnet,digits = 0)
head(test.nnet2)
#confusionMatrix(data = test.nnet2,reference = test$CLASS)
?nnet
?nnet
install.packages("neuralnet")

##M3


library(neuralnet)
ctrain<-colnames(train)[-256]
length(ctrain)
f<-paste(ctrain,collapse = "+")
f<-as.formula(paste("CLASS~",f))
f
train.nnet2<-neuralnet(formula = f,data = train,hidden = c(10,5,6),threshold = 0.01,
                       rep = 4,lifesign = "full",algorithm = "rprop+",err.fct = "ce",
                       act.fct = "logistic",linear.output = F)

nn <- neuralnet(f,data=train,hidden=c(10,5,3),linear.output=T)

nn1 <- neuralnet(f,data=train,hidden=5,linear.output=T)

pred_train<-compute(x = train.nnet2,covariate = train[-256])
head(pred_train$net.result)

pred_train<-round(pred_train$net.result,digits = 0)











head(pred_train)
conf<-confusionMatrix(data = pred_train,reference = train$CLASS)
conf
pred_test<-compute(x = train.nnet2,covariate = test[-256])
pred_test<-round(pred_test$net.result)
conf2<-confusionMatrix(data = pred_test,reference = test$CLASS)
conf2
??train.nnet2






#mlp function from RSNNS package
library(RSNNS)
set.seed(45)
mod3<-mlp(train[,-256], test[,-256], size=10,linOut=T)
train.nnet<-mlp(CLASS~.,train,size=3,rang=0.05,Hess=FALSE,decay=15e-4,maxit=450)





#import the function from Github
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')



#plot each model
plot.nnet(nn)
plot.nnet(train.nnet)

##################################################################################

## KNN Algorithm.




rm(list = ls(all=T))

getwd()
setwd("G:/INTERNSHIP DATA")
list.files()
read.csv("Youtube01-Psy.csv",header = T,sep = ",",stringsAsFactors = F)->psy
read.csv("Youtube02-KatyPerry.csv",header = T,sep = ",",stringsAsFactors = F)->katyperry
read.csv("Youtube03-LMFAO.csv",header = T,sep = ",",stringsAsFactors = F)->LMFAO
read.csv("Youtube04-Eminem.csv",header = T,sep = ",",stringsAsFactors = F)->Eminem
read.csv("Youtube05-Shakira.csv",header = T,sep = ",",stringsAsFactors = F)->Shakira
realdata <- rbind(Eminem,katyperry,LMFAO,psy,Shakira)
realdata

str(realdata)
#remove meaningless attributes in data.
realdata[-c(1:3)]->san_data
san_data
names(san_data)
sum(is.na(san_data))
length(which(!complete.cases(san_data)))
table(san_data$CLASS)

prop.table(table(san_data$CLASS))

#no.of words in particular Content.
san_data$CONTENT_length=nchar(san_data$CONTENT)
summary(san_data$CONTENT_length)

library(tm)
getSources()
getReaders()

you_spam=Corpus(VectorSource(san_data$CONTENT))
print(you_spam)
inspect(you_spam[1:5])

###### clean the corpus

#translate all letters to lower case
clean_corpus <- tm_map(you_spam, tolower)
# remove numbers
clean_corpus <- tm_map(clean_corpus, removeNumbers)
# remove punctuation
clean_corpus <- tm_map(clean_corpus, removePunctuation)
inspect(clean_corpus[1:3])
length(stopwords("english"))
stopwords()[1:10]
clean_corpus <- tm_map(clean_corpus, removeWords, stopwords())

clean_corpus <- tm_map(clean_corpus, stripWhitespace)

#corpus.stemmed <- tm_map(clean_corpus, stemDocument)


inspect(clean_corpus[1:3])

yt_dtm <- DocumentTermMatrix(clean_corpus)
inspect(yt_dtm[1:4, 30:35])

as.data.frame(data.matrix(yt_dtm),stringsAsFactors = F)->mat.sandata
cbind(mat.sandata,san_data$CLASS)->mat.sandata

colnames(mat.sandata)[ncol(mat.sandata)]<-"class"
names(mat.sandata)

set.seed(123)

sample(1:nrow(mat.sandata),nrow(mat.sandata)*.60)->traindata
traindata

train <- sample(nrow(mat.sandata), ceiling(nrow(mat.sandata) * .50))
test <- (1:nrow(mat.sandata))[- train]
mat.sandata[,"class"]->cl

mat.sandata[,!colnames(mat.sandata)%in%"class"]->modeldata
library(class)

knn.pred <- knn(modeldata[train, ], modeldata[test, ], cl[train])

predict(knn.pred,train)

plot(knn.pred)

conf.mat <- table("Predictions" = knn.pred, Actual = cl[test])
conf.mat

conf.mat1 <- table("Predictions" = knn.pred, Actual = cl[train])
conf.mat1


########################################################################################

## Naive Bayes


getwd()
setwd("G:/INTERNSHIP DATA")
list.files()
read.csv("Youtube01-Psy.csv",header = T,sep = ",",stringsAsFactors = F)->psy
read.csv("Youtube02-KatyPerry.csv",header = T,sep = ",",stringsAsFactors = F)->katyperry
read.csv("Youtube03-LMFAO.csv",header = T,sep = ",",stringsAsFactors = F)->LMFAO
read.csv("Youtube04-Eminem.csv",header = T,sep = ",",stringsAsFactors = F)->Eminem
read.csv("Youtube05-Shakira.csv",header = T,sep = ",",stringsAsFactors = F)->Shakira
realdata <- rbind(Eminem,katyperry,LMFAO,psy,Shakira)
realdata

realdata[-c(1:3)]->san_data
san_data

#View(san_data)
names(san_data)

#sum(is.na(san_data))
length(which(!complete.cases(san_data)))
table(san_data$CLASS)

prop.table(table(san_data$CLASS))


san_data$CONTENT_length=nchar(san_data$CONTENT)
summary(san_data$CONTENT_length)


library(tm)

getReaders()
getSources()

you_spam=Corpus(VectorSource(san_data$CONTENT))
print(you_spam)
inspect(you_spam[1:5])

clean_corpus <- tm_map(you_spam, tolower)

clean_corpus <- tm_map(clean_corpus, removeNumbers)

clean_corpus <- tm_map(clean_corpus, removePunctuation)

inspect(clean_corpus[1:3])

stopwords()[1:10]
clean_corpus <- tm_map(clean_corpus, removeWords, stopwords())

clean_corpus <- tm_map(clean_corpus, stripWhitespace)

inspect(clean_corpus[1:3])

yt_dtm <- DocumentTermMatrix(clean_corpus)
inspect(yt_dtm[1:4, 30:35])

spam_indices <- which(san_data$CLASS == "1")
spam_indices[1:3]
ham_indices <- which(san_data$CLASS == "0")
ham_indices[1:3]

library(wordcloud)

wordcloud(clean_corpus[ham_indices], min.freq=20)


wordcloud(clean_corpus[spam_indices], min.freq=40)


yt_raw_train <- san_data[1:1000,]

yt_raw_test <- san_data[1001:1956,]



yt_dtm_train <- yt_dtm[1:1000,]
yt_dtm_test <- yt_dtm[1001:1956,]
yt_corpus_train <- clean_corpus[1:1000]
yt_corpus_test <- clean_corpus[1001:1956]

spam <- subset(yt_raw_train, CLASS == "1")

ham <- subset(yt_raw_train, CLASS == "0")

five_times_words <- findFreqTerms(yt_dtm_train, 5)
length(five_times_words)
five_times_words[1:5]

yt_train <- DocumentTermMatrix(yt_corpus_train, control=list(dictionary = five_times_words))

yt_test <- DocumentTermMatrix(yt_corpus_test, control=list(dictionary = five_times_words))

convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

yt_train <- apply(yt_train, 2, convert_count)
yt_test <- apply(yt_test, 2, convert_count)

library(e1071)

naiveBayes(yt_train,factor(yt_raw_train$CLASS))->nb

library(gmodels)

#CrossTable(nb, yt_raw_train$CLASS)


class(nb)

pred <- predict(nb, newdata=yt_test)

pred
CrossTable(pred,yt_raw_test$CLASS)

table(pred,yt_raw_test$CLASS)->tb

confusionMatrix(tb)
which(pred!=yt_raw_test$CLASS)

yt_raw_test$CLASS[c(55,217)]
pred[c(55,217)]


yt_raw_test$CONTENT[c(55,217)]




























