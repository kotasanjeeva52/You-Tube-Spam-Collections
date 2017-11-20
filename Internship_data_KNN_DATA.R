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







library(e1071)

svm(modeldata[train,],modeldata[test,])->sv

predict(sv,test)

(accuracy <- sum(diag(conf.mat))/length(test) * 100)

cbind(knn.pred,modeldata[test,])->sandata.test.perd
sandata.test.perd
summary(knn.pred)

as.data.frame(modeldata[train])->data
library(rpart)
library(caret)
rpart(class~.,data =data)























