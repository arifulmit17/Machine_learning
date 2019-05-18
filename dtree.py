#A DescisionTreeClassifier for Bag of words and Tf-Idf encoding
import modeling as md
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
def dtree(train, train_y,valid_y,my_tags,xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars):
   xtrain_count=train[0]
   xvalid_count=train[1]
   from sklearn.tree import DecisionTreeClassifier
   # training a DescisionTreeClassifier on Bag of words
   predictions = md.train_model(DecisionTreeClassifier(max_depth = 2), xtrain_count, train_y,xvalid_count)
   cm = confusion_matrix(valid_y, predictions)
   print (cm)
   print('Bow Dtree accuracy %s' % accuracy_score(predictions, valid_y))
   print(classification_report(valid_y, predictions,target_names=my_tags))
   # training a DescisionTreeClassifier  on Word Level TF IDF Vectors
   predictions = md.train_model(DecisionTreeClassifier(max_depth = 2), xtrain_tfidf, train_y, xvalid_tfidf)
   cm = confusion_matrix(valid_y, predictions)
   print (cm)
   print('Word level TF IDF Vectors Dtree accuracy %s' % accuracy_score(predictions, valid_y))
   print(classification_report(valid_y,predictions,target_names=my_tags))

   # training a DescisionTreeClassifier  on Ngram Level TF IDF Vectors
   predictions = md.train_model(DecisionTreeClassifier(max_depth = 2), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
   cm = confusion_matrix(valid_y, predictions)
   print (cm)
   print('Ngram Level TF IDF Vectors Dtree accuracy %s' % accuracy_score(predictions, valid_y))
   print(classification_report(valid_y,predictions,target_names=my_tags))

   # training a DescisionTreeClassifier  on Character Level TF IDF Vectors
   predictions = md.train_model(DecisionTreeClassifier(max_depth = 2), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
   cm = confusion_matrix(valid_y, predictions)
   print (cm)
   print('Character Level TF IDF Vectors Dtree accuracy %s' % accuracy_score(predictions, valid_y))
   print(classification_report(valid_y,predictions,target_names=my_tags))
