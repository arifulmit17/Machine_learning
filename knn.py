#A KNN Classifier for Bag of words and Tf-Idf encoding
import modeling as md
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
def knn(xtrain_count,xvalid_count, train_y,valid_y,my_tags,xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars):

    from sklearn.neighbors import KNeighborsClassifier
    # training a KNN classifier on Bag of words
    predictions = md.train_model(KNeighborsClassifier(n_neighbors = 7), xtrain_count, train_y,xvalid_count)
    cm = confusion_matrix(valid_y, predictions)
    print (cm)
    print('For Bow KNN accuracy %s' % accuracy_score(predictions, valid_y))
    print(classification_report(valid_y,predictions,target_names=my_tags))

    # training a KNN classifier on Word Level TF IDF Vectors 
    predictions =md.train_model(KNeighborsClassifier(n_neighbors = 7), xtrain_tfidf, train_y, xvalid_tfidf)
    cm = confusion_matrix(valid_y, predictions)
    print (cm)
    print('For Word Level TF IDF Vectors KNN accuracy %s' % accuracy_score(predictions, valid_y))
    print(classification_report(valid_y,predictions,target_names=my_tags))

    # training a KNN classifier on Ngram Level TF IDF Vectors
    predictions = md.train_model(KNeighborsClassifier(n_neighbors = 7), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    cm = confusion_matrix(valid_y, predictions)
    print (cm)
    print('For Ngram Level TF IDF Vectors KNN accuracy %s' % accuracy_score(predictions, valid_y))
    print(classification_report(valid_y,predictions,target_names=my_tags))

    # training a KNN classifier on Ngram Level TF IDF Vectors
    predictions = md.train_model(KNeighborsClassifier(n_neighbors = 7), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    cm = confusion_matrix(valid_y, predictions)
    print (cm)
    print('For Character Level TF IDF Vectors KNN accuracy %s' % accuracy_score(predictions, valid_y))
    print(classification_report(valid_y,predictions,target_names=my_tags))

