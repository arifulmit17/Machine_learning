#A SVM Classifier for Bag of words and Tf-Idf encoding
import modeling as md
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
def Svm(xtrain_count,xvalid_count,train_y,valid_y,my_tags,xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars):
    from sklearn.svm import SVC
    # SVM on Bag of words
    predictions = md.train_model(SVC(kernel = 'linear', C = 1), xtrain_count, train_y,xvalid_count)
    cm = confusion_matrix(valid_y, predictions)
    print (cm)
    print('SVM Bow accuracy %s' % accuracy_score(predictions, valid_y))
    print(classification_report(valid_y, predictions,target_names=my_tags))
    # SVM on Word Level TF IDF Vectors
    predictions = md.train_model(SVC(kernel = 'linear', C = 1), xtrain_tfidf, train_y, xvalid_tfidf)
    print ("SVM, WordLevel TF-IDF: ", accuracy_score(predictions, valid_y))
    cm = confusion_matrix(valid_y, predictions)
    print (cm)
    print(classification_report(valid_y,predictions,target_names=my_tags))
    # SVM on Character Level TF IDF Vectors
    predictions = md.train_model(SVC(kernel = 'linear', C = 1), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    print ("SVM, CharLevel Vectors: ", accuracy_score(predictions, valid_y))
    cm = confusion_matrix(valid_y, predictions)
    print (cm)
    print(classification_report(valid_y,predictions,target_names=my_tags))
    # SVM on Ngram Level TF IDF Vectors
    predictions = md.train_model(SVC(kernel = 'linear', C = 1), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    print ("SVM, N-Gram Vectors: ",  accuracy_score(predictions, valid_y))
    cm = confusion_matrix(valid_y, predictions)
    print (cm)
    print(classification_report(valid_y,predictions,target_names=my_tags))

