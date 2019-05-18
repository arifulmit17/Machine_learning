#A Svm_sgd Classifier for Bag of words and Tf-Idf encoding
import modeling as md
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
def SGD_Svm(xtrain_count,xvalid_count,train_y,valid_y,my_tags,xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars):

    # training a SVM_sgd classifier on Bag of words
    from sklearn.linear_model import SGDClassifier
    predictions = md.train_model(SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None), xtrain_count, train_y,xvalid_count)
    cm = confusion_matrix(valid_y, predictions)
    print (cm)
    print('SVM_SGD for Bow accuracy %s' % accuracy_score(predictions, valid_y))
    print(classification_report(valid_y,predictions,target_names=my_tags))

    # SVM SGDClassifier on Word Level TF IDF Vectors
    predictions = md.train_model(SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None), xtrain_tfidf, train_y, xvalid_tfidf)
    print ("SVM_sgd, WordLevel TF-IDF: ", accuracy_score(predictions, valid_y))
    cm = confusion_matrix(valid_y, predictions)
    print (cm)
    print(classification_report(valid_y,predictions,target_names=my_tags))
    #SVM SGDClassifier on Ngram Level TF IDF Vectors
    predictions = md.train_model(SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    print ("SVM_sgd, N-Gram Vectors: ", accuracy_score(predictions, valid_y))
    cm = confusion_matrix(valid_y, predictions)
    print (cm)
    print(classification_report(valid_y,predictions,target_names=my_tags))
    # SVM SGDClassifier on Character Level TF IDF Vectors
    predictions = md.train_model(SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    print ("SVM_sgd, CharLevel Vectors: ", accuracy_score(predictions, valid_y))
    cm = confusion_matrix(valid_y, predictions)
    print (cm)
    print(classification_report(valid_y,predictions,target_names=my_tags))
