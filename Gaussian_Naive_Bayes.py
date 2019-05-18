#  Gaussian Naive Bayes classifier using Bag of words and TF-IDF encoding
import modeling as md
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
def Gaussian_Naive_Bayes(xtrain_count,xvalid_count,train_y,valid_y,my_tags,xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars):
    from sklearn.naive_bayes import GaussianNB
    xtrain_count=xtrain_count.toarray()
    xvalid_count=xvalid_count.toarray()

    # training a Gaussian Naive Bayes classifier for Bag of words
    predictions = md.train_model(GaussianNB(), xtrain_count, train_y,xvalid_count)
    cm = confusion_matrix(valid_y,predictions)
    print (cm)
    print('gnb accuracy for Bow %s' % accuracy_score(predictions, valid_y))
    print(classification_report(valid_y, predictions,target_names=my_tags))

    # training a Gaussian Naive Bayes classifier for word level TF-IDF
    xtrain_tfidf=xtrain_tfidf.toarray()
    xvalid_tfidf=xvalid_tfidf.toarray()
    predictions = md.train_model(GaussianNB(), xtrain_tfidf, train_y,xvalid_tfidf)
    cm = confusion_matrix(valid_y,predictions)
    print (cm)
    print('gnb accuracy for word level TF-IDF %s' % accuracy_score(predictions, valid_y))
    print(classification_report(valid_y, predictions,target_names=my_tags))
    
    # training a Gaussian Naive Bayes classifier for N-gram TF-IDF
    xtrain_tfidf_ngram=xtrain_tfidf_ngram.toarray()
    xvalid_tfidf_ngram=xvalid_tfidf_ngram.toarray()
    predictions = md.train_model(GaussianNB(), xtrain_tfidf_ngram, train_y,xvalid_tfidf_ngram)
    cm = confusion_matrix(valid_y,predictions)
    print (cm)
    print('gnb accuracy for N-gram TF-IDF %s' % accuracy_score(predictions, valid_y))
    print(classification_report(valid_y, predictions,target_names=my_tags))

    # training a Gaussian Naive Bayes classifier for character level TF-IDF
    xtrain_tfidf_ngram_chars=xtrain_tfidf_ngram_chars.toarray()
    xvalid_tfidf_ngram_chars=xvalid_tfidf_ngram_chars.toarray()
    predictions = md.train_model(GaussianNB(), xtrain_tfidf_ngram_chars, train_y,xvalid_tfidf_ngram_chars)
    cm = confusion_matrix(valid_y,predictions)
    print (cm)
    print('gnb accuracy for character level TF-IDF %s' % accuracy_score(predictions, valid_y))
    print(classification_report(valid_y, predictions,target_names=my_tags))

