# -*- coding: UTF-8 -*-
import sys
from sklearn.datasets.base import Bunch
import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
reload(sys)
sys.setdefaultencoding('utf-8')

def readf(path):
    with open(path, "rb") as fout:
        text = fout.read()
    return text

def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

def _writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)

def vector_space(stopword_path,data_set,tfidfobj,trainobj=None):

    stopwords = readf(stopword_path).splitlines()
    with open(data_set, "rb") as fobj:
        bunch = pickle.load(fobj)
    #bunch = _readbunchobj(data_set)
    tfidfbunch = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[], vocabulary={})

    if trainobj is not None:
    	with open(trainobj, "rb") as fobj:
            trainbunch = pickle.load(fobj)
        #trainbunch = _readbunchobj(trainobj)
        tfidfbunch.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stopwords, sublinear_tf=True, max_df=0.5, vocabulary=trainbunch.vocabulary)
        tfidfbunch.tdm = vectorizer.fit_transform(bunch.contents)

    else:
        vectorizer = TfidfVectorizer(stop_words=stopwords, sublinear_tf=True, max_df=0.5)
        tfidfbunch.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfbunch.vocabulary = vectorizer.vocabulary_

    with open(tfidfobj, "wb") as fobj:
        pickle.dump(tfidfbunch, fobj)
    #_writebunchobj(tfidfobj, tfidfbunch)
    print "tfidf词向量空间实例创建完毕"

if __name__ == '__main__':

    stopword_path = "wordbag/stop_words.txt"
    data_set = "wordbag/data_set.dat"
    tfidfobj = "wordbag/tfdif.dat"
    vector_space(stopword_path,data_set,tfidfobj)
