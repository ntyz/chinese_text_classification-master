# -*- coding: UTF-8 -*-
import sys
import cPickle as pickle
from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.externals import joblib
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import train_test_split 
from sklearn import metrics
reload(sys)
sys.setdefaultencoding('utf-8')

# 读取bunch对象
def readobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

# 导入训练集
trainpath = "wordbag/tfdif.dat"
train_set = readobj(trainpath)
X_train,X_test, y_train, y_test = train_test_split(train_set.tdm, train_set.label, test_size = 0.3)
#print X_train[0,:], y_train[0]
print "split done"
# 导入测试集
#testpath = "train_word_bag/tfdifspace.dat"#"test_word_bag/testspace.dat"
#test_set = _readbunchobj(testpath)

# 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高
NB_clf = MultinomialNB(alpha=0.01).fit(X_train, y_train)
joblib.dump(NB_clf, "model/NB.pkl")

#lr_clf = LogisticRegression(penalty='l2', solver ='lbfgs', multi_class='multinomial', max_iter=800,  C=0.2 )
#lr_clf.fit(X_train,y_train)
#joblib.dump(lr_clf, "model/lr.pkl")

#svm_clf = SVC(kernel='rbf', probability=True)  
#svm_clf.fit(X_train,y_train) 
#joblib.dump(svm_clf, "model/svm.pkl")
# 预测分类结果
predicted = NB_clf.predict(X_test)
#predicted = lr_clf.predict(X_test)
#predicted = svm_clf.predict(X_test)
for flabel,expct_cate in zip(y_test, predicted):
    if flabel != expct_cate:
        print "实际类别:",flabel," -->预测类别:",expct_cate


print "预测完成"

# 计算分类精度：
#print '*******Analysis Model Result*********'
def metrics_result(actual, predict):
    print '精度:{0:.3f}'.format(metrics.precision_score(actual, predict,average='weighted'))
    print '召回:{0:0.3f}'.format(metrics.recall_score(actual, predict,average='weighted'))
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict,average='weighted'))

metrics_result(y_test, predicted)
