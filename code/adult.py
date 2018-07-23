# -*- coding: utf-8 -*-

import time
import matplotlib
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from pandas.tools.plotting import parallel_coordinates,andrews_curves
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers=('s','x','o','^','v')
    colors=('blue','red','green','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),
                        np.arange(x2_min,x2_max,resolution))
    Z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z=Z.reshape(xx1.shape)

    cm=plt.cm.get_cmap('bwr')

    sc=plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cm)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    plt.colorbar(sc)

    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],
                    alpha=0.8,c=cmap(idx),
                    marker=markers[idx],label=cl)

class mv():
    def majority_voting(self,data):
        row=len(data.axes[0])
        col=len(data.axes[1])
        new_class=[]
        for i in range(0,row):
               count_zero=0
               count_one=0
               count_two=0
               count_three=0
               for j in range(0,col):
                    if   (data.ix[i,j] == 0):
                             count_zero = count_zero + 1
                    elif (data.ix[i,j]==1):
                             count_one = count_one + 1
                    elif (data.ix[i,j]==2):
                             count_two = count_two + 1
                    elif (data.ix[i,j]==3):
                             count_three = count_three +1
                             
               bigger={"count_zero":count_zero, "count_one":count_one, "count_two":count_two,"count_three":count_three}
               help_max=max(bigger.iterkeys(), key=lambda k: bigger[k])
               if (help_max == "count_zero"):
                   new_class.append(0)
               elif (help_max == "count_one"):
                   new_class.append(1)
               elif (help_max == "count_two"):
                   new_class.append(2)
               elif (help_max == "count_three"):
                   new_class.append(3)  
        return new_class


#选择属性
variables=['age','workclass','fnlwgt','education','education_num',\
           'marital-status','occupation','relationship','race','sex',\
           'capital-gain','capital-loss','hours-per-week','native-country','income']

#读取数据
classificator=pd.read_csv('C:/Users/SKY2/Desktop/adult/adult/data/data_withtitle.csv')
validator=pd.read_csv('C:/Users/SKY2/Desktop/adult/adult/data/test_withtitle.csv')
trainData=classificator[variables]
testData=validator[variables]

#map工资收入
train_income_mapping={
    '<=50K':0,
    '>50K':1
}
test_income_mapping={
    '<=50K.':0,
    '>50K.':1
}
#map 性别
sex_mapping={
    'Male':0,
    'Female':1
}

#数据预处理并清洗数据
trainData['income']=trainData['income'].map(train_income_mapping)
testData['income']=testData['income'].map(test_income_mapping)
trainData['sex']=trainData['sex'].map(sex_mapping)
testData['sex']=testData['sex'].map(sex_mapping)
trainData=trainData.dropna()
testData=testData.dropna()





#method 2 使用pd自带的函数
trainData=pd.get_dummies(trainData,sparse=False)
testData=pd.get_dummies(testData,sparse=False)


trainRows=random.sample(trainData.index,20000)
testRows=random.sample(testData.index,1000)

work_train=trainData.ix[trainRows]

train_x=work_train.drop('income',axis=1)
train_x=train_x.drop('native-country_Holand-Netherlands',axis=1)
train_x=train_x.values
train_y=work_train['income']
train_y=train_y.values

work_test=testData.ix[testRows]

test_x=work_test.drop('income',axis=1)
test_y=work_test['income']
test_x=test_x.values
test_y=test_y.values
#标准化  分类结果从75%提升到了80%以上
train_x_std=StandardScaler().fit_transform(train_x)
test_x_std=StandardScaler().fit_transform(test_x)

pcan = 2
#PCA降维
pca=PCA(n_components=pcan)
train_x_pca=pca.fit_transform(train_x_std)
test_x_pca=pca.transform(test_x_std)


start_CPU = time.clock()
# pipe_lr = Pipeline([('scl',StandardScaler()),
#                     ('pca',PCA(n_components=2)),
#                     ('cfl',svm.SVC())])
#
# pipe_lr.fit(train_x,train_y)
#
# print ("pipe " ,pipe_lr.score(test_x,test_y))

#使用 SVM, Naive Bayes, Stochastic Gradient Descent
print ('Training SVM')
clf_svm=svm.SVC(kernel='rbf',gamma=1.0,C=3.0)
clf_svm.fit(train_x_pca,train_y)
print ('Testing SVM')
predict_svm=clf_svm.predict(test_x_pca)



print ('Training BNB')
clf_bnb=GaussianNB()
clf_bnb.fit(train_x_pca,train_y)
print ('Testing BNB')
predict_bnb=clf_bnb.predict(test_x_pca)



print ('Training SGD')
clf_SGD=SGDClassifier(penalty='l1')
clf_SGD.fit(train_x_pca,train_y)
print ('Testing SGD')
predict_SGD=clf_SGD.predict(test_x_pca)


#组合分类器
voting=mv()
ensemble=pd.DataFrame({"svm":predict_svm,"bnb":predict_bnb,"SGD":predict_SGD})
ensemble["voting"]=voting.majority_voting(ensemble)
ensemble["original"]=list(test_y)
print ensemble

#得分
score_svm=accuracy_score(list(test_y),predict_svm)*100
print "accuracy_svm:%f "%score_svm
score_bayes=accuracy_score(list(test_y),predict_bnb)*100
print "accuracy_binomial naivebayes:%f "%score_bayes
score_SGD=accuracy_score(list(test_y),predict_SGD)*100
print "accuracy_SGD:%f "%score_SGD
score_ensemble=accuracy_score(list(test_y),list(ensemble["voting"]))*100
print "accuracy_ensemble:%f "%score_ensemble

print("svm classification")
print(metrics.classification_report(test_y,predict_svm))
print("bnb classification")
print(metrics.classification_report(test_y,predict_bnb))
print("SGD classification")
print(metrics.classification_report(test_y,predict_SGD))

print('confusion matrix')
print(metrics.confusion_matrix(test_y,predict_svm))
print(metrics.confusion_matrix(test_y,predict_bnb))
print(metrics.confusion_matrix(test_y,predict_SGD))

print("pca n = %d\n"%pcan)

#long running




end_CPU = time.clock()
print("Method all: %f CPU seconds" % (end_CPU - start_CPU))

#画
plt_svm=plt.subplot(311)
plt_SGD=plt.subplot(312)
plt_BNB=plt.subplot(313)

plt.sca(plt_svm)
plot_decision_regions(test_x_pca,test_y,classifier=clf_svm)
plt_svm.set_ylabel('PCA_Y')
plt_svm.set_title("SVM result",fontsize=16)
plt_svm.legend()

plt.sca(plt_SGD)
plot_decision_regions(test_x_pca,test_y,classifier=clf_SGD)
plt_SGD.set_ylabel('PCA_Y')
plt_SGD.set_title("SGD result",fontsize=16)
plt_SGD.legend()

plt.sca(plt_BNB)
plot_decision_regions(test_x_pca,test_y,classifier=clf_bnb)
plt_BNB.set_xlabel('PCA_X')
plt_BNB.set_ylabel('PCA_Y')
plt_BNB.set_title("BNB result",fontsize=16)
plt_BNB.legend()

plt.show()
