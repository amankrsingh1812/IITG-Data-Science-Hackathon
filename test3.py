import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
count_vect= CountVectorizer()
train=pd.read_csv("./train_file.csv")
test=pd.read_csv("./test_file.csv")

train = train.replace(np.nan,"0")
test = test.replace(np.nan,"0")
X=train[train.columns[-4]]
y=train[train.columns[-1]]
Xt=test[test.columns[-3]]
X1 = count_vect.fit_transform(X)

Xt1 = count_vect.transform(Xt)


from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer()
XF = tf_transformer.fit_transform(X1)
Xtf= tf_transformer.fit_transform(Xt1)

#print(Xtf)

from sklearn.linear_model import SGDClassifier


#from sklearn.naive_bayes import MultinomialNB
clf = XGBClassifier( learning_rate =0.05,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27).fit(XF, y)

predicted = clf.predict(Xtf)

sol = pd.DataFrame()
sol['ID']=test[test.columns[0]]
sol['MaterialType']=predicted
sol.to_csv('./TestFinal.csv', index = False)

print(sol)
# making predictions on the testing set 
#y_pred = knn.predict(Xt1) 
