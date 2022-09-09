# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk 
import re
import string
from subprocess import call
from sklearn.datasets import load_iris
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score,  precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from collections import Counter
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
datasets = pd.read_csv('../jupyter/spam.csv', encoding='latin-1')
datasets.head(5)


# %%
datasets.tail(5)


# %%
datasets.isnull().sum()


# %%
datasets.shape


# %%
datasets.drop_duplicates(inplace=True)


# %%
datasets.shape


# %%
datasets.dropna()


# %%
datasets.head(5)


# %%
datasets.tail(5)


# %%
datasets.shape


# %%
datasets.columns


# %%
datasets = datasets.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)


# %%
datasets.columns


# %%
datasets.isnull().sum()


# %%
datasets.isnull()


# %%
count_Class=pd.value_counts(datasets["v1"], sort= True)
count_Class.plot(kind= 'bar', color= ["blue", "orange"])
plt.title('Bar chart')
plt.show()


# %%
count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()


# %%
count1 = Counter(" ".join(datasets[datasets['v1']=='ham']["v2"]).split()).most_common(20)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in non-spam", 1 : "count"})
count2 = Counter(" ".join(datasets[datasets['v1']=='spam']["v2"]).split()).most_common(20)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "words in spam", 1 : "count_"})


# %%
count1


# %%
df1.head(5)


# %%
count2


# %%
df2.head(5)


# %%
df1.plot.bar(legend = False)
y_pos = np.arange(len(df1["words in non-spam"]))
plt.xticks(y_pos, df1["words in non-spam"])
plt.title('More frequent words in non-spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()


# %%
df2.plot.bar(legend = False, color = 'orange')
y_pos = np.arange(len(df2["words in spam"]))
plt.xticks(y_pos, df2["words in spam"])
plt.title('More frequent words in spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()


# %%
def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words


# %%
f = CountVectorizer(analyzer=process_text, ngram_range=(1,1))
X = f.fit_transform(datasets["v2"])
features_names = (f.get_feature_names())
print(np.shape(X))
print(X)
print(X.toarray())


# %%
Tf = TfidfTransformer().fit(X)
TfIdf = Tf.transform(X)


# %%
print(TfIdf)
print(TfIdf.shape)


# %%
df_idf = pd.DataFrame(Tf.idf_, index=features_names,columns=["idf_weights"])

df_idf.sort_values(by=['idf_weights'])


# %%
datasets["v1"]=datasets["v1"].map({'spam':1,'ham':0})
X_train, X_test, y_train, y_test = model_selection.train_test_split(TfIdf, datasets['v1'], test_size=0.30, random_state=42)
print([np.shape(X_train), np.shape(X_test)])


# %%
print("X_train : ",type(X_train)," X_test : ",type(X_test))
print("")
print("y_train : ",type(y_train)," y_test : ",type(y_test))


# %%
X_train.toarray()


# %%
X_test.toarray()


# %%
y_train


# %%
y_test


# %%
#Multinomial Naive Bayes
Bayes = naive_bayes.MultinomialNB().fit(X_train, y_train)


# %%
#MultiNomial Naive Bayes Training Phase


# %%
Bptr = Bayes.predict(X_train)
Bptr


# %%
Bactr = y_train.values
Bactr


# %%
Bconfs = confusion_matrix(Bactr, Bptr)


# %%
pd.DataFrame(data = Bconfs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
BaS = accuracy_score(Bactr, Bptr)
BaS


# %%
BpS = precision_score(Bactr, Bptr)
BpS


# %%
BrS = recall_score(Bactr, Bptr)
BrS


# %%
BfS = f1_score(Bactr, Bptr)
BfS


# %%
Nmtr = pd.DataFrame(
    {
        "Accuracy" : [BaS],
        "Precision": [BpS],
        "Recall": [BrS],
        "F1-Score": [BfS],
    }
)
Nmtr


# %%
Bclasr = classification_report(Bactr, Bptr)
print(Bclasr)


# %%
#MultiNomial Naive Bayes Testing Phase


# %%
Bptx = Bayes.predict(X_test)
Bptx


# %%
Bactx = y_test.values
Bactx


# %%
Bconfs = confusion_matrix(Bactx, Bptx)


# %%
pd.DataFrame(data = Bconfs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
BaS = accuracy_score(Bactx, Bptx)
BaS


# %%
BpS = precision_score(Bactx, Bptx)
BpS


# %%
BrS = recall_score(Bactx, Bptx)
BrS


# %%
BfS = f1_score(Bactx, Bptx)
BfS


# %%
Nmtx = pd.DataFrame(
    {
        "Accuracy" : [BaS],
        "Precision": [BpS],
        "Recall": [BrS],
        "F1-Score": [BfS],
    }
)
Nmtx


# %%
Bclasr = classification_report(Bactx, Bptx)
print(Bclasr)


# %%
#Decision Tree with gini-index
Decision_Tree = DecisionTreeClassifier(criterion='gini').fit(X_train, y_train)


# %%
#Decision Tree Training Phase


# %%
Dptr = Decision_Tree.predict(X_train)
Dptr


# %%
Dactr = y_train.values
Dactr


# %%
Dconfs = confusion_matrix(Dactr, Dptr)


# %%
pd.DataFrame(data = Dconfs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
DaS = accuracy_score(Dactr, Dptr)
DaS


# %%
DpS = precision_score(Dactr, Dptr)
DpS


# %%
DrS = recall_score(Dactr, Dptr)
DrS


# %%
DfS = f1_score(Dactr, Dptr)
DfS


# %%
Nmtr = pd.DataFrame(
    {
        "Accuracy" : [DaS],
        "Precision": [DpS],
        "Recall": [DrS],
        "F1-Score": [DfS],
    }
)
Nmtr


# %%
Dclasr = classification_report(Dactr, Dptr)
print(Dclasr)


# %%
#Decision Tree Testing Phase


# %%
Dptx = Decision_Tree.predict(X_test)
Dptx


# %%
Dactx = y_test.values
Dactx


# %%
Dconfs = confusion_matrix(Dactx, Dptx)


# %%
pd.DataFrame(data = Dconfs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
DaS = accuracy_score(Dactx, Dptx)
DaS


# %%
DpS = precision_score(Dactx, Dptx)
DpS


# %%
DrS = recall_score(Dactx, Dptx)
DrS


# %%
DfS = f1_score(Dactx, Dptx)
DfS


# %%
Nmtx = pd.DataFrame(
    {
        "Accuracy" : [DaS],
        "Precision": [DpS],
        "Recall": [DrS],
        "F1-Score": [DfS],
    }
)
Nmtx


# %%
Dclasr = classification_report(Dactx, Dptx)
print(Dclasr)


# %%
#Decision Tree with entropy
Decision_Tree = DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)


# %%
#Decision Tree Training Phase


# %%
Dptr = Decision_Tree.predict(X_train)
Dptr


# %%
Dactr = y_train.values
Dactr


# %%
Dconfs = confusion_matrix(Dactr, Dptr)


# %%
pd.DataFrame(data = Dconfs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
DaS = accuracy_score(Dactr, Dptr)
DaS


# %%
DpS = precision_score(Dactr, Dptr)
DpS


# %%
DrS = recall_score(Dactr, Dptr)
DrS


# %%
DfS = f1_score(Dactr, Dptr)
DfS


# %%
Nmtr = pd.DataFrame(
    {
        "Accuracy" : [DaS],
        "Precision": [DpS],
        "Recall": [DrS],
        "F1-Score": [DfS],
    }
)
Nmtr


# %%
Dclasr = classification_report(Dactr, Dptr)
print(Dclasr)


# %%
#Decision Tree Testing Phase


# %%
Dptx = Decision_Tree.predict(X_test)
Dptx


# %%
Dactx = y_test.values
Dactx


# %%
Dconfs = confusion_matrix(Dactx, Dptx)


# %%
pd.DataFrame(data = Dconfs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
DaS = accuracy_score(Dactx, Dptx)
DaS


# %%
DpS = precision_score(Dactx, Dptx)
DpS


# %%
DrS = recall_score(Dactx, Dptx)
DrS


# %%
DfS = f1_score(Dactx, Dptx)
DfS


# %%
Nmtx = pd.DataFrame(
    {
        "Accuracy" : [DaS],
        "Precision": [DpS],
        "Recall": [DrS],
        "F1-Score": [DfS],
    }
)
Nmtx


# %%
Dclasr = classification_report(Dactx, Dptx)
print(Dclasr)


# %%
#Support Vector Classifier
model_svm = svm.SVC().fit(X_train, y_train)


# %%
#Support Vector Classifier Training Phase


# %%
Sptr = model_svm.predict(X_train)
Sptr


# %%
Sactr = y_train.values
Sactr


# %%
Sconfs = confusion_matrix(Sactr, Sptr)


# %%
pd.DataFrame(data = Sconfs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
SaS = accuracy_score(Sactr, Sptr)
SaS


# %%
SpS = precision_score(Sactr, Sptr)
SpS


# %%
SrS = recall_score(Sactr, Sptr)
SrS


# %%
SfS = f1_score(Sactr, Sptr)
SfS


# %%
Nmtr = pd.DataFrame(
    {
        "Accuracy" : [SaS],
        "Precision": [SpS],
        "Recall": [SrS],
        "F1-Score": [SfS],
    }
)
Nmtr


# %%
Sclasr = classification_report(Sactr, Sptr)
print(Sclasr)


# %%
#Support Vector Classifier Testing Phase


# %%
Sptx = model_svm.predict(X_test)
Sptx


# %%
Sactx = y_test.values
Sactx


# %%
Sconfs = confusion_matrix(Sactx, Sptx)


# %%
pd.DataFrame(data = Sconfs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
SaS = accuracy_score(Sactx, Sptx)
SaS


# %%
SpS = precision_score(Sactx, Sptx)
SpS


# %%
SrS = recall_score(Sactx, Sptx)
SrS


# %%
SfS = f1_score(Sactx, Sptx)
SfS


# %%
Nmtr = pd.DataFrame(
    {
        "Accuracy" : [SaS],
        "Precision": [SpS],
        "Recall": [SrS],
        "F1-Score": [SfS],
    }
)
Nmtr


# %%
Sclasr = classification_report(Sactx, Sptx)
print(Sclasr)


# %%
#Random Forest with gini-index
ranfor_tree = RandomForestClassifier(n_estimators=31,criterion='gini').fit(X_train, y_train)


# %%
#Radnom Forest Classifier Training Phase


# %%
Rptr = ranfor_tree.predict(X_train)
Rptr


# %%
Ractr = y_train.values
Ractr


# %%
Rconfs = confusion_matrix(Ractr, Rptr)


# %%
pd.DataFrame(data = Rconfs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
RaS = accuracy_score(Ractr, Rptr)
RaS


# %%
RpS = precision_score(Ractr, Rptr)
RpS


# %%
RrS = recall_score(Ractr, Rptr)
RrS


# %%
RfS = f1_score(Ractr, Rptr)
RfS


# %%
Nmtr = pd.DataFrame(
    {
        "Accuracy" : [RaS],
        "Precision": [RpS],
        "Recall": [RrS],
        "F1-Score": [RfS],
    }
)
Nmtr


# %%
Rclasr = classification_report(Ractr, Rptr)
print(Rclasr)


# %%
#Random Forest Classifier Testing Phase


# %%
Rptx = ranfor_tree.predict(X_test)
Rptx


# %%
Ractx = y_test.values
Ractx


# %%
Rconfs = confusion_matrix(Ractx, Rptx)


# %%
pd.DataFrame(data = Rconfs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
RaS = accuracy_score(Ractx, Rptx)
RaS


# %%
RpS = precision_score(Ractx, Rptx)
RpS


# %%
RrS = recall_score(Ractx, Rptx)
RrS


# %%
RfS = f1_score(Ractx, Rptx)
RfS


# %%
Nmtr = pd.DataFrame(
    {
        "Accuracy" : [RaS],
        "Precision": [RpS],
        "Recall": [RrS],
        "F1-Score": [RfS],
    }
)
Nmtr


# %%
Rclasr = classification_report(Ractx, Rptx)
print(Rclasr)


# %%
#Random Forest with entropy
ranfor_tree = RandomForestClassifier(n_estimators=10,criterion='entropy').fit(X_train, y_train)
#est = ranfor_tree.estimators_[0]
#cn = datasets['v1']


# %%
#with open("est1.dot", "w") as f:
#    f = export_graphviz(est, filled=True, rotate=True, rounded=True, max_depth=6, out_file=f)


# %%
#import pydot
#(graph,) = pydot.graph_from_dot_file('../jupyter/est1.dot')
#graph.write_png('somefile1.png')


# %%
#Radnom Forest Classifier Training Phase


# %%
Rptr = ranfor_tree.predict(X_train)
Rptr


# %%
Ractr = y_train.values
Ractr


# %%
Rconfs = confusion_matrix(Ractr, Rptr)


# %%
pd.DataFrame(data = Rconfs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
RaS = accuracy_score(Ractr, Rptr)
RaS


# %%
RpS = precision_score(Ractr, Rptr)
RpS


# %%
RrS = recall_score(Ractr, Rptr)
RrS


# %%
RfS = f1_score(Ractr, Rptr)
RfS


# %%
Nmtr = pd.DataFrame(
    {
        "Accuracy" : [RaS],
        "Precision": [RpS],
        "Recall": [RrS],
        "F1-Score": [RfS],
    }
)
Nmtr


# %%
Rclasr = classification_report(Ractr, Rptr)
print(Rclasr)


# %%
#Random Forest Classifier Testing Phase


# %%
Rptx = ranfor_tree.predict(X_test)
Rptx


# %%
Ractx = y_test.values
Ractx


# %%
Rconfs = confusion_matrix(Ractx, Rptx)


# %%
pd.DataFrame(data = Rconfs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
RaS = accuracy_score(Ractx, Rptx)
RaS


# %%
RpS = precision_score(Ractx, Rptx)
RpS


# %%
RrS = recall_score(Ractx, Rptx)
RrS


# %%
RfS = f1_score(Ractx, Rptx)
RfS


# %%
Nmtr = pd.DataFrame(
    {
        "Accuracy" : [RaS],
        "Precision": [RpS],
        "Recall": [RrS],
        "F1-Score": [RfS],
    }
)
Nmtr


# %%
Rclasr = classification_report(Ractx, Rptx)
print(Rclasr)


# %%
#Adaboost Classifier 
Ada_class = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=62).fit(X_train, y_train)


# %%
#Adaboost Classifier training phase


# %%
Aptr = Ada_class.predict(X_train)
Aptr


# %%
Aactr = y_train.values
Aactr


# %%
Aconfs = confusion_matrix(Aactr, Aptr)


# %%
pd.DataFrame(data = Aconfs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
AaS = accuracy_score(Aactr, Aptr)
AaS


# %%
ApS = precision_score(Aactr, Aptr)
ApS


# %%
ArS = recall_score(Aactr, Aptr)
ArS


# %%
AfS = f1_score(Aactr, Aptr)
AfS


# %%
Nmtr = pd.DataFrame(
    {
        "Accuracy" : [AaS],
        "Precision": [ApS],
        "Recall": [ArS],
        "F1-Score": [AfS],
    }
)
Nmtr


# %%
Aclasr = classification_report(Aactr, Aptr)
print(Aclasr)


# %%
#Adaboost Classifier testing phase


# %%
Aptx = Ada_class.predict(X_test)
Aptx


# %%
Aactx = y_test.values
Aactx


# %%
Aconfs = confusion_matrix(Aactx, Aptx)


# %%
pd.DataFrame(data = Aconfs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
AaS = accuracy_score(Aactx, Aptx)
AaS


# %%
ApS = precision_score(Aactx, Aptx)
ApS


# %%
ArS = recall_score(Aactx, Aptx)
ArS


# %%
AfS = f1_score(Aactx, Aptx)
AfS


# %%
Nmtr = pd.DataFrame(
    {
        "Accuracy" : [AaS],
        "Precision": [ApS],
        "Recall": [ArS],
        "F1-Score": [AfS],
    }
)
Nmtr


# %%
Aclasr = classification_report(Aactx, Aptx)
print(Aclasr)


# %%
#Bagging Classifier with decision tree
Bag_class =  BaggingClassifier(n_estimators=100).fit(X_train, y_train)


# %%
#Bagging Classifier Training phase


# %%
Bag_ptr = Bag_class.predict(X_train)
Bag_ptr


# %%
Bag_actr = y_train.values
Bag_actr


# %%
Bag_confs = confusion_matrix(Bag_actr, Bag_ptr)


# %%
pd.DataFrame(data = Bag_confs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
Bag_aS = accuracy_score(Bag_actr, Bag_ptr)
Bag_aS


# %%
Bag_pS = precision_score(Bag_actr, Bag_ptr)
Bag_pS


# %%
Bag_rS = recall_score(Bag_actr, Bag_ptr)
Bag_rS


# %%
Bag_fS = f1_score(Bag_actr, Bag_ptr)
Bag_fS


# %%
Nmtr = pd.DataFrame(
    {
        "Accuracy" : [Bag_aS],
        "Precision": [Bag_pS],
        "Recall": [Bag_rS],
        "F1-Score": [Bag_fS],
    }
)
Nmtr


# %%
Bag_clasr = classification_report(Bag_actr, Bag_ptr)
print(Bag_clasr)


# %%
#Bagging Classifier Testing phase


# %%
Bag_ptx = Bag_class.predict(X_test)
Bag_ptx


# %%
Bag_actx = y_test.values
Bag_actx


# %%
Bag_confs = confusion_matrix(Bag_actx, Bag_ptx)


# %%
pd.DataFrame(data = Bag_confs, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


# %%
Bag_aS = accuracy_score(Bag_actx, Bag_ptx)
Bag_aS


# %%
Bag_pS = precision_score(Bag_actx, Bag_ptx)
Bag_pS


# %%
Bag_rS = recall_score(Bag_actx, Bag_ptx)
Bag_rS


# %%
Bag_fS = f1_score(Bag_actx, Bag_ptx)
Bag_fS


# %%
Nmtr = pd.DataFrame(
    {
        "Accuracy" : [Bag_aS],
        "Precision": [Bag_pS],
        "Recall": [Bag_rS],
        "F1-Score": [Bag_fS],
    }
)
Nmtr


# %%
Bag_clasr = classification_report(Bag_actx, Bag_ptx)
print(Bag_clasr)


# %%


