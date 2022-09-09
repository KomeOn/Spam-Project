import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk 
import re
import string
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score,  precision_score, recall_score
from sklearn import feature_extraction, model_selection
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import shuffle
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import hypeparameter_tune as hp
from pickle import dump, load
import warnings
warnings.filterwarnings("ignore")


def dataset_prop(files):
    fpath = files['path']
    ds = pd.read_csv(fpath)
    print("Shape : ", ds.shape)
    print("Columns : ", ds.columns)
    print("Head (5) : ", ds.head(5))
    print("Tail (5) : ", ds.tail(5))

    count1 = Counter(" ".join(ds[ds['spam']==0]["text"]).split()).most_common(20)
    df1 = pd.DataFrame.from_dict(count1)
    df1 = df1.rename(columns={0: "words in non-spam", 1 : "count"})
    count2 = Counter(" ".join(ds[ds['spam']==1]["text"]).split()).most_common(20)
    df2 = pd.DataFrame.from_dict(count2)
    df2 = df2.rename(columns={0: "words in spam", 1 : "count_"})
    print("DF1 : ", df1, "\ncount : ", count1)
    print("DF2 : ",df2,  "\ncount : ", count2)

def model_assessment(u_classify, y_data, predicted_class):
    mod_ass = dict()
    conf_mtrx = confusion_matrix(y_data,predicted_class)
    acc_score = accuracy_score(y_data,predicted_class)
    prec_score = precision_score(y_data,predicted_class)
    rec_score = recall_score(y_data,predicted_class)
    f1_scr = f1_score(y_data,predicted_class)
    print('confusion matrix : ',conf_mtrx)
    print('accuracy : ',acc_score)
    print('precision : ',prec_score)
    print('recall : ',rec_score)
    print('f-Score : ',f1_scr)
    mod_ass.update({
        "confusion" : conf_mtrx,
        "accuracy" : acc_score,
        "precision" : prec_score,
        "recall" : rec_score,
        "f1" : f1_scr
    })
    return mod_ass
 
def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

def features_transform(mail, dtrain, var1):
    bow = CountVectorizer(analyzer=process_text)
    bow_transformer = bow.fit(dtrain)
    messages_bow = bow_transformer.transform(mail)
    print('\nsparse matrix shape:', messages_bow.shape)
    print('number of non-zeros:', messages_bow.nnz) 
    print('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])), '\n')
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)
    if var1 in ["RFC", "BAG", "ADA", "VOTE"]:
        sav_fit(vr1=bow.vocabulary_, var=var1)
    return messages_tfidf

def train_classifier(clf, f_train, l_train, typ):
    model = clf.fit(f_train, l_train)
    file = str(typ)+'.pkl'
    print("name : ",file)
    dump(model, open(file, 'wb'))

def sav_fit(vr1, var):
    fname = 'vect'+str(var)+'.pkl'
    dump(vr1,open(fname, 'wb'))

def rfc_classifier(files, var):
    res_df = list()
    fpath = files['path']
    datasets = pd.read_csv(fpath)
    datasets = datasets.dropna()
    datasets.drop_duplicates(inplace=True)
    dsets = shuffle(datasets)
    if int(var) == 1:
        d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
        dtrain_msg = features_transform(mail=d_train, dtrain=d_train, var1='RFC')
        n_est = hp.RFC_class(files)
        model_rf=RandomForestClassifier(n_estimators=n_est['n_trees'],criterion='entropy', random_state=111)
        train_classifier(model_rf, dtrain_msg, l_train, typ="RFC")
        model_rf.fit(dtrain_msg, l_train)
        pred_train = model_rf.predict(dtrain_msg)
        mnb_dict = model_assessment(u_classify='Random Forest', y_data=l_train, predicted_class=pred_train)
        return mnb_dict
    elif int(var) == 2:
        print("Inside testing phase : ")
        d_test = dsets['text']         
        model_rf = load(open('pickle_files/RFC.pkl', 'rb'))
        vect = load(open('pickle_files/vectRFC.pkl', 'rb'))
        tf = TfidfTransformer()
        load_vect = CountVectorizer(vocabulary=vect)
        last = len(d_test)
        for i in range(0, last):
            if d_test.get(i) != None:
                tup = [d_test[i],]
                dtest_msg = tf.fit_transform(load_vect.fit_transform(tup))
                pred_test = model_rf.predict(dtest_msg)
                pred = model_rf.predict_proba(dtest_msg)
                res_df.append((i+1, [pred[0][0], pred[0][1], pred_test[0]]))
        df = pd.DataFrame.from_items(res_df, orient='index', columns=['Class O', 'Class 1', 'Result'])
        print(df.head(15))
        return df
    else:
        d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
        model_rf = load(open('pickle_files/RFC.pkl', 'rb'))
        vect = load(open('pickle_files/vectRFC.pkl', 'rb'))
        tf = TfidfTransformer()
        load_vect = CountVectorizer(vocabulary=vect)
        dtest_msg = tf.fit_transform(load_vect.fit_transform(d_test))
        pred_test = model_rf.predict(dtest_msg)
        mnb_dict = model_assessment(u_classify='Random Forest', y_data=l_test, predicted_class=pred_test)
        return mnb_dict

def bagging_classifier(files, var):
    res_df = list()
    fpath = files['path']
    datasets = pd.read_csv(fpath)
    datasets = datasets.dropna()
    datasets.drop_duplicates(inplace=True)
    dsets = shuffle(datasets)
    if int(var) == 1:
        d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
        dtrain_msg = features_transform(mail=d_train, dtrain=d_train, var1='BAG')
        n_est = hp.BAG_class(files)
        bag_class =  BaggingClassifier(n_estimators=n_est['n_classifiers'], random_state=111)
        train_classifier(bag_class, dtrain_msg, l_train, typ="BAG")
        bag_class.fit(dtrain_msg, l_train)
        pred_train = bag_class.predict(dtrain_msg)
        mnb_dict = model_assessment(u_classify='Bagging', y_data=l_train, predicted_class=pred_train)
        return mnb_dict
    elif int(var) == 2:
        print("Inside testing phase : ")
        d_test = dsets['text']         
        bag_class = load(open('pickle_files/BAG.pkl', 'rb'))
        vect = load(open('pickle_files/vectBAG.pkl', 'rb'))
        tf = TfidfTransformer()
        load_vect = CountVectorizer(vocabulary=vect)
        last = len(d_test)
        for i in range(0, last):
            if d_test.get(i) != None:
                tup = [d_test[i],]
                dtest_msg = tf.fit_transform(load_vect.fit_transform(tup))
                pred_test = bag_class.predict(dtest_msg)
                pred = bag_class.predict_proba(dtest_msg)
                res_df.append((i+1, [pred[0][0], pred[0][1], pred_test[0]]))
        df = pd.DataFrame.from_items(res_df, orient='index', columns=['Class O', 'Class 1', 'Result'])
        print(df.head(15))
        return df
    else:
        d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
        bag_class = load(open('pickle_files/BAG.pkl', 'rb'))
        vect = load(open('pickle_files/vectBAG.pkl', 'rb'))
        tf = TfidfTransformer()
        load_vect = CountVectorizer(vocabulary=vect)
        dtest_msg = tf.fit_transform(load_vect.fit_transform(d_test))
        pred_test = bag_class.predict(dtest_msg)
        mnb_dict = model_assessment(u_classify='Bagging EM', y_data=l_test, predicted_class=pred_test)
        return mnb_dict

def adaB_classifier(files, var):
    res_df = list()
    fpath = files['path']
    datasets = pd.read_csv(fpath)
    datasets = datasets.dropna()
    datasets.drop_duplicates(inplace=True)
    dsets = shuffle(datasets)
    if int(var) == 1:
        d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
        dtrain_msg = features_transform(mail=d_train, dtrain=d_train, var1='ADA')
        n_est = hp.AB_class(files)
        ada_class = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n_est['n_classifiers'], random_state=111)
        train_classifier(ada_class, dtrain_msg, l_train, typ="ADA")
        ada_class.fit(dtrain_msg, l_train)
        pred_train = ada_class.predict(dtrain_msg)
        mnb_dict = model_assessment(u_classify='AdaBoosting EM', y_data=l_train, predicted_class=pred_train)
        return mnb_dict
    elif int(var) == 2:
        print("Inside training phase : ")
        d_test = dsets['text']       
        ada_class = load(open('pickle_files/ADA.pkl', 'rb'))
        vect = load(open('pickle_files/vectADA.pkl', 'rb'))
        tf = TfidfTransformer()
        load_vect = CountVectorizer(vocabulary=vect)
        last = len(d_test)
        for i in range(0, last):
            if d_test.get(i) != None:
                tup = [d_test[i],]
                dtest_msg = tf.fit_transform(load_vect.fit_transform(tup))
                pred_test = ada_class.predict(dtest_msg)
                pred = ada_class.predict_proba(dtest_msg)
                res_df.append((i+1, [pred[0][0], pred[0][1], pred_test[0]]))
        df = pd.DataFrame.from_items(res_df, orient='index', columns=['Class O', 'Class 1', 'Result'])
        print(df.head(15))
        return df
    else:
        d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
        ada_class = load(open('pickle_files/ADA.pkl', 'rb'))
        vect = load(open('pickle_files/vectADA.pkl', 'rb'))
        tf = TfidfTransformer()
        load_vect = CountVectorizer(vocabulary=vect)
        dtest_msg = tf.fit_transform(load_vect.fit_transform(d_test))
        pred_test = ada_class.predict(dtest_msg)
        mnb_dict = model_assessment(u_classify='AdaBoosting EM', y_data=l_test, predicted_class=pred_test)
        return mnb_dict

def voting_classifier(files, var):
    res_df = list()
    fpath = files['path']
    datasets = pd.read_csv(fpath)
    datasets = datasets.dropna()
    datasets.drop_duplicates(inplace=True)
    dsets = shuffle(datasets)
    if int(var) == 1:
        d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
        dtrain_msg = features_transform(mail=d_train, dtrain=d_train, var1='VOTE')
        n_est1 = hp.BAG_class(files)
        n_est2 = hp.RFC_class(files)
        n_est3 = hp.AB_class(files)
        alp_dict = hp.MNB_class(files)
        bag_class =  BaggingClassifier(n_estimators=100)
        model_rf=RandomForestClassifier(n_estimators=20,criterion='entropy')
        ada_class = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=62)
        modelMNB = naive_bayes.MultinomialNB()
        eclf = VotingClassifier(estimators=[('BgC', bag_class), ('RF', model_rf), ('Ada', ada_class), ('MNB', modelMNB) ], voting='soft')
        train_classifier(eclf, dtrain_msg, l_train, typ="VOTE")
        eclf.fit(dtrain_msg, l_train)
        pred_train = eclf.predict(dtrain_msg)
        mnb_dict = model_assessment(u_classify='Voting EM', y_data=l_train, predicted_class=pred_train)
        return mnb_dict
    elif int(var) == 2:
        print("Inside training phase : ")
        d_test = dsets['text']
        eclf = load(open('pickle_files/VOTE.pkl', 'rb'))
        vect = load(open('pickle_files/vectVOTE.pkl', 'rb'))
        tf = TfidfTransformer()
        load_vect = CountVectorizer(vocabulary=vect)
        last = len(d_test)
        for i in range(0, last):
            if d_test.get(i) != None:
                tup = [d_test[i],]
                dtest_msg = tf.fit_transform(load_vect.fit_transform(tup))
                pred_test = eclf.predict(dtest_msg)
                pred = eclf.predict_proba(dtest_msg)
                res_df.append((i+1, [pred[0][0], pred[0][1], pred_test[0]]))
        df = pd.DataFrame.from_items(res_df, orient='index', columns=['Class O', 'Class 1', 'Result'])
        print(df.head(15))
        return df
    else:
        d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
        eclf = load(open('pickle_files/VOTE.pkl', 'rb'))
        vect = load(open('pickle_files/vectVOTE.pkl', 'rb'))
        tf = TfidfTransformer()
        load_vect = CountVectorizer(vocabulary=vect)
        dtest_msg = tf.fit_transform(load_vect.fit_transform(d_test))
        pred_test = eclf.predict(dtest_msg)
        mnb_dict = model_assessment(u_classify='Voting EM', y_data=l_test, predicted_class=pred_test)
        return mnb_dict

def menu(files):
    menu = '''
                Ensemble Classifiers : 
            ---------------------------

            1. Random Forest Classifier 
            2. Bagging Classifier
            3. AdaBoosting Classifier
            4. Voting Classifier


           '''
    print(menu)
    print("\n\nEnsemble Classifiers training......")
    print("\nRandom Forest Classifier Training.....")
    rfc_classifier(files, 1)
    print("\nBagging Classifier Training.....")
    bagging_classifier(files, 1)
    print("\nAdaBoost Classifier Training.....")
    adaB_classifier(files, 1)
    print("\nVoting Classifier Training.....")
    voting_classifier(files, 1)



