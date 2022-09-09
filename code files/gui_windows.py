from tkinter import *
# import tkinter
# import numpy as np
from tkinter.filedialog import *
# from tkinter import ttk
import tkinter.messagebox
import sys
import os
import chardet
import codecs
import pandas as pd
import basic_classifiers as bc
import ensemble_classif as em
import hypeparameter_tune as hpt 
import comapre1 as cp
from pickle import dump, load
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandastable import Table, TableModel


main_win = Tk()
main_win.title("Spam Classifier Software")

top_f = Frame(main_win, width=580, padx=10, pady=10, bg="#29BF89", borderwidth=4, relief=RAISED)
top_f.pack(side=TOP, expand=True, fill='both')
mid_f = Frame(main_win, width=580, padx=10, pady=10, bg="#0083BB", borderwidth=4, relief=RAISED)
mid_f.pack(expand=True, fill='both')
bottom_f = Frame(main_win, width=580, padx=10, pady=10, bg="#0083BB", borderwidth=4, relief=RAISED)
bottom_f.pack(expand=True, fill='both')
last_f = Frame(main_win, width=580, padx=10, pady=10, bg="#29838B", borderwidth=4, relief=RAISED)
last_f.pack(side=BOTTOM, expand=True, fill='both')


f_dir = dict()


def  classify_window(var1):
    m_dict = dict()
    if var1 == 'mnb':
        top_l = Toplevel()
        top_l.title("Multinomial Naive Bayes Window")
        
        lr1 = LabelFrame(top_l, padx=3, pady=5, text="Multinomial Naive Bayes Descripton", borderwidth=5, relief=RIDGE)
        lr1.pack()
        det = """
                Multinomial Naive Bayes is a specialized version of Naive Bayes that is designed more for text documents. Whereas simple naive 
                Bayes would model a document as the presence and absence of particular words, multinomial naive bayes explicitly models the word 
                counts and adjusts the underlying calculations to deal with in.
              """
        l1 = Label(lr1, padx=3, pady=5, text=det)
        l1.pack(expand=True, fill="both")
        
        lr2 = LabelFrame(top_l, padx=3, pady=5, text="Classifiers buttons", borderwidth=5, relief=RAISED)
        lr2.pack(expand=True, fill="both")
        bt1 = Button(lr2, padx=3, pady=5, text="Train with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: MNB_f(1))
        bt1.grid(row=0, column=0, sticky="NEWS")
        bt2 = Button(lr2, padx=3, pady=5, text="Test with an unseen dataset", borderwidth=5, relief=RAISED, command=lambda: MNB_f(2))
        bt2.grid(row=0, column=1, sticky="NEWS")
        bt3 = Button(lr2, padx=3, pady=5, text="Test with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: MNB_f(3))
        bt3.grid(row=0, column=2, sticky="NEWS")        

    elif var1 == 'svm':
        top_l = Toplevel()
        top_l.title("Support Vector Machine Window")
        
        lr1 = LabelFrame(top_l, padx=3, pady=5, text="Support Vector Machine Descripton", borderwidth=5, relief=RIDGE)
        lr1.pack()
        det = """
                Support vector machines (SVM) are a group of supervised learning methods that can be applied to classification or regression. 
                Support vector machines represent an extension to nonlinear models of the generalized portrait algorithm developed by Vladimir 
                Vapnik . SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, even when the 
                data are not otherwise linearly separable. A separator between the categories is found, then the data are transformed in such a 
                way that the separator could be drawn as a hyperplane.
              """
        l1 = Label(lr1, padx=3, pady=5, text=det)
        l1.pack(expand=True, fill="both")
        
        lr2 = LabelFrame(top_l, padx=3, pady=5, text="Classifiers buttons", borderwidth=5, relief=RAISED)
        lr2.pack(expand=True, fill="both")
        bt1 = Button(lr2, padx=3, pady=5, text="Train with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: SVM_f(1))
        bt1.grid(row=0, column=0, sticky="NEWS")
        bt2 = Button(lr2, padx=3, pady=5, text="Test with an unseen dataset", borderwidth=5, relief=RAISED, command=lambda: SVM_f(2))
        bt2.grid(row=0, column=1, sticky="NEWS")
        bt3 = Button(lr2, padx=3, pady=5, text="Test with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: SVM_f(3))
        bt3.grid(row=0, column=2, sticky="NEWS") 

    elif var1 == 'knn':
        top_l = Toplevel()
        top_l.title("K-Nearest Neighbor Window")
        
        lr1 = LabelFrame(top_l, padx=3, pady=5, text="K-Nearest Neighbor Descripton", borderwidth=5, relief=RIDGE)
        lr1.pack()
        det = """
                The K-Nearest Neighbor (KNN) Classifier is a simple classifier that works well on basic recognition problems, however it can be 
                slow for real-time prediction if there are a large number of training examples and is not robust to noisy data. The KNN algorithm 
                is part of the GRT classification modules . K nearest neighbors is a simple algorithm that stores all available cases and 
                classifies new cases based on a similarity measure (e.g., distance functions).
              """
        l1 = Label(lr1, padx=3, pady=5, text=det)
        l1.pack(expand=True, fill="both")
        
        lr2 = LabelFrame(top_l, padx=3, pady=5, text="Classifiers buttons", borderwidth=5, relief=RAISED)
        lr2.pack(expand=True, fill="both")
        bt1 = Button(lr2, padx=3, pady=5, text="Train with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: KNN_f(1))
        bt1.grid(row=0, column=0, sticky="NEWS")
        bt2 = Button(lr2, padx=3, pady=5, text="Test with an unseen dataset", borderwidth=5, relief=RAISED, command=lambda: KNN_f(2))
        bt2.grid(row=0, column=1, sticky="NEWS")
        bt3 = Button(lr2, padx=3, pady=5, text="Test with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: KNN_f(3))
        bt3.grid(row=0, column=2, sticky="NEWS")

    elif var1 == 'dt':
        top_l = Toplevel()
        top_l.title("Decision Tree Window")
        
        lr1 = LabelFrame(top_l, padx=3, pady=5, text="Decision Tree Descripton", borderwidth=5, relief=RIDGE)
        lr1.pack()
        det = """
                Decision Tree Classifier is a simple and widely used classification technique. It applies a straitforward idea to solve the 
                classification problem. Decision Tree Classifier poses a series of carefully crafted questions about the attributes of the test 
                record. A decision tree algorithm is a decision support system. It uses a model that is tree-like decisions and their possible 
                consequences which includes - chance event outcomes, resource costs, and utility.
              """
        l1 = Label(lr1, padx=3, pady=5, text=det)
        l1.pack(expand=True, fill="both")
        
        lr2 = LabelFrame(top_l, padx=3, pady=5, text="Classifiers buttons", borderwidth=5, relief=RAISED)
        lr2.pack(expand=True, fill="both")
        bt1 = Button(lr2, padx=3, pady=5, text="Train with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: DTree_f(1))
        bt1.grid(row=0, column=0, sticky="NEWS")
        bt2 = Button(lr2, padx=3, pady=5, text="Test with an unseen dataset", borderwidth=5, relief=RAISED, command=lambda: DTree_f(2))
        bt2.grid(row=0, column=1, sticky="NEWS")
        bt3 = Button(lr2, padx=3, pady=5, text="Test with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: DTree_f(3))
        bt3.grid(row=0, column=2, sticky="NEWS")
        
    elif var1 == 'rfc':
        top_l = Toplevel()
        top_l.title("Random Forest Window")
        lr1 = LabelFrame(top_l, padx=3, pady=5, text="Random Forest Descripton", borderwidth=5, relief=RIDGE)
        lr1.pack()
        det = """
                A random forest classifier. A random forest is a meta estimator (i.e. an ensemble learning method) that fits a number of 
                decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and 
                control over-fitting. Random Forest algorithm is a supervised classification algorithm. We can see it from its name, which is 
                to create a forest by some way and make it random. There is a direct relationship between the number of trees in the forest and 
                the results it can get: the larger the number of trees, the more accurate the result.
              """
        l1 = Label(lr1, padx=3, pady=5, text=det)
        l1.pack(expand=True, fill="both")
        
        lr2 = LabelFrame(top_l, padx=3, pady=5, text="Classifiers buttons", borderwidth=5, relief=RAISED)
        lr2.pack(expand=True, fill="both")
        bt1 = Button(lr2, padx=3, pady=5, text="Train with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: RFC_f(1))
        bt1.grid(row=0, column=0, sticky="NEWS")
        bt2 = Button(lr2, padx=3, pady=5, text="Test with an unseen dataset", borderwidth=5, relief=RAISED, command=lambda: RFC_f(2))
        bt2.grid(row=0, column=1, sticky="NEWS")
        bt3 = Button(lr2, padx=3, pady=5, text="Test with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: RFC_f(3))
        bt3.grid(row=0, column=2, sticky="NEWS")

    elif var1 == 'ada':
        top_l = Toplevel()
        top_l.title("AdaBoosting Window")
        lr1 = LabelFrame(top_l, padx=3, pady=5, text="AdaBoosting Descripton", borderwidth=5, relief=RIDGE)
        lr1.pack()
        det = """
                AdaBoost, short for Adaptive Boosting, is a machine learning meta-algorithm formulated by Yoav Freund and Robert Schapire. An 
                AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional 
                copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that 
                subsequent classifiers focus more on difficult cases.
              """
        l1 = Label(lr1, padx=3, pady=5, text=det)
        l1.pack(expand=True, fill="both")
        
        lr2 = LabelFrame(top_l, padx=3, pady=5, text="Classifiers buttons", borderwidth=5, relief=RAISED)
        lr2.pack(expand=True, fill="both")
        bt1 = Button(lr2, padx=3, pady=5, text="Train with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: Ada_f(1))
        bt1.grid(row=0, column=0, sticky="NEWS")
        bt2 = Button(lr2, padx=3, pady=5, text="Test with an unseen dataset", borderwidth=5, relief=RAISED, command=lambda: Ada_f(2))
        bt2.grid(row=0, column=1, sticky="NEWS")
        bt3 = Button(lr2, padx=3, pady=5, text="Test with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: Ada_f(3))
        bt3.grid(row=0, column=2, sticky="NEWS")
        
    elif var1 == 'bag':
        top_l = Toplevel()
        top_l.title("Bagging Window")
        lr1 = LabelFrame(top_l, padx=3, pady=5, text="Bagging Descripton", borderwidth=5, relief=RIDGE)
        lr1.pack()
        det = """
                Bootstrap Aggregation (or Bagging for short), is a simple and very powerful ensemble method.An ensemble method is a technique 
                that combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any 
                individual model.Bootstrap Aggregation is a general procedure that can be used to reduce the variance for those algorithm that 
                have high variance. A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the 
                original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.
              """
        
        l1 = Label(lr1, padx=3, pady=5, text=det)
        l1.pack(expand=True, fill="both")
        
        lr2 = LabelFrame(top_l, padx=3, pady=5, text="Classifiers buttons", borderwidth=5, relief=RAISED)
        lr2.pack(expand=True, fill="both")
        bt1 = Button(lr2, padx=3, pady=5, text="Train with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: Bag_f(1))
        bt1.grid(row=0, column=0, sticky="NEWS")
        bt2 = Button(lr2, padx=3, pady=5, text="Test with an unseen dataset", borderwidth=5, relief=RAISED, command=lambda: Bag_f(2))
        bt2.grid(row=0, column=1, sticky="NEWS")
        bt3 = Button(lr2, padx=3, pady=5, text="Test with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: Bag_f(3))
        bt3.grid(row=0, column=2, sticky="NEWS")
        
    elif var1 == 'vot':
        top_l = Toplevel()
        top_l.title("Voting Window")
        lr1 = LabelFrame(top_l, padx=3, pady=5, text="Voting Descripton", borderwidth=5, relief=RIDGE)
        lr1.pack()
        det = """
                A Voting Classifier is a machine learning model that trains on an ensemble of numerous models and predicts an output (class) 
                based on their highest probability of chosen class as the output.
                It simply aggregates the findings of each classifier passed into Voting Classifier and predicts the output class based on the 
                highest majority of voting. The idea is instead of creating separate dedicated models and finding the accuracy for each them, 
                we create a single model which trains by these models and predicts output based on their combined majority of voting for each 
                output class.
              """
        l1 = Label(lr1, padx=3, pady=5, text=det)
        l1.pack(expand=True, fill="both")
        
        lr2 = LabelFrame(top_l, padx=3, pady=5, text="Classifiers buttons", borderwidth=5, relief=RAISED)
        lr2.pack(expand=True, fill="both")
        bt1 = Button(lr2, padx=3, pady=5, text="Train with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: Vote_f(1))
        bt1.grid(row=0, column=0, sticky="NEWS")
        bt2 = Button(lr2, padx=3, pady=5, text="Test with an unseen dataset", borderwidth=5, relief=RAISED, command=lambda: Vote_f(2))
        bt2.grid(row=0, column=1, sticky="NEWS")
        bt3 = Button(lr2, padx=3, pady=5, text="Test with a supervised dataset", borderwidth=5, relief=RAISED, command=lambda: Vote_f(3))
        bt3.grid(row=0, column=2, sticky="NEWS")

def file_prop(fname, ftype, fencode, fpath):
    print("File Name : ",fname)
    print("File Type : ",ftype)
    print("File Encoded : ",fencode)
    print("File Path : ",fpath)    

def find_encode(fname):
    r_file = open(fname, 'rb').read()
    result = chardet.detect(r_file)
    charenc = result['encoding']
    return charenc

def change_encode(fname, encding):
    try: 
        with open(fname, 'r', encoding=encding) as f, open("s1.csv", 'w', encoding='utf-8') as e:
            text = f.read() # for small files, for big use chunks
            e.write(text)
        os.remove(fname) # remove old encoding file
        os.rename("s1.csv", fname) # rename new encoding
    except UnicodeDecodeError:
        print('Decode Error')
    except UnicodeEncodeError:
        print('Encode Error')            

def file_process():
    global f_dir
    filename = askopenfilename(filetypes=[('CSV', '*.csv',), ('Excel', ('*.xls', '*.xlsx')),('Text File', '*.txt')])
    if len(filename) != 0:
        fencode = find_encode(filename)
        files = filename.split('/')
        fname = files[-1:]
        fname = "".join(fname)
        frmt = fname.split('.')
        frmt = frmt[-1:]
        frmt = "".join(frmt)
        if fencode not in ["utf-8","ascii"] and frmt not in ['xls', 'xlsx', 'txt']:
            reason = "Unsupported encoded file : "+str(fencode)
            tkinter.messagebox.showinfo("Reason", reason)
            tkinter.messagebox.showerror("Error", "Unable to open")
            resp = tkinter.messagebox.askquestion("Different Encoding","Do you want to encode it in to the supported format and try again?")
            if resp == 'yes' or resp == 'Yes' or resp == 'YES':
                change_encode(fname, fencode)
                fencode = find_encode(filename)
                if fencode not in ["utf-8","ascii"]:
                    sys.exit()
                else:
                    file_prop(filename, fname, frmt, fencode)
                    return (fname, frmt)
            else:
                resq = tkinter.messagebox.askquestion("Different File","Do you want to open another file and try again?")
                if resq == 'yes' or resq == 'Yes' or resq == 'YES':
                    file_process()
                else:
                    sys.exit()
                pass
            return 0
        elif frmt in ['xls', 'xlsx', 'txt'] or fencode in ['utf-8', 'ascii']:
            f_dir.update({
                'name' : fname,
                'format' : frmt,
                'encode' : fencode,
                'path' : filename
            })
            file_prop(fname, frmt, fencode, filename)
            return f_dir
    else:
        print("Empty Selection\n Unable to proceed\n Please select a file to open!!")
        file_process()


def train():
    TopL = Toplevel()
    TopL.title("Train Window for all classifiers")
    l1 = LabelFrame(TopL, padx=5, pady=3, borderwidth=4, relief=RAISED, text="Basic Classifiers")
    l1.pack(expand=True, fill="both")
    
    tt1 = Label(l1, padx=3, pady=2, fg="#d3d3d3", text="Multinomial Naive Bayes Classifier")
    tt1.pack(expand=True)
    tt2 = Label(l1, padx=3, pady=2, fg="#d3d3d3", text="Support Vector Classifier")
    tt2.pack(expand=True)
    tt3 = Label(l1, padx=3, pady=2, fg="#d3d3d3", text="Decision Tree Classifier")
    tt3.pack(expand=True)
    tt4 = Label(l1, padx=3, pady=2, fg="#d3d3d3", text="K-Nearest Neighbor Classifier")
    tt4.pack(expand=True)

    l2 = LabelFrame(TopL, padx=5, pady=3, borderwidth=4, relief=RAISED, text="Ensemble Methods")
    l2.pack(expand=True, fill="both")

    tt5 = Label(l2, padx=3, pady=2, fg="#d3d3d3", text="Random Forest Ensemble Method")
    tt5.grid(row=0)
    tt6 = Label(l2, padx=3, pady=2, fg="#d3d3d3", text="Bagging Ensemble Method")
    tt6.grid(row=1)
    tt7 = Label(l2, padx=3, pady=2, fg="#d3d3d3", text="AdaBoosting Ensemble Method")
    tt7.grid(row=2)
    tt8 = Label(l2, padx=3, pady=2, fg="#d3d3d3", text="Voting Ensemble Method")
    tt8.grid(row=3)
    
    r1 = bc.Multi_NB(f_dir, 1)
    if r1 != None:
        tt1.configure(fg="#008000")
    r2 = bc.SVM(f_dir, 1)
    if r2 != None:
        tt2.configure(fg="#008000")
    r3 = bc.D_Tree(f_dir, 1)
    if r3 != None:
        tt3.configure(fg="#008000")
    r4 = bc.KNC(f_dir, 1)
    if r4 != None:
        tt4.configure(fg="#008000")

    r5 = em.rfc_classifier(f_dir, 1)
    if r5 != None:
        tt5.configure(fg="#008000")
    r6 = em.bagging_classifier(f_dir, 1)
    if r6 != None:
        tt6.configure(fg="#008000")
    r7 = em.adaB_classifier(f_dir, 1)
    if r7 != None:
        tt7.configure(fg="#008000")
    r8 = em.voting_classifier(f_dir, 1)
    if r8 != None:
        tt8.configure(fg="#008000")

def MNB_f(var):
    if int(var) == 1:
        r2 = bc.Multi_NB(f_dir, 1)
        confs_matrx = r2['confusion']
        accuracy = r2['accuracy']
        precision = r2['precision']
        recall = r2['recall']
        f1 = r2['f1']

        r1 = bc.dataset_prop(f_dir)
        df1 = pd.DataFrame(data=list(r1['Ham']), columns=['Word', 'Count'])
        df2 = pd.DataFrame(data=list(r1['Spam']), columns=['Word', 'Count'])
        df1.index = [x for x in range(1, len(df1.values)+1)]
        df2.index = [x for x in range(1, len(df2.values)+1)]
        df1.index.name = 'id'
        df2.index.name = 'id'
        df4 = df1.copy(deep=True)
        df5 = df2.copy(deep=True)

        ccl1 = r1['CClass'][0]
        ccl2 = r1['CClass'][1]
        ccl1_series = pd.Series(ccl1)
        ccl2_series = pd.Series(ccl2)
        fr = {'Ham': ccl1_series, 'Spam': ccl2_series}
        df3 = pd.DataFrame(fr)
        
        top_l1 = Toplevel()
        top_l1.title("Multinomial Naive Bayes Training Result Window 1")

        l1 = LabelFrame(top_l1, padx=5, pady=2, text="Graphs", borderwidth=4, relief=RAISED)
        l1.pack(expand=True, fill="both")

        figure1 = plt.Figure(figsize=(5,4), dpi=100)
        ax1 = figure1.add_subplot(111)
        bar1 = FigureCanvasTkAgg(figure1, l1)
        bar1.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df1 = df1[['Word','Count']].groupby('Word').sum()
        df1.plot(kind='bar', legend=True, ax=ax1)
        ax1.set_title('Number of Non-Spam Words') 
        
        figure2 = plt.Figure(figsize=(5,4), dpi=100)
        ax2 = figure2.add_subplot(111) 
        bar2 = FigureCanvasTkAgg(figure2, l1)
        bar2.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df2 = df2[['Word','Count']].groupby('Word').sum()
        df2.plot(kind='bar', legend=True, ax=ax2)
        ax2.set_title('Number of Spam Words')

        figure3 = plt.Figure(figsize=(5,4), dpi=100) 
        ax3 = figure3.add_subplot(111)
        bar3 = FigureCanvasTkAgg(figure3, l1)
        bar3.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df3 = df3[['Ham','Spam']].sum()
        df3.plot(kind='bar', legend=True, ax=ax3)
        ax3.set_title('Number of Ham words vs Spam words')
        
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Results", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill="both")
        
        gr1 = Frame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr1.grid(row=0, column=0)
        l30 = Label(gr1, text="Confusion Matrix")
        l30.grid(row=0)
        l31 = Label(gr1, text=confs_matrx)
        l31.grid(row=1)

        gr2 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr2.grid(row=0, column=1)
        l32 = Label(gr2, text="Accuracy")
        l32.grid(row=0)
        l33 = Label(gr2, text=accuracy)
        l33.grid(row=1) 

        gr3 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr3.grid(row=0, column=2)
        l34 = Label(gr3, text="Precision")
        l34.grid(row=0)
        l35 = Label(gr3, text=precision)
        l35.grid(row=1)

        gr4 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr4.grid(row=0, column=3)
        l36 = Label(gr4, text="Recall")
        l36.grid(row=0)
        l37 = Label(gr4, text=recall)
        l37.grid(row=1)

        gr5 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr5.grid(row=0, column=3)
        l38 = Label(gr4, text="F1-score")
        l38.grid(row=0)
        l39 = Label(gr4, text=f1)
        l39.grid(row=1)

    elif int(var) == 2:
        r2 = bc.Multi_NB(f_dir, 2)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        dic = {0: 'Ham', 1: 'Spam'}
        r2['Result'] = r2['Result'].map(dic)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        top_l1 = Toplevel()
        top_l1.title("Mutlinomial Naive Bayes Result Info")    
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Table", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill='both')
        pt = Table(l2, dataframe=r2, showtoolbar=True, showstatusbar=True)
        pt.autoResizeColumns()
        pt.show()

    else:
        r3 = bc.Multi_NB(f_dir, 3)
        print("r1 : ", r3)

def SVM_f(var):
    if int(var) == 1:
        r2 = bc.SVM(f_dir, 1)
        confs_matrx = r2['confusion']
        accuracy = r2['accuracy']
        precision = r2['precision']
        recall = r2['recall']
        f1 = r2['f1']

        r1 = bc.dataset_prop(f_dir)
        df1 = pd.DataFrame(data=list(r1['Ham']), columns=['Word', 'Count'])
        df2 = pd.DataFrame(data=list(r1['Spam']), columns=['Word', 'Count'])
        df1.index = [x for x in range(1, len(df1.values)+1)]
        df2.index = [x for x in range(1, len(df2.values)+1)]
        df1.index.name = 'id'
        df2.index.name = 'id'
        df4 = df1.copy(deep=True)
        df5 = df2.copy(deep=True)

        ccl1 = r1['CClass'][0]
        ccl2 = r1['CClass'][1]
        ccl1_series = pd.Series(ccl1)
        ccl2_series = pd.Series(ccl2)
        fr = {'Ham': ccl1_series, 'Spam': ccl2_series}
        df3 = pd.DataFrame(fr)
        
        top_l1 = Toplevel()
        top_l1.title("Support Vector Machine Training Result Window 1")

        l1 = LabelFrame(top_l1, padx=5, pady=2, text="Graphs", borderwidth=4, relief=RAISED)
        l1.pack(expand=True, fill="both")

        figure1 = plt.Figure(figsize=(5,4), dpi=100)
        ax1 = figure1.add_subplot(111)
        bar1 = FigureCanvasTkAgg(figure1, l1)
        bar1.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df1 = df1[['Word','Count']].groupby('Word').sum()
        df1.plot(kind='bar', legend=True, ax=ax1)
        ax1.set_title('Number of Non-Spam Words') 
        
        figure2 = plt.Figure(figsize=(5,4), dpi=100)
        ax2 = figure2.add_subplot(111) 
        bar2 = FigureCanvasTkAgg(figure2, l1)
        bar2.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df2 = df2[['Word','Count']].groupby('Word').sum()
        df2.plot(kind='bar', legend=True, ax=ax2)
        ax2.set_title('Number of Spam Words')

        figure3 = plt.Figure(figsize=(5,4), dpi=100) 
        ax3 = figure3.add_subplot(111)
        bar3 = FigureCanvasTkAgg(figure3, l1)
        bar3.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df3 = df3[['Ham','Spam']].sum()
        df3.plot(kind='bar', legend=True, ax=ax3)
        ax3.set_title('Number of Ham words vs Spam words')
        
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Results", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill="both")
        
        gr1 = Frame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr1.grid(row=0, column=0)
        l30 = Label(gr1, text="Confusion Matrix")
        l30.grid(row=0)
        l31 = Label(gr1, text=confs_matrx)
        l31.grid(row=1)

        gr2 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr2.grid(row=0, column=1)
        l32 = Label(gr2, text="Accuracy")
        l32.grid(row=0)
        l33 = Label(gr2, text=accuracy)
        l33.grid(row=1) 

        gr3 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr3.grid(row=0, column=2)
        l34 = Label(gr3, text="Precision")
        l34.grid(row=0)
        l35 = Label(gr3, text=precision)
        l35.grid(row=1)

        gr4 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr4.grid(row=0, column=3)
        l36 = Label(gr4, text="Recall")
        l36.grid(row=0)
        l37 = Label(gr4, text=recall)
        l37.grid(row=1)

        gr5 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr5.grid(row=0, column=3)
        l38 = Label(gr4, text="F1-score")
        l38.grid(row=0)
        l39 = Label(gr4, text=f1)
        l39.grid(row=1)

    elif int(var) == 2:
        r2 = bc.SVM(f_dir, 2)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        dic = {0: 'Ham', 1: 'Spam'}
        r2['Result'] = r2['Result'].map(dic)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        top_l1 = Toplevel()
        top_l1.title("Support Vector Machine Result Info")    
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Table", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill='both')
        pt = Table(l2, dataframe=r2, showtoolbar=True, showstatusbar=True)
        pt.autoResizeColumns()
        
        pt.show()
    else:
        r3 = bc.SVM(f_dir, 3)
        print("3: ", r3)

def KNN_f(var):
    if int(var) == 1:
        r2 = bc.KNC(f_dir, 1)
        confs_matrx = r2['confusion']
        accuracy = r2['accuracy']
        precision = r2['precision']
        recall = r2['recall']
        f1 = r2['f1']

        r1 = bc.dataset_prop(f_dir)
        df1 = pd.DataFrame(data=list(r1['Ham']), columns=['Word', 'Count'])
        df2 = pd.DataFrame(data=list(r1['Spam']), columns=['Word', 'Count'])
        df1.index = [x for x in range(1, len(df1.values)+1)]
        df2.index = [x for x in range(1, len(df2.values)+1)]
        df1.index.name = 'id'
        df2.index.name = 'id'
        df4 = df1.copy(deep=True)
        df5 = df2.copy(deep=True)

        ccl1 = r1['CClass'][0]
        ccl2 = r1['CClass'][1]
        ccl1_series = pd.Series(ccl1)
        ccl2_series = pd.Series(ccl2)
        fr = {'Ham': ccl1_series, 'Spam': ccl2_series}
        df3 = pd.DataFrame(fr)
        
        top_l1 = Toplevel()
        top_l1.title("K-Nearest Neighbor Training Result Window 1")

        l1 = LabelFrame(top_l1, padx=5, pady=2, text="Graphs", borderwidth=4, relief=RAISED)
        l1.pack(expand=True, fill="both")

        figure1 = plt.Figure(figsize=(5,4), dpi=100)
        ax1 = figure1.add_subplot(111)
        bar1 = FigureCanvasTkAgg(figure1, l1)
        bar1.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df1 = df1[['Word','Count']].groupby('Word').sum()
        df1.plot(kind='bar', legend=True, ax=ax1)
        ax1.set_title('Number of Non-Spam Words') 
        
        figure2 = plt.Figure(figsize=(5,4), dpi=100)
        ax2 = figure2.add_subplot(111) 
        bar2 = FigureCanvasTkAgg(figure2, l1)
        bar2.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df2 = df2[['Word','Count']].groupby('Word').sum()
        df2.plot(kind='bar', legend=True, ax=ax2)
        ax2.set_title('Number of Spam Words')

        figure3 = plt.Figure(figsize=(5,4), dpi=100) 
        ax3 = figure3.add_subplot(111)
        bar3 = FigureCanvasTkAgg(figure3, l1)
        bar3.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df3 = df3[['Ham','Spam']].sum()
        df3.plot(kind='bar', legend=True, ax=ax3)
        ax3.set_title('Number of Ham words vs Spam words')
        
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Results", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill="both")
        
        gr1 = Frame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr1.grid(row=0, column=0)
        l30 = Label(gr1, text="Confusion Matrix")
        l30.grid(row=0)
        l31 = Label(gr1, text=confs_matrx)
        l31.grid(row=1)

        gr2 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr2.grid(row=0, column=1)
        l32 = Label(gr2, text="Accuracy")
        l32.grid(row=0)
        l33 = Label(gr2, text=accuracy)
        l33.grid(row=1) 

        gr3 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr3.grid(row=0, column=2)
        l34 = Label(gr3, text="Precision")
        l34.grid(row=0)
        l35 = Label(gr3, text=precision)
        l35.grid(row=1)

        gr4 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr4.grid(row=0, column=3)
        l36 = Label(gr4, text="Recall")
        l36.grid(row=0)
        l37 = Label(gr4, text=recall)
        l37.grid(row=1)

        gr5 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr5.grid(row=0, column=3)
        l38 = Label(gr4, text="F1-score")
        l38.grid(row=0)
        l39 = Label(gr4, text=f1)
        l39.grid(row=1)
    
    elif int(var) == 2:
        r2 = bc.KNC(f_dir, 2)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        dic = {0: 'Ham', 1: 'Spam'}
        r2['Result'] = r2['Result'].map(dic)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        top_l1 = Toplevel()
        top_l1.title("K-Nearest Neighbor Result Info")    
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Table", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill='both')
        pt = Table(l2, dataframe=r2, showtoolbar=True, showstatusbar=True)
        pt.autoResizeColumns()
        pt.show()

    else:
        r3 = bc.KNC(f_dir, 3)
        print("3: ", r3)

def DTree_f(var):
    if int(var) == 1:
        r2 = bc.D_Tree(f_dir, 1)
        confs_matrx = r2['confusion']
        accuracy = r2['accuracy']
        precision = r2['precision']
        recall = r2['recall']
        f1 = r2['f1']

        r1 = bc.dataset_prop(f_dir)
        df1 = pd.DataFrame(data=list(r1['Ham']), columns=['Word', 'Count'])
        df2 = pd.DataFrame(data=list(r1['Spam']), columns=['Word', 'Count'])
        df1.index = [x for x in range(1, len(df1.values)+1)]
        df2.index = [x for x in range(1, len(df2.values)+1)]
        df1.index.name = 'id'
        df2.index.name = 'id'
        df4 = df1.copy(deep=True)
        df5 = df2.copy(deep=True)

        ccl1 = r1['CClass'][0]
        ccl2 = r1['CClass'][1]
        ccl1_series = pd.Series(ccl1)
        ccl2_series = pd.Series(ccl2)
        fr = {'Ham': ccl1_series, 'Spam': ccl2_series}
        df3 = pd.DataFrame(fr)
        
        top_l1 = Toplevel()
        top_l1.title("Decision Tree Training Result Window 1")

        l1 = LabelFrame(top_l1, padx=5, pady=2, text="Graphs", borderwidth=4, relief=RAISED)
        l1.pack(expand=True, fill="both")

        figure1 = plt.Figure(figsize=(5,4), dpi=100)
        ax1 = figure1.add_subplot(111)
        bar1 = FigureCanvasTkAgg(figure1, l1)
        bar1.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df1 = df1[['Word','Count']].groupby('Word').sum()
        df1.plot(kind='bar', legend=True, ax=ax1)
        ax1.set_title('Number of Non-Spam Words') 
        
        figure2 = plt.Figure(figsize=(5,4), dpi=100)
        ax2 = figure2.add_subplot(111) 
        bar2 = FigureCanvasTkAgg(figure2, l1)
        bar2.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df2 = df2[['Word','Count']].groupby('Word').sum()
        df2.plot(kind='bar', legend=True, ax=ax2)
        ax2.set_title('Number of Spam Words')

        figure3 = plt.Figure(figsize=(5,4), dpi=100) 
        ax3 = figure3.add_subplot(111)
        bar3 = FigureCanvasTkAgg(figure3, l1)
        bar3.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df3 = df3[['Ham','Spam']].sum()
        df3.plot(kind='bar', legend=True, ax=ax3)
        ax3.set_title('Number of Ham words vs Spam words')
        
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Results", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill="both")
        
        gr1 = Frame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr1.grid(row=0, column=0)
        l30 = Label(gr1, text="Confusion Matrix")
        l30.grid(row=0)
        l31 = Label(gr1, text=confs_matrx)
        l31.grid(row=1)

        gr2 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr2.grid(row=0, column=1)
        l32 = Label(gr2, text="Accuracy")
        l32.grid(row=0)
        l33 = Label(gr2, text=accuracy)
        l33.grid(row=1) 

        gr3 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr3.grid(row=0, column=2)
        l34 = Label(gr3, text="Precision")
        l34.grid(row=0)
        l35 = Label(gr3, text=precision)
        l35.grid(row=1)

        gr4 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr4.grid(row=0, column=3)
        l36 = Label(gr4, text="Recall")
        l36.grid(row=0)
        l37 = Label(gr4, text=recall)
        l37.grid(row=1)

        gr5 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr5.grid(row=0, column=3)
        l38 = Label(gr4, text="F1-score")
        l38.grid(row=0)
        l39 = Label(gr4, text=f1)
        l39.grid(row=1)
    
    elif int(var) == 2:
        r2 = bc.D_Tree(f_dir, 2)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        dic = {0: 'Ham', 1: 'Spam'}
        r2['Result'] = r2['Result'].map(dic)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        top_l1 = Toplevel()
        top_l1.title("Decision Tree Result Info")    
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Table", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill='both')
        pt = Table(l2, dataframe=r2, showtoolbar=True, showstatusbar=True)
        pt.autoResizeColumns()
        pt.show()

    else:
        r3 = bc.D_Tree(f_dir, 3)
        print("3: ", r3)

def RFC_f(var):
    if int(var) == 1:
        r2 = em.rfc_classifier(f_dir, 1)
        confs_matrx = r2['confusion']
        accuracy = r2['accuracy']
        precision = r2['precision']
        recall = r2['recall']
        f1 = r2['f1']

        r1 = bc.dataset_prop(f_dir)
        df1 = pd.DataFrame(data=list(r1['Ham']), columns=['Word', 'Count'])
        df2 = pd.DataFrame(data=list(r1['Spam']), columns=['Word', 'Count'])
        df1.index = [x for x in range(1, len(df1.values)+1)]
        df2.index = [x for x in range(1, len(df2.values)+1)]
        df1.index.name = 'id'
        df2.index.name = 'id'
        df4 = df1.copy(deep=True)
        df5 = df2.copy(deep=True)

        ccl1 = r1['CClass'][0]
        ccl2 = r1['CClass'][1]
        ccl1_series = pd.Series(ccl1)
        ccl2_series = pd.Series(ccl2)
        fr = {'Ham': ccl1_series, 'Spam': ccl2_series}
        df3 = pd.DataFrame(fr)
        
        top_l1 = Toplevel()
        top_l1.title("Random Forest Training Result Window 1")

        l1 = LabelFrame(top_l1, padx=5, pady=2, text="Graphs", borderwidth=4, relief=RAISED)
        l1.pack(expand=True, fill="both")

        figure1 = plt.Figure(figsize=(5,4), dpi=100)
        ax1 = figure1.add_subplot(111)
        bar1 = FigureCanvasTkAgg(figure1, l1)
        bar1.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df1 = df1[['Word','Count']].groupby('Word').sum()
        df1.plot(kind='bar', legend=True, ax=ax1)
        ax1.set_title('Number of Non-Spam Words') 
        
        figure2 = plt.Figure(figsize=(5,4), dpi=100)
        ax2 = figure2.add_subplot(111) 
        bar2 = FigureCanvasTkAgg(figure2, l1)
        bar2.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df2 = df2[['Word','Count']].groupby('Word').sum()
        df2.plot(kind='bar', legend=True, ax=ax2)
        ax2.set_title('Number of Spam Words')

        figure3 = plt.Figure(figsize=(5,4), dpi=100) 
        ax3 = figure3.add_subplot(111)
        bar3 = FigureCanvasTkAgg(figure3, l1)
        bar3.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df3 = df3[['Ham','Spam']].sum()
        df3.plot(kind='bar', legend=True, ax=ax3)
        ax3.set_title('Number of Ham words vs Spam words')
        
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Results", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill="both")
        
        gr1 = Frame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr1.grid(row=0, column=0)
        l30 = Label(gr1, text="Confusion Matrix")
        l30.grid(row=0)
        l31 = Label(gr1, text=confs_matrx)
        l31.grid(row=1)

        gr2 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr2.grid(row=0, column=1)
        l32 = Label(gr2, text="Accuracy")
        l32.grid(row=0)
        l33 = Label(gr2, text=accuracy)
        l33.grid(row=1) 

        gr3 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr3.grid(row=0, column=2)
        l34 = Label(gr3, text="Precision")
        l34.grid(row=0)
        l35 = Label(gr3, text=precision)
        l35.grid(row=1)

        gr4 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr4.grid(row=0, column=3)
        l36 = Label(gr4, text="Recall")
        l36.grid(row=0)
        l37 = Label(gr4, text=recall)
        l37.grid(row=1)

        gr5 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr5.grid(row=0, column=3)
        l38 = Label(gr4, text="F1-score")
        l38.grid(row=0)
        l39 = Label(gr4, text=f1)
        l39.grid(row=1)

    elif int(var) == 2:
        r2 = em.rfc_classifier(f_dir, 2)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        dic = {0: 'Ham', 1: 'Spam'}
        r2['Result'] = r2['Result'].map(dic)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        top_l1 = Toplevel()
        top_l1.title("Random Forest Result Info")    
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Table", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill='both')
        pt = Table(l2, dataframe=r2, showtoolbar=True, showstatusbar=True)
        pt.autoResizeColumns()
        pt.show()

    else:
        r3 = em.rfc_classifier(f_dir, 3)
        print("3: ", r3)

def Bag_f(var):
    if int(var) == 1:
        r2 = em.bagging_classifier(f_dir, 1)
        confs_matrx = r2['confusion']
        accuracy = r2['accuracy']
        precision = r2['precision']
        recall = r2['recall']
        f1 = r2['f1']

        r1 = bc.dataset_prop(f_dir)
        df1 = pd.DataFrame(data=list(r1['Ham']), columns=['Word', 'Count'])
        df2 = pd.DataFrame(data=list(r1['Spam']), columns=['Word', 'Count'])
        df1.index = [x for x in range(1, len(df1.values)+1)]
        df2.index = [x for x in range(1, len(df2.values)+1)]
        df1.index.name = 'id'
        df2.index.name = 'id'
        df4 = df1.copy(deep=True)
        df5 = df2.copy(deep=True)

        ccl1 = r1['CClass'][0]
        ccl2 = r1['CClass'][1]
        ccl1_series = pd.Series(ccl1)
        ccl2_series = pd.Series(ccl2)
        fr = {'Ham': ccl1_series, 'Spam': ccl2_series}
        df3 = pd.DataFrame(fr)
        
        top_l1 = Toplevel()
        top_l1.title("Bagging (Bootstrap Aggregation) Training Result Window 1")

        l1 = LabelFrame(top_l1, padx=5, pady=2, text="Graphs", borderwidth=4, relief=RAISED)
        l1.pack(expand=True, fill="both")

        figure1 = plt.Figure(figsize=(5,4), dpi=100)
        ax1 = figure1.add_subplot(111)
        bar1 = FigureCanvasTkAgg(figure1, l1)
        bar1.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df1 = df1[['Word','Count']].groupby('Word').sum()
        df1.plot(kind='bar', legend=True, ax=ax1)
        ax1.set_title('Number of Non-Spam Words') 
        
        figure2 = plt.Figure(figsize=(5,4), dpi=100)
        ax2 = figure2.add_subplot(111) 
        bar2 = FigureCanvasTkAgg(figure2, l1)
        bar2.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df2 = df2[['Word','Count']].groupby('Word').sum()
        df2.plot(kind='bar', legend=True, ax=ax2)
        ax2.set_title('Number of Spam Words')

        figure3 = plt.Figure(figsize=(5,4), dpi=100) 
        ax3 = figure3.add_subplot(111)
        bar3 = FigureCanvasTkAgg(figure3, l1)
        bar3.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df3 = df3[['Ham','Spam']].sum()
        df3.plot(kind='bar', legend=True, ax=ax3)
        ax3.set_title('Number of Ham words vs Spam words')
        
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Results", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill="both")
        
        gr1 = Frame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr1.grid(row=0, column=0)
        l30 = Label(gr1, text="Confusion Matrix")
        l30.grid(row=0)
        l31 = Label(gr1, text=confs_matrx)
        l31.grid(row=1)

        gr2 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr2.grid(row=0, column=1)
        l32 = Label(gr2, text="Accuracy")
        l32.grid(row=0)
        l33 = Label(gr2, text=accuracy)
        l33.grid(row=1) 

        gr3 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr3.grid(row=0, column=2)
        l34 = Label(gr3, text="Precision")
        l34.grid(row=0)
        l35 = Label(gr3, text=precision)
        l35.grid(row=1)

        gr4 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr4.grid(row=0, column=3)
        l36 = Label(gr4, text="Recall")
        l36.grid(row=0)
        l37 = Label(gr4, text=recall)
        l37.grid(row=1)

        gr5 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr5.grid(row=0, column=3)
        l38 = Label(gr4, text="F1-score")
        l38.grid(row=0)
        l39 = Label(gr4, text=f1)
        l39.grid(row=1)
    
    elif int(var) == 2:
        r2 = em.bagging_classifier(f_dir, 2)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        dic = {0: 'Ham', 1: 'Spam'}
        r2['Result'] = r2['Result'].map(dic)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        top_l1 = Toplevel()
        top_l1.title("Bagging Result Info")    
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Table", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill='both')
        pt = Table(l2, dataframe=r2, showtoolbar=True, showstatusbar=True)
        pt.autoResizeColumns()
        pt.show()

    else:
        r3 = em.bagging_classifier(f_dir, 3)
        print("3: ", r3)

def Ada_f(var):
    if int(var) == 1:
        r2 = em.adaB_classifier(f_dir, 1)
        confs_matrx = r2['confusion']
        accuracy = r2['accuracy']
        precision = r2['precision']
        recall = r2['recall']
        f1 = r2['f1']

        r1 = bc.dataset_prop(f_dir)
        df1 = pd.DataFrame(data=list(r1['Ham']), columns=['Word', 'Count'])
        df2 = pd.DataFrame(data=list(r1['Spam']), columns=['Word', 'Count'])
        df1.index = [x for x in range(1, len(df1.values)+1)]
        df2.index = [x for x in range(1, len(df2.values)+1)]
        df1.index.name = 'id'
        df2.index.name = 'id'
        df4 = df1.copy(deep=True)
        df5 = df2.copy(deep=True)

        ccl1 = r1['CClass'][0]
        ccl2 = r1['CClass'][1]
        ccl1_series = pd.Series(ccl1)
        ccl2_series = pd.Series(ccl2)
        fr = {'Ham': ccl1_series, 'Spam': ccl2_series}
        df3 = pd.DataFrame(fr)
        
        top_l1 = Toplevel()
        top_l1.title("AdaBoosting Training Result Window 1")

        l1 = LabelFrame(top_l1, padx=5, pady=2, text="Graphs", borderwidth=4, relief=RAISED)
        l1.pack(expand=True, fill="both")

        figure1 = plt.Figure(figsize=(5,4), dpi=100)
        ax1 = figure1.add_subplot(111)
        bar1 = FigureCanvasTkAgg(figure1, l1)
        bar1.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df1 = df1[['Word','Count']].groupby('Word').sum()
        df1.plot(kind='bar', legend=True, ax=ax1)
        ax1.set_title('Number of Non-Spam Words') 
        
        figure2 = plt.Figure(figsize=(5,4), dpi=100)
        ax2 = figure2.add_subplot(111) 
        bar2 = FigureCanvasTkAgg(figure2, l1)
        bar2.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df2 = df2[['Word','Count']].groupby('Word').sum()
        df2.plot(kind='bar', legend=True, ax=ax2)
        ax2.set_title('Number of Spam Words')

        figure3 = plt.Figure(figsize=(5,4), dpi=100) 
        ax3 = figure3.add_subplot(111)
        bar3 = FigureCanvasTkAgg(figure3, l1)
        bar3.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df3 = df3[['Ham','Spam']].sum()
        df3.plot(kind='bar', legend=True, ax=ax3)
        ax3.set_title('Number of Ham words vs Spam words')
        
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Results", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill="both")
        
        gr1 = Frame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr1.grid(row=0, column=0)
        l30 = Label(gr1, text="Confusion Matrix")
        l30.grid(row=0)
        l31 = Label(gr1, text=confs_matrx)
        l31.grid(row=1)

        gr2 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr2.grid(row=0, column=1)
        l32 = Label(gr2, text="Accuracy")
        l32.grid(row=0)
        l33 = Label(gr2, text=accuracy)
        l33.grid(row=1) 

        gr3 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr3.grid(row=0, column=2)
        l34 = Label(gr3, text="Precision")
        l34.grid(row=0)
        l35 = Label(gr3, text=precision)
        l35.grid(row=1)

        gr4 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr4.grid(row=0, column=3)
        l36 = Label(gr4, text="Recall")
        l36.grid(row=0)
        l37 = Label(gr4, text=recall)
        l37.grid(row=1)

        gr5 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr5.grid(row=0, column=3)
        l38 = Label(gr4, text="F1-score")
        l38.grid(row=0)
        l39 = Label(gr4, text=f1)
        l39.grid(row=1)
    
    elif int(var) == 2:
        r2 = em.adaB_classifier(f_dir, 2)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        dic = {0: 'Ham', 1: 'Spam'}
        r2['Result'] = r2['Result'].map(dic)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        top_l1 = Toplevel()
        top_l1.title("AdaBoosting Result Info")    
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Table", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill='both')
        pt = Table(l2, dataframe=r2, showtoolbar=True, showstatusbar=True)
        pt.autoResizeColumns()
        pt.show()

    else:
        r3 = em.adaB_classifier(f_dir, 3)
        print("3: ", r3)

def Vote_f(var):
    if int(var) == 1:
        r2 = em.voting_classifier(f_dir, 1)
        confs_matrx = r2['confusion']
        accuracy = r2['accuracy']
        precision = r2['precision']
        recall = r2['recall']
        f1 = r2['f1']

        r1 = bc.dataset_prop(f_dir)
        df1 = pd.DataFrame(data=list(r1['Ham']), columns=['Word', 'Count'])
        df2 = pd.DataFrame(data=list(r1['Spam']), columns=['Word', 'Count'])
        df1.index = [x for x in range(1, len(df1.values)+1)]
        df2.index = [x for x in range(1, len(df2.values)+1)]
        df1.index.name = 'id'
        df2.index.name = 'id'
        df4 = df1.copy(deep=True)
        df5 = df2.copy(deep=True)

        ccl1 = r1['CClass'][0]
        ccl2 = r1['CClass'][1]
        ccl1_series = pd.Series(ccl1)
        ccl2_series = pd.Series(ccl2)
        fr = {'Ham': ccl1_series, 'Spam': ccl2_series}
        df3 = pd.DataFrame(fr)
        
        top_l1 = Toplevel()
        top_l1.title("Voting Training Result Window 1")

        l1 = LabelFrame(top_l1, padx=5, pady=2, text="Graphs", borderwidth=4, relief=RAISED)
        l1.pack(expand=True, fill="both")

        figure1 = plt.Figure(figsize=(5,4), dpi=100)
        ax1 = figure1.add_subplot(111)
        bar1 = FigureCanvasTkAgg(figure1, l1)
        bar1.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df1 = df1[['Word','Count']].groupby('Word').sum()
        df1.plot(kind='bar', legend=True, ax=ax1)
        ax1.set_title('Number of Non-Spam Words') 
        
        figure2 = plt.Figure(figsize=(5,4), dpi=100)
        ax2 = figure2.add_subplot(111) 
        bar2 = FigureCanvasTkAgg(figure2, l1)
        bar2.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df2 = df2[['Word','Count']].groupby('Word').sum()
        df2.plot(kind='bar', legend=True, ax=ax2)
        ax2.set_title('Number of Spam Words')

        figure3 = plt.Figure(figsize=(5,4), dpi=100) 
        ax3 = figure3.add_subplot(111)
        bar3 = FigureCanvasTkAgg(figure3, l1)
        bar3.get_tk_widget().pack(side=LEFT, fill=BOTH)
        df3 = df3[['Ham','Spam']].sum()
        df3.plot(kind='bar', legend=True, ax=ax3)
        ax3.set_title('Number of Ham words vs Spam words')
        
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Results", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill="both")
        
        gr1 = Frame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr1.grid(row=0, column=0)
        l30 = Label(gr1, text="Confusion Matrix")
        l30.grid(row=0)
        l31 = Label(gr1, text=confs_matrx)
        l31.grid(row=1)

        gr2 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr2.grid(row=0, column=1)
        l32 = Label(gr2, text="Accuracy")
        l32.grid(row=0)
        l33 = Label(gr2, text=accuracy)
        l33.grid(row=1) 

        gr3 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr3.grid(row=0, column=2)
        l34 = Label(gr3, text="Precision")
        l34.grid(row=0)
        l35 = Label(gr3, text=precision)
        l35.grid(row=1)

        gr4 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr4.grid(row=0, column=3)
        l36 = Label(gr4, text="Recall")
        l36.grid(row=0)
        l37 = Label(gr4, text=recall)
        l37.grid(row=1)

        gr5 = LabelFrame(l2, padx=5, pady=3, borderwidth=4, relief=RAISED)
        gr5.grid(row=0, column=3)
        l38 = Label(gr4, text="F1-score")
        l38.grid(row=0)
        l39 = Label(gr4, text=f1)
        l39.grid(row=1)
    
    elif int(var) == 2:
        r2 = em.voting_classifier(f_dir, 2)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        dic = {0: 'Ham', 1: 'Spam'}
        r2['Result'] = r2['Result'].map(dic)
        print("Val : ",r2)
        print("\ntype : ",type(r2))
        top_l1 = Toplevel()
        top_l1.title("Voting Result Info")    
        l2 = LabelFrame(top_l1, padx=5, pady=2, text="Table", borderwidth=4, relief=RAISED)
        l2.pack(expand=True, fill='both')
        pt = Table(l2, dataframe=r2, showtoolbar=True, showstatusbar=True)
        pt.autoResizeColumns()
        pt.show()

    else:
        r3 = em.voting_classifier(f_dir, 3)
        print("3: ", r3)


def compared1():
    #Dataframe which is returned from the comparison program
    #Result is plotted on the screen
    df = cp.cll_all(f_dir)
    
    top_l = Toplevel()
    top_l.title("Comparison window")
    l1 = LabelFrame(top_l, padx=5, pady=2, text="Graph", borderwidth=4, relief=RAISED)
    l1.pack(expand=True, fill='both')
    ll1 = Frame(l1, padx=2, pady=2, borderwidth=4, relief=RAISED)
    ll1.grid(row=0, column=0)
    figure1 = plt.Figure(figsize=(6,4), dpi=100)
    ax1 = figure1.add_subplot(111)
    chart_type = FigureCanvasTkAgg(figure1, ll1)
    chart_type.get_tk_widget().pack(expand=True)
    df1 = df[['Classifier','Score1','Score2','Score3','Score4']].groupby('Classifier').sum()
    df1.plot(kind='bar', legend=True, ax=ax1, colormap="Accent")
    ax1.set_title('Classifiers Comparison')
    
    ll2 = Frame(l1, padx=2, pady=2, borderwidth=4, relief=RAISED)
    ll2.grid(row=0, column=1)
    figure2 = plt.Figure(figsize=(6,4), dpi=100)
    ax2 = figure2.add_subplot(111)
    chart_type = FigureCanvasTkAgg(figure2, ll2)
    chart_type.get_tk_widget().pack(expand=True)
    df2 = df[['Classifier','Score1','Score2','Score3','Score4']].groupby('Classifier').sum()
    df2.plot(kind='line', legend=True, ax=ax2, marker='o', fontsize=10)
    ax2.set_title('Classifiers Comparison')
    
    l2 = LabelFrame(top_l, padx=5, pady=2, text="Table", borderwidth=4, relief=RAISED)
    l2.pack(expand=True, fill='both')
    pt = Table(l2, dataframe=df, showtoolbar=True, showstatusbar=True)
    pt.autoResizeColumns()
    pt.show()

def wai(f1, f2, f3, f4, f5):
    f1.configure(fg="#29BF89", text="Dataset acquired")
    f2.configure(fg="#29BF89", text="Compatibility checked")
    f3.configure(fg="#29BF89", text="No duplicates found")
    f4.configure(fg="#29BF89", text="No Null/NAN values found")
    f5.configure(fg="#29Bf89", text="Ready for classification")


def main():
    
    #Top Layer
    tf_title = LabelFrame(top_f, padx=5, pady=2, width=590, text="File Sector")
    tf_title.pack(expand=True, fill='both')
    tf_title.grid_columnconfigure(0, weight=15)
    tf_title.grid_columnconfigure(1, weight=15)

    f_dis = Frame(tf_title, padx=5, pady=2, width=580, borderwidth=4, relief=SUNKEN)
    f_dis.grid(row=0, column=0, sticky="nesw")
    f_nm = Label(f_dis, padx=5, pady=2, text="Dataset insertion here")
    f_nm.grid(row=0, sticky=E)
    f_open = Button(f_dis, padx=3, pady=3, text="Open", command=lambda :file_process())
    f_open.grid(row=1)
    f_dis.grid_rowconfigure(0, weight=5)
    f_dis.grid_rowconfigure(1, weight=5)
    
    f_det = Frame(tf_title, padx=5, pady=2, borderwidth=4, relief=SUNKEN)
    f_det.grid(row=0, column=1, sticky="nesw")
    f_l1 = Label(f_det, padx=5, pady=2, fg='#d3d3d3', text="Acquiring dataset")
    f_l1.grid(row=0 )
    f_l2 = Label(f_det, padx=5, pady=2, fg='#d3d3d3', text="Checking compatibility")
    f_l2.grid(row=1 )
    f_l3 = Label(f_det, padx=5, pady=2, fg='#d3d3d3', text="Checking for duplicates")
    f_l3.grid(row=2 )
    f_l4 = Label(f_det, padx=5, pady=2, fg='#d3d3d3', text="Checking for NAN/Null values")
    f_l4.grid(row=3 )
    f_l5 = Label(f_det, padx=5, pady=2, fg='#d3d3d3', text="Not ready for classification")
    f_l5.grid(row=4 )

    #Middle Layer
    mf_title = LabelFrame(mid_f, padx=5, pady=2, width=590, text="Basic Classifier Sector")
    mf_title.pack(expand=TRUE)
    
    left_gr = Frame(mf_title, padx=5, pady=2, width=590, borderwidth=4, relief=SUNKEN)
    left_gr.grid(row=0, column=0)
    mnb = Label(left_gr, padx=5,  pady=2, text="Multinomial Naive Bayes")
    mnb.grid(row=0 )
    mnb_btt = Button(left_gr, padx=5, pady=2, text="Test", command=lambda: classify_window('mnb'))
    mnb_btt.grid(row=1 )
    svm = Label(left_gr, padx=5, pady=2, text="Support Vector Machine")
    svm.grid(row=2 )
    svm_btt = Button(left_gr, padx=5, pady=2, text="Test", command=lambda: classify_window('svm'))
    svm_btt.grid(row=3 )

    right_gr = Frame(mf_title, padx=5, pady=2, width=590, borderwidth=4, relief=SUNKEN)
    right_gr.grid(row=0, column=1)
    knn = Label(right_gr, padx=5,  pady=2, text="K-Nearest Neighbors")
    knn.grid(row=0)
    knn_btt = Button(right_gr, padx=5, pady=2, text="Test", command=lambda: classify_window('knn'))
    knn_btt.grid(row=1)
    dt = Label(right_gr, padx=5, pady=2, text="Decision Tree")
    dt.grid(row=2)
    dt_btt = Button(right_gr, padx=5, pady=2, text="Test", command=lambda: classify_window('dt'))
    dt_btt.grid(row=3)   

    #Bottom Layer
    bf_title = LabelFrame(bottom_f, padx=5, pady=2, width=590, text="Ensemble Methods Sector")
    bf_title.pack(expand=TRUE)
    
    left_gr = Frame(bf_title, padx=5, pady=2, width=590, borderwidth=4, relief=SUNKEN)
    left_gr.grid(row=0, column=0)
    rfc = Label(left_gr, padx=5, pady=2, text="Random Forest")
    rfc.grid(row=0)
    rfc_btt = Button(left_gr, padx=5, pady=2, text="Test", command=lambda: classify_window('rfc'))
    rfc_btt.grid(row=1)
    bag = Label(left_gr, padx=5, pady=2, text="Bagging Method")
    bag.grid(row=2)
    bag_btt = Button(left_gr, padx=5, pady=2, text="Test", command=lambda: classify_window('bag'))
    bag_btt.grid(row=3)

    right_gr = Frame(bf_title, padx=5, pady=2, width=590, borderwidth=4, relief=SUNKEN)
    right_gr.grid(row=0, column=1)
    ada = Label(right_gr, padx=5,  pady=2, text="AdaBoosting Method")
    ada.grid(row=0)
    ada_btt = Button(right_gr, padx=5, pady=2, text="Test", command=lambda: classify_window('ada'))
    ada_btt.grid(row=1)
    vot = Label(right_gr, padx=5, pady=2, text="Voting Method")
    vot.grid(row=2)
    vot_btt = Button(right_gr, padx=5, pady=2, text="Test", command=lambda: classify_window('vot'))
    vot_btt.grid(row=3)
    
    #Last Layer
    lt_title = LabelFrame(last_f, padx=5, pady=2, width=590, text="Other")
    lt_title.pack(expand=TRUE)

    left_gr1 = Frame(lt_title, padx=5, pady=2, width=590, borderwidth=4, relief=SUNKEN)
    left_gr1.grid(row=0, column=0)

    ll1 = Label(left_gr1, padx=5, pady=2, text="Train All")
    ll1.grid(row=0)
    bb1 = Button(left_gr1, padx=5, pady=2, text="Train", command=train)
    bb1.grid(row=1)

    right_gr1 = Frame(lt_title, padx=5, pady=2, width=590, borderwidth=4, relief=SUNKEN)
    right_gr1.grid(row=0, column=1)
    
    ll2 = Label(right_gr1, padx=5, pady=2, text="Comparison All")
    ll2.grid(row=0)
    bb2 = Button(right_gr1, padx=5, pady=2, text="Comparison", command=compared1)
    bb2.grid(row=1)

    main_win.mainloop()
    

if __name__ == "__main__":
    main()
