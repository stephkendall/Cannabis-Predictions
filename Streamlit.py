import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve
from sklearn.metrics import auc, classification_report, confusion_matrix
from sklearn import tree, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.externals.six import StringIO 
# import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA  
from sklearn.linear_model import LogisticRegression

@st.cache
def loadData():
    df = pd.read_csv('/Users/stephaniekendall/Desktop/Errthang/Flatiron/projects/Cannabis-Predictions/CSV Files/final.csv')
    # drop secondary index column along with ratings
    return df

# Basic preprocessing required for all the models.  
def preprocessing(df):
    # Assign X and y
    
    #Train Test Split on imbalanced classes
    features=df.drop(columns=['name','type'])
    trainn=df.drop(columns=['name'])
    target=df.type
    
    # handle class imbalance with SMOTE

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(features, target)
    

    # Train Test Split

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25,random_state=42)
    trainx=X_train[selectedfeatures]
    testx=X_test[selectedfeatures]
    return X_train, X_test, y_train, y_test



# Training Decission Tree for Classification
@st.cache(suppress_st_warning=True)
def decisionTree(X_train, X_test, y_train, y_test):
    # Train the model
    tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)

    return score, report, tree


# Training Neural Network for Classification.
@st.cache(suppress_st_warning=True)
def neuralNet(X_train, X_test, y_train, y_test):
    # Scalling the data before feeding it to the Neural Network.
    scaler = StandardScaler()  
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)
    # Instantiate the Classifier and fit the model.
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score1 = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    
    return score1, report, clf


# Training KNN Classifier
@st.cache(suppress_st_warning=True)
def Knn_Classifier(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)

    return score, report, clf


# Accepting user data for predicting its Member Type
def accept_user_data():
    thc = st.text_input("Enter the THC Content: ")
    cbd = st.text_input("Enter the CBD Content: ")
    taste = st.text_input("Enter the Taste Profile: ")
    user_prediction_data = np.array([thc,cbd,taste]).reshape(1,-1)

    return user_prediction_data



def main():
    st.title("Prediction of Cannabis Class")
    data = loadData()
    X_train, X_test, y_train, y_test = preprocessing(data)

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
        st.subheader("Showing raw data---->>>")
        st.write(data.head())


    # ML Section
    choose_model = st.sidebar.selectbox("Choose the ML Model",
        ["NONE","Decision Tree", "Neural Network", "K-Nearest Neighbours"])

    if(choose_model == "Decision Tree"):
        score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
        st.text("Accuracy of Decision Tree model is: ")
        st.write(score,"%")
        st.text("Report of Decision Tree model is: ")
        st.write(report)

        try:
                if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
                    user_prediction_data = accept_user_data() 
                    pred = tree.predict(user_prediction_data)
                    st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 
        except:
                pass

    elif(choose_model == "Neural Network"):
        score, report, clf = neuralNet(X_train, X_test, y_train, y_test)
        st.text("Accuracy of Neural Network model is: ")
        st.write(score,"%")
        st.text("Report of Neural Network model is: ")
        st.write(report)

        try:
            if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
                user_prediction_data = accept_user_data()
                scaler = StandardScaler()  
                scaler.fit(X_train)  
                user_prediction_data = scaler.transform(user_prediction_data)
                pred = clf.predict(user_prediction_data)
                st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 
        except:
            pass

    elif(choose_model == "K-Nearest Neighbours"):
        score, report, clf = Knn_Classifier(X_train, X_test, y_train, y_test)
        st.text("Accuracy of K-Nearest Neighbour model is: ")
        st.write(score,"%")
        st.text("Report of K-Nearest Neighbour model is: ")
        st.write(report)

        try:
            if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
                user_prediction_data = accept_user_data() 
                pred = clf.predict(user_prediction_data)
                st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 
        except:
            pass
    
if __name__ == "__main__":
    main() 