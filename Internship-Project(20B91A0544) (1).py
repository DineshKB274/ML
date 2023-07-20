#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore")

# Set to display all the columns in dataset
pd.set_option("display.max_columns", None)

# Import psql to run queries 
import pandasql as psql


# In[2]:


reports = pd.read_csv(r"C:\Users\Dinesh Kumar\Desktop\Internship\database.csv", header=0)
reports_bk=reports.copy()
reports_bk.head()


# In[3]:


reports.info()


# In[4]:


reports.isnull().sum()


# In[5]:


reports.columns


# In[6]:


#statistical measures about data
reports.describe()


# In[1]:


reports['Crime Solved'].value_counts()


# In[8]:


reports['Crime Solved']=reports['Crime Solved'].replace('Yes','1')
reports['Crime Solved']=reports['Crime Solved'].replace('No','0')
reports['Crime Solved']=reports['Crime Solved'].astype(int)
reports['Perpetrator Age']=reports['Perpetrator Age'].replace(' ','0')
reports['Perpetrator Age']=reports['Perpetrator Age'].astype(int)


# In[9]:


#count target or dependent variable by 0 & 1
reports_count=reports['Crime Solved'].value_counts()
print('Class 0:',reports_count[0])
print('Class 1:',reports_count[1])
print('Proportion:',round(reports_count[0]/reports_count[1],2),':1')
print('Total records:',len(reports))


# In[10]:


reports


# In[11]:


#data visualization using bar graph
reports['Year'].value_counts()
a=reports['Year'].value_counts().keys()
b=reports['Year'].value_counts().tolist()
plt.bar(a,b,color='purple')
plt.title('Distribution Of Crimes',color='red',size=15)
plt.xlabel('YEAR',color='red',size=12)
plt.ylabel('Number Of Incidents',color='red',size=12)
plt.show()


# In[12]:


#data visualization using scatterplot
plt.scatter(a,b,color='purple')
plt.title('Distribution Of Crimes',color='red',size=15)
plt.xlabel('YEAR',color='red',size=12)
plt.ylabel('Number Of Incidents',color='red',size=12)
plt.show()


# In[13]:


reports['Incident'].value_counts()


# In[14]:


#data visualization using histogram
plt.hist(reports['Incident'],color='green')
plt.title('Distribution Of Incidents',color='red',size=15)
plt.show()


# In[15]:


del reports['Record ID']


# In[16]:


#use label encoder to handle categorical data
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
reports['Agency Name']=LE.fit_transform(reports[['Agency Name']])
reports['Agency Code']=LE.fit_transform(reports[['Agency Code']])
reports['Agency Type']=LE.fit_transform(reports[['Agency Type']])
reports['City']=LE.fit_transform(reports[['City']])
reports['State']=LE.fit_transform(reports[['State']])
reports['Month']=LE.fit_transform(reports[['Month']])
reports['Crime Type']=LE.fit_transform(reports[['Crime Type']])
reports['Victim Sex']=LE.fit_transform(reports[['Victim Sex']])
reports['Victim Race']=LE.fit_transform(reports[['Victim Race']])
reports['Victim Ethnicity']=LE.fit_transform(reports[['Victim Ethnicity']])
reports['Perpetrator Sex']=LE.fit_transform(reports[['Perpetrator Sex']])
reports['Perpetrator Race']=LE.fit_transform(reports[['Perpetrator Race']])
reports['Perpetrator Ethnicity']=LE.fit_transform(reports[['Perpetrator Ethnicity']])
reports['Relationship']=LE.fit_transform(reports[['Relationship']])
reports['Record Source']=LE.fit_transform(reports[['Record Source']])
reports['Weapon']=LE.fit_transform(reports[['Weapon']])


# In[17]:


reports


# In[18]:


# Identify the independent and Target (dependent) variables

IndepVar = []
for col in reports.columns:
    if col != 'Crime Solved':
        IndepVar.append(col)

TargetVar = 'Crime Solved'

x = reports[IndepVar]
y = reports[TargetVar]


# In[19]:


x


# In[20]:


y


# In[21]:


#Random oversampling can be implemented using RandomOverSampler class
from imblearn.over_sampling import RandomOverSampler
oversample=RandomOverSampler(sampling_strategy=0.5)
x_over,y_over=oversample.fit_resample(x,y)
print(x_over.shape)
print(y_over.shape)


# In[22]:


reports


# In[23]:


# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, stratify=y, random_state = 42)
x.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[24]:


cols1=['City','Year','Victim Age','Perpetrator Age']


# In[25]:


# Scaling the features by using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
mmscaler = MinMaxScaler(feature_range=(0, 1))

x_train[cols1] = mmscaler.fit_transform(x_train[cols1])
x_train = pd.DataFrame(x_train)

x_test[cols1] = mmscaler.fit_transform(x_test[cols1])
x_test = pd.DataFrame(x_test)


# In[26]:


HTResults=pd.read_csv(r"C:\Users\Dinesh Kumar\Desktop\Internship\HTResults.csv")
HTResults.head()


# In[27]:


# Build the Calssification models and compare the results
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb

# Create objects of classification algorithm with default hyper-parameters
ModelLR = LogisticRegression()
ModelDC = DecisionTreeClassifier()
ModelRF = RandomForestClassifier()
ModelET = ExtraTreesClassifier()
ModelKNN = KNeighborsClassifier(n_neighbors=5)

modelBAG = BaggingClassifier(base_estimator=None, n_estimators=100, max_samples=1.0, max_features=1.0,bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,n_jobs=None, random_state=None, verbose=0)
ModelGB = GradientBoostingClassifier(loss='deviance', learning_rate=0.1,n_estimators=100, subsample=1.0,criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None,random_state=None,max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False,validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
ModelLGB = lgb.LGBMClassifier()
ModelGNB = GaussianNB()
# Evalution matrix for all the algorithms

 
MM = [ModelLR, ModelDC, ModelRF, ModelET,ModelKNN, modelBAG,ModelGB, ModelLGB, ModelGNB]
for models in MM:
            
     #Train the model training dataset
    
    models.fit(x_train, y_train)
    
    # Prediction the model with test dataset
    
    y_pred = models.predict(x_test)
    y_pred_prob = models.predict_proba(x_test)
    
    # Print the model name
    
    print('Model Name: ', models)
    
    # confusion matrix in sklearn

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # actual values

    actual = y_test

    # predicted values

    predicted = y_pred

    # confusion matrix

    matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
    print('Confusion matrix : \n', matrix)

    # outcome values order in sklearn

    tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
    print('Outcome values : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy

    C_Report = classification_report(actual,predicted,labels=[1,0])

    print('Classification report : \n', C_Report)

    # calculating the metrics
    
    sensitivity = round(tp/(tp+fn), 3);
    specificity = round(tn/(tn+fp), 3);
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
    balanced_accuracy = round((sensitivity+specificity)/2, 3);
    
    precision = round(tp/(tp+fp), 3);
    f1Score = round((2*tp/(2*tp + fp + fn)), 3);

    # Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
    # A model with a score of +1 is a perfect model and -1 is a poor model

    from math import sqrt

    mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

    print('Accuracy :', round(accuracy*100, 2),'%')
    print('Precision :', round(precision*100, 2),'%')
    print('Recall :', round(sensitivity*100,2), '%')
    print('F1 Score :', f1Score)
    print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
    print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
    #print('MCC :', MCC)

    # Area under ROC curve 

    from sklearn.metrics import roc_curve, roc_auc_score

    print('roc_auc_score:', round(roc_auc_score(actual, y_pred), 3))
    
    # ROC Curve
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    Model_roc_auc = roc_auc_score(actual, y_pred)
    fpr, tpr, thresholds = roc_curve(actual, models.predict_proba(x_test)[:,1])
    plt.figure()
    #
    plt.plot(fpr, tpr, label= 'Classification Model' % Model_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    print('-----------------------------------------------------------------------------------------------------')
    #----------------------------------------------------------------------------------------------------------
    new_row = {'Model Name' : models,
               'True_Positive': tp,
               'False_Negative': fn, 
               'False_Positive': fp, 
               'True_Negative': tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC': MCC,
               'ROC_AUC_Score':roc_auc_score(actual, y_pred),
               'Balanced Accuracy':balanced_accuracy}
    HTResults = HTResults.append(new_row, ignore_index=True)


# In[28]:


HTResults.head(10)

