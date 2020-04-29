""""this is the ml project to classify the breast cancer type if it is melignent or benile using 
our ml classifier model vector machine in this project we have taken the data provided by the sklearn
lib consisting of the 30 features and on the basis of these 30 feature we are classifying that a
tumer is melignent of benile  here the target value 0 means meligenant and 1 means begign"""

#importing the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing the data from the sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

#making the dataframe from the dataset imported 
df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns = np.append(cancer['feature_names'],['target_names']))

#visualising the data
#we use pairplot to see the relation between the various variables
sns.pairplot(df_cancer,hue = 'target_names',vars= ['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness', 'mean compactness'])
#we use the countplot to count the number of the different types of 
sns.countplot(df_cancer['target_names'])
#used scattered plot to plot the points of the mean radius and mean parimeter in 2-d
sns.scatterplot(x = 'mean radius',y = 'mean perimeter', hue = 'target_names' , data = df_cancer)
#using heatmap to see the corelation between the different features
plt.figure(figsize = (20,10))
sns.heatmap(df_cancer.corr(),annot = True)

#training the model
x = df_cancer.drop(['target_names'],axis = 1)
y = df_cancer['target_names']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)

from sklearn.svm import SVC
from sklearn.metrics import classification_report , confusion_matrix
svc_model  = SVC()
svc_model.fit(X_train,y_train)

#evaluating the model
y_pred = svc_model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)

#improving the model usign the feature scalling and the paramenter of svc 
x_train_min = X_train.min()
rangge = (X_train - x_train_min).max()
X_train_scaled = (X_train - x_train_min)/rangge
sns.scatterplot(x = X_train_scaled['mean radius'],y = X_train_scaled['mean perimeter'])


x_test_min = X_test.min()
rangge = (X_test - x_test_min).max()
X_test_scaled = (X_test - x_test_min)/rangge

svc_model.fit(X_train_scaled,y_train)
y_pred1 = svc_model.predict(X_test_scaled)
cm1 = confusion_matrix(y_test,y_pred1)
print(classification_report(y_test,y_pred1))


#improving the model using best c and gamma values
param_grid = { 'C' : [0.1,1,10,100] , 'gamma'  : [1,0.1,0.01,0.001] , 'kernel' : ['rbf']}
from sklearn.model_selection import GridSearchCV
grid  = GridSearchCV(SVC(),param_grid,refit = True,verbose = 4)
grid.fit(X_train_scaled,y_train)
grid.best_estimator_
grid_predicted = grid.predict(X_test_scaled)
cm3 = confusion_matrix(y_test,grid_predicted)
print(classification_report(y_test,grid_predicted))




