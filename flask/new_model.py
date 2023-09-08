import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('diabetes.csv')
dataset_x = dataset.iloc[:,[1,4,5,7]].values
dataset_y = dataset.iloc[:,8].values
sc = MinMaxScaler(feature_range=(0,1))
dataset_scaled = sc.fit_transform(dataset_x)
dataset_scaled = pd.DataFrame(dataset_scaled)
x = dataset_scaled
y = dataset_y
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42,stratify=dataset['outcome'])
from sklearn.svm import SVC
svc_ = SVC(kernel='linear',random_state=42)
svc_.fit(x_train,y_train)