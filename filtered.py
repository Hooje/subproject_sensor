
import pickle
import numpy as np 
import pandas as pd
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

with open('combined.pickle', 'rb') as f:
    Xy = pickle.load(f)
x = Xy[:,:-1]
y = Xy[:,-1]

index = np.arange(len(x[0]))
flag = 0

for i in range(len(x)):
    sample = x[i].reshape(-1,1)
    
    time = (index+1).reshape(-1,1)
    id_array = np.array([i]*100).reshape(-1,1)
    tmp_x = np.concatenate((id_array, time), axis = 1)
    #print(new_x.shape)
    #print(x.shape)
    tmp_x = np.concatenate((tmp_x, sample), axis = 1)
    if flag == 0:
        new_x = tmp_x
        flag = 1
    else:
        new_x = np.concatenate((new_x, tmp_x), axis = 0)
    
    #print(new_x)
    #input()
data_df = pd.DataFrame(new_x)
print(data_df)
features = extract_features(data_df, column_id=0, column_sort=1)


impute(features)

filtered_features = select_features(features, y)
#filtered_features

input(filtered_features.columns)

X_feature_train, X_feature_test, y_train, y_test = train_test_split(features, y, test_size=0.5)
X_filtered_train, X_filtered_test = X_feature_train[filtered_features.columns], X_feature_test[filtered_features.columns]

classifier_feature = RandomForestClassifier()
classifier_feature.fit(X_feature_train, y_train)
yt =  classifier_feature.predict(X_feature_test)
print(y_test)
print(yt)
print(classification_report(y_test, classifier_feature.predict(X_feature_test)))

classifier_filtered = RandomForestClassifier()
classifier_filtered.fit(X_filtered_train, y_train)
yt = classifier_filtered.predict(X_filtered_test)
print(y_test)
print(yt)
print(classification_report(y_test, classifier_filtered.predict(X_filtered_test)))