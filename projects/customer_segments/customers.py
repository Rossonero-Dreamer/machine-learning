import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#global config
np.random.seed(22)
#matplotlib.style.use('ggplot')

#load the dataset with 6 features
#Fresh
#Milk
#Grocery
#Frozen
#Detergent
#Delicatessen
data = pd.read_csv("customers.csv")
data.drop(['Channel','Region'], axis=1, inplace=True)

#describe data
print("Describe Raw Data")
print(data.describe())
print(data.corr())
#print(data.info())

#obtain some samples for further analysis
indices = [20,40,80]
samples = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop=True)
#print(samples)

#study relevance of a specific feature
feature = 'Grocery'
new_data = data.drop([feature],axis=1)
X_train, X_test, y_train, y_test = train_test_split(new_data, data[feature], test_size=0.25,
                                                    random_state=22)
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
score = regressor.score(X_test, y_test)
print('score for {}:{}'.format(feature,score))

#visualize all features
#pd.plotting.scatter_matrix(data, alpha=0.3, figsize=(14,8), diagonal="kde")
#plt.show()

#data preprocessing to handle skewness
log_data = data.apply(np.log)
sample_data = samples.apply(np.log)
print("Describe Log Data")
print(log_data.describe())

#remove outliers
features = log_data.keys()
for feature in features:
    q1 = np.percentile(log_data[feature],25)
    q3 = np.percentile(log_data[feature],75)
    step = (q3-q1)*1.5
    print("Display outliers for {}".format(feature))
    outliers = log_data[~((log_data[feature]>=q1-step)&(log_data[feature]<=q3+step))]
    print(outliers)

outliers = [38,57,65,66,75,81,86,95,96,98,109,128,137,142,145,154,161,171,175,183,184,187,193,203,218,233,264,285,289,304,305,325,338,343,353,355,356,357,412,420,429,429]
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop=True)
print(good_data.head())


