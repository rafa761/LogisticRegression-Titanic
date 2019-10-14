import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

titanic_data = pd.read_csv("titanic.csv")
titanic_data.head(10)

# Number of passengers in Original titanic_data
print("# of passengers in original data: " + str(len(titanic_data.index)))

# Analyzing Data
sns.countplot(x="Survived", data=titanic_data)

sns.countplot(x="Survived", hue="Sex", data=titanic_data)

sns.countplot(x="Survived", hue="Pclass", data=titanic_data)

titanic_data["Age"].plot.hist()

titanic_data["Fare"].plot.hist(bins=20, figsize=(10,5))

titanic_data.info()

sns.countplot(x="SibSp", data=titanic_data)

# Data Wrangling
# Verify if there is any null value in DataSet
titanic_data.isnull()
# Print the number of invalid values
titanic_data.isnull().sum()

sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap="viridis")

sns.boxplot(x="Pclass", y="Age", data=titanic_data)

titanic_data.head(5)

titanic_data.drop("Cabin", axis=1, inplace=True)
titanic_data.head(5)

# Drop all "Not a Number"
titanic_data.dropna(inplace=True)

sns.heatmap(titanic_data.isnull(), yticklabels=False, cbar=False, cmap="viridis")

# Now the DataSet does not contain any null values
titanic_data.isnull().sum()

titanic_data.head(2)

sex = pd.get_dummies(titanic_data["Sex"], drop_first=True)
sex.head()

embark = pd.get_dummies(titanic_data["Embarked"], drop_first=True)
embark.head()

Pcl = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
Pcl.head()

# Concatenate this new categorical information on the DataSet
titanic_data = pd.concat([titanic_data, sex, embark, Pcl], axis=1)
titanic_data.head()

# Drop the uncessary columns
titanic_data.drop(["PassengerId", "Pclass", "Name", "Sex", "Ticket", "Embarked"], axis=1, inplace=True)
titanic_data.head()

# Train Data
X = titanic_data.drop("Survived", axis=1)
y = titanic_data["Survived"]

# sklearn.cross_validation has been deprecated. The function train_test_split can now be found here: from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)

# Acuracy Check
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
