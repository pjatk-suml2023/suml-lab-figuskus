# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OO0OS3KK7z1dvCV3UOzbun8j_kQGwUNa

# Zadanie 1
"""

import pandas as pd

data = pd.read_csv("DSP_6.csv")

missing_data = data.isnull().sum()
print(missing_data)

"""# Zadanie 2"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("DSP_6.csv")

plt.figure(figsize=(8, 6))
data['FamilyMembers'].value_counts().plot(kind='bar')
plt.title('Liczba członków rodziny na pokładzie')
plt.xlabel('Liczba członków rodziny')
plt.ylabel('Liczba pasażerów')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(data['Fare'], bins=20, color='skyblue')
plt.title('Rozkład opłat')
plt.xlabel('Opłata')
plt.ylabel('Liczba pasażerów')
plt.show()

"""# Zadanie 3"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("DSP_6.csv")

X = data.drop(columns='target_column')
y = data['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

"""# Zadanie 4"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("DSP_2.csv")

X = data.drop(columns='HeartDisease')
y = data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)