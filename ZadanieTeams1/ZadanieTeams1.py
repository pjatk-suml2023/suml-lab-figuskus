# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1V7A6yoCQwY3uS4VzGtyPkxyGE19veddI

# Import bibliotek
"""

import os
import pandas as pd

!pip install tensorflow

from platform import python_version
python_version()

"""# Przygotowanie danych do analizy"""

os.getcwd()

#base_data = pd.read_csv("/content/drive/MyDrive/DSP_4.csv")

os.chdir("/content/drive/MyDrive/SUML/datasets/")

df = pd.read_csv("DSP_4.csv", sep=";")

pd.set_option('display.max_columns',10)
pd.set_option('display.max_colwidth',10)
pd.set_option('display.max_rows',10)

display(df)

print(df)

df.columns

df.isnull().any()

print(df["wiek"].mean())

print(df["wzrost"].mean())

df_2 = df.fillna(df.mean())

df_3 = df.fillna(df.median())
print(df["wiek"].median())
print(df["wzrost"].median())
df_4 = df.fillna(0)
display(df_3)
display(df_4)

display(df_2)

df_2.isnull().any()

"""# Podstawowe statystyki opisowe"""

print(round(df["wiek"].mean(),2))
print(df["wiek"].median())
print(df["wiek"].max())
print(df["wiek"].min())
print(df["wiek"].var())

df["wiek"].max() - df["wiek"].min()

df["wiek"].quantile([.25,.5,.75])

round(df["wiek"].std(),2)

round(df.describe(),2)

df["wiek"].groupby(df["objawy"]).describe()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import pearsonr

df.corr()

sns.heatmap(df.corr())

plt.figure(figsize=(10,5))
heatmap = sns.heatmap(df.corr(),vmin=-1,vmax=1,annot=True)
heatmap.set_title("Analiza korelacji",fontdict={"fontsize":14}, pad=12)

"""# Zad1"""

df = pd.read_csv("DSP_4.csv", sep=";")

pd.set_option('display.max_columns',10)
pd.set_option('display.max_colwidth',10)
pd.set_option('display.max_rows',10)

display(df)

"""# Zad2"""

df_3 = df.fillna(df.median())
print(df["wiek"].median())
print(df["wzrost"].median())
df_4 = df.fillna(0)
display(df_3)
display(df_4)

"""# Zad3"""

df5 = pd.read_csv("DSP_5.csv", sep=";")

df5.isnull().any()

df5 = df5.fillna(df5.mean())

print(round(df5["hp"].mean(),2))
print(df5["hp"].median())
print(df5["hp"].max() - df5["hp"].min())
print(df5["hp"].var())

df5.corr()

"""Związek między "mpg" a "wt" jest bardzo znaczący. Współczynnik korelacji wynosi -0,864418, co oznacza, że istnieje silna, ujemna zależność. Innymi słowy, im mniejszy jest "wt", tym wyższa jest wartość "mpg". Jest to ważna informacja dla osób zainteresowanych oszczędnością paliwa.

# Zad4
"""

df8 = pd.read_csv("DSP_8.csv", sep=",")

pd.set_option('display.max_rows',100)

print(df8)

print("liczba kolumn:", df8.columns.size)
print("liczba wierszy:", len(df8))

# a)
num_columns = df8.shape[1]
column_names = df8.columns.tolist()

# b)

# c)
missing_data = df8.isnull().sum()

# d)
mean_age_female = df8[df8['Sex'] == 'Female']['Age'].mean()
std_age_female = df8[df8['Sex'] == 'Female']['Age'].std()

mean_age_male = df8[df8['Sex'] == 'Male']['Age'].mean()
std_age_male = df8[df8['Sex'] == 'Male']['Age'].std()

# e)
percentage_male = (df8['Sex'] == 'Male').mean() * 100

# f)
females_age_45_to_50 = ((df8['Sex'] == 'Female') & (df8['Age'] >= 45) & (df8['Age'] <= 50)).sum()

# g)
subset_data = df8[df8['RestingECG'] == 'Normal']
correlation_matrix = subset_data.corr()

# Printy

print("a) Liczba kolumn (wraz z ich nazwami):")
print(f"   Liczba kolumn: {num_columns}")
print(f"   Nazwy kolumn: {', '.join(column_names)}\n")

print("b) Liczba wierszy (obserwacji):")
print(f"   Liczba wierszy: {num_rows}\n")

print("c) Ewentualne braki danych:")
print(missing_data)
print("\n")

print("d) Średni wiek i odchylenie standardowe w grupie kobiet i mężczyzn:")
print(f"   Średni wiek kobiet: {mean_age_female:.2f}, Odchylenie standardowe kobiet: {std_age_female:.2f}")
print(f"   Średni wiek mężczyzn: {mean_age_male:.2f}, Odchylenie standardowe mężczyzn: {std_age_male:.2f}\n")

print("e) Odsetek mężczyzn w zbiorze danych:")
print(f"   Odsetek mężczyzn: {percentage_male:.2f}%\n")

print("f) Liczba kobiet w wieku od 45 do 50 lat:")
print(f"   Liczba kobiet w wieku 45-50 lat: {females_age_45_to_50}\n")

print("g) Korelacje pomiędzy zmiennymi dla osób z EKG w czasie spoczynku w normie:")
print(correlation_matrix)