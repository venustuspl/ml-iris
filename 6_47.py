import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
        'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = pd.read_csv(r"/home/tom/PycharmProjects/data/housing.data",
                   sep=' +', engine='python', header=None,
                   names=cols)

sns.boxplot(x=data['CHAS'], y=data['MEDV'])
sns.barplot(x=data['CHAS'], y=data['MEDV'])

sns.barplot(x=data['CRIM'], y=data['MEDV'])

corr_matrix = np.corrcoef(data.values.T)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data=corr_matrix,
            annot=True,
            square=True,
            fmt='.2f',
            xticklabels=cols,
            yticklabels=cols)

sns.pairplot(data[cols])
plt.show()

cols = ['LSTAT', 'RM', 'PTRATIO', 'INDUS', 'TAX', 'NOX', 'MEDV']

fig, ax = plt.subplots(figsize=(12, 12))
sns.pairplot(data[cols])
plt.show()