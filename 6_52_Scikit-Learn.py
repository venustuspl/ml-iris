import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
        'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = pd.read_csv(r"/home/tom/PycharmProjects/data/housing.data",
                   sep=' +', engine='python', header=None,
                   names=cols)

X = data.drop('MEDV', axis=1)
y = data['MEDV'].values

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X1_train = X_train
    X1_test = X_test

    X2_train = X_train.loc[:, ['LSTAT', 'RM', 'PTRATIO', 'INDUS', 'TAX', 'NOX']]
    X2_test = X_test.loc[:, ['LSTAT', 'RM', 'PTRATIO', 'INDUS', 'TAX', 'NOX']]

    lr1 = LinearRegression(normalize=True)
    lr1.fit(X1_train, y_train)
    score_all_columns = lr1.score(X1_test, y_test)

    lr2 = LinearRegression()
    lr2.fit(X2_train, y_train)
    score_selected_columns = lr2.score(X2_test, y_test)

    print(score_all_columns, score_selected_columns)
