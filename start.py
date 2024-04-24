# Zaczynamy od załadowania bibliotek. Te najpopularniejsze to
# pandas - do pracy z danymi
# matplotlib - do rysowania wykresow
# sklearn - zawierający gotowe funkcje modelujące dane
import matplotlib.pyplot as plt
# %matplotlib inline -> plt.show()
import inline
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# tutaj ładujemy dane do obiektu data frame z biblioteki pandas
# plik CSV nie posiada nagłówka dlatego header=None
# kolumnom nadajemy nazwy korzystając z parametru names
# W skryptach ML dane trzeba skądś pobrać, stad znajomość polecenia
# read_csv jest super przydatna

iris = pd.read_csv(r"/home/tom/PycharmProjects/data/iris.data",
                   header=None,
                   names=['petal length', 'petal width',
                          'sepal length', 'sepal width', 'species'])