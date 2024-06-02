# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "~/Desktop/ML-for-beginners/data/raw/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)


## SUMMARIZING DATA

# Number of rows (instances) + columns (attributes)
print(dataset.shape)

# First 20 rows of data
print(dataset.head(20))

# Statistical summary of each column
print(dataset.describe())

# Class distribution of number of rows per class
print(dataset.groupby('class').size())


## DATA VISUALIZATION

# Plot: box and whiskers
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# Plot: histogram
dataset.hist()
plt.show()

# Plot: scatterplots
scatter_matrix(dataset)
plt.show() # high correlation + predictable data


## EVALUATION OF DATA

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

"""
Training data now in X_train and Y_train for preparing models
and X_validation and Y_validation sets to use later.
"""