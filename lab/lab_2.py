# Thành viên:
#   Phan Thế Giang 17021236 
#   Nguyễn Huy Hoàng 17020052
#   Phan Quang Hưng 17021270

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

def visualize_iris_data(iris,x_index=0,y_index=1):
  formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
  plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
  plt.colorbar(ticks=[0, 1, 2], format=formatter)
  plt.xlabel(iris.feature_names[x_index])
  plt.ylabel(iris.feature_names[y_index])
  plt.tight_layout()
  plt.show()

def train(x,y):
  clf = RandomForestClassifier(max_depth=2, bootstrap=False, criterion='entropy')
  clf.fit(x,y)
  return clf

def test(clf,x,y_true):
  y_predict = clf.predict(x)
  return accuracy_score(y_true,y_predict)

def run_model(x,y):
  loo = LeaveOneOut()
  score = 0
  for train_index, test_index in loo.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = train(x_train,y_train)
    score += test(clf,x_test,y_test)
  print(f'accuracy: {score/x.shape[0]}')

if __name__ == "__main__":
     
  iris = load_iris()
  visualize_iris_data(iris)

  x = iris.data
  y = iris.target
  run_model(x,y)




