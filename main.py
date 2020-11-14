
import pandas as pd

from sklearn import datasets
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.cluster import MeanShift 
# import seaborn as sns
import matplotlib.pyplot as plt 



colleges = pd.read_csv('College_data.csv',index_col = 0)
print(colleges.info())

x = colleges[["Apps","Grad.Rate"]].values  

print(x)

ms = MeanShift() 
y_hc = ms.fit_predict(x) 

# print(y_hc)

plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.title('University Data')
plt.xlabel('Applications')
plt.ylabel('Graduate Rates')
plt.legend()
plt.show()

#  iris = datasets.load_iris()
# print(iris.info())

# columns = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width','Petal Length', 'Petal Width']) 
# xx = columns[["Sepal Length","Sepal Width"]]
# x = columns[["Sepal Length","Sepal Width"]].values  
                                 
# print(columns)

# plt.scatter(xx["Sepal Length"],xx["Sepal Width"])
# plt.show()

#  ms = MeanShift() 
# y_hc = ms.fit_predict(x) 


# plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
# plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
# plt.title('Iris Dataset')
# plt.xlabel('Sepal Length')
# plt.ylabel('Sepal Width')
# plt.legend()
# plt.show()


