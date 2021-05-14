import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

clf=[
     make_pipeline(StandardScaler(), SVC(gamma='auto')),
     make_pipeline(StandardScaler(), SVC(gamma='auto')),
     make_pipeline(StandardScaler(), SVC(gamma='auto')),
     make_pipeline(StandardScaler(), SVC(gamma='auto')),
     make_pipeline(StandardScaler(), SVC(gamma='auto')),
     make_pipeline(StandardScaler(), SVC(gamma='auto')),
     make_pipeline(StandardScaler(), SVC(gamma='auto')),
     make_pipeline(StandardScaler(), SVC(gamma='auto')),
     make_pipeline(StandardScaler(), SVC(gamma='auto')),
     make_pipeline(StandardScaler(), SVC(gamma='auto')),
     make_pipeline(StandardScaler(), SVC(gamma='auto')),
     make_pipeline(StandardScaler(), SVC(gamma='auto')),
     make_pipeline(StandardScaler(), SVC(gamma='auto'))]

x=[[],[],[],[],[],[],[],[],[],[],[],[],[]]
y=[5, 1, 0, 5, 16, 17, 3, 9, 17, 3, 2, 1, 5, 0, 24, 1, 24, 15, 13, 1, 2, 13, 5, 24, 1, 13, 3, 15, 3, 2, 17, 3, 19, 13, 15, 1, 22, 0, 5, 0, 4, 1, 24, 14, 14, 16, 13, 5, 15, 16, 0, 18, 4, 15, 13, 9, 14, 3, 19, 5, 13, 17, 1, 14, 11, 24, 19, 5, 24, 3, 1, 9, 0, 14, 1, 24, 2, 2, 5, 5, 1, 0, 2, 18, 16, 17, 13, 3, 2, 14, 24, 22, 13, 4, 3, 18, 24, 4, 5, 2, 18, 19, 5, 16, 2, 22, 24, 4, 9, 15, 4, 9, 2, 3, 14, 19, 3, 3, 0, 5, 2, 3, 19, 2, 3, 24, 22, 11, 15, 19, 9, 9, 5, 1, 14, 0, 24, 24, 2, 9, 13, 15, 11, 0, 15, 16, 5, 2, 22, 16, 18, 13, 13, 15, 24, 5, 3, 19, 13, 19, 5, 4, 5, 17, 15, 13, 11, 15, 14, 2, 16, 19, 22, 24, 1, 1, 14, 16, 3, 1, 1, 9, 2, 24, 3, 2, 13, 4, 15, 22, 11, 9, 24, 2, 17, 13, 13, 22, 18, 1, 3, 3, 3, 11, 15, 15, 17, 4, 2, 13, 13, 1, 15, 18, 15, 11, 1, 15, 1, 4, 22, 11, 24, 5, 24, 15, 18, 17, 22, 18, 11, 16, 11, 15, 18, 24, 17, 5, 13, 2]

f=open('x.txt','rb')
while True:
    try:
        a=pickle.load(f)
        for i in range(13):
            x[i].append(a[i].detach().numpy())
    except EOFError as e:
        break

accuracy=[]

print(np.array(y[:200]).shape)

for i in range(len(clf)):
  clf[i].fit(np.array(x[i][:200]), np.array(y[:200]))
  y_pred=clf[i].predict(np.array(x[i][200:]))
  accuracy.append(len(np.where((y_pred-np.array(y[200:]))==0)[0])/len(y[200:]))

plt.plot(range(len(accuracy)),accuracy)
plt.show()