import math
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

fx=open('x.txt','rb')
fy=open('y.txt','rb')
while True:
    try:
        a=pickle.load(fx)
        for i in range(13):
            x[i].append(a[i].detach().numpy())
    except EOFError as e:
        break
y=pickle.load(fy)

split=0.2
split=math.floor(len(y)*split)
accuracy=[]

for i in range(len(clf)):
  clf[i].fit(np.array(x[i][:split]), np.array(y[:split]))
  y_pred=clf[i].predict(np.array(x[i][split:]))
  accuracy.append(len(np.where((y_pred-np.array(y[split:]))==0)[0])/len(y[split:]))

plt.plot(range(len(accuracy)),accuracy)
plt.show()