#Mean featured logistic regression

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import linear_model,svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

scaler = MinMaxScaler()
#import data and normalization
data = genfromtxt('training_log.csv',delimiter=',')
X = data[:,:2]
Y = data[:,-1]
avg = np.mean(X,axis=0)
dev = np.std(X,axis=0)
X_normal = (X-avg)/dev
#X_normal = scaler.fit_transform(X)
test_data = genfromtxt('test_log.csv',delimiter=',')
X_test = test_data[:,:2]
X_test_normal = (X_test-avg)/dev
#X_test_normal = scaler.transform(X_test)
X_test1 = X_test_normal[:,0]
X_test2 = X_test_normal[:,1]
Y2 = test_data[:,-1]
h=0.02 # stepsize in the mesh
X_train = X_normal[:,0]
logreg = linear_model.LogisticRegression(C=1)
svmreg = svm.SVC(kernel='rbf',C=1.8,gamma=7).fit(X_train.reshape(len(X_train),1),Y)
#svmreg = svm.SVC(kernel='rbf', C=0.5,gamma=2)

# we create an instance of Neighbours Classifier and fit the data
#logreg.fit(X_train.reshape(len(X_train),1), Y)
#svmreg.fit(X_train.reshape(len(X_train),1), Y)

# Plot decision boundary. Assign each point in the mesh [x_min,xmax]*[y_min,y_max]
x_min, x_max = X_normal[:, 0].min() - .5, X_normal[:, 0].max() + .5
y_min, y_max = X_normal[:, 1].min() - .5, X_normal[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
Z = svmreg.predict(np.c_[xx.ravel()].reshape(len(np.c_[xx.ravel()]),1))
#Z = svmreg.predict(np.c_[xx.ravel(), yy.ravel()])
# Put result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4,3))
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Paired)

#Predict on test set
Z1 = svmreg.predict(X_normal[:,0].reshape(len(X_normal[:,0]),1))
Z2 = svmreg.predict(X_test_normal[:,0].reshape(len(X_test_normal[:,0]),1))

#Calculate accuracy, precision, recall
train_acc, train_pre, train_rec = accuracy_score(Y, Z1), precision_score(Y, Z1),recall_score(Y, Z1)
test_acc, test_pre, test_rec = accuracy_score(Y2, Z2), precision_score(Y2, Z2), recall_score(Y2, Z2) 
print 'training set: accuracy %10.2f, precision %10.2f, recall %10.2f' % (train_acc,train_pre,train_rec)
print 'test set: accuracy %10.2f, precision %10.2f, recall %10.2f' % (test_acc, test_pre,test_rec)

#Plot training or test set
#plt.scatter(X_test1[Y2 == 1], X_test2[Y2 == 1], c='b', s = 50, marker = '^',label='Unstable')
#plt.scatter(X_test1[Y2 == 0], X_test2[Y2 == 0], c='r', s = 50, marker = 's',label='Stable')
plt.scatter(X_normal[:,0][Y == 1], X_normal[:,1][Y == 1], c='b',s=50, marker = '^',label='Unstable')
plt.scatter(X_normal[:,0][Y == 0], X_normal[:,1][Y == 0], c='r',s=50, marker = 's',label='Stable')
leg = plt.legend(loc = 'upper right',fancybox=True,fontsize='x-small')
leg.get_frame().set_alpha(0.5)
#plt.scatter(X_normal[:,0],X_normal[:,1], c=Y,s=40, edgecolors='k',cmap=plt.cm.Paired)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.xlabel('Normalized Energy')
plt.ylabel('Normalized Strength')
plt.title('Training set')
#plt.savefig('Test_trans.png',bbox_inches='tight',transparent= True)
plt.show()
