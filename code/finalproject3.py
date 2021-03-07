import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import statsmodels.api as sm 
from sklearn.model_selection import train_test_split

##一樣用之前做好的class
class LogisticRegression():
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.l=0.001
        w=np.zeros(x.shape[1])
        self.w=w.reshape(-1,1)
    def _sigmoid(self,z):
        return 1/(1 + np.exp(-z))
    def gradient_mse(self):
        mse=[]
        for k in range(2500,2700):
            for _ in range(1,k,1):
                y_pred=self._sigmoid(np.dot(self.x,self.w))
                y_loss=y_pred-y
                self.w=self.w-self.l*(np.dot(self.x.T,y_loss))*(1/self.x.shape[0])
            pred=self._sigmoid(np.dot(self.x,self.w))
            sse=np.square(pred-self.y)
            ok=np.mean(sse)
            mse.append(ok)
        self.mse=mse
        return self.mse
    def get_best_epochs(self):
        min_mse=float(np.min(self.mse))
        for i in range( len(self.mse) ) :
            if(self.mse[i]==min_mse):
                min_term=i-1
        best_epochs=2500+min_term
        self.best_epochs=best_epochs
        return self.best_epochs
    def gradient(self):
        for _ in range(self.best_epochs):
            y_pred=self._sigmoid(np.dot(self.x,self.w))
            y_loss=y_pred-self.y
            self.w=self.w-self.l*(np.dot(self.x.T,y_loss))*(1/self.x.shape[0])
        return self.w
    def predict(self,x_data):
        y_pred=self._sigmoid(np.dot(x_data,self.w))
        return y_pred
#-----------------------------clean_data----------------------------------------------------#
labeled=pd.read_csv(r"/Users/chen-lichiang/Desktop/2020HW/財務計量經濟學/計量期末報告/train.csv")
##清理掉資料有缺陷的地方
##分一下資拆成train, validation
labeled = labeled[~labeled["Age"].isna()]
train, validation = train_test_split(labeled, test_size=0.3, random_state=123)
x = train.loc[:, ["Fare", "Age"]].values
x=sm.add_constant(x)
y = train.loc[:, "Survived"].values
y=y.reshape(-1,1)
#------------------------build model-------------------------------------------------------#
model=LogisticRegression(x,y)
min_mse=model.gradient_mse()
best_term=model.get_best_epochs()
w=model.gradient()
prediction=model.predict(x)
w=w.reshape(x.shape[1])
print(w)
#------------------------plot_data-------------------------------------------------------#
def sigmoid(z):
  return 1/(1 + np.exp(-z))
data=pd.read_csv(r"/Users/chen-lichiang/Desktop/2020HW/財務計量經濟學/計量期末報告/train.csv")
data = data.dropna()
survived=data[data["Survived"]==1]
dead=data[data["Survived"]==0]
x1=data["Fare"].values
x2=data["Age"].values
x1_min=np.min(x1)
x1_max=np.max(x1)
x2_min=np.min(x2)
x2_max=np.max(x2)
x1=np.linspace(x1_min-5,x1_max+5,1000)
x2=np.linspace(x2_min-5,x2_max+5,1000)
##用msehgrid 將x1,x2轉成2d的型態
xx1,xx2=np.meshgrid(x1,x2)
ones=np.ones(xx1.shape)
x_grid = np.concatenate([ones.reshape(-1,1), xx1.reshape(-1, 1), xx2.reshape(-1, 1)], axis=1)
y_pred=np.dot(x_grid,w)
##gradient decent 所跑出來的dataset做預測
y_pred_sigmoid=sigmoid(y_pred)
y_pred_sigmoid=y_pred_sigmoid.reshape(xx1.shape)
##做預測為會發生的地方設為1,沒有的地方設為0
zz=np.where(y_pred_sigmoid>=0.5,1,0)
cmap = cm.get_cmap("Spectral")

plt.figure()
plt.title("example: titanic")
plt.scatter(survived["Fare"], survived["Age"], label="Survived", marker="o", color="b")
plt.scatter(dead["Fare"], dead["Age"], label="Dead", marker="x", color="r")
##用contourf 在二維型態上畫出邊界條件
plt.contourf(xx1,xx2,zz, alpha=0.6,cmap=cmap)
plt.xlabel("Fare")
plt.ylabel("Age")
plt.legend()
plt.grid()
plt.show()
