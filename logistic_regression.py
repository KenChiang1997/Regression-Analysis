import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 

class LogisticRegression():
    def __init__(self,x,y):   ## 定義我class 裡的parameters
        self.x=x
        self.y=y
        self.l=0.005
        w=np.zeros(x.shape[1])
        self.w=w.reshape(-1,1)
    def _sigmoid(self,z):     ## logit function 
        return 1/(1 + np.exp(-z))
    def gradient_mse(self):   ## 我這邊先固定learning_rate=0.005,找出在每一次iteration下每一次的mse,當然我已經先找過這筆dataset才知道要設在1350~1500之間,這樣跑起來比較快><
        mse=[]
        for k in range(1350,1500):
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
    def get_best_epochs(self): ##跟之前功課模擬的一樣找出最小mse的那個點並當作我的iteration 次數
        min_mse=float(np.min(self.mse))
        for i in range( len(self.mse) ):
            if(self.mse[i]==min_mse):
                min_term=i-1
        best_epochs=1350+min_term
        self.best_epochs=best_epochs
        return self.best_epochs
    def gradient(self):         ##梯度下降iteration 演算法
        for _ in range(1,self.best_epochs,1):
            y_pred=self._sigmoid(np.dot(self.x,self.w))
            y_loss=y_pred-self.y
            self.w=self.w-self.l*(np.dot(self.x.T,y_loss))*(1/self.x.shape[0])
        return self.w
    def predict(self,x_data):   ##用梯度下降得到的權重w去做prediction
        y_pred=self._sigmoid(np.dot(x_data,self.w))
        return y_pred
#-----------------------------clean_data----------------------------------------------------#
#主要是將data換成我notation上的樣子
data=pd.read_excel(r"/Users/chen-lichiang/Desktop/2020HW/財務計量經濟學/計量期末報告/xilathon.xlsx",index_col="dose")
print(data)
x=np.array(data.index)
y=np.array(data["death"].values)
x=x.reshape(-1,1)
x=sm.add_constant(x)  ##用statsmodel.sm 的fnc 在x_dataset裡加入一排 (1*n)的vector
y=y.reshape(-1,1)
w=np.zeros(x.shape[1])
w=w.reshape(-1,1)
#------------------------build model-------------------------------------------------------#
#用我class裡的fnc
model=LogisticRegression(x,y)
min_mse=model.gradient_mse()
best_term=model.get_best_epochs()
w=model.gradient()
prediction=model.predict(x)
print(w)


