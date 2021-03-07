import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
##一樣用之前做好的class
class LogisticRegression():
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.l=0.005
        w=np.zeros(x.shape[1])
        self.w=w.reshape(-1,1)
    def _sigmoid(self,z):
        return 1/(1 + np.exp(-z))
    def gradient_mse(self):
        mse=[]
        self.theta=[]
        self.j_history=[]
        self.j_omg_history=[]
        for k in range(1350,1450):
            for _ in range(1,k,1):
                y_pred=self._sigmoid(np.dot(self.x,self.w))
                y_loss=y_pred-y
                self.w=self.w-self.l*(np.dot(self.x.T,y_loss))*(1/self.x.shape[0])
                self.theta.append(float(self.w[1]))
                a=np.dot(self.y.T,np.log(self._sigmoid(np.dot(self.x,self.w))))    
                b=np.dot((1-self.y).T,np.log(1-self._sigmoid(np.dot(self.x,self.w)))) 
                self.j_history.append((-1/self.x.shape[0])* (float(a)+float(b)) )
                c=np.sum(np.square(y_loss))
                self.j_omg_history.append((1/self.x.shape[0])*c)
            pred=self._sigmoid(np.dot(self.x,self.w))
            sse=np.square(pred-self.y)
            ok=np.mean(sse)
            mse.append(ok)
        self.mse=mse
        return self.mse ,self.theta, self.j_history ,self.j_omg_history
    def get_best_epochs(self):
        min_mse=float(np.min(self.mse))
        for i in range(len(self.mse)):
            if(self.mse[i]==min_mse):
                min_term=i-1
        best_epochs=1350+min_term
        self.best_epochs=best_epochs
        return self.best_epochs
    def gradient(self):
        for _ in range(1,self.best_epochs,1):
            y_pred=self._sigmoid(np.dot(self.x,self.w))
            y_loss=y_pred-self.y
            self.w=self.w-self.l*(np.dot(self.x.T,y_loss))*(1/self.x.shape[0])
        return self.w
    def predict(self,x_data):
        y_pred=self._sigmoid(np.dot(x_data,self.w))
        return y_pred
#-----------------------------clean_data----------------------------------------------------#
data=pd.read_excel(r"/Users/chen-lichiang/Desktop/2020HW/財務計量經濟學/計量期末報告/xilathon.xlsx",index_col="dose")
x=np.array(data.index)
y=np.array(data["death"].values)
x=x.reshape(-1,1)
x=sm.add_constant(x)
y=y.reshape(-1,1)
w=np.zeros(x.shape[1])
w=w.reshape(-1,1)
#------------------------build model-------------------------------------------------------#
model=LogisticRegression(x,y)
mse_all=model.gradient_mse()
mse=mse_all[0]
theta=mse_all[1]
j_history=mse_all[2]
j_omg=mse_all[3]
best_term=model.get_best_epochs()
w=model.gradient()
prediction=model.predict(x)
print(w)
#-----------------------plot_data1---------------------------------------------------------#
##這一個主要畫的是邊界決策線
plt.figure()
plt.scatter(data.index,data["death"])
plt.title("data")
plt.xlabel("Xilation_dose")
plt.ylabel("death_numbers")
plt.plot(data.index,prediction,label="prediction_line",color="r")
plt.plot(data.index,np.dot(x,w),label="boundary_line",color="green",linestyle="--",alpha=0.5)
plt.ylim(0,1)
plt.legend()
#-----------------------plot_data2---------------------------------------------------------#
# 主要是在畫cost function調整前,調整後的差別,目前為為after log adjust,所以是達到最佳解的狀態
x=theta
y=j_history
y_min=min(y)
for i in range(len(y)):
    if(y[i]==y_min):
        term=i
min_theta=x[term]
plt.figure()
plt.title("convex")
plt.plot(x,y,label="after log adjust")
plt.ylabel("J(theta)")
plt.xlabel("theta")
plt.hlines(y=y_min,xmin=0,xmax=min_theta,color="red",linestyle="--")
plt.axvline(x=min_theta,ymin=0,ymax=0.06,color="red",linestyle="--")
plt.scatter(min_theta,y_min)
plt.grid()
plt.legend()
plt.show()
