import pandas as pd
import sklearn
import sklearn.linear_model


size=[5,10,12,14,18,30,33,55,65,80,100,150]
distance=[50,20,70,100,200,150,30,50,70,35,40,20]
price=[300,400,450,800,1200,1400,2000,2500,2800,3000,3500,9000]

series_dict={'X1':size,'X2':distance,'y':price}
df=pd.DataFrame(series_dict)

train = df.iloc[0:10,:]
X = train[['X1','X2']]
y = train[['y']]


#test = df.iloc[10:,:]
#X_test = test[['X1','X2']]
#y_test = test[['y']]


#X_test = df.iloc[10:,:]
#y_test = df.iloc[0:10,:]

regr=sklearn.linear_model.LinearRegression()
regr.fit(X, y)
regr.predict(X)


regr.coef_
