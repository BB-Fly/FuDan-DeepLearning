import sklearn.linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import linear
import lasso
import ridge

from pre_work import *

def test():
    x_train, y_train = load_trainset()
    x_test, y_test = load_testset()

    plt.figure(figsize=(12, 6))                      
    plt.plot(y_test,label='True')

    his_line = sklearn.linear_model.LinearRegression()
    his_ridge = sklearn.linear_model.Ridge()
    his_lasso = sklearn.linear_model.Lasso()

    my_line = linear.Linear()
    my_ridge = ridge.Ridge()
    my_lasso = lasso.Lasso()

    his_line.fit(x_train, y_train)
    his_ridge.fit(x_train, y_train)
    his_lasso.fit(x_train, y_train)
    my_line.fit(x_train, y_train)
    my_ridge.fit(x_train, y_train)
    my_lasso.fit(x_train, y_train)

    y1= his_line.predict(x_test)
    plt.plot(y1 ,label='Line')

    y2 = his_ridge.predict(x_test)
    plt.plot(y2,label='Ridge')

    y3 = his_lasso.predict(x_test)
    plt.plot(y3,label='Lasso')

    y4= my_line.predict(x_test)
    plt.plot(y4 ,label='my_Line')

    y5 = my_ridge.predict(x_test)
    plt.plot(y5,label='my_Ridge')

    y6 = my_lasso.predict(x_test)
    plt.plot(y6,label='my_Lasso')




    plt.legend()

    plt.show() 

    s1=r2_score(y_test,y1)
    s2=r2_score(y_test,y2)
    s3=r2_score(y_test,y3)
    s4=r2_score(y_test,y4)
    s5=r2_score(y_test,y5)
    s6=r2_score(y_test,y6)

    print('model score:\nline:{}\nridge:{}\nlasso:{}\nmy_line:{}\nmy_ridge:{}\nmy_lasso:{}\n'.format(s1,s2,s3,s4,s5,s6))


if __name__ == '__main__':
    test()