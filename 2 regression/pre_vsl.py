from pre_work import load_dataset
import matplotlib.pyplot as plt

def pre_vsl():
    titles = ['CRIM','ZN','INDUS','CHAS','NOX','AGE','DIS','RAD','TAX','RTRATIO','B-1000','LSTAT','MEDV']

    x, y = load_dataset()

    plt.figure(figsize=(12,9))

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.scatter(x[:,i],y)
        # plt.xlabel(titles[i])
        # plt.ylabel("value")
        plt.title(str(i+1)+"."+titles[i]+"-value")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    pre_vsl()