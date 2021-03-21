import pandas as pd
import numpy as np

data=pd.read_excel('input data.xlsx')
x=data.iloc[:,0:2]
y=data.iloc[:,2]

def main(input,output):
    input['x0']=1 #将b放进w中，b=w*x0
    w=np.zeros([1,input.columns.size])
    while True:
        for i in range(len(input)):
            temp_x=input.iloc[i].values
            temp_y=output[i]
            label=np.dot(temp_x,w.T)
            stop=True
            if np.sign(label)[0] != temp_y:
                w = w + temp_y* temp_x
                stop=False
                break
        if stop:
            break
    return w


print(main(x,y))