import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def cal_cfm(pred,label,typeacc,ncls):
    pred=pred.cpu().detach().numpy()
    label=label.cpu().detach().numpy()
    
    pred=np.argmax(pred,1)
    cfm=confusion_matrix(label,pred,labels=np.arange(ncls))
    for i in range(0, label.size):
        typeacc[0,label[i]] += 1
        if pred[i] == label[i]:
            typeacc[1, label[i]] += 1 
    print(labels)
    print(cfm)
    return cfm