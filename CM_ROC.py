# -*- coding: utf-8 -*-
"""
Confusion Matrix and ROC curve
@author: s151385
"""
def CM(true,predicted):
    from sklearn.metrics import confusion_matrix
    import itertools
    import matplotlib.pyplot as plt
    import numpy as np
    cm = confusion_matrix(true, predicted)
    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    classes = {'benign':0, 'melanoma':1}
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def ROC(true,predictions): 
    from sklearn.metrics import roc_curve, auc 
    import matplotlib.pyplot as plt    
    fpr= dict()
    tpr= dict()
    roc_auc =dict()
    fpr,tpr, _ = roc_curve(true,predictions)
    roc_auc = auc(fpr, tpr)
        
# =============================================================================
#     plt.figure()
#     plt.plot(fpr, tpr, lw=2,
#              label='ROC curve  (area = {f:.2f})'.format( f=roc_auc))
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     plt.show()
# =============================================================================
    return(roc_auc)