import numpy as np
import sklearn.metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import seaborn as sns
import matplotlib
import time
matplotlib.use('Agg')
import pdb
import matplotlib.pyplot as plt
nmi = normalized_mutual_info_score
ari = adjusted_rand_score

print(plt.get_backend())

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

def acc(y_true, y_pred, dataset, to_log, m,n):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    timestr = time.strftime("%Y%m%d-%H%M%S")

    #Saving the y_true and y_pred for later
    y_true_orig = y_true
    y_pred_orig = y_pred


    #Metrics Excluding the 0th class 
    ind_nonzero = []

    for i in range(len(y_true)):
        if (y_true[i] > 0):
            ind_nonzero.append(i)

    # y_true_new = [y_true[i] for i in range(len(y_true)) if y_true[i] > 0]
    # y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    # y_true = y_true_new

    # # pdb.set_trace()

    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)

    # # D = max(y_pred.max(), y_true.max()) + 1
    # # D = max(y_pred.max(), y_true.max()) 
    # # w = np.zeros((D, D), dtype=np.int64)

    # # for i in range(y_pred.size):
    # #     w[y_pred[i], y_true[i]] += 1

    # true_classes = np.unique(y_true)
    # pred_classes = np.unique(y_pred)

    # print("Without Zeros")
    # print(true_classes)
    # print(pred_classes)

    # D = max(len(true_classes), len(pred_classes)) 
    # w = np.zeros((D, D), dtype=np.int64)

    # # pdb.set_trace()

    # for p in range(len(true_classes)):
    #     for q in range(len(pred_classes)):
    #         indices_true = [k for (k, val) in enumerate(y_true) if (true_classes[p] == val)]
    #         indices_pred = [l for (l, val1) in enumerate(y_pred) if (pred_classes[q] == val1)]
    #         w[p,q] = len(intersection(indices_pred, indices_true))

    # from sklearn.utils.linear_assignment_ import linear_assignment
    
    # ind = linear_assignment(- w) #also try w.max - w

    # print(ind)

    # y_pred_new = np.zeros(y_pred.size)

    # for i in range(y_pred.size):
    #     y_pred_new[i] = ind[y_pred[i],1 ]

    true_classes = np.unique(y_true_orig)
    pred_classes = np.unique(y_pred_orig)

    # print("With Zeros")

    D = max(len(true_classes), len(pred_classes)) 
    w = np.zeros((D, D), dtype=np.int64)

    # pdb.set_trace()

    for p in range(len(true_classes)):
        for q in range(len(pred_classes)):
            indices_true = [k for (k, val) in enumerate(y_true_orig) if (true_classes[p] == val)]
            indices_pred = [l for (l, val1) in enumerate(y_pred_orig) if (pred_classes[q] == val1)]
            w[p,q] = len(intersection(indices_pred, indices_true))

    from sklearn.utils.linear_assignment_ import linear_assignment
    
    ind = linear_assignment(w.max()-w) #also try w.max - w

    # print(ind)

    y_pred_new = np.zeros(y_pred_orig.size)

    for i in range(y_pred_orig.size):
        y_pred_new[i] = ind[y_pred_orig[i],1 ]
    
    
    if (to_log == 1):

        y_true_new = [y_true[i] for i in range(len(y_true)) if y_true[i] > 0]
        y_pred_no_zero = [y_pred_new[i] for i in range(len(y_true)) if y_true[i] > 0]



        #Plot of confusion matrix
        sns.set(font_scale=3)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true_new, y_pred_no_zero)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        np.save('confusion_mat/'+timestr+'_' + dataset+'_confusion_mat', confusion_matrix)

        #Predicted Maps
        fig = plt.figure(figsize=(16, 14))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
        plt.title("Confusion matrix", fontsize=30)
        plt.ylabel('True label', fontsize=25)
        plt.xlabel('Clustering label', fontsize=25)
        fig.savefig("confusion_mat/{}_{}_confusion_mat_fig.png" .format(timestr,dataset))

        # y_true1 = y_true_orig.reshape((m,n))

        # cmap = plt.cm.jet
        # norm = plt.Normalize(vmin=y_true1.min(), vmax=y_true1.max())
        # image1 = cmap(norm(y_true1))

        # plt.imsave("spatial_results/{}_{}_fig_gt.png" .format(timestr,dataset), image1)


        y_pred1 = np.zeros(y_true_orig.size)
        
        for i in range(len(ind_nonzero)):
            y_pred1[ind_nonzero[i]] = y_pred_no_zero[i]

        y_pred1 = y_pred1.reshape((m,n))
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=y_pred1.min(), vmax=y_pred1.max())
        image = cmap(norm(y_pred1))


        plt.imsave("spatial_results/{}_{}_fig_predicted.png" .format(timestr,dataset), image)

    #Metrics Including the 0th class 
       
    acc =  sum([w[i, j] for i, j in ind]) * 1.0 / y_pred_orig.size  

    if (to_log == 1):
        #Plot of confusion matrix
        sns.set(font_scale=3)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true_orig, y_pred_new)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        np.save('confusion_mat_with_zero/'+timestr+'_' + dataset+'_confusion_mat', confusion_matrix)

        #Predicted Maps
        fig = plt.figure(figsize=(16, 14))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
        plt.title("Confusion matrix", fontsize=30)
        plt.ylabel('True label', fontsize=25)
        plt.xlabel('Clustering label', fontsize=25)
        fig.savefig("confusion_mat_with_zero/{}_{}_confusion_mat_fig.png" .format(timestr,dataset))

        # y_true1 = y_true_orig.reshape((m,n))

        # cmap = plt.cm.jet
        # norm = plt.Normalize(vmin=y_true1.min(), vmax=y_true1.max())
        # image1 = cmap(norm(y_true1))

        # plt.imsave("spatial_results/{}_{}_fig_gt.png" .format(timestr,dataset), image1)


        y_pred_new = y_pred_new.reshape((m,n))
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=y_pred_new.min(), vmax=y_pred_new.max())
        image = cmap(norm(y_pred_new))


        plt.imsave("spatial_results/{}_{}_{}_fig_predicted.png" .format(timestr,dataset,acc), image)
    

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred_new.size