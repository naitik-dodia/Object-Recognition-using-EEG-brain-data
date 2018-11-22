import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
import scipy.io as sp
from sklearn import preprocessing

sam_sizes = [2,4,6,8]

def load_data_binary(file_name):
    data = sp.loadmat(file_name)
    lab = data['categoryLabels'].transpose()
    lab = np.reshape(lab, (lab.shape[0]*lab.shape[1]))
    matrix = data['X_3D'].transpose(2,0,1)
    electrode = []
    cat_labels = []
    for i in range(matrix.shape[0]):
        if(lab[i]==2):
            electrode.append(matrix[i])
            cat_labels.append(1)
        elif(lab[i]==6):
            electrode.append(matrix[i])
            cat_labels.append(2)
    electrode = np.array(electrode)
    cat_labels = np.array(cat_labels)
    matrix = electrode
    matrix = np.reshape(matrix,(matrix.shape[0], matrix.shape[1]*matrix.shape[2]))
    matrix = preprocessing.scale(matrix)
    return electrode, matrix, cat_labels

def load_data(file_name):
    data = sp.loadmat(file_name)
    print(data.keys())
    matrix = data['X_3D'].transpose(2,0,1)
    electrode = data['X_3D'].transpose(2,0,1)
    matrix = np.reshape(matrix,(matrix.shape[0], matrix.shape[1]*matrix.shape[2]))
    matrix = preprocessing.scale(matrix)
    cat_labels = data['categoryLabels'].transpose()
    cat_labels = np.reshape(cat_labels, (cat_labels.shape[0]*cat_labels.shape[1]))
    return electrode, matrix, cat_labels

def split_sampling(matrix, cat_labels, k = 0.8):
    n = matrix.shape[0]
    indices = np.random.permutation(n)
    training_idx, test_idx = indices[:int(k*n)], indices[int(k*n):]
    #print(int(k*n))
    x_train, x_test = matrix[training_idx], matrix[test_idx]
    y_train, y_test = cat_labels[training_idx], cat_labels[test_idx]
    #print(indices)
    x_train = np.reshape(x_train, newshape = (x_train.shape[0],1))
    x_test = np.reshape(x_test, newshape = (x_test.shape[0],1))
    return x_train, y_train, x_test, y_test

def split_sampling2D(matrix, cat_labels, k = 0.8):
    n = matrix.shape[0]
    indices = np.random.permutation(n)
    training_idx, test_idx = indices[:int(k*n)], indices[int(k*n):]
    #print(int(k*n))
    x_train, x_test = matrix[training_idx,:], matrix[test_idx,:]
    y_train, y_test = cat_labels[training_idx], cat_labels[test_idx]
    #print(indices)
    return x_train, y_train, x_test, y_test

def lda_with_shrinkage(x_train, y_train, x_test, y_test):
    clf = LDA(solver = 'eigen',shrinkage = True)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return accuracy_score(pred,y_test)
    
def separate_by_class(x , y):
    num_classes = np.max(y) - np.min(y) + 1
    class_wise_data = []
    for i in range(num_classes):
        class_idx = [j for j in range(x.shape[0]) if(y[j]==i+1)]
        #print(class_idx)
        #print(y[class_idx])
        class_wise_data.append(x[class_idx,:])
    return class_wise_data

def cross_validation(matrix, cat_labels, k = 0.8, niter = 10):
    accuracy = []
    for i in range(niter):
        if(len(matrix.shape)>1):
            x_train, y_train, x_test, y_test = split_sampling2D(matrix, cat_labels)
        else:
            x_train, y_train, x_test, y_test = split_sampling(matrix, cat_labels)
        acc = lda_with_shrinkage(x_train, y_train,x_test, y_test)
        accuracy.append(acc)
    return np.mean(accuracy)

def full_univariate(electrode, cat_labels):
    electrode = electrode.transpose(1,2,0)
    sh0 = electrode.shape[0]
    sh1 = electrode.shape[1]
    sh2 = electrode.shape[2]
    acc = []
    for l in range(sh0):
        temp = []
        for k in range(sh1):
            x_tr = []; y_tr = []
            for i in range(sh2):
                x_tr.append(electrode[l][k][i])
                y_tr.append(cat_labels[i])
            x_tr = np.array(x_tr)
            y_tr = np.array(y_tr)
            # use x_tr & y_tr to get accuracy and store it in acc_score

            temp.append(cross_validation(x_tr,y_tr))
        acc.append(temp)
    # you have acc as 2D matrix (of size 124*32) with every value denoting accuracy accordingly.
    # Plot graph of accuracy.
    electrode = electrode.transpose(2,0,1)
    return acc

def run_cross_val_with_aug(matrix, cat_labels, niter=8, k=0.8):
    sam_sizes = [2,4,6,8]
    acc_mat = []
    for size in sam_sizes:
        acc1 = cross_validation_with_augmentation(matrix,cat_labels,sample_size = size, with_replacement = True, k = k, niter = niter)
        acc2 = cross_validation_with_augmentation(matrix,cat_labels,sample_size = size, with_replacement = False, k = k, niter = niter)
        acc_mat.append([acc1,acc2])
    return acc_mat

def channel_wise(electrode, cat_labels):
    electrode = electrode.transpose(1,2,0)
    sh0 = electrode.shape[0]
    sh1 = electrode.shape[1]
    sh2 = electrode.shape[2]
    acc = []
    acc_augment = []
    for l in range(sh0):
        x_tr = []; y_tr = []
        for i in range(sh2):
            temp = []
            for k in range(sh1):
                temp.append(electrode[l][k][i])
            x_tr.append(temp)
            y_tr.append(cat_labels[i])
        x_tr = np.array(x_tr)
        y_tr = np.array(y_tr)
        # use x_tr & y_tr to get accuracy and store it in acc_score
        acc_score = cross_validation(x_tr, y_tr)#lda_with_shrinkage(x_train, y_train, x_test, y_test)
        acc.append(acc_score)
        acc_score_augment = run_cross_val_with_aug(x_tr, y_tr, 8, 0.8)
        acc_augment.append(acc_score_augment)
    # you have acc as 2D matrix (of size 124) with every value denoting accuracy accordingly.
    # Plot graph of accuracy.
    electrode = electrode.transpose(2,0,1)
    return acc, acc_augment

def time_wise(electrode, cat_labels):
    electrode = electrode.transpose(1,2,0)
    sh0 = electrode.shape[0]
    sh1 = electrode.shape[1]
    sh2 = electrode.shape[2]
    acc = []
    for k in range(sh1):
        x_tr = []; y_tr = []
        for i in range(sh2):
            temp = []
            for l in range(sh0):
                temp.append(electrode[l][k][i])
            x_tr.append(temp)
            y_tr.append(cat_labels[i])
        x_tr = np.array(x_tr)
        y_tr = np.array(y_tr)
        # use x_tr & y_tr to get accuracy and store it in acc_score
        #x_train, y_train, x_test, y_test = split_sampling2D(x_tr, y_tr)
        acc_score = cross_validation(matrix = x_tr, cat_labels = y_tr) #lda_with_shrinkage(x_train, y_train, x_test, y_test)
        # use x_tr & y_tr to get accuracy and store it in acc_score
        acc.append(acc_score)
    # you have acc as 2D matrix (of size 32) with every value denoting accuracy accordingly.
    # Plot graph of accuracy.
    electrode = electrode.transpose(2,0,1)
    return acc


#Augment data

def augment(x,y, sample_size=4, with_replacement = True, k = 0.8):
    num_classes = np.max(y) - np.min(y) + 1
    x_train, y_train, x_test, y_test = split_sampling2D(x, y, k)
    if(with_replacement):
        class_wise_xtrain = separate_by_class(x_train, y_train)
        class_wise_xtest = separate_by_class(x_test, y_test)
        
        augmented_xtrain = []
        augmented_xtest = []
        augmented_ytrain = []
        augmented_ytest = []
        for i in range(num_classes):
            class_augmented_xtrain = []
            class_augmented_xtest = []
            for j in range(len(class_wise_xtrain[i])):
                #print(len(class_wise_xtrain[i])-1)
                idxs = np.random.random_integers(low = 0,high = len(class_wise_xtrain[i])-1, size=sample_size)
                #print(idxs)
                vec = np.mean(class_wise_xtrain[i][idxs,:], axis=0)
                class_augmented_xtrain.append(vec)
            
            augmented_xtrain.extend(class_augmented_xtrain)
            augmented_ytrain.extend(np.ones(len(class_wise_xtrain[i]))+i)
            
            for j in range(len(class_wise_xtest[i])):
                idxs = np.random.random_integers(len(class_wise_xtest[i])-1, size=sample_size)
                class_augmented_xtest.append(np.mean(class_wise_xtrain[i][idxs,:], axis=0))
            augmented_xtest.extend(class_augmented_xtest)
            augmented_ytest.extend(np.ones(len(class_wise_xtest[i]))+i)
            
        augmented_xtrain = np.squeeze(np.array(augmented_xtrain))
        augmented_xtest = np.squeeze(np.array(augmented_xtest))
        return augmented_xtrain, np.array(augmented_ytrain), augmented_xtest, np.array(augmented_ytest)
    else:
        class_wise_xtrain = separate_by_class(x_train, y_train)
        class_wise_xtest = separate_by_class(x_test, y_test)
        
        augmented_xtrain = []
        augmented_xtest = []
        augmented_ytrain = []
        augmented_ytest = []
        
        for i in range(num_classes):
            class_augmented_xtrain = []
            class_augmented_xtest = []
            indices = np.random.permutation(len(class_wise_xtrain[i]))
            
            for j in range(int(len(class_wise_xtrain[i])/sample_size)):
                #print(np.min([(j+1)*sample_size, len(indices)-1]))
                idx = indices[j*sample_size:np.min([(j+1)*sample_size, len(indices)-1])]
                vec = np.mean(class_wise_xtrain[i][idx,:], axis=0)
                #print(vec)
                class_augmented_xtrain.append(vec)
                
            augmented_xtrain.extend(class_augmented_xtrain)
            augmented_ytrain.extend(np.ones(len(class_augmented_xtrain))+i)
            
            indices = np.random.permutation(len(class_wise_xtest[i]))
            
            for j in range(int(len(class_wise_xtest[i])/sample_size)):
                idx = indices[j*sample_size:np.min([(j+1)*sample_size, len(indices)-1])]
                vec = np.mean(class_wise_xtest[i][idx,:], axis=0)
                class_augmented_xtest.append(vec)
                
            augmented_xtest.extend(class_augmented_xtest)
            augmented_ytest.extend(np.ones(len(class_augmented_xtest))+i)
            
        augmented_xtrain = np.squeeze(np.array(augmented_xtrain))
        augmented_xtest = np.squeeze(np.array(augmented_xtest))
        return augmented_xtrain, np.array(augmented_ytrain), augmented_xtest, np.array(augmented_ytest)

def cross_validation_with_augmentation(matrix,cat_labels,sample_size = 4, with_replacement = True, k = 0.8, niter = 10):
    acc = []
    for i in range(niter):
        augmented_xtrain, augmented_ytrain, augmented_xtest, augmented_ytest = augment(matrix, 
                                                                                       cat_labels,
                                                                                       sample_size=sample_size, 
                                                                                       with_replacement=with_replacement, 
                                                                                       k = k)
        temp_acc = lda_with_shrinkage(augmented_xtrain, augmented_ytrain, augmented_xtest, augmented_ytest)
        acc.append(temp_acc)
    return np.mean(acc)

