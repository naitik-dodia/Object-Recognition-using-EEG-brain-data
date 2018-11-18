import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score


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

