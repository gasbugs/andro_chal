#python2.7
# command
# sudo python decompileApk.py /folder

import os
import sys
import subprocess
from multiprocessing import Pool

import pandas as pd

smali_1gram_f = "1st_smali_1gram.csv"
smali_4gram_f = "1st_smali_4gram.csv"
java_function_f = "1st_java_function.csv"
permission_f = "1st_permission.csv"
y_f = "1st_y.csv"

def decompile(walk_dir, pool=16):
        command1 = []
        command2 = []
        command3 = []

    
        if walk_dir[-1] is not '/':
            walk_dir += '/'

        o_dir = walk_dir[:-1] + "_apktools/"
        os.system('mkdir '+ o_dir)
        #-------------------------------------decompile----------------------------
        print("start decompiling:", walk_dir)
        i = 0
        once = True
        for root, subdirs, files in os.walk(walk_dir):
            for file in files:
                if file.endswith(".apk") or file.endswith(".vir"):
                    i=i+1
                    #print(str(i)+"-"+file )
                    if file.endswith(".apk"):
                        o_file = root+file
                    else :
                        o_file = root+file.split('.')[0]+".apk"
                        os.system("mv {} {} ".format(root+file, o_file))

                    command1.append("apktool d {} -o {} -f > /dev/null".format(o_file, o_dir+file.split('.')[0]))
                    command2.append("unzip -n {} -d {} > /dev/null".format(o_file, o_dir+file.split('.')[0]))
                    command3.append('JAVA_OPTS="-Xmx16G" jadx -j 1 -d {} {}'.format(o_dir+file.split('.')[0]+'/out > /dev/null', 
                                                                                    o_dir+file.split('.')[0]+"/classes.dex"))
                    
        with Pool(pool) as p:
            i = 0
            len_command = len(command3)
            
            i = 0
            print('[*]start decompile apk to smali')
            
            for result in p.imap(os.system, command1):
                i += 1
                if i%10==0:
                    print('{}%           \r'.format(i/len_command))
            print('{}%           \r'.format(i/len_command))
               
            i = 0
            print('[*]start extract classes.dex from apk')
            
            for result in p.imap(os.system, command2):
                i += 1
                if i%10==0:
                    print('processing: {}%           \r'.format(i/len_command * 100))
            print('processing:{}%           \r'.format(i/len_command * 100))
             
            i = 0
            print('[*]start decompile dex to java')
            
            for result in p.imap(os.system, command3):
                i += 1
                if i%10==0:
                    print('processing:{}%           \r'.format(i/len_command * 100))
            print('processing:{}%           \r'.format(i/len_command * 100))
            
            

def machine_learning_for_class(X_train, X_test, Y_train, Y_test):
    from sklearn.svm import SVC
    svc = SVC()
    svc.fit(X_train, Y_train)
    svc_result = svc.score(X_test,Y_test)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    knn_result = knn.score(X_test,Y_test)

    from sklearn.svm import LinearSVC
    lsvc = LinearSVC()
    lsvc.fit(X_train, Y_train)
    lsvc_result = lsvc.score(X_test,Y_test)

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, Y_train)
    nb_result = clf.score(X_test,Y_test)

    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, Y_train)
    decision_tree_result = clf.score(X_test,Y_test)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X_train, Y_train)
    random_forest_result = clf.score(X_test,Y_test)
    
    #nn_result = keras_nn_for_class(X_train, X_test, Y_train, Y_test)

    result = {
        'svm_svc' : svc_result,
        'knn' : knn_result,
        'lsvc' : lsvc_result,
        'GaussianNB' : nb_result,
        'DecisionTree' : decision_tree_result,
        'RandomForest' : random_forest_result#,
        #"nn_result" : nn_result
    }
    return result

def machine_learning_for_family(X_train, X_test, Y_train, Y_test):
    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    knn_result = knn.score(X_test,Y_test)

    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, Y_train)
    decision_tree_result = clf.score(X_test,Y_test)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X_train, Y_train)
    random_forest_result = clf.score(X_test,Y_test)
    
    #nn_result = keras_nn_for_family(X_train, X_test, Y_train, Y_test)

    result = {
        'knn' : knn_result,
        'DecisionTree' : decision_tree_result,
        'RandomForest' : random_forest_result#,
        #"nn_result" : nn_result
    }
    return result

def keras_nn_for_class(X_train, X_test, Y_train, Y_test, dropout = 0.5, batch_size = 32):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    
    input_node = X_train.shape[1]
    
    layer = []
    batch_size = 16
    
    train_generator = generator(X_train, Y_train, batch_size=batch_size)
    validation_generator = generator(X_test, Y_test, batch_size=batch_size)
    
    input_shape =(X_train.shape[1],)  # Trimmed image format
    #print("input_shape:", input_shape)
        
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Dense(input_node, activation='relu', input_shape=input_shape , kernel_initializer='normal'))
    layer.append(input_node)
    model.add(Dropout(dropout)) # for preventing overfit
    layer.append("Dropout")
    input_node = input_node//2
    
    
    while(input_node > 1):
        model.add(Dense(input_node, activation='relu',  kernel_initializer='normal'))
        layer.append(input_node)
        model.add(Dropout(dropout)) # for preventing overfit
        layer.append("Dropout")
        input_node = input_node//2
        
        
    model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
    layer.append(1)
    #print('layer:', layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #########################################################################
    # train a model

    history_object = model.fit_generator(train_generator, steps_per_epoch=len(X_train)/batch_size,
                        validation_data=validation_generator,
                        validation_steps=len(X_test)/batch_size, epochs=10,verbose=0)
    
    loss, score = model.evaluate_generator(validation_generator, steps=len(X_test)/batch_size, max_queue_size=10, workers=1, use_multiprocessing=True)
    return score

def generator(X, y, batch_size=32):
    from sklearn.utils import shuffle
    
    num_samples = len(X)
    
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_X = X[offset:offset+batch_size]
            batch_y = y[offset:offset+batch_size]
        
            yield shuffle(batch_X, batch_y)

def keras_nn_for_family(X_train, X_test, Y_train, Y_test, dropout = 0.5, batch_size = 32):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    
    input_node = X_train.shape[1]
    layer = []
    
    
    train_generator = generator(X_train, Y_train, batch_size=batch_size)
    validation_generator = generator(X_test, Y_test, batch_size=batch_size)
    
    input_shape =(X_train.shape[1],)  # Trimmed image format
    #print("input_shape:", input_shape)
        
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Dense(input_node, activation='relu', input_shape=input_shape , kernel_initializer='normal'))
    layer.append(input_node)
    model.add(Dropout(dropout)) # for preventing overfit
    layer.append("Dropout")
    input_node = input_node//2
    
    
    while(input_node > 10):
        model.add(Dense(input_node, activation='relu',  kernel_initializer='normal'))
        layer.append(input_node)
        model.add(Dropout(dropout)) # for preventing overfit
        layer.append("Dropout")
        input_node = input_node//2
        
        
    model.add(Dense(10, activation='softmax', kernel_initializer='normal'))
    layer.append(10)
    #print('layer:', layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #########################################################################
    # train a model

    history_object = model.fit_generator(train_generator, steps_per_epoch=len(X_train)/batch_size,
                        validation_data=validation_generator,
                        validation_steps=len(X_test)/batch_size, epochs=10, verbose=0)
    
    loss, score = model.evaluate_generator(validation_generator, steps=len(X_test)/batch_size, max_queue_size=10, workers=1, use_multiprocessing=True)
    return score

def generator(X, y, batch_size=32):
    from sklearn.utils import shuffle
    
    num_samples = len(X)
    
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_X = X[offset:offset+batch_size]
            batch_y = y[offset:offset+batch_size]
        
            yield shuffle(batch_X, batch_y)

import pandas as pd

def loadDataset(X=0, feature_scailing=True):
    
    if X not in [1,2,3,4,5,6,7]:
        
        message = '''\
please insert number to X:
    if X == 1:
        X = ['java']
    elif X == 2:
        X = ['smali']
    elif X == 3:
        X = ['permission']
    elif X == 4:
        X = ['permission', 'java']
    elif X == 5:
        X = ['permission', 'smali']
    elif X == 6:
        X = ['java', 'smali']
    elif X == 7:
        X = ['permission', 'java', 'smali']'''
        print(message)
        return False
    
    database = []
    
    if X == 1:
        X = ['java']
    elif X == 2:
        X = ['smali']
    elif X == 3:
        X = ['permission']
    elif X == 4:
        X = ['permission', 'java']
    elif X == 5:
        X = ['permission', 'smali']
    elif X == 6:
        X = ['java', 'smali']
    elif X == 7:
        X = ['permission', 'java', 'smali']
    
    
    #######################
    # Y
    Y = pd.read_csv('1st_y.csv')
    Y.index = Y['filename']
    del Y['filename']
    Y = Y.sort_index()
    Y = Y.fillna(0)
    Y_class = Y['class']
    Y_fam = Y[Y['class']==1].drop(['class'],axis=1)
    
    if 'permission' in X:
        ######################
        # Permissions
        premissions_X = pd.read_csv(permission_f)
        premissions_X.index = premissions_X['Unnamed: 0']
        del premissions_X['Unnamed: 0']
        premissions_X = premissions_X.sort_index()
        premissions_X = premissions_X.fillna(0)
        database.append(premissions_X)
        
    if 'java' in X:
        ##############################
        # java function
        java_X = pd.read_csv(java_function_f)
        java_X.index = java_X['Unnamed: 0']
        del java_X['Unnamed: 0']
        java_X = java_X.sort_index()
        java_X = java_X.fillna(0)
        database.append(java_X)
    
    if 'smali' in X:
        ##############################
        # smali 4gram
        #smali_X = pd.read_csv("bm_df_smali_4gram_X.csv")
        #smali_X.index = smali_X['Unnamed: 0']
        #del smali_X['Unnamed: 0']
        #smali_X = smali_X.sort_index()
        #smali_X = smali_X.fillna(0)
        #database.append(smali_X)
        
        ##############################
        # smali one word function
        smali_X_1 = pd.read_csv(smali_1gram_f)
        smali_X_1.index = smali_X_1['Unnamed: 0']
        del smali_X_1['Unnamed: 0']
        smali_X_1 = smali_X_1.sort_index()
        smali_X_1 = smali_X_1.fillna(0).T
        database.append(smali_X_1)

    X_class = pd.concat(database, axis=1).fillna(0)
    X_family = X_class.loc[Y_fam.index].fillna(0)
    Y_class = Y_class.loc[X_class.index].fillna(0)
    Y_fam = Y_fam.loc[X_family.index].fillna(0)

    
    if feature_scailing:
        # Feature scailing
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_class = sc_X.fit_transform(X_class.values)
        sc_X = StandardScaler()
        X_family = sc_X.fit_transform(X_family.values)
        
    return X_class, X_family, Y_class, Y_fam

def predict_class(X_class, y_class, X):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, Y_train)
    decision_tree_result = clf.predict(X)
    return decision_tree_result

def predict_family(X_family, y_family, X):
    pass

def loadSample():
    pass

if __name__ == "__main__":
    
    X_class, X_family, y_class, y_family = loadDataset(7, feature_scailing=False)
    print(X_class.shape)
    print(X_family.shape)
    print(y_class.shape)
    print(y_family.shape)
    
    X = loadSample()
    
    result_class = predict_class(X_class, y_class, X)
    result_family = predict_family(X_family, y_family, X)
    
    df = pd.concat([result_class,result_family])
    print(df.shape)

    
    