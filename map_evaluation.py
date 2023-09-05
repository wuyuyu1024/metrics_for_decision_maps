import os
import sys
sys.path.append('./sdbm')
sys.path.append('./deepview')
#import cupy as cp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
# import seaborn as sns
import time 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import accuracy score
from sklearn.metrics import accuracy_score

from umap import UMAP

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
## import constant bias
from tensorflow.keras.initializers import Constant



from ssnp import SSNP
from deepview import DeepView

from deepview.fisher_metric import calculate_fisher_new
from tqdm import tqdm




class P_wrapper:
    def __init__(self, NNinv_Keras=None, deepview=None, ssnp=None):
        self.backup = (NNinv_Keras, deepview, ssnp)
        self.deepview = deepview 
        self.ssnp = ssnp
        self.NNinv_Keras = NNinv_Keras
 
      
    def fit(self, X, y=None, clf=None, lam=0.65, **kwargs):
        start_time = time.time()
        # check is X is in [0,1] range
      
        if self.deepview:
            self.deepview = DeepView(pred_fn=clf.predict_proba, classes=set(y), max_samples=5000, batch_size=1000, data_shape=X.shape[1:], 
                                    n=5, lam=lam)

            self.deepview.add_samples_new(X, y)
            self.P = self.deepview.mapper
            self.P_inverse = self.deepview.inverse
        elif self.ssnp:
            # print('device: ', tf.test.gpu_device_name())
            self.ssnp = SSNP(patience=5, opt='adam', bottleneck_activation='linear', verbose=0)
            self.ssnp.fit(X, y, **kwargs)

        elif self.NNinv_Keras:           
            self.P = UMAP(metric='euclidean', n_components=2, random_state=42)
            self.P_inverse = NNinv_keras()

            embedding = self.P.fit_transform(X)
            self.P_inverse.fit(embedding, X, **kwargs)

            embedding = self.P.fit_transform(X)
            self.P_inverse.fit(embedding, X, **kwargs)

        self.train_time = time.time() - start_time
        print('fitting time: ', self.train_time)
        return self.train_time
            
    def transform(self, X):
        if self.deepview:
            ## calculate new distances
            new_discr, new_eucl = calculate_fisher_new(self.deepview.model, X,  self.deepview.samples, 
                self.deepview.n, self.deepview.batch_size, self.deepview.n_classes, self.deepview.verbose) 
            eucl_distances = new_discr
            discr_distances = new_eucl
            eucl_scale = 1. / self.deepview.eucl_distances.max()
            fisher_scale = 1. / self.deepview.discr_distances.max()
            eucl = eucl_distances * eucl_scale * self.deepview.lam
            fisher = discr_distances * fisher_scale * (1.-self.deepview.lam)
            stacked = np.dstack((fisher, eucl))
            distances =  stacked.sum(-1)
            ## now transform
            embedding = self.P.transform(distances)
            return embedding
        
        elif self.ssnp:
            return self.ssnp.transform(X)
        else:
            return self.P.transform(X)

    def inverse_transform(self, embedding, **kwargs):
        if self.ssnp:
            X_reconstuct = self.ssnp.inverse_transform(embedding, **kwargs)
        else:
            X_reconstuct =  self.P_inverse.transform(embedding, **kwargs)
        return X_reconstuct

    def inverse_timer(self, n_samples):
        start_time = time.time()
        random_embedding = np.random.rand(n_samples, 2)
        self.P_inverse.transform(random_embedding)
        time_taken = time.time() - start_time
        print(f'inverse time with {n_samples} samples: ', time_taken)
        return time_taken

    def reset(self):
        self.NNinv_Keras, self.deepview, self.ssnp = self.backup
        self.train_time = 0
        # if hass attr self.P_inverse
        if hasattr(self, 'P_inverse'):
            self.P_inverse = None
            self.P = None

   
         
#### Keras version NNinv
class NNinv_keras:
    def __init__(self, bias=0.01):
        self.model = None
        self.bias = bias
        tf.random.set_seed(42)

    def create_model(self, n_dim):
        model = keras.Sequential(
            [
                keras.Input(shape=2),
                layers.Dense(2048, activation="relu", kernel_initializer='he_uniform', bias_initializer=Constant(self.bias)), #
                layers.Dense(2048, activation="relu", kernel_initializer='he_uniform', bias_initializer=Constant(self.bias)),
                layers.Dense(2048, activation="relu", kernel_initializer='he_uniform', bias_initializer=Constant(self.bias)),
                layers.Dense(2048, activation="relu", kernel_initializer='he_uniform', bias_initializer=Constant(self.bias)),
                layers.Dense(n_dim, activation="sigmoid", kernel_initializer='he_normal', bias_initializer=Constant(self.bias)),
            ]
        )
        return model

    def fit(self, X_2d, X, epochs=300, batch_size=128, verbose=0, early_stop=True, patience=5):
        self.scaler = MinMaxScaler()
        X_2d = self.scaler.fit_transform(X_2d)
        if self.model is None:
            self.model = self.create_model(X.shape[1])
            self.model.compile(loss='mean_squared_error', optimizer='adam')
        if early_stop:
            early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
            self.model.fit(X_2d, X, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[early_stop])
        else:
            self.model.fit(X_2d, X, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def transform(self, X_2d):
        X_2d = self.scaler.transform(X_2d)
        return self.model.predict(X_2d)

    def reset(self):
        self.model = None


    

class Evaluator:
    def __init__(self, classifiers=None, projecters=None, Datasets=None):
        """
        classifiers: dict of trained classifiers with name as key
        projecters: dict of trained projecters with name as key
        Datasets: dict of datasets with name as key
        """
        # pass
        self.columns =  ['Dataset', 'Decision_Map', 'Classifier', 
            "$ACC_C^t$", "$ACC_M$^t", '$Cons_d$^t', 'Avg. Train',
            "$ACC_C^T$", "$ACC_M^T$ Test",   '$Cons_d^T$ Test',  "Avg. Test", 
            "Grad Avg.", "Grad Min", "Grad Max",  "Norm Grad Avg.", 
            "S_mean", "$Cons_p$", \
            "Time Fitting", "Time Evaluation"
            ]
        self.df = pd.DataFrame(columns=self.columns)
         

    def evaluate_all(self, classifiers, projecters, datasets, read_from_file=False, save_name='eval_results.csv', save_path='results'):
        if read_from_file:
            self.df = pd.read_csv(read_from_file, index_col=None)
        "return a dataframe of evaluation results"
        for dataset_name, dataset in datasets.items():
            for clf_name, clf in classifiers.items(): 
                if clf_name == 'Random Forest':
                    clf.set_params(random_state=0)
                for proj_name, proj in projecters.items():    ## changed the order of loops                                
                    print(proj_name, clf_name, dataset_name) 

                    if clf_name == "SVM" and proj_name == "DeepView":
                        print(f"skipping {clf_name} on {dataset_name} with {proj_name}")
                        continue
   
                    print('------------------------------------')
                    time0 = time.time()
                    ##########################
                    print(f'training {clf_name} on {dataset_name} with {proj_name}')
                    clf.fit(dataset[0], dataset[2])
                    # # classifiers[clf_name] = clf
                    # print(f'test score: {clf.score(dataset[1], dataset[3])}')
            
                    time_fitting = proj.fit(dataset[0], dataset[2], clf)
                    # projecters[proj_name] = proj
                    ##########################
                    time1 = time.time()
                    # time_fitting = time1 - time0

                    print(f'evaluating {dataset_name} with {proj_name} and {clf_name}')
                    prefix = [dataset_name, proj_name, clf_name]
                    results = self.evaluate_one(clf, proj, dataset, proj_name, clf_name, dataset_name, save_path)

                    time_eval = time.time() - time1
                    results.append(time_fitting)
                    results.append(time_eval)
                   
                    self.df.loc[len(self.df)] = prefix + results

                    proj.reset()
                    print("=====================================================")

                    # if ./eval_results/ does not exist, create it
                    if not os.path.exists('./global_metrics_evaluation/'):
                        os.makedirs('./global_metrics_evaluation/')
                    self.df.to_csv('./global_metrics_evaluation/'+save_name, index=False)

        return self.df
                  

    def evaluate_one(self, clf, projecters, data, proj_name, clf_name, dataset_name, save_path='results'):
        """
        return a list of evaluation results:
        [acc_train, acc_train_proj, cons_train, (acc_train + acc_train_proj + cons_train)/3,
        acc_test, acc_test_proj, cons_test, (acc_test + acc_test_proj + cons_test)/3,
        gradient_mean, D_min, D_max, norm_gradient_mean,
        S_mean, one_round_error, MSEs[0]]

        """
        acc_test, acc_test_proj, cons_test = -1, -1, -1
        gradient_mean, D_min, D_max, norm_gradient_mean = -1, -1, -1, -1
        # print(f'evaluating {dataset_name} with {proj_name} and {clf_name}')
        ### unpack data
        X_train, X_test, y_train, y_test = data
        # get the results
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)

        # ===================
        if not os.path.exists(f'./{save_path}/data/'):
            os.makedirs(f'./{save_path}/data/')

        #### train data
        if projecters.ssnp:
            X_train_2d = projecters.transform(X_train)
        else:
            X_train_2d = projecters.P.embedding_
        X_train_rec = projecters.inverse_transform(X_train_2d)
        y_train_pred_rec = clf.predict(X_train_rec)
        np.save(f'./{save_path}/data/{dataset_name}_{proj_name}_{clf_name}_X_train_rec.npy', X_train_rec)
        np.save(f'./{save_path}/data/{dataset_name}_{proj_name}_{clf_name}_X_train_2d.npy', X_train_2d)
        np.save(f'./{save_path}/data/{dataset_name}_{proj_name}_{clf_name}_y_train_pred.npy', y_train_pred)
        np.save(f'./{save_path}/data/{dataset_name}_{proj_name}_{clf_name}_y_train_pred_rec.npy', y_train_pred_rec)
        del X_train_rec
  

        #### test data
        X_test_2d = projecters.transform(X_test)
        X_test_rec = projecters.inverse_transform(X_test_2d)
        # y_test_pred = clf.predict(X_test)
        y_test_pred_rec = clf.predict(X_test_rec)
        np.save(f'./{save_path}/data/{dataset_name}_{proj_name}_{clf_name}_X_test_rec.npy', X_test_rec)
        np.save(f'./{save_path}/data/{dataset_name}_{proj_name}_{clf_name}_X_test_2d.npy', X_test_2d)
        np.save(f'./{save_path}/data/{dataset_name}_{proj_name}_{clf_name}_y_test_pred.npy', y_test_pred)
        np.save(f'./{save_path}/data/{dataset_name}_{proj_name}_{clf_name}_y_test_pred_rec.npy', y_test_pred_rec) # forgot before
        del X_test_rec
     

        acc_train_proj = accuracy_score(y_train, y_train_pred_rec)
        acc_test_proj = accuracy_score(y_test, y_test_pred_rec)

        cons_train = accuracy_score(y_train_pred, y_train_pred_rec)
        cons_test = accuracy_score(y_test_pred, y_test_pred_rec)


        # ===================
        print('calculating gradient map')

        grid_size = 150

        gradient_mean, norm_gradient_mean, D, ndgrid_rec = self.get_gradient_new(projecters, X_train_2d, grid_size)
        D_min = np.min(D)
        D_max = np.max(D)
        if not os.path.exists(f'./{save_path}/GM/'):
                os.makedirs(f'./{save_path}/GM/')
        np.save(f'./{save_path}/GM/{dataset_name}_{proj_name}_{clf_name}_GM.npy', D)
        del D


        print('calculating probability map')
        
        proba_label = self.get_prob_map(projecters, clf, X_train_2d, grid_size, ndgrid_rec)
        if not os.path.exists(f'./{save_path}/DM/'):
            os.makedirs(f'./{save_path}/DM/')
        np.save(f'./{save_path}/DM/{dataset_name}_{proj_name}_{clf_name}_DM.npy', proba_label)
        del proba_label
        del ndgrid_rec

        self.xy = None

        ##=========================== class stability ============================
        print('calculating class stability')
    
        N_round = 10
        S, label_list, MSEs, _ = self.class_stability_all(projecters, clf, X_train_2d, grid=100, N=N_round)
        # print(S.shape)
        # alterd_label = label_list[-1]
        #################
        if not os.path.exists(f'./{save_path}/CS/'):
            os.makedirs(f'./{save_path}/CS/')
        np.save(f'./{save_path}/CS/{dataset_name}_{proj_name}_{clf_name}_CS.npy', S)
        np.save(f'./{save_path}/CS/{dataset_name}_{proj_name}_{clf_name}_CS_label.npy', label_list)

        S_mean = np.mean(S, axis=0) / N_round
        one_round_error = accuracy_score(label_list[0], label_list[1])
        

        return [acc_train, acc_train_proj, cons_train, (acc_train + acc_train_proj + cons_train)/3, 
                acc_test, acc_test_proj, cons_test, (acc_test + acc_test_proj + cons_test)/3,
                gradient_mean, D_min, D_max, norm_gradient_mean,
                S_mean, one_round_error]


    def projection_miss(self,clf, projecters, x, y=None):
    # project x to the latent space and then back to the original space
        # time0 = time.time()
        x2d = projecters.transform(x)
        xnd = projecters.inverse_transform(x2d)
        pred2d = clf.predict(xnd)
        y_pred = clf.predict(x)
        # return the index where pred2d != y_pred
        ind = np.where(pred2d == y_pred)[0]
        return len(ind)/len(y_pred)

    def clf_missclass(self, clf, projecters, x, y):
        # time0 = time.time()
        y_pred = clf.predict(x)
        # print('time: clf_missclass:', time.time() - time0)
        ind = np.where(y_pred == y)[0]
        return len(ind) / len(y)

    def dbm_missclass(self, clf, projecters, x, y):
        # time0 = time.time()
        # project x to the latent space and then back to the original space
        x2d = projecters.transform(x)
        xnd = projecters.inverse_transform(x2d)
        pred2d = clf.predict(xnd)
        return len(np.where(pred2d == y)[0]) / len(y)

    def get_gradient_mean(self, projecters=None, x2d=None, grid=100):
        # make grid
        # x2d = projecters.transform(x)
        x_max, x_min = np.max(x2d[:,0]), np.min(x2d[:,0])
        y_max, y_min = np.max(x2d[:,1]), np.min(x2d[:,1])
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid), np.linspace(y_min, y_max, grid)) # make it 100*100 to reduce the computation
        self.xy = np.c_[xx.ravel(), yy.ravel()]
        # get the gradient
        pixel_width = (x_max - x_min) / grid
        pixel_height = (y_max - y_min) / grid
        Dx = projecters.inverse_transform(self.xy + np.array([pixel_width, 0])) - projecters.inverse_transform(self.xy - np.array([pixel_width, 0]))
        Dy = projecters.inverse_transform(self.xy + np.array([0, pixel_height])) - projecters.inverse_transform(self.xy - np.array([0, pixel_height]))
        Dx = Dx / (2 * pixel_width)
        Dy = Dy / (2 * pixel_height)
        # get the gradient norm
        D = np.sqrt(np.sum(Dx**2, axis=1) + np.sum(Dy**2, axis=1))  
        norm_D = D / np.max(D)
        return D.mean(), norm_D.mean(), D

    def get_gradient_new(self, projecters=None, x2d=None, grid=100):
        # make grid
        # x2d = projecters.transform(x)
        
        x_max, x_min = np.max(x2d[:,0]), np.min(x2d[:,0])
        y_max, y_min = np.max(x2d[:,1]), np.min(x2d[:,1])
        # pixel_width = (x_max - x_min) / grid
        # pixel_height = (y_max - y_min) / grid
        pixel_width =  1/grid
        pixel_height = 1/grid

        grid_pad = grid + 2 

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_pad), np.linspace(y_min, y_max, grid_pad)) # make it 100*100 to reduce the computation
        xy = np.c_[xx.ravel(), yy.ravel()]
        # get the gradient
        ndgrid_padding = projecters.inverse_transform(xy)
        print(ndgrid_padding.shape)
        
        # ndgrid_rec = ndgrid_rec
        ndgrid_padding = ndgrid_padding.reshape(grid_pad, grid_pad, -1)
        ndgrid_rec = ndgrid_padding[1:-1, 1:-1, :]

        Dx = ndgrid_padding[2:, 1:-1] - ndgrid_padding[:-2, 1:-1]
        Dy = ndgrid_padding[1:-1, 2:] - ndgrid_padding[1:-1, :-2]
        Dx = Dx / (2 * pixel_width)
        Dy = Dy / (2 * pixel_height)
        # get the gradient norm
        D = np.sqrt(np.sum(Dx**2, axis=2) + np.sum(Dy**2, axis=2))
        D = D.reshape(-1)
        norm_D = D / np.max(D)
        return D.mean(), norm_D.mean(), D, ndgrid_rec
        

    def get_prob_map(self, proj, clf, X_train_2d, grid=100, ndgrid_rec=None):
        """Get probability map for the classifier
        """
        if ndgrid_rec is None:
         
            x_max, x_min = np.max(X_train_2d[:,0]), np.min(X_train_2d[:,0])
            y_max, y_min = np.max(X_train_2d[:,1]), np.min(X_train_2d[:,1])
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid), np.linspace(y_min, y_max, grid)) # make it 100*100 to reduce the computation
            xy = np.c_[xx.ravel(), yy.ravel()]

            scaler2d = MinMaxScaler().fit(X_train_2d)
            inverse_scaler = scaler2d.inverse_transform(xy)#.astype('float32')
            inversed = proj.inverse_transform(inverse_scaler)#.astype('float32')
        else:
            # print('skip inverseprojection with shape:', ndgrid_rec.shape)
            inversed = ndgrid_rec.reshape(-1, ndgrid_rec.shape[-1])
            print('inversed shape:', inversed.shape)
        
        probs = clf.predict_proba(inversed)
        ###
        alpha = np.amax(probs, axis=1)
        labels = probs.argmax(axis=1)
        return alpha, labels

    def class_stability(self, proj, clf, X_train_2d, grid=100, N=10): 
                 
        x_max, x_min = np.max(X_train_2d[:,0]), np.min(X_train_2d[:,0])
        y_max, y_min = np.max(X_train_2d[:,1]), np.min(X_train_2d[:,1])
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid), np.linspace(y_min, y_max, grid)) # make it 100*100 to reduce the computation
        xy = np.c_[xx.ravel(), yy.ravel()]

        pixel_nd0 = proj.inverse_transform(xy)
        label_0 = clf.predict(pixel_nd0)
        pixel_nd_next = pixel_nd0
        MSEs = []
        
        S = np.zeros(xy.shape[0]) + N
        label_list = [label_0]
        ind_left = np.arange(xy.shape[0])
        pix_left = []
        pix_left.append(len(ind_left))
        new_labels = label_0.copy()
        new_pixel_nd0 = pixel_nd0.copy()

        for i in tqdm(range(N)):
            if len(ind_left) == 0:
                print('all points are changed')
                break
            pixel_nd_next = proj.inverse_transform(proj.transform(new_pixel_nd0[ind_left]))
            label_next = clf.predict(pixel_nd_next)

            ## reduce calculation
            new_labels[ind_left] = label_next
            new_pixel_nd0[ind_left] = pixel_nd_next
            label_list.append(new_labels.copy()) ## add index
            
            MSE = np.mean((new_pixel_nd0 - pixel_nd0)**2)

            MSEs.append(MSE)
            ind_changed = np.where(new_labels != label_0)[0]
            ind_changed = np.intersect1d(ind_changed, ind_left)
            ind_left = np.setdiff1d(ind_left, ind_changed)
            pix_left.append(len(ind_left))

            S[ind_changed] = i+1
        return S, label_list, MSEs, new_labels, pix_left

    def class_stability_all(self, proj, clf, X_train_2d, grid=100, N=10):          
        x_max, x_min = np.max(X_train_2d[:,0]), np.min(X_train_2d[:,0])
        y_max, y_min = np.max(X_train_2d[:,1]), np.min(X_train_2d[:,1])
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid), np.linspace(y_min, y_max, grid)) # make it 100*100 to reduce the computation
        xy = np.c_[xx.ravel(), yy.ravel()]

        pixel_nd0 = proj.inverse_transform(xy)
        label_0 = clf.predict(pixel_nd0)
        pixel_nd_next = pixel_nd0
        MSEs = []
        
        S = np.zeros(xy.shape[0]) + N
        label_list = [label_0]
        ind_left = np.arange(xy.shape[0])
        pix_left = []
        pix_left.append(len(ind_left))
        # new_labels = label_0.copy()
        # new_pixel_nd0 = pixel_nd0.copy()

        for i in tqdm(range(N)):
            if len(ind_left) == 0:
                print('all points are changed')
                break
            pixel_nd_next = proj.inverse_transform(proj.transform(pixel_nd_next))
            label_next = clf.predict(pixel_nd_next)

            ## reduce calculation
   
            label_list.append(label_next) ## add index
            
            MSE = np.mean((pixel_nd_next - pixel_nd0)**2)

            MSEs.append(MSE)
            ind_changed = np.where(label_next != label_0)[0]
            ind_changed = np.intersect1d(ind_changed, ind_left)
            ind_left = np.setdiff1d(ind_left, ind_changed)
            pix_left.append(len(ind_left))

            S[ind_changed] = i+1
        return S, label_list, MSEs, pix_left



        

class MapBuilder:
    def __init__(self, clf, projecters, X_train, y_train=None, grid=100, X_train_2d=None):
        self.clf = clf
        self.projecters = projecters
        if X_train_2d != None:
            self.X_train_2d = X_train_2d
        else:
            try:
                self.X_train_2d = projecters.P.embedding_
            except:
                self.X_train_2d = projecters.transform(X_train) 

        self.y_train = y_train

        self.scaler2d = MinMaxScaler().fit(self.X_train_2d)
        # self.scalernd = scalernd
        self.xx, self.yy = self.make_meshgrid(grid=grid)
        print('calculating probability map')
        self.map_res = self.get_prob_map()
        self.gradient_res = None
        # self.inversed_feature_res = self.inversed_feature()


    def make_meshgrid(self, x=np.array([0,1]), y=np.array([0,1]), grid=300):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 0.0, x.max() + 0.0
        y_min, y_max = y.min() - 0.0, y.max() + 0.0
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid),
                            np.linspace(y_min, y_max, grid))
        return xx, yy
    
    def get_gradient_map(self, xy=None, step=0.01):
        """
        for each pixel, the gradient is calculated by
        Dx(y) = (B(y + (w, 0)) - B(y - (w, 0)))/ 2w 
        Dy(y) = (B(y + (0, h)) - B(y - (0, h)))/ 2h 
        D(y) = squrt(‖Dx(y)‖**2 + ‖Dy(y)‖**2)
        
        where y is a point in the 2D projection space and w and h are a pixel's width and height, respectively.
        B is projecters.inverse_transform() function.
        """
        print("calculating gradient map")
        if xy is None:
            xx, yy = self.xx, self.yy
            xy = np.c_[xx.ravel(), yy.ravel()].astype('float32')
        else:
            xy = xy
        # step = (xy.max() - xy.min()) / int(np.sqrt(xy.shape[0]))
        # inversed = self.projecters.inverse_transform(inverse_scaler).astype('float32')
        Dx = self.projecters.inverse_transform(self.scaler2d.inverse_transform(xy + [step, 0])) - self.projecters.inverse_transform(self.scaler2d.inverse_transform(xy - [step, 0]))
        Dy = self.projecters.inverse_transform(self.scaler2d.inverse_transform(xy + [0, step])) - self.projecters.inverse_transform(self.scaler2d.inverse_transform(xy - [0, step]))
        Dx /= (2 * step / self.scaler2d.scale_[0])
        Dy /= (2 * step / self.scaler2d.scale_[1])
        # np.sqrt(l2 norm of Dx + l2 norm of Dy)
        D = np.sqrt(np.sum(Dx**2, axis=1) + np.sum(Dy**2, axis=1))  
        self.gradient_res = D  
        # print D mean
        print(np.mean(D))
        return D

    def plot_gradient_map(self, ax=None, cmap=None, step=0.01):
        """Plot probability map for the classifier
        """
        if ax is None:
            ax = plt.gca()
        if self.gradient_res is not None:
            D = self.gradient_res
        else:
            xx, yy = self.xx, self.yy
            D = self.get_gradient_map(step=step)
            D = D.reshape(xx.shape)
            D = np.flip(D, axis=0)
            
        if cmap is None:
            cmap = cm.get_cmap('viridis')

        ax.imshow(D, cmap=cmap, extent=[xx.min(), xx.max(), yy.min(), yy.max()])  
        # ax.pcolormesh(xx, yy, D, cmap=cmap)
        # cbar for gradient
        # norm = colors.Normalize(vmin=0, vmax=1)
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # cbar = plt.colorbar(sm)
        # cbar.set_label('Gradient')
        ax.set_xticks([])
        ax.set_yticks([])
        # aspect square
        ax.set_aspect('equal')
        return ax

    def get_prob_map(self):
        """Get probability map for the classifier
        """
        xx, yy = self.xx, self.yy
        inverse_scaler = self.scaler2d.inverse_transform(np.c_[xx.ravel(), yy.ravel()]).astype('float32')
        inversed = self.projecters.inverse_transform(inverse_scaler).astype('float32')
        probs = self.clf.predict_proba(inversed)
        alpha = np.amax(probs, axis=1)
        labels = probs.argmax(axis=1)
        # alpha = alpha.reshape(xx.shape)
        # labels = labels.reshape(xx.shape)
        # self.map_res = (alpha, labels, xx, yy)
        return alpha, labels
    
    
    def plot_boundary(self, ax=None,):
        """Plot probability map for the classifier
        """
        if not self.map_res:
            alpha, labels= self.get_prob_map()
        else:
            alpha, labels= self.map_res
        if ax is None:
            ax = plt.gca()
        xx, yy = self.xx, self.yy
        labels_normlized = labels/self.clf.classes_.max()
        # labels = labels.reshape(xx.shape)
        ax.contour(xx, yy, labels.reshape(xx.shape), levels=(np.arange(self.clf.classes_.max() + 2) - 0.5),
                    linewidths=1, colors="k", antialiased=True)
        return ax


    def plot_prob_map(self, ax=None, cmap=cm.tab10, epsilo=0.85, proba=True, ture_map=False):
        """Plot probability map for the classifier
        """
        if not self.map_res:
            alpha, labels= self.get_prob_map()
        else:
            alpha, labels= self.map_res
        if ax is None:
            ax = plt.gca()
        xx, yy = self.xx, self.yy
        labels_normlized = labels/self.clf.classes_.max()
        map = cmap(labels_normlized)
        if proba:
            map[:, 3] = alpha 
        map[:, 3] *= epsilo  # plus a float to control the transparency
        map =  map.reshape(xx.shape[0], xx.shape[1], 4)
        map = np.flip(map, 0)
        ax.imshow(map, interpolation='nearest', aspect='auto', extent=[xx.min(), xx.max(), yy.min(), yy.max()])
        ax.set_facecolor('black')
        ax.set_xticks([])
        ax.set_yticks([])
        # set lim
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        return ax

    def plot_training_data(self, ax=None, cmap=cm.tab10, size=10, alpha=0.95):
        """Plot probability map for the classifier
        """
        if ax is None:
            ax = plt.gca()
        X_2d = self.scaler2d.transform(self.X_train_2d).astype('float32')
        # ax.scatter(X_2d[:, 0], X_2d[:, 1], marker='.', s=10, edgecolors='k', c=self.y_train, cmap=cmap)
        colors = cmap(self.y_train/max(self.y_train))
        for i in set(self.y_train):
            ax.scatter(X_2d[self.y_train==i, 0], X_2d[self.y_train==i, 1], marker='.', s=size, edgecolors=None, c=colors[self.y_train==i], label=i)
        ax.legend()
        return ax



