import pdb
import sys

import numpy as np

from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, generic_utils
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, ZeroPadding3D
from keras.layers.core import Reshape, Dropout
from keras.models import model_from_yaml

print("Imported keras")

sys.path.append("../LorentzLayer")

from cola import CoLa
from lola import LoLa
from sola import SoLa
#from ala import ALa

#
# Prepare Jet Image
#

def to_image_2d(df):
    foo =  np.expand_dims(np.expand_dims(df[ ["img_{0}".format(i) for i in range(40*40)]], axis=-1).reshape(-1,40,40), axis=-1)        
    
    return foo

#
# Prepare Constituents
#

def sort_energy(ret):

    batch_index = np.arange(np.shape(ret)[0])[:,np.newaxis,np.newaxis]
    feature_index = np.repeat(np.arange(ret.shape[1])[np.newaxis,:,np.newaxis],ret.shape[0],axis=0)
    sorting_index = ret[:,0].argsort()
    sorting_index = np.repeat(np.expand_dims(sorting_index,1),ret.shape[1],axis=1)
    return ret[batch_index, feature_index, sorting_index]

def to_constit(df, n_constit, n_features):
    brs = []

    if n_features == 4:
        feat_list =  ["E","PX","PY","PZ"] 
    elif n_features == 5:
        feat_list =  ["E","PX","PY","PZ","C"] 
    elif n_features == 8:
        feat_list =  ["E","PX","PY","PZ","C","VX", "VY", "VZ"] 

    brs += ["{0}_{1}".format(feature,constit) for feature in feat_list for constit in range(n_constit)]

    ret = np.expand_dims(df[brs],axis=-1).reshape(-1, n_features, n_constit)
    

    ret = ret/500.


    # # Random sort inputs
    # ret = np.swapaxes(ret,1,2) # b f c -> b c f
    # for _ in range(n_constit):
    #     np.random.shuffle(ret[_]) # resort constits for
    # ret = np.swapaxes(ret,1,2) # b c f -> b f c

    if n_features == 5:
        ret[:,4,:] = ret[:,4,:] * 500.
        ret[:,4,:] = pow(ret[:,4,:],2)

    if n_features == 8:
        ret[:,4,:] = ret[:,4,:] * 500.
        ret[:,4,:] = pow(ret[:,4,:],2)
        #ret[:,5,:] = ret[:,5,:] * 500.
        #ret[:,6,:] = ret[:,6,:] * 500.
        #ret[:,7,:] = ret[:,7,:] * 500.


    #ret[:,4:,:30] = np.zeros((512,4,30))
        
    return ret



def turbo_boost(ret):
    """ Receive a tensor and do a Lorentz boost. 

    Shape is (event, features, particles)

    First the total jet is calculated as the sum over all particles per event.

    Then all particles are boosted into that frame
    """

    # total jet
    p_jet = np.sum(ret,axis=2)

    # boost direction = -1 * (px,py,pz)/E
    b =-1. * p_jet[:,1:]/np.expand_dims(p_jet[:,0],1).repeat(3,axis=1)
    
    # magnitude of boost vector
    bmag2=np.sum(b*b,axis=1)

    # and gamma factor
    gamma = 1/np.sqrt(1.-bmag2) 

    # E/3-vector component of inital object
    old_E    = ret[:,0]
    old_3vec = ret[:,1:]

    # x' = x + (gamma-1)/bmag2 * b.x * b + gamma * E * b

    # term2 = (gamma-1)/bmag2 * b.x * b    
    term2 = (gamma-1)/bmag2 

    b_in_x = np.einsum('ij,ijk->ik',b, old_3vec)
    term2 = np.einsum('i,ik->ik',term2, b_in_x)
    term2 = np.einsum('ik,ij->ijk', term2,b)

    # term3 = gamma * E * b
    term3 = np.einsum('i,ik->ik',gamma,old_E)
    term3 = np.einsum('ik,ij->ijk',term3, b)

    # Now put it together
    # x' = x + (gamma-1)/bmag2 * b.x * b + gamma * E * b
    new_3vec = old_3vec + term2 + term3

    # t' = gamma * (t + b.x)
    new_E = np.expand_dims(np.einsum('i,ij->ij', gamma, (old_E + b_in_x)),1)

    new_4vec = np.concatenate((new_E, new_3vec), axis=1)

    ret = (np.nan_to_num(new_4vec))

    ret = sort_energy(ret)

    return ret
 


def to_constit_boost(df, n_constit, n_features):
    brs = []

    if n_features == 4:
        feat_list =  ["E","PX","PY","PZ"] 
    else:
        print("n_features={0} not implemented. Quitting.".format(n_features))
        sys.exit()

    brs += ["{0}_{1}".format(feature,constit) for feature in feat_list for constit in range(n_constit)]
    ret = np.expand_dims(df[brs],axis=-1).reshape(-1, n_features, n_constit)
    ret = ret/500.

    return turbo_boost(ret)


#
# 2D ConvNet
#


def model_caps(params):

    model, eval_model, manipulate_model = CapsNet(input_shape=(40,40,1),
                                                  n_class=2,
                                                  routings=3)

    eval_model.load_weights("trained_model.h5")
    print eval_model.summary()


    return eval_model



def model_shih(params):
    model = Sequential()
    model.add(Conv2D(128,(4,4),padding='same', activation='relu',input_shape=(40,40,1),data_format = "channels_last"))
    model.add(Conv2D(64,(4,4),padding='same', activation='relu',data_format = "channels_last"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, data_format = "channels_last"))
    model.add(Conv2D(64,(4,4),padding='same', activation='relu',data_format = "channels_last"))
    model.add(Conv2D(64,(4,4),padding='same', activation='relu',data_format = "channels_last"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, data_format = "channels_last"))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.summary()

    return model



def model_2d(params):

    activ = lambda : Activation('relu')
    model = Sequential()

    nclasses = params["n_classes"]


    model.add(Conv2D(32,(3,3),padding='same',data_format='channels_last', activation="relu",input_shape=(40, 40, 1)))
    model.add(Conv2D(32,(3,3),padding='same',data_format='channels_last', activation="relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,(3,3),padding='same',data_format='channels_last', activation="relu"))
    model.add(Conv2D(32,(3,3),padding='same',data_format='channels_last', activation="relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,(3,3),padding='same',data_format='channels_last', activation="relu"))
    model.add(Conv2D(32,(3,3),padding='same',data_format='channels_last', activation="relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    
    model.add(Dense(400,activation="relu"))
    model.add(Dense(200,activation="relu"))
    model.add(Dense(100,activation="relu"))
    model.add(Dense(50,activation="relu"))
    model.add(Dense(10,activation="relu"))

    model.add(Dense(nclasses))
    model.add(Activation('softmax'))

    return model

#
# FCN
#

def model_fcn(params):

    activ = lambda : Activation('relu')
    model = Sequential()
    
    model.add(Flatten(input_shape=(4,params["n_constit"])))

    #model.add(Dense(320, activation='relu'))
    #model.add(Dense(320, activation='relu'))
    #model.add(Dense(160, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(params["n_classes"]))
    model.add(Activation('softmax'))

    return model

#
# LoLa
#

def model_lola(params):

    model = Sequential()

    model.add(CoLa(input_shape = (params["n_features"], params["n_constit"]),
                   add_total   = False,
                   add_eye     = True,
                   debug       =  False,
                   n_out_particles = 10))

    model.add(LoLa( 
        #input_shape = (params["n_features"], params["n_constit"]),
        train_metric = False,
        es  = 1,
        xs  = 1,
        ys  = 1,
        zs  = 1,                 
        cs  = 0, 
        vxs = 0, 
        vys = 0,
        vzs = 0,        
        ms  = 1,                 
        pts = 0,                 
        n_train_es  = 0,
        n_train_ms  = 0,
        n_train_pts = 0,        
        n_train_sum_dijs   = 0,
        n_train_min_dijs   = 0))

    model.add(Flatten())


    model.add(Dense(400))
    model.add(Activation('relu'))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(params["n_classes"], activation='softmax'))

    return model
