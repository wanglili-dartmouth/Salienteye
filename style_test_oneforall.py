from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
from keras.models import Model
from keras.applications import vgg19
from keras import backend as K
from keras.applications.vgg19 import VGG19



def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img




def gram_matrix(x):
    #print(x.shape)
    x.resize(x.shape[1]*x.shape[2],x.shape[3])
    #print(x.shape)
    grams = np.array((x.T).dot(x))
    #print(grams.shape)    
    return grams



def style_loss(style, combination):

    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return ((S - C) ** 2).sum().sum() / (4. * (channels ** 2) * (size ** 2))



img_nrows = 224
img_ncols = 224
train_len=100
test_len=50
valid_len=30
dataset=['thephotosociety_train','cats_of_instagram_train','travelalberta_train','clarklittle_train']
testset=['thephotosociety_test','cats_of_instagram_test','travelalberta_test','clarklittle_test']


total=np.zeros(len(dataset))
feature_layers = ['block1_pool','block4_pool']
layer_weight=[0.01,1]
base_model = VGG19(weights='imagenet')
model = [ Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output) for layer_name in feature_layers ]

for testi in range(len(testset)):
    total=np.zeros(len(dataset))
    for i in range(51,test_len+51):
        new_image = preprocess_image(testset[testi]+"//"+str(i)+".jpg")
        loss=np.zeros(len(dataset))
        for j in range(len(dataset)):
            heap=np.array([])
            for k in range(train_len):
                tmploss=0
                style_reference_image =preprocess_image(dataset[j]+"//"+str(101+k)+".jpg")
                #print(dataset[j]+"//"+str(101+k)+".jpg")
                for ii in range(len(feature_layers)):
                    
                    new_image_features=model[ii].predict(new_image)
                    style_reference_features=model[ii].predict(style_reference_image)
                    tmploss+= style_loss(new_image_features, style_reference_features)*layer_weight[ii]
                    #print(ii," ",tmploss)
                heap=np.insert(heap,len(heap),tmploss)
            heap=np.sort(heap)
            loss[j]=np.sum(heap[0:valid_len])
        total[np.argmin(loss)]+=1

    print("---------------------------------------------------")
    print(total)
