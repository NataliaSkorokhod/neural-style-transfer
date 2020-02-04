from __future__ import print_function
import tensorflow as tf
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
from keras.applications import vgg19
from keras import backend as K

# VGG19 summary for reference:

"""_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 3, 224, 224)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 64, 224, 224)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 64, 224, 224)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 64, 112, 112)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 128, 112, 112)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 128, 112, 112)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 128, 56, 56)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 256, 56, 56)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 256, 56, 56)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 256, 56, 56)       590080    
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 256, 56, 56)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 256, 28, 28)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 512, 28, 28)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 512, 28, 28)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 512, 28, 28)       2359808   
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 512, 28, 28)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 512, 14, 14)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 512, 14, 14)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 512, 14, 14)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 512, 14, 14)       2359808   
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 512, 14, 14)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 512, 7, 7)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000   
"""

base_image_path = 'sakura.JPG'
style_reference_image_path1 = 'watercolor1.jpg'
style_reference_image_path2 = 'watercolor2.jpg'
style_reference_image_path3 = 'watercolor3.jpg'

style_coeff = np.array([0.5, 0.4, 0.1]) # weights given to each of the style reference paintings

iterations = 200

# these are the weights of the different loss components
style_weight = 0.5            # style image contribution to created image
content_weight = 0.5          # content image contribution to created image

# dimensions of the generated picture.
width, height = load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

# util function to open, resize and format pictures into appropriate tensors

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# util function to convert a tensor into a valid image

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel. This depends on the NN training set, in this case ImageNet
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# get tensor representations of our images
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image1 = K.variable(preprocess_image(style_reference_image_path1))
style_reference_image2 = K.variable(preprocess_image(style_reference_image_path2))
style_reference_image3 = K.variable(preprocess_image(style_reference_image_path3))

# this will contain our generated image.
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

# combine the 3 images into a single Keras tensor
input_tensor = K.concatenate([base_image,
                              style_reference_image1,
                              style_reference_image2,
                              style_reference_image3,
                              combination_image], axis=0)

# build the VGG19 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
print('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])


# the gram matrix of an image tensor (feature-wise outer product)

def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1))) # turns it into channels_first and then flattens the rest
    gram = K.dot(features, K.transpose(features))
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image

def content_loss(base, combination):
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(combination - base)) / (4.0 * channels * size)

# combine these loss functions

loss = K.variable(0.0)
layer_features = outputs_dict['block4_conv4']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[-1, :, :, :]
loss = loss + content_weight * content_loss(base_image_features,
                                            combination_features)

feature_layers = [
    ('block4_conv1', 0.2),
    ('block4_conv2', 0.3),
    ('block3_conv1', 0.5)]

for layer_name, layer_coeff in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1:-1, :, :, :]
    combination_features = layer_features[-1, :, :, :]

    for i in range(0, 3):
        loss = loss + style_weight * layer_coeff * style_coeff[i] * style_loss(style_reference_features[i], combination_features)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)


def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
x = preprocess_image(base_image_path)

name_prefix = 'output/created_image'

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                    fprime=evaluator.grads, maxfun=20)

    print('Current loss value:', min_val)

    # save generated image every 10 iterations
    if i % 10 == 0:
        img = deprocess_image(x.copy())
        fname = name_prefix + '_at_iteration_%d.png' % i
        save_img(fname, img)
        print('Image saved at', fname)

    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))