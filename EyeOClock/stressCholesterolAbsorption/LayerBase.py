from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D, UpSampling2D, Cropping2D
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, Dense, AveragePooling2D
from tensorflow.keras.layers import LeakyReLU, ELU, Conv2DTranspose
from tensorflow.keras import backend as K
from tensorflow.keras.backend import resize_images
from tensorflow.keras.layers import Lambda

from EyeOClock.stressCholesterolAbsorption.instance_normalization import InstanceNormalization

class LayerBase:
    def activation(self, x, type='relu', name='act'):
        if type not in ['relu', 'relu6', 'h-swish', 'h-sigmoid', 'sigmoid', 'softmax', 'tanh', 'leaky-relu', 'elu']:
            raise NotImplementedError(type + ' type is not available. ex) relu, relu6, h-swish, h-sigmoid, sigmoid')

        func = {
            'relu': Activation('relu', name=name+'_relu')(x),
            'relu6': Activation(self.relu6, name=name+'_relu6')(x),
            'h-swish': Activation(self.hardSwish, name=name+'_h-swish')(x),
            'sigmoid': Activation('sigmoid', name=name+'_sigmoid')(x),
            'hard-sigmoid': Activation('hard_sigmoid', name=name+'_hard-sigmoid')(x),
            'softmax': Activation('softmax', name=name+'_softmax')(x),
            'tanh': Activation('tanh', name=name+'_tanh')(x),
            'leaky-relu': LeakyReLU(alpha=0.01, name=name+'_leaky-relu')(x),
            'elu': ELU(alpha=1.0, name=name+'_elu')(x)
        }

        return func.get(type)

    def relu6(self, x):
        return K.relu(x, max_value=6)

    def hardSwish(self, x):
        return x * K.relu(x + 3, max_value=6) / 6

    def conv2d_transpose(self, x, filters, kernel=3, strides=1, padding=True, act=None, name='Conv2DTranspose'):
        x = Conv2DTranspose(filters=filters, kernel_size=kernel, strides=strides, padding='same' if padding else 'valid', activation=act, name=name)(x)

        return x

    def conv2d(self, x, filters, kernel=3, strides=1, padding=True, dilationRate=1, act=None, useBias=False, name='conv2d'):
        x = Conv2D(filters, kernel_size=kernel, strides=strides, padding='same' if padding else 'valid', dilation_rate=dilationRate, use_bias=useBias, activation=act, name=name)(x)

        return x

    def depthConv2d(self, x, kernel=3, strides=1, padding=True, depthMultiplier=1, act=None, useBias=False, name='depthConv2d'):
        x = DepthwiseConv2D(kernel_size=kernel, strides=strides, padding='same' if padding else 'valid', depth_multiplier=depthMultiplier, activation=act, use_bias=useBias, name=name)(x)

        return x

    def depthSepConv2d(self, x, filters, kernel=3, strides=1, padding=True, dilationRate=1, depthMultiplier=1, act='relu', useBias=False, name='depthSepConv2d'):
        x = self.depthConv2d(x, kernel=kernel, strides=strides, padding=padding, depthMultiplier=depthMultiplier, useBias=useBias, name=name + '_depthConv2d')
        x = self.activation(x, type=act, name=name + '_act1')
        x = self.conv2d(x, filters, kernel=1, strides=1, padding=padding, dilationRate=dilationRate, useBias=useBias, name=name + '_pointConv2d')
        x = self.activation(x, type=act, name=name + '_act2')

        return x

    def normalization(self, x, type='bn', name='norm'):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        func = {
            'bn': BatchNormalization(axis=channel_axis, name=name + '_bn')(x),
            'in': InstanceNormalization(axis=channel_axis, name=name + '_in')(x)
            # 'gn': tfa.layers.GroupNormalization(groups=4, axis=channel_axis, name=name + '_gn')(x)
        }

        return func.get(type)

    def normConv2d(self, x, filters, kernel=3, strides=1, padding=True, dilationRate=1, act='relu', useBias=False, normType='bn', name='normConv2d'):
        x = self.conv2d(x, filters, kernel=kernel, strides=strides, padding=padding, dilationRate=dilationRate, useBias=useBias, name=name+'_conv2d')
        x = self.normalization(x, type=normType, name=name+'_norm')
        x = self.activation(x, type=act, name=name+'_act')

        return x

    def normDepthSepConv2d(self, x, filters, kernel=3, strides=1, padding=True, dilationRate=1, depthMultiplier=1, act='relu', useBias=False, normType='bn', name='normDepthSepConv2d'):
        x = self.depthConv2d(x, kernel=kernel, strides=strides, padding=padding, depthMultiplier=depthMultiplier, useBias=useBias, name=name+'_depthConv2d')
        x = self.normalization(x, type=normType, name=name+'_norm1')
        x = self.activation(x, type=act, name=name+'_act1')
        x = self.conv2d(x, filters, kernel=1, strides=1, padding=padding, dilationRate=dilationRate, useBias=useBias, name=name+'_pointConv2d')
        x = self.normalization(x, type=normType, name=name + '_norm2')
        x = self.activation(x, type=act, name=name + '_act2')

        return x

    def maxPoll2d(self, x, poolSize=2, strides=2, padding=False, name='maxPool2d'):
        return MaxPooling2D(pool_size=poolSize, strides=strides, padding='same', name=name)(x) if padding else MaxPooling2D(pool_size=poolSize, strides=strides, padding='valid', name=name)(x)

    def ap2d(self, x, pool_size=2, strides=2, padding='same', name='ap2d'):
        return AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name)(x)

    def gap2d(self, x, name='gap2d'):
        return GlobalAveragePooling2D(name=name)(x)

    def gmp2d(self, x, name='gmp2d'):
        return GlobalMaxPooling2D(name=name)(x)

    def dense(self, x, units=2, act='softmax', useBias=False, name='dense'):
        return Dense(units=units, activation=act, use_bias=useBias, name=name)(x)

    def upSampling2d(self, x, size=2, data_format='channels_last', interpolation='nearest', name='upsampling2d'):
        return UpSampling2D(size=size, data_format=data_format, interpolation=interpolation, name=name)(x)

    def cropping2d(self, x, cropping=(2, 2), data_format='channels_last', name='cropping2d'):
        return Cropping2D(cropping=cropping, data_format=data_format, name=name)(x)

    def resizeImage(self, x, height_factor, width_factor, interpolation='nearest', data_format='channels_last'):
        return Lambda(lambda x: resize_images(x, height_factor=height_factor, width_factor=width_factor, interpolation=interpolation, data_format=data_format))(x)

    def resizeImageBilinear(self, x, height, width):
        return Lambda(lambda x: K.tf.image.resize_bilinear(x, (height, width), align_corners=True))(x)
