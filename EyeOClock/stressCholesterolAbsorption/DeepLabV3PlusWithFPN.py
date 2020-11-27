from tensorflow.keras.layers import concatenate, Add, Multiply, Lambda, Reshape
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from EyeOClock.stressCholesterolAbsorption.LayerBase import LayerBase
# from model.ResNest import ResNest
# from config import flags
# from Regularization.DropBlock import DropBlock2D
#
# class ResNest(LayerBase):
#
#     def __init__(self, inputShape):
#         self.inputShape = inputShape
#
#     def model(self):
#         self.inputs = Input(shape=self.inputShape)
#         x = self.conv2d(self.inputs, 16, name='resnestConv1')
#         x = self.normalization(x, name='resnestNorm1')
#         x = self.activation(x, type='elu', name='resnestElu1')
#         low_feature = self.ap2d(x, pool_size=3, name='resnestAvgPool1')
#
#         x = self.se_resnest_block(low_feature, 32, name='resnest_block1')
#         x = self.se_resnest_block(x, 32, name='resnest_block2')
#         mid_feature = self.ap2d(x, pool_size=3, name='resnestAvgPool2')
#
#         x = self.se_resnest_block(mid_feature, 64, name='resnest_block3')
#         x = self.se_resnest_block(x, 64, name='resnest_block4')
#         x = DropBlock2D(block_size=5, keep_prob=0.75)(x)
#         high_feature = self.ap2d(x, pool_size=3, name='resnestAvgPool3')
#
#         x = self.se_resnest_block(high_feature, 96, name='resnest_block5')
#         x = self.se_resnest_block(x, 96, name='resnest_block6')
#         x = DropBlock2D(block_size=5, keep_prob=0.75)(x)
#         x = self.ap2d(x, pool_size=3, name='resnestAvgPool4')
#
#         x = self.resnest_block(x, 128, name='resnest_block7')
#         # x = self.ap2d(x, pool_size=3, name='resnestAvgPool5')
#
#         # [n, h, w, c] = x.shape
#         # x = Reshape([h*w*c])(x)
#         # x = self.dense(x, units=flags.classes, name='resnest_dense')
#
#         return x, low_feature, mid_feature, high_feature
#
#     def resnest_block(self, x, outChannel, name='resnest_block'):
#         convtmp = self.conv2d(x, outChannel, kernel=flags.ksize, name=name+'_Conv1')
#         convtmp = self.normalization(convtmp, name=name+'_Norm1')
#         convtmp = self.activation(convtmp, type='elu', name=name+'_Elu1')
#         convtmp = self.conv2d(convtmp, outChannel, kernel=flags.ksize, name=name+'_Conv2')
#         convtmp = self.normalization(convtmp, name=name+'_Norm2')
#         convtmp = self.activation(convtmp, type='elu', name=name+'_Elu2')
#
#         concats = None
#         for idx_k in range(flags.kpaths):
#             cardinal = self.cardinal(convtmp, outChannel=outChannel//2, name=name+'_Card_'+str(idx_k))
#             if (idx_k == 0): concats = cardinal
#             else: concats = concatenate([concats, cardinal], axis=3)
#
#         concats = self.conv2d(concats, outChannel, kernel=1, name=name+'_ConvCC')
#         concats = concats + convtmp
#
#         if (x.shape[-1] != concats.shape[-1]):
#             x = self.conv2d(x, outChannel, kernel=1, name=name+'_Conv3')
#             x = self.normalization(x, name=name+'_Norm3')
#             x = self.activation(x, type='elu', name=name+'_Elu3')
#
#         return concats + x
#
#     def se_resnest_block(self, x, outChannel, name='se_resnest_block'):
#         primeChannel = outChannel // 2
#         doublePrimeChannel = outChannel // 4
#
#         convtmp = self.conv2d(x, outChannel, kernel=flags.ksize, name=name + '_Conv1')
#         convtmp = self.normalization(convtmp, name=name + '_Norm1')
#         convtmp = self.activation(convtmp, type='elu', name=name + '_Elu1')
#         convtmp = self.conv2d(convtmp, outChannel, kernel=flags.ksize, name=name + '_Conv2')
#         convtmp = self.normalization(convtmp, name=name + '_Norm2')
#         convtmp = self.activation(convtmp, type='elu', name=name + '_Elu2')
#
#         splits = []
#         for idx_r in range(flags.radix):
#             splits.append(self.split(x, outChannel=primeChannel, name=name+'_split_'+str(idx_r)))
#
#         poolings = []
#         for idx_r in range(flags.radix):
#             poolings.append(self.gap2d(splits[idx_r], name=name+'_gap_'+str(idx_r)))
#
#         for idx_r in range(flags.radix):
#             r_x = self.dense(poolings[idx_r], units=doublePrimeChannel, act=None, name=name + '_R_dense_'+str(idx_r))
#             r_x = Lambda(lambda x: K.expand_dims(x, 1))(r_x)
#             r_x = Lambda(lambda x: K.expand_dims(x, 1))(r_x)
#             r_x = self.normalization(r_x, name=name + '_R_Norm_'+str(idx_r))
#             r_x = self.activation(r_x, type='relu', name=name + '_R_relu_'+str(idx_r))
#
#             r_x = Reshape([doublePrimeChannel])(r_x)
#             r_x = self.dense(r_x, units=primeChannel*flags.radix, act=None, name=name + '_R_dense_D_' + str(idx_r))
#             r_x = Lambda(lambda x: K.expand_dims(x, 1))(r_x)
#             r_x = Lambda(lambda x: K.expand_dims(x, 1))(r_x)
#             r_x = self.normalization(r_x, name=name + '_R_Norm_D_' + str(idx_r))
#             r_x = self.activation(r_x, type='relu', name=name + '_R_relu_D_' + str(idx_r))
#
#             r_x = Reshape([primeChannel*flags.radix])(r_x)
#             if (flags.radix == 1): poolings[idx_r] = self.dense(r_x, units=primeChannel, act='sigmoid', name=name + '_rS_dense_' + str(idx_r))
#             elif (flags.radix > 1): poolings[idx_r] = self.dense(r_x, units=primeChannel, act='softmax', name=name+'_rS_dense_'+str(idx_r))
#
#         output = None
#         for split, pooling in zip(splits, poolings):
#             if output is None: output = Multiply()([split, pooling])
#             else: output = Add()([output, Multiply()([split, pooling])])
#
#         output = self.conv2d(output, outChannel, kernel=1, name=name + '_Conv_Output')
#         output = Add()([output, convtmp])
#
#         x = self.se_block(x, outChannel, ratio=16, name=name+'_seBlock')
#
#         return Add()([output, x])
#
#     def se_block(self, x, outChannel, ratio, name='se_block'):
#         squeeze = self.gap2d(x, name=name+'_Gap')
#
#         excitation = self.dense(squeeze, units=outChannel / ratio, act='relu', name=name+'_Dense1')
#         excitation = self.dense(excitation, units=outChannel, act='sigmoid', name=name+'_Dense2')
#
#         excitation = Lambda(lambda x: K.expand_dims(x, 1))(excitation)
#         excitation = Lambda(lambda x: K.expand_dims(x, 1))(excitation)
#
#         if x.shape[-1] != outChannel:
#             x = self.conv2d(x, outChannel, kernel=1, name=name + '_SE')
#             x = self.normalization(x, name=name + '_SE')
#             x = self.activation(x, type='elu', name=name + '_SE')
#
#         scale = Multiply()([x, excitation])
#
#         return scale
#
#     def split(self, x, outChannel, name="split"):
#         cardinals = None
#         for idx_k in range(flags.kpaths):
#             cardinal = self.cardinal(x, outChannel, name=name+"_cardinal_"+str(idx_k))
#             if idx_k == 0: cardinals = cardinal
#             else: cardinals = concatenate([cardinals, cardinal])
#
#         return cardinals
#
#     def cardinal(self, x, outChannel, name='cardinal'):
#         outChannel_conv11 = int(outChannel / flags.radix / flags.kpaths)
#         outChannel_convkk = int(outChannel / flags.kpaths)
#
#         conv11 = self.conv2d(x, outChannel_conv11, kernel=1, name=name + '_Conv11')
#         conv11 = self.normalization(conv11, name=name + '_Norm11')
#         conv11 = self.activation(conv11, type='elu', name=name + '_Elu11')
#
#         convkk = self.conv2d(conv11, outChannel_convkk, kernel=flags.ksize, name=name + '_Convkk')
#         convkk = self.normalization(convkk, name=name + '_Normkk')
#         convkk = self.activation(convkk, type='elu', name=name + '_Elukk')
#
#         return convkk
#
#     def split_attention(self, x, inChannel, name='split_attention'):
#         i_holder = None
#         for idx_s, split in enumerate(x):
#             if (idx_s == 0): i_holder = split
#             else: i_holder += split
#
#         # i_x = self.gap2d(i_holder, name=name+'_I_gap2d')
#         # i_x = Lambda(lambda x: K.expand_dims(x, 1))(i_x)
#         # i_x = Lambda(lambda x: K.expand_dims(x, 1))(i_x)
#         #
#         # i_x = self.conv2d(i_x, inChannel//2, kernel=1, name=name+'_I_Conv11')
#         # i_x = self.normalization(i_x, name=name+'_I_Norm')
#         # i_x = self.activation(i_x, type='elu', name=name+'_I_Elu')
#         #
#         # o_holder = None
#         # for idx_s in range(len(x)):
#         #     o_x = self.conv2d(i_x, inChannel, kernel=1, name=name+'_O_Conv11_'+str(idx_s))
#         #
#         #     if (flags.radix == 1): o_x = self.activation(o_x, type='sigmoid', name=name+'_O_Sigmoid_'+str(idx_s))
#         #     elif (flags.radix > 1): o_x = self.activation(o_x, type='softmax', name=name+'_O_Softmax_'+str(idx_s))
#         #
#         #     if (idx_s == 0): o_holder = x[idx_s] * o_x
#         #     else: o_holder += x[idx_s] * o_x
#         i_x = self.gap2d(i_holder, name=name+'_I_gap2d')
#
#         i_x = self.dense(i_x, units=inChannel//2, act=None, name=name+'_I_dense')
#         i_x = Lambda(lambda x: K.expand_dims(x, 1))(i_x)
#         i_x = Lambda(lambda x: K.expand_dims(x, 1))(i_x)
#         i_x = self.normalization(i_x, name=name+'_I_Norm')
#         i_x = self.activation(i_x, type='relu', name=name+'_I_relu')
#
#         o_x_list = []
#         for idx_s in range(len(x)):
#             r_i_x = Reshape([inChannel // 2])(i_x)
#             if (flags.radix == 1): o_x = self.dense(r_i_x, units=inChannel, act='sigmoid', name=name + '_O_dense_' + str(idx_s))
#             elif (flags.radix > 1): o_x = self.dense(r_i_x, units=inChannel, act='softmax', name=name+'_O_dense_'+str(idx_s))
#             o_x = Lambda(lambda x: K.expand_dims(x, 1))(o_x)
#             o_x = Lambda(lambda x: K.expand_dims(x, 1))(o_x)
#             o_x_list.append(o_x)
#
#         o_holder = None
#         i = 0
#         for input, o_x in zip(x, o_x_list):
#             if (i == 0): o_holder = input * o_x
#             else: o_holder += input * o_x
#
#         return o_holder

class DeepLabV3PlusWithFPN(LayerBase):

    def __init__(self, inputShape=(480, 640, 3)):
        self.inputShape = inputShape
        self.dimRatio = 4  # todo default: 8
        self.dim = [4, 6, 8, 16, 32, 91, 128, 198, 256]

    def atrous_spatial_pyramid_pooling(self, x, depth=256):
        input_shape = K.int_shape(x)[1:3]
        atrous_rates = [2, 4, 6]  # todo default: [6, 12, 18]

        # x_1x1 = self.conv2d(x, depth, kernel=1, name='aspp_conv_1x1')
        # x_3x3_1 = self.conv2d(x, depth, dilationRate=atrous_rates[0], padding=True, act='leaky-relu', name='aspp_conv_3x3_1')
        # x_3x3_2 = self.conv2d(x, depth, dilationRate=atrous_rates[1], padding=True, act='leaky-relu', name='aspp_conv_3x3_2')
        # x_3x3_3 = self.conv2d(x, depth, dilationRate=atrous_rates[2], padding=True, act='leaky-relu', name='aspp_conv_3x3_3')
        x_1x1 = self.normDepthSepConv2d(x, depth, kernel=1, name='aspp_conv_1x1')
        x_3x3_1 = self.normDepthSepConv2d(x, depth, dilationRate=atrous_rates[0], padding=True, act='leaky-relu', name='aspp_conv_3x3_1')
        x_3x3_2 = self.normDepthSepConv2d(x, depth, dilationRate=atrous_rates[1], padding=True, act='leaky-relu', name='aspp_conv_3x3_2')
        x_3x3_3 = self.normDepthSepConv2d(x, depth, dilationRate=atrous_rates[2], padding=True, act='leaky-relu', name='aspp_conv_3x3_3')

        x_pooling = self.ap2d(x, strides=1, name='aspp_ap')
        # x_pooling = Lambda(lambda x: K.expand_dims(x, 1))(x_pooling)
        # x_pooling = Lambda(lambda x: K.expand_dims(x, 1))(x_pooling)  #todo (b_size, channels) -> (b_size, 1, 1, channels)

        x_pooling = self.normDepthSepConv2d(x_pooling, depth, kernel=1, padding=True, act='leaky-relu', name='aspp_conv_pooling')
        # x_pooling = self.conv2d(x_pooling, depth, kernel=1, padding=True, act='leaky-relu', name='aspp_conv_pooling')
        # x_pooling = self.upSampling2d(x_pooling, size=(15, 20), interpolation='bilinear', name='aspp_upsampling2d')
        # x_pooling = self.resizeImageBilinear(x_pooling, input_shape[0], input_shape[1])

        x = concatenate([x_1x1, x_3x3_1, x_3x3_2, x_3x3_3, x_pooling], axis=-1, name='aspp_concat')
        x = self.normDepthSepConv2d(x, depth, kernel=1, act='leaky-relu', padding=True, name='aspp_output')

        return x

    # def segmentNet(self):
    #     #todo Encoder
    #     # base_net, low_features = self.xception()
    #     resnest = ResNest(inputShape=self.inputShape)
    #     base_net, low_feature, mid_feature, high_feature = resnest.model()
    #     encoder_output = self.atrous_spatial_pyramid_pooling(base_net, depth=int(self.dim[4]*self.dimRatio))
    #
    #     low_feature = self.conv2d(low_feature, int(self.dim[1]*self.dimRatio), kernel=1, padding=True, name='low_feature_conv2d')
    #     mid_feature = self.conv2d(mid_feature, int(self.dim[2]*self.dimRatio), kernel=1, padding=True, name='mid_feature_conv2d')
    #     high_feature = self.conv2d(high_feature, int(self.dim[3]*self.dimRatio), kernel=1, padding=True, name='high_feature_conv2d')
    #
    #     #todo Decoder
    #     # net = self.resizeImage(encoder_output, 4, 4)
    #     # net = self.upSampling2d(encoder_output, 4, interpolation='bilinear', name='decode_upsampling2d_1')
    #     net = self.resizeImage(encoder_output, 2, 2, interpolation='bilinear')
    #     net = concatenate([net, high_feature], axis=-1, name='low_concat')
    #     net = self.normDepthSepConv2d(net, int(self.dim[4]*self.dimRatio), act='leaky-relu', padding=True, name='decoder_conv2d1')
    #     net = self.resizeImage(net, 2, 2, interpolation='bilinear')
    #     net = concatenate([net, mid_feature], axis=-1, name='mid_concat')
    #     net = self.normDepthSepConv2d(net, int(self.dim[4] * self.dimRatio), act='leaky-relu', padding=True, name='decoder_conv2d2')
    #     net = self.resizeImage(net, 2, 2, interpolation='bilinear')
    #     net = concatenate([net, low_feature], axis=-1, name='high_concat')
    #
    #     net = self.conv2d(net, 1, kernel=1, padding=True, name='decoder_output')
    #     net = self.resizeImage(net, 2, 2, interpolation='bilinear')
    #     net = self.activation(net, type='sigmoid', name='decoder_logits')
    #
    #     model = Model(resnest.inputs, net, name='deeplabv3plus')
    #
    #     return model