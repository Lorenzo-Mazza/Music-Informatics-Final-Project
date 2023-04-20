import functools
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Add, Input, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.layers import GlobalAveragePooling1D,GlobalAveragePooling2D

BATCHNORM_L2 = 3e-4

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from original paper
    momentum=0.9,
    beta_regularizer=tf.keras.regularizers.l2(BATCHNORM_L2),
    gamma_regularizer=tf.keras.regularizers.l2(BATCHNORM_L2)
    )

l1_l2 = tf.keras.regularizers.l1_l2


def main_block(x, filters, n, strides, dropout, l1=0., l2=0.):
    # Normal part
    x_res = Conv2D(filters, (5, 5), strides=strides, padding="same", kernel_regularizer=l1_l2(l1=l1, l2=l2))(
        x)  # , kernel_regularizer=l2(5e-4)
    x_res = BatchNormalization()(x_res)
    x_res = Activation('elu')(x_res)
    x_res = Conv2D(filters, (5, 5), padding="same", kernel_regularizer=l1_l2(l1=l1, l2=l2))(x_res)
    # Alternative branch
    x = Conv2D(filters, (5, 5),padding="same", strides=strides, kernel_regularizer=l1_l2(l1=l1, l2=l2))(x)
    # Merge Branches
    x = Add()([x_res, x])

    for i in range(n - 1):
        # Residual connection
        x_res = BatchNormalization()(x)
        x_res = Activation('elu')(x_res)
        x_res = Conv2D(filters, (5, 5), padding="same", kernel_regularizer=l1_l2(l1=l1, l2=l2))(x_res)
        # Apply dropout if given
        if dropout: x_res = Dropout(dropout)(x)
        # Second part
        x_res = BatchNormalization()(x_res)
        x_res = Activation('elu')(x_res)
        x_res = Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l1_l2(l1=l1, l2=l2))(x_res)
        # Merge branches
        x = Add()([x, x_res])

    # Inter block part
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    return x


def build_wrn(input_dims, n, k, dropout=None, l1=0., l2=0., output_dim=24):
    """ Builds the model. Params:
            - n: number of layers. WRNs are of the form WRN-N-K
                 It must satisfy that (N-4)%6 = 0
            - k: Widening factor. WRNs are of the form WRN-N-K
                 It must satisfy that K%2 = 0
            - input_dims: input dimensions for the model
            - output_dim: output dimensions for the model
            - dropout: dropout rate - default=0 (not recomended >0.3)
            - act: activation function - default=relu. Build your custom
                   one with keras.backend (ex: swish, e-swish)
    """
    # Ensure n & k are correct
    assert (n - 4) % 6 == 0
    assert k % 2 == 0
    n = (n - 4) // 6
    # This returns a tensor input to the model
    inputs = Input(shape=(input_dims))
    x=tf.keras.layers.Reshape(input_dims+(1,))(inputs)

    scaled_l2 = l2
    scaled_l1 = l1
    kernel_regularizer = l1_l2(l1=scaled_l1, l2=scaled_l2)
    # Head of the model
    x = Conv2D(8, (5, 5), padding="same", kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # 3 Blocks (normal-residual)
    x = main_block(x, 8 * k, n, (1, 1), dropout, l2=l2, l1=l1)  # 0
    x = main_block(x, 16 * k, n, (2, 2), dropout, l2=l2, l1=l1)  # 1
    x = main_block(x, 32 * k, n, (2, 2), dropout, l2=l2, l1=l1)  # 2
    #x = BatchNormalization()(x)

    # Final part of the model
    #x = Flatten()(x)
    #x = tf.keras.layers.Reshape((input_dims[0], -1))(x)
    #x = GlobalAveragePooling1D()(x)

    #x=tf.keras.layers.Reshape((-1,48))(x)  # for 151
    x = GlobalAveragePooling2D()(x) # for 151
    x= Dense(48, activation='elu')(x)

    #x=GlobalAveragePooling2D()(x)
    outputs = Dense(output_dim,
                    activation='softmax',
                    kernel_regularizer=l1_l2(l1=scaled_l1, l2=scaled_l2),
                    bias_regularizer=l1_l2(l1=scaled_l1, l2=scaled_l2)
                    )(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
