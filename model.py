import tensorflow as tf


def get_model():
    inputs = tf.keras.Input(shape=(128, 128, 3))

    vgg16_blocks = tf.keras.applications.VGG16(input_tensor=inputs,
                                               include_top=False,
                                               weights='imagenet')
    outputs = vgg16_blocks.output
    for i in range(1, 3):
        outputs = tf.keras.layers.Conv2D(filters=256,
                                         kernel_size=3,
                                         padding='same',
                                         name='block6_conv' + str(i),
                                         activation='relu')(outputs)

    # outputs = tf.keras.layers.MaxPooling2D(name='block6_pool')(outputs)
    # outputs = tf.keras.layers.Conv2D(filters=512,
    #                                  kernel_size=3,
    #                                  padding='same',
    #                                  strides=2,
    #                                  activation='relu')(outputs)
    outputs = tf.keras.layers.Flatten(name='flatten')(outputs)
    outputs = tf.keras.layers.Dense(name='predictions',
                                    units=2)(outputs)
    outputs = tf.keras.layers.Softmax()(outputs)
    m = tf.keras.Model(inputs=inputs, outputs=outputs, name='binary_vgg16')
    return m


def get_small_model(filters_size, l2):
    l2_norm = tf.keras.regularizers.L2(l2=l2)
    inputs = tf.keras.Input(shape=(128, 128, 3))
    after_first_filters_size = filters_size[1:]
    outputs = tf.keras.layers.Conv2D(filters=filters_size[0],
                                     kernel_size=3,
                                     strides=2,
                                     padding='same',
                                     activation='relu',
                                     kernel_regularizer=l2_norm,
                                     bias_regularizer=l2_norm)(inputs)
    for filter_ in after_first_filters_size:
        outputs = tf.keras.layers.Conv2D(filters=filter_,
                                         kernel_size=3,
                                         strides=2,
                                         padding='same',
                                         activation='relu',
                                         kernel_regularizer=l2_norm,
                                         bias_regularizer=l2_norm)(outputs)

    outputs = tf.keras.layers.Flatten()(outputs)
    outputs = tf.keras.layers.Dense(units=2,
                                    kernel_regularizer=l2_norm,
                                    bias_regularizer=l2_norm)(outputs)
    outputs = tf.keras.layers.Softmax()(outputs)

    m = tf.keras.Model(inputs=inputs, outputs=outputs, name='binary_small_model')
    return m

# m_ = get_model()
# m_.summary()
