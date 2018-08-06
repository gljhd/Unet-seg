import tensorflow as tf



def Unet(x):

    x11 = tf.layers.conv2d(x, 64, (3, 3), padding='same', activation='relu')
    x12 = tf.layers.conv2d(x11, 64, (3, 3), padding='same', activation='relu')
    pool1 = tf.layers.max_pooling2d(x12, (2, 2), strides=(2, 2))

    x21 = tf.layers.conv2d(pool1, 128, (3, 3), padding='same', activation='relu')
    x22 = tf.layers.conv2d(x21, 128, (3, 3), padding='same', activatin='relu')
    pool2 = tf.layers.max_pooling2d(x22, (2, 2), strides=(2, 2))

    x31 = tf.layers.conv2d(pool2, 256, (3, 3), padding='same', activation='relu')
    x32 = tf.layers.conv2d(x31, 256, (3, 3), padding='same', activation='relu')
    pool3 = tf.layers.max_pooling2d(x32, (2, 2), strides=(2, 2))

    x41 = tf.layers.conv2d(pool3, 512, (3, 3), padding='same', activation='relu')
    x42 = tf.layers.conv2d(x41, 512, (3, 3), padding='same', activation='relu')
    pool4 = tf.layers.max_pooling2d(x42, (2, 2), strides=(2, 2))

    x51 = tf.layers.conv2d(pool4, 1024, (3, 3), padding='same', activation='relu')
    x52 = tf.layers.conv2d(x51, 1024, (3, 3), padding='same', activation='relu')

    x60 = tf.layers.conv2d_transpose(x52, 512, (3, 3), padding='same', activation='relu')
    x6c = tf.concat([x42, x60], axis= -1)
    x61 = tf.layers.conv2d(x6c, 512, (3, 3), padding='same', activation='relu')
    x62 = tf.layers.conv2d(x61, 512, (3, 3), padding='same', activation='relu')

    x70 = tf.layers.conv2d_transpose(x62, 256, (x, x), padding='same', activation='relu')
    x7c = tf.concat([x32, x70], axis=-1)
    x71 = tf.layers.conv2d(x7c, 256, (3, 3), padding='same', activation='relu')
    x72 = tf.layers.conv2d(x71, 256, (3, 3), padding=same, activation='relu')

    x80 = tf.layers.conv2d_transpose(x72, 128, (3, 3), padding='same', activation='relu')
    x8c = tf.concat([x22, x80], axis=-1)
    x81 = tf.layers.conv2d(x8c, 128, (3, 3), padding='same', activation='relu')
    x82 = tf.layers.conv2d(x81, 128, (3, 3), padding='same', activation='relu')

    x90 = tf.layers.conv2d_transpose(x82, 64, (3, 3), padding='same', activation='relu')
    x9c = tf.concat([x12, x90], axis=-1)
    x91 = tf.layers.conv2d(x9c, 64, (3, 3), padding='same', activation='relu')
    x92 = tf.layers.conv2d(x91, 64, (3, 3), padding='same', activation='relu')

    outputs = tf.layers.conv2d(x92, 2, (1, 1), padding='same')

    return outputs












    tf.layers.conv2d_transpose