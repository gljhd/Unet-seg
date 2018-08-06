import tensorflow as tf

def Res_Conv2d(x_, filters, filter_size, name, padding='same'):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x_, filters, filter_size, padding=padding)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters, filter_size, padding=padding)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.add(x_, x)
        return x

def block1(x, depth, filters, filter_size, name):
    with tf.variable_scope(name):
        for i in range(depth):
            x = Res_Conv2d(x, filters, filter_size=filter_size, name=str(i))
        x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2))
        return x

def Conv2d(inputs, filters, filter_size, name, padding='same'):
    with tf.variable_scope(name):
        output = tf.layers.conv2d(inputs, filters, filter_size, name=name, padding=padding)
        output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)
        return output

def ResUnet51(x):

    x1 = block1(x, 3, 64, (3, 3), 'block1')
    x2 = block1(x1, 4, 128, (3, 3), 'block2')
    x3 = block1(x2, 5, 256, (3, 3), 'block3')
    x4 = block1(x3, 6, 512, (3, 3), 'block4')

    x5 = Conv2d(x4, 1024, (3, 3), name='connect1')
    x5 = Conv2d(x5, 1024, (3, 3), name='connect2')
    x5 = tf.layers.conv2d_transpose(x5, 512, (3, 3), activation='relu', name='up1')

    x6 = tf.concat([x4, x5],axis=-1)
    x6 = block1(x6, 6, 512, (3, 3), name='block6')
    x6 = tf.layers.conv2d_transpose(x6, 256, (3, 3), activation='relu', name='up2')

    x7 = tf.concat([x3, x6], axis=-1)
    x7 = block1(x7, 5, 256, (3, 3), name='block7')
    x7 = tf.layers.conv2d_transpose(x7, 4, 128, (3, 3), padding='same', name='up3')

    x8 = tf.concat([x2, x7], axis=-1)
    x8 = block1(x8, 4, 128, (3, 3), name='block8')
    x8 = tf.layers.conv2d_transpose(x8, 64, (3, 3), activation='relu', name='up4')

    x9 = tf.concat([x1, x8], axis=-1)
    x9 = block1(x9, 3, 64, (3, 3), name='block9')

    outputs = tf.layers.conv2d(x9, 2, (1, 1), name='outputs', padding='same')

    return outputs