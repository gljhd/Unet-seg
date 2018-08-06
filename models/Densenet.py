import tensorflow as tf


def dense_block(x, blocks, name):
    """A dense block.

        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
    with tf.variable_scope(name):
        for i in range(blocks):
            x = conv_block(x, 32, name=str(i))
        return x


def


def Densenet():
