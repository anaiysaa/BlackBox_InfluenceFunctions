import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

def _weights_init(shape, dtype=None):
    # Kaiming He initialization
    fan_in = shape[0] * shape[1] * shape[2] if len(shape) == 3 else shape[0]
    stddev = tf.sqrt(2. / fan_in)
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

class LambdaLayer(layers.Layer):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def call(self, x):
        return self.lambd(x)

class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(planes, kernel_size=3, strides=stride, 
                                  padding='same', use_bias=False,
                                  kernel_initializer=_weights_init)
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(planes, kernel_size=3, strides=1,
                                  padding='same', use_bias=False,
                                  kernel_initializer=_weights_init)
        self.bn2 = layers.BatchNormalization()

        self.shortcut = tf.keras.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x: 
                    tf.pad(x[:, :, ::2, ::2] if K.image_data_format() == 'channels_first' 
                          else x[:, ::2, ::2, :], 
                    [[0, 0], 
                     [planes//4, planes//4] if K.image_data_format() == 'channels_first' else [0, 0],
                     [0, 0], 
                     [0, 0] if K.image_data_format() == 'channels_first' else [planes//4, planes//4]]))
            elif option == 'B':
                self.shortcut = tf.keras.Sequential([
                    layers.Conv2D(self.expansion * planes, kernel_size=1, 
                                 strides=stride, use_bias=False,
                                 kernel_initializer=_weights_init),
                    layers.BatchNormalization()
                ])

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        
        shortcut_out = self.shortcut(x, training=training) if hasattr(self.shortcut, '__call__') else self.shortcut(x)
        out = out + shortcut_out
        out = tf.nn.relu(out)
        return out

class ResNet(Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = layers.Conv2D(16, kernel_size=3, strides=1, 
                                  padding='same', use_bias=False,
                                  kernel_initializer=_weights_init)
        self.bn1 = layers.BatchNormalization()
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.linear = layers.Dense(num_classes, kernel_initializer=_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers_list = []
        for stride in strides:
            layers_list.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return tf.keras.Sequential(layers_list)

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.avg_pool(out)
        out = self.linear(out)
        return out

# Model definitions
def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])

def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])

def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])

def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])

def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

def test(net):
    total_params = sum([tf.keras.backend.count_params(w) for w in net.trainable_weights])
    print("Total number of params", total_params)
    
    # Count layers (conv + linear layers with weights)
    num_layers = sum([1 for w in net.trainable_weights if len(w.shape) > 1])
    print("Total layers", num_layers)

def config():
    return resnet20()

# Example usage
if __name__ == "__main__":
    # Create model
    model = resnet20()
    
    # Build model with sample input
    model.build(input_shape=(None, 32, 32, 3))
    
    # Test with sample data
    sample_input = tf.random.normal((1, 32, 32, 3))
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
    
    # Test model summary and parameters
    model.summary()
    test(model)