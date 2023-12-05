import tensorflow as tf

class Network:

# Build a physics informed neural network (PINN) model for the steady Navier-Stokes equations.

    def __init__(self):
       
        self.activations = {
            'swish': self.swish,
        }

    def swish(self, x):
        
# Swish activation function.
    
        return x * tf.math.sigmoid(x)
    

    def build(self, num_inputs=2, layers=[32, 16, 16, 32], activation='swish', num_outputs=2):
        """
        Build a PINN model for the steady Navier-Stokes equation with input shape (x,y) and output shape (psi, p).

        Args:
            num_inputs: number of input variables. Default is 2 for (x, y).
            layers: number of hidden layers.
            activation: activation function in hidden layers.
            num_outpus: number of output variables. Default is 2 for (psi, p).

        Returns:
            keras network model
        """

        # input layer
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        # hidden layers
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=self.activations[activation],
                kernel_initializer='glorot_normal')(x)
        # output layer
        outputs = tf.keras.layers.Dense(num_outputs,
            kernel_initializer='glorot_normal')(x)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    
