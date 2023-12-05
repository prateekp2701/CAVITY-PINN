import tensorflow as tf
from layer import GradientLayer

class PINN:
    
#Build a physics informed neural network (PINN) model for the steady Navier-Stokes equation.

    def __init__(self, network, rho=1, nu=0.01):
        """
        Args:
            network: keras network model with input (x, y) and output (psi, p).
            rho: density.
            nu: viscosity.
        """

        self.network = network
        self.rho = rho
        self.nu = nu
        self.grads = GradientLayer(self.network)

    def build(self):
        """
        Build a PINN model for the steady Navier-Stokes equation.

        Returns:
            PINN model for the steady Navier-Stokes equation with
                input: [ (x, y) relative to equation,
                         (x, y) relative to boundary condition ],
                output: [ (u, v) relative to equation (must be zero),
                          (psi, psi) relative to boundary condition (psi is duplicated because outputs require the same dimensions),
                          (u, v) relative to boundary condition ]
        """

        # equation input: (x, y)
        xy_e = tf.keras.layers.Input(shape=(2,))
        # boundary condition
        xy_b = tf.keras.layers.Input(shape=(2,))

        # compute gradients relative to equation
        _, p_grads, u_grads, v_grads = self.grads(xy_e)
        _, p_x, p_y = p_grads
        u, u_x, u_y, u_xx, u_yy = u_grads
        v, v_x, v_y, v_xx, v_yy = v_grads
        # compute equation loss
        u_e = u*u_x + v*u_y + p_x/self.rho - self.nu*(u_xx + u_yy)
        v_e = u*v_x + v*v_y + p_y/self.rho - self.nu*(v_xx + v_yy)
        uv_e = tf.concat([u_e, v_e], axis=-1)

        # compute gradients relative to boundary condition
        psi_b, _, u_grads_b, v_grads_b = self.grads(xy_b)
        # compute boundary condition loss
        psi_b = tf.concat([psi_b, psi_b], axis=-1)
        uv_b = tf.concat([u_grads_b[0], v_grads_b[0]], axis=-1)

        # build the PINN model for the steady Navier-Stokes equation
        return tf.keras.models.Model(
            inputs=[xy_e, xy_b], outputs=[uv_e, psi_b, uv_b])
