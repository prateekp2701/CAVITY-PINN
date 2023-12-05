import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):

# Custom layer to compute derivatives for the steady Navier-Stokes equation.


    def __init__(self, model, **kwargs):

        self.model = model
        super().__init__(**kwargs)

    def call(self, xy):
        """
        Computing derivatives for loss calculation

        Args:
            xy: input variable.

        Returns:
            psi: stream function.
            p_grads: pressure and its gradients.
            u_grads: u and its gradients.
            v_grads: v and its gradients.
        """

        x, y = [ xy[..., i, tf.newaxis] for i in range(xy.shape[-1]) ]
        with tf.GradientTape(persistent=True) as g3:
            g3.watch(x)
            g3.watch(y)
            with tf.GradientTape(persistent=True) as g2:
                g2.watch(x)
                g2.watch(y)
                with tf.GradientTape(persistent=True) as g:
                    g.watch(x)
                    g.watch(y)
                    psi_p = self.model(tf.concat([x, y], axis=-1))
                    psi = psi_p[..., 0, tf.newaxis]
                    p   = psi_p[..., 1, tf.newaxis]
                u   =  g.batch_jacobian(psi, y)[..., 0]
                v   = -g.batch_jacobian(psi, x)[..., 0]
                p_x =  g.batch_jacobian(p,   x)[..., 0]
                p_y =  g.batch_jacobian(p,   y)[..., 0]
                del g
            u_x = g2.batch_jacobian(u, x)[..., 0]
            u_y = g2.batch_jacobian(u, y)[..., 0]
            v_x = g2.batch_jacobian(v, x)[..., 0]
            v_y = g2.batch_jacobian(v, y)[..., 0]
            del g2
        u_xx = g3.batch_jacobian(u_x, x)[..., 0]
        u_yy = g3.batch_jacobian(u_y, y)[..., 0]
        v_xx = g3.batch_jacobian(v_x, x)[..., 0]
        v_yy = g3.batch_jacobian(v_y, y)[..., 0]
        del g3

        p_grads = p, p_x, p_y
        u_grads = u, u_x, u_y, u_xx, u_yy
        v_grads = v, v_x, v_y, v_xx, v_yy

        return psi, p_grads, u_grads, v_grads
