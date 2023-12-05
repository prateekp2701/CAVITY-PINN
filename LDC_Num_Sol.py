import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class LDC_Num_Sol:
    def __init__(self, RE):
        self.RE = RE
        self.N_POINTS = 100
        self.DOMAIN_SIZE = 1.0
        self.N_ITERATIONS = 10000
        self.TIME_STEP_LENGTH = 0.001
        self.KINEMATIC_VISCOSITY = 1.0/RE
        self.DENSITY = 1.0
        self.HORIZONTAL_VELOCITY_TOP = 1.0

        self.N_PRESSURE_POISSON_ITERATIONS = 50
        self.STABILITY_SAFETY_FACTOR = 0.5
        self.element_length = self.DOMAIN_SIZE / (self.N_POINTS - 1)
        self.x = np.linspace(0.0, self.DOMAIN_SIZE, self.N_POINTS)
        self.y = np.linspace(0.0, self.DOMAIN_SIZE, self.N_POINTS)

        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.u_prev = np.zeros_like(self.X)
        self.v_prev = np.zeros_like(self.X)
        self.p_prev = np.zeros_like(self.X)


    def central_difference_x(self,f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[1:-1, 2:  ]
            -
            f[1:-1, 0:-2]
        ) / (
            2 * self.element_length
        )
        return diff
    
    def central_difference_y(self,f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[2:  , 1:-1]
            -
            f[0:-2, 1:-1]
        ) / (
            2 * self.element_length
        )
        return diff
    
    def laplace(self,f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[1:-1, 0:-2]
            +
            f[0:-2, 1:-1]
            -
            4
            *
            f[1:-1, 1:-1]
            +
            f[1:-1, 2:  ]
            +
            f[2:  , 1:-1]
        ) / (
            self.element_length**2
        )
        return diff
    
    def solve(self):
        maximum_possible_time_step_length = (
            0.5 * self.element_length**2 / self.KINEMATIC_VISCOSITY
        )
        if self.TIME_STEP_LENGTH > self.STABILITY_SAFETY_FACTOR * maximum_possible_time_step_length:
            raise RuntimeError("Stability is not guarenteed")

        
        for _ in tqdm(range(self.N_ITERATIONS)):
            d_u_prev__d_x = self.central_difference_x(self.u_prev)
            d_u_prev__d_y = self.central_difference_y(self.u_prev)
            d_v_prev__d_x = self.central_difference_x(self.v_prev)
            d_v_prev__d_y = self.central_difference_y(self.v_prev)
            laplace__u_prev = self.laplace(self.u_prev)
            laplace__v_prev = self.laplace(self.v_prev)

            # Perform a tentative step by solving the momentum equation without the
            # pressure gradient
            u_tent = (
                self.u_prev
                +
                self.TIME_STEP_LENGTH * (
                    -
                    (
                        self.u_prev * d_u_prev__d_x
                        +
                        self.v_prev * d_u_prev__d_y
                    )
                    +
                    self.KINEMATIC_VISCOSITY * laplace__u_prev
                )
            )
            v_tent = (
                self.v_prev
                +
                self.TIME_STEP_LENGTH * (
                    -
                    (
                        self.u_prev * d_v_prev__d_x
                        +
                        self.v_prev * d_v_prev__d_y
                    )
                    +
                    self.KINEMATIC_VISCOSITY * laplace__v_prev
                )
            )

            # Velocity Boundary Conditions: Homogeneous Dirichlet BC everywhere
            # except for the horizontal velocity at the top, which is prescribed
            u_tent[0, :] = 0.0
            u_tent[:, 0] = 0.0
            u_tent[:, -1] = 0.0
            u_tent[-1, :] = self.HORIZONTAL_VELOCITY_TOP
            v_tent[0, :] = 0.0
            v_tent[:, 0] = 0.0
            v_tent[:, -1] = 0.0
            v_tent[-1, :] = 0.0


            d_u_tent__d_x = self.central_difference_x(u_tent)
            d_v_tent__d_y = self.central_difference_y(v_tent)

            # Compute a pressure correction by solving the pressure-poisson equation
            rhs = (
                self.DENSITY / self.TIME_STEP_LENGTH
                *
                (
                    d_u_tent__d_x
                    +
                    d_v_tent__d_y
                )
            )

            for _ in range(self.N_PRESSURE_POISSON_ITERATIONS):
                p_next = np.zeros_like(self.p_prev)
                p_next[1:-1, 1:-1] = 1/4 * (
                    +
                    self.p_prev[1:-1, 0:-2]
                    +
                    self.p_prev[0:-2, 1:-1]
                    +
                    self.p_prev[1:-1, 2:  ]
                    +
                    self.p_prev[2:  , 1:-1]
                    -
                    self.element_length**2
                    *
                    rhs[1:-1, 1:-1]
                )

                # Pressure Boundary Conditions: Homogeneous Neumann Boundary
                # Conditions everywhere except for the top, where it is a
                # homogeneous Dirichlet BC
                p_next[:, -1] = p_next[:, -2]
                p_next[0,  :] = p_next[1,  :]
                p_next[:,  0] = p_next[:,  1]
                p_next[-1, :] = 0.0

                self.p_prev = p_next
            

            d_p_next__d_x = self.central_difference_x(p_next)
            d_p_next__d_y = self.central_difference_y(p_next)

            # Correct the velocities such that the fluid stays incompressible
            u_next = (
                u_tent
                -
                self.TIME_STEP_LENGTH / self.DENSITY
                *
                d_p_next__d_x
            )
            v_next = (
                v_tent
                -
                self.TIME_STEP_LENGTH / self.DENSITY
                *
                d_p_next__d_y
            )

            # Velocity Boundary Conditions: Homogeneous Dirichlet BC everywhere
            # except for the horizontal velocity at the top, which is prescribed
            u_next[0, :] = 0.0
            u_next[:, 0] = 0.0
            u_next[:, -1] = 0.0
            u_next[-1, :] = self.HORIZONTAL_VELOCITY_TOP
            v_next[0, :] = 0.0
            v_next[:, 0] = 0.0
            v_next[:, -1] = 0.0
            v_next[-1, :] = 0.0


            # Advance in time
            self.u_prev = u_next
            self.v_prev = v_next
            self.p_prev = p_next
            
        return u_next,v_next,p_next
