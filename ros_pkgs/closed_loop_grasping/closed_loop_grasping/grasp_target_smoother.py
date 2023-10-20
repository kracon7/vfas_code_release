import math
import torch


def rotation_to_quaternion(R):
    """
    Convert a rotation matrix to quaternion
    
    Parameters
    ----------
    R : tensor of shape (3, 3)
        Rotation matrix

    Returns
    -------
    q : tensor of shape (4,)
        The unit quaternion (w, x, y, z)
    """
    
    r11, r12, r13 = R[0][0], R[0][1], R[0][2]
    r21, r22, r23 = R[1][0], R[1][1], R[1][2]
    r31, r32, r33 = R[2][0], R[2][1], R[2][2]

    # computing four sets of solutions
    qw_1 = torch.sqrt(1 + r11 + r22 + r33)
    u1 = 1/2 * torch.tensor([qw_1,
                                (r32-r23)/qw_1,
                                (r13-r31)/qw_1,
                                (r21-r12)/qw_1
                            ])

    qx_2 = torch.sqrt(1 + r11 - r22 - r33)
    u2 = 1/2 * torch.tensor([(r32-r23)/qx_2,
                                qx_2,
                                (r12+r21)/qx_2,
                                (r31+r13)/qx_2
                            ])

    qy_3 = torch.sqrt(1 - r11 + r22 - r33)
    u3 = 1/2 * torch.tensor([(r13-r31)/qy_3,
                                (r12+r21)/qy_3,
                                qy_3,
                                (r23+r32)/qy_3
                            ])

    qz_4 = torch.sqrt(1 - r11 - r22 + r33)
    u4 = 1/2* torch.tensor([(r21-r12)/qz_4,
                            (r31+r13)/qz_4,
                            (r32+r23)/qz_4,
                            qz_4
                            ])

    U = [u1,u2,u3,u4]
    idx = torch.tensor([r11+r22+r33, r11, r22, r33]).argmax()
    q = U[idx]
    assert torch.abs(torch.norm(q) - 1) < 1e-4
    if q[0] < 0:
        q = -q
    return q


def quaternion_to_rotation(q):
    """
    Convert a quaternion to rotation matrix
    
    Parameters
    ----------
    q : tensor of shape (4,)
        Unit quaternion (w, x, y, z)

    Returns
    -------
    R : tensor of shape (3, 3)
        The rotation matrix
    """
    
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    R = torch.tensor([[1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
                        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]])
    assert torch.norm(torch.eye(3) - R.T @ R) < 1e-4
    return R


def interpolate_quaternion(q1, q2, w):
    """
    Interpolate the quaternion
    
    Parameters
    ----------
    q1 : torch.tensor 
        (qw, qx, qy, qz)
    q2 : torch.tensor 
        (qw, qx, qy, qz)
    w : float
        Weight between two poses

    Returns
    -------
    p : torch.tensor 
        (qw, qx, qy, qz)
    """
    # get sphere angle between two quaternions
    omega = torch.acos(torch.clamp(2 * (q1 @ q2)**2 - 1, min=-1, max=1))
    if omega > 1e-3:
        # get interpolated quaternion
        if q1 @ q2 < 0:
            q1 = -q1
        p = (q1 * torch.sin((1-w)*omega) + q2 * torch.sin(w*omega)) / torch.sin(omega)
    else:
        p = q1
    
    # normalize p
    p = p / torch.norm(p)

    if p[0] < 0:
        p = -p

    return p


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class EuclideanOneEuroFilter:
    '''
    One Euro filter in Euclidean space
    '''
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * torch.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        #print("Translation a: ", a)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
    

class SO3OneEuroFilter:
    '''
    One Euro filter for SO3 rotations
    '''
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """
        x0 -- quaternion (qw, qx, qy, qz)
        """
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)

        omega = torch.acos(torch.clamp(2 * (x @ self.x_prev)**2 - 1, min=-1, max=1))
        dx = omega / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * torch.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)

        # print(x, self.x_prev)
        x_hat = interpolate_quaternion(self.x_prev, x, a)
        # x_hat = self.exponential_smoothing(a, x, self.x_prev)
        # if omega > 1:
        # print("Rotation omega: %.3f, a: %.3f"%(omega, a))
        
        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
    

class SE3OneEuroFilter:
    '''
    One Euro filter for SE3 transformations
    '''
    def __init__(self, t0, T0, 
                 device,
                 min_cutoff_r = 0.001, 
                 min_cutoff_t = 0.004,
                 beta_r = 1.5,
                 beta_t = 5,
                 d_cutoff_r = 1.0,
                 d_cutoff_t = 1.0,
                 ):
        x0 = T0[:3, 3]
        q0 = rotation_to_quaternion(T0[:3, :3])
        self.device = device
        self.translation_filter = EuclideanOneEuroFilter(t0, x0, 
                                                         torch.zeros(3, device=self.device), 
                                                         min_cutoff_t,
                                                         beta_t,
                                                         d_cutoff_t, 
                                                         )
        self.rotation_filter = SO3OneEuroFilter(t0, q0,
                                                0, 
                                                min_cutoff_r,
                                                beta_r,
                                                d_cutoff_r,
                                               )

    def __call__(self, t: float, T: torch.tensor):
        x = T[:3, 3]
        q = rotation_to_quaternion(T[:3, :3])

        x_hat = self.translation_filter(t, x)
        q_hat = self.rotation_filter(t, q)

        T_hat = torch.eye(4, dtype=T.dtype, device=T.device)
        T_hat[:3, :3] = quaternion_to_rotation(q_hat)
        T_hat[:3, 3] = x_hat

        return T_hat

    
