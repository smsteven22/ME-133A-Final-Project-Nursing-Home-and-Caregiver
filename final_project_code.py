import rclpy
import numpy as np

from math import *

from hw5code.GeneratorNode      import GeneratorNode
from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *

from hw5code.KinematicChain     import KinematicChain

class Trajectory():
    def __init__(self, node):
        self.right_leg_chain = KinematicChain(node, 'pelvis', 'r_foot', self.right_leg_intermediate_joints())
        self.left_leg_chain = KinematicChain(node, 'pelvis', 'l_foot', self.left_leg_intermediate_joints())
        
        # self.q0 = np.array([0.0, 0.0, -0.312, 0.678, -0.366, 0.0, 0.0, 0.0, -0.312, 0.678, -0.366, 0.0]).reshape(-1, 1) # Shallow squat
        self.q0 = np.array([0.0, 0.0, -1.126, 1.630, -0.504, 0.0, 0.0, 0.0, -1.126, 1.630, -0.504, 0.0]).reshape(-1, 1) # Deep squat
        self.qdot_max = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

        # self.p0 = np.array([-0.034242, -0.1115, -0.83125]).reshape(-1, 1) # W.R.T Pelvis
        # self.p_final = np.array([0.40741, -0.11154, -0.52203]).reshape(-1, 1) # W.R.T Pelvis

        # self.p0 = np.array([-0.0000010414, -0.233, -0.00000000015678, 0.0, 0.0, 0.0]).reshape(-1, 1) # Shallow squat W.R.T l_foot
        self.p0 = np.array([-0.0000010421, -0.22301, -0.00000000027629]).reshape(-1, 1) # Deep squat W.R.T l_foot
        self.p_final = np.array([0.4628, -0.22304, 0.30902]).reshape(-1, 1) # W.R.T l_foot

        

        self.q = self.q0
        self.lam = 20

    def jointnames(self):
        return[  'back_bkx', 'back_bky', 'back_bkz', 

                'l_arm_elx', 'l_arm_ely', 'l_arm_shx', 'l_arm_shz',
                'l_arm_wrx', 'l_arm_wry', 'l_arm_wry2', 

                'l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_kny', 'l_leg_aky',  'l_leg_akx',

                'neck_ry',

                'r_arm_elx', 'r_arm_ely', 'r_arm_shx', 'r_arm_shz',
                'r_arm_wrx', 'r_arm_wry', 'r_arm_wry2',

                'r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky',  'r_leg_akx'
             ]

    def right_leg_intermediate_joints(self):
        return ['r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky',  'r_leg_akx']
    
    def left_leg_intermediate_joints(self):
        return ['l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_kny', 'l_leg_aky',  'l_leg_akx']
        
    
    def evaluate(self, t, dt):
        t_mod = t % 6

        if t_mod < 3:
            (pd, vd) = goto(t_mod, 3.0, self.p0, self.p_final)
        else:
            (pd, vd) = goto(t_mod-3.0, 3.0, self.p_final, self.p0)  

        Rd = Reye()
        wd = np.zeros((3,1))

        qlast_right = (self.q[0:6, 0]).reshape(-1,1)
        qlast_left = (self.q[6:, 0]).reshape(-1, 1)

        qdot_max_right = self.qdot_max[0:6]
        qdot_max_left = self.qdot_max[6:]

        qdot_max_right_diag = np.diag(qdot_max_right)
        qdot_max_left_diag = np.diag(qdot_max_left)

        (p_right, R_right, Jv_right, Jw_right) = self.right_leg_chain.fkin(qlast_right)
        (p_left, R_left, Jv_left, Jw_left) = self.left_leg_chain.fkin(qlast_left)

        new_p_right = (-1 * (np.transpose(R_left) @ p_left)) + (np.transpose(R_left) @ p_right)

        J_right = np.vstack((Jv_right, Jw_right))
        v_right = np.vstack((vd, wd))
        e_right = np.vstack((ep(pd, new_p_right), eR(Rd, R_right)))
        J_pinv_right = np.linalg.pinv(J_right @ qdot_max_right_diag)

        qdot_right = qdot_max_right_diag @ J_pinv_right @ (v_right + self.lam * e_right)
        q_right = qlast_right + dt * qdot_right

        J_left = np.vstack((Jv_left, Jw_left))
        v_left = np.vstack((vd, wd))
        e_left = np.vstack((ep(pd, p_left), eR(Rd, R_left)))
        J_pinv_left = np.linalg.pinv(J_left @ qdot_max_left_diag)

        qdot_left = qdot_max_left_diag @ J_pinv_left @ (v_left + self.lam * e_left)
        q_left = qlast_left + dt * qdot_left

        qdot_before_left = np.zeros((10, 1))
        qdot_between_left_right = np.zeros((8,1))

        qdot_first_half = np.vstack((qdot_before_left, qdot_left))
        qdot_second_half = np.vstack((qdot_between_left_right, qdot_right))
        qdot = np.vstack((qdot_first_half, qdot_second_half))

        q_before_left = np.zeros((10, 1))
        q_between_left_right = np.zeros((8, 1))

        q_first_half = np.vstack((q_before_left, q_left))
        q_second_half = np.vstack((q_between_left_right, q_right))
        q = np.vstack((q_first_half, q_second_half))

        # Because it is singular, it's losing a DOF --> use secondary task to pull knee forward

        self.q = np.vstack((q_right, q_left))
        
        return (q.flatten().tolist(), qdot.flatten().tolist())
        
            

#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = GeneratorNode('generator', 100, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
