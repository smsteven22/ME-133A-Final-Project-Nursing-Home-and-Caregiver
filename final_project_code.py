import rclpy
import numpy as np

from math import *

from hw5code.GeneratorNode      import GeneratorNode
from hw5code.TransformHelpers   import *
from hw5code.TrajectoryUtils    import *

from hw5code.KinematicChain     import KinematicChain

class Trajectory():
    def __init__(self, node):
        # change tip/world
        self.chain = KinematicChain(node, 'pelvis', 'r_foot', self.intermediate_joints())
        
        self.q0 = np.array([0.0, 0.0, -0.312, 0.678, -0.366, 0.0]).reshape(-1, 1)
        
        self.p0 = np.array([-0.034242, -0.1115, -0.83125]).reshape(-1, 1)
        self.p_final = np.array([0.40741, -0.11154, -0.52203]).reshape(-1, 1)

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

    def intermediate_joints(self):
        return ['r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky',  'r_leg_akx']
        
    
    def evaluate(self, t, dt):
        t_mod = t % 6

        if t_mod < 3:
            (pd, vd) = goto(t_mod, 3.0, self.p0, self.p_final)
        else:
            (pd, vd) = goto(t_mod-3.0, 3.0, self.p_final, self.p0)  

        Rd = Reye()
        wd = np.array([0, 0, 0]).reshape(-1, 1)
        
        qlast = self.q
        (p, R, Jv, Jw) = self.chain.fkin(qlast)

        J = np.vstack((Jv, Jw))
        v = np.vstack((vd, wd))
        e = np.vstack((ep(pd, p), eR(Rd, R)))
        J_pinv = np.linalg.pinv(J)

        # Because it is singular, it's losing a DOF --> use secondary task to pull knee forward

        qdot_r_leg = J_pinv @ (v + self.lam * e)
        q_r_leg = self.q + dt * qdot_r_leg

        qdot_full = np.zeros((24,1))
        qdot = np.vstack((qdot_full, qdot_r_leg))

        q_before_l_leg = np.zeros((10,1))
        q_between_l_r_legs = np.zeros((8,1))

        q_first_half = np.vstack((q_before_l_leg, self.q0))
        q_second_half = np.vstack((q_between_l_r_legs, q_r_leg))
        q = np.vstack((q_first_half, q_second_half))
        
        self.q = q_r_leg
        
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