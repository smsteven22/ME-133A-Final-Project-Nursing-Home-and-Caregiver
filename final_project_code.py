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
        
        """
        self.q0 = np.array([0.0, 0.0, 0.0, 
                   
                   0.0, 0.0, 0.0, 0.0, 
                   0.0, 0.0, 0.0, 

                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                   
                   0.0, 

                   0.0, 0.0, 0.0, 0.0, 
                   0.0, 0.0, 0.0, 
                   
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
        """
        
        self.q0 = np.zeros((6,1))
        
        self.p0 = np.array([0.00015731, -0.1115, -0.86201]).reshape(-1, 1)
        # self.R0 = Reye()

        # self.q_final = np.array([0.0, 0.0, 0.0, 
                        
        #                0.0, 0.0, 0.0, 0.0, 
        #                0.0, 0.0, 0.0, 

        #                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 

        #                0.0, 

        #                0.0, 0.0, 0.0, 0.0, 
        #                0.0, 0.0, 0.0, 
                   
        #                0.0, 0.0, 0.0, -1.612, 0.0, 1.612]).reshape(-1, 1)
        self.p_final = np.array([0.4257, -0.11154, -0.52242]).reshape(-1, 1)

        self.q = self.q0
        # self.x = self.p0
        self.lam = 20

    def jointnames(self):
        return[  'back_bkx', 'back_bky', 'back_bkz', 

                'l_arm_elx', 'l_arm_ely', 'l_arm_shx', 'l_arm_shz',
                'l_arm_wrx', 'l_arm_wry', 'l_arm_wry2', 

                'l_leg_akx', 'l_leg_aky', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_hpz', 'l_leg_kny',

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

        # FIX
        qdot_6 = J_pinv @ (v + self.lam * e)
        q_6 = self.q + dt * qdot_6

        # print(f"qdot_6 Length: {np.shape(qdot_6)}")
        # print(f"q_6 Length: {np.shape(q_6)}")

        # FIX
        qdot_full = np.zeros((24,1))
        q_full = np.zeros((24, 1))

        # print(f"qdot_full Length: {np.shape(qdot_full)}")
        # print(f"q_full Length: {np.shape(q_full)}")
        
        #qdot_full = np.zeros((30,1))
        #q_full = np.zeros((30, 1))
        
        #qdot_full[24:30, 0] = qdot_6
        #q_full[24:30, 0] = q_6

        qdot = np.vstack((qdot_full, qdot_6))
        q = np.vstack((q_full, q_6))

        # print(f"qdot Length: {np.shape(qdot)}")
        # print(f"q Length: {np.shape(q)}")

        # print(f"jointnames Length: {len(self.jointnames())}")
        # print(f"q list Length: {len(q.flatten().tolist())}")
        # print(f"qdot list Length: {qdot.flatten().tolist()}")
        
        self.q = q_6
        
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