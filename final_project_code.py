import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.time import Duration

from geometry_msgs.msg import Point, Vector3, Quaternion
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import JointState

from asyncio import Future

from hw5code.TransformHelpers import *
from hw5code.TrajectoryUtils import *
from hw5code.KinematicChain import KinematicChain

class SoccerNode(Node):
    # Initialization
    def __init__(self, name, rate, Trajectory):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Set up a trajectory.
        self.trajectory = Trajectory(self)
        self.jointnames = self.trajectory.jointnames()

        # Add a publisher to send the joint commands.
        self.pub_joint = self.create_publisher(JointState, '/joint_states', 10)

        # Prepare the marker publisher (latching for new subscribers)
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=20)
        self.pub_marker = self.create_publisher(MarkerArray, '/visualization_marker_array', quality)

        # Initialize the ball position, velocity, set the acceleration.
        self.radius = 0.1

        self.aball = np.array([0.0, 0.0, -9.81]).reshape((3,1))
        # robot can only handle initial ball z velocities of approx 0 < v_z_0 < 3.7 (meters per second)
        # visualizer handles better if restricted to 0 < v_z_0 < 2 (meters per second) and -2 < v_x_0 < 0 
        self.vball = np.array([-2.0, 0.0,  2]).reshape((3,1))
        # robot can only handle xforward of approx 0 < xforward < 0.7
        xforward = 0.4
        self.pball = np.array([xforward + (self.vball[0,0] * -3), self.trajectory.p0[1,0], self.radius]).reshape((3,1))

        # Manual ball kinematic equations to generate pkick
        self.pkick = np.array([self.pball[0,0] + (3 * self.vball[0,0]), self.pball[1,0],  0.0]).reshape((3,1))

        # velocity slowdown (and frankly more robust) re-approach to finding pkick
        # only holds for initial ball position of z = 0:
        ttokick = 3
        bouncetime = (-2 * self.vball[2,0]) / self.aball[2,0]
        tlast = ttokick % bouncetime
        zkick = (self.vball[2,0] * tlast) + (0.5 * self.aball[2,0] * (tlast**2))
        self.pkick[2,0] = zkick

        self.trajectory.p_final = self.pkick
        
        # Create the sphere marker.
        diam = 2 * self.radius
        self.marker = Marker()
        self.marker.header.frame_id = "l_foot"
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.marker.action = Marker.ADD
        self.marker.ns = "point"
        self.marker.id = 1
        self.marker.type = Marker.SPHERE
        self.marker.pose.orientation = Quaternion()
        self.marker.pose.position = Point_from_p(self.pball)
        self.marker.scale = Vector3(x=diam, y=diam, z=diam)
        self.marker.color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=1.0)

        # Create the marker array Message.
        self.mark = MarkerArray()
        self.mark.markers.append(self.marker)

        # Create a future object to signal when the trajectory ends,
        # i.e. no longer returns useful data.
        self.future = Future()

        # Set up the timing so (t=0) will occur in the first update
        # cycle (dt) from now.
        self.dt = 1.0 / float(rate)
        self.t = -self.dt
        self.start = self.get_clock().now() + Duration(seconds=self.dt)

        # Create a timer to keep calculating/sending commands.
        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" % (self.dt, rate))

    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()

    # Spin
    def spin(self):
        # Keep running (taking care of the timer callbacks and message 
        # passing), until interrupted or the trajectory is complete
        # (as signaled by the future object).
        rclpy.spin_until_future_complete(self, self.future)

        # Report the reason for shutting down.
        if self.future.done():
            self.get_logger().info("Stopping: " + self.future.result())
        else:
            self.get_logger().info("Stopping: Interrupted")

    # Update - send a new joint command every time step.
    def update(self):
        # To avoid any time jitter enforce a constant time step and
        # integrate to get the current time.
        self.t += self.dt

        # Integrate the velocity, then the position.
        self.vball += self.dt * self.aball
        self.pball += self.dt * self.vball
        
        # Check for right foot
        if (abs(self.pball[0,0] - self.trajectory.p_right[0,0]) < (self.radius)) and (abs(self.pball[1,0] - self.trajectory.p_right[1,0]) < (self.radius)) and (abs(self.pball[2,0] - self.trajectory.p_right[2,0]) < (self.radius)):
            # self.pball[0,0] = self.radius + (self.radius - self.pball[0,0])
            self.pball[0,0] = (self.pball[0,0]) + self.radius
            self.vball[0,0] *= -1.0
        elif self.pball[2, 0] < self.radius:
            # Check for the ground
            # important lines, do not delete or comment out
            self.pball[2,0] = self.radius + (self.radius - self.pball[2,0])
            self.vball[2,0] *= -1.0

        # Determine the corresponding ROS time (seconds since 1970).
        now = self.start + Duration(seconds=self.t)

        # Compute the desired joint positions and velocities for this time.
        desired = self.trajectory.evaluate(self.t, self.dt)
        if desired is None:
            self.future.set_result("Trajectory has ended")
            return
        (q, qdot) = desired

        # Check the results.
        if not (isinstance(q, list) and isinstance(qdot, list )):
            self.get_logger().warn("(q) and (qdot) must be python lists!")
            return
        if not (len(q) == len(self.jointnames)):
            self.get_logger().warn("(q) must be same length as jointnames!")
            return
        if not (len(qdot) == len(self.jointnames)):
            self.get_logger().warn("(qdot) must be same length as (q)!")
            return
        if not (isinstance(q[0], float) and isinstance(qdot[0], float)):
            self.get_logger().warn("Flatten NumPy arrays before makiing lists!")
            return
        
        # Build up a joint command message and publish.
        cmdmsg = JointState()
        cmdmsg.header.stamp = now.to_msg() # Current time for ROS
        cmdmsg.name = self.jointnames # List of joint names
        cmdmsg.position = q # List of joint positions
        cmdmsg.velocity = qdot # List of joint velocities
        self.pub_joint.publish(cmdmsg)

        # Update the marker message and publish.
        # Increment marker id to show path of the ball:
        # self.marker.id += 1
        self.marker.header.stamp = now.to_msg()
        self.marker.pose.position = Point_from_p(self.pball)
        self.pub_marker.publish(self.mark)

class Trajectory():
    def __init__(self, node):
        self.right_leg_chain = KinematicChain(node, 'pelvis', 'r_foot', self.right_leg_intermediate_joints())
        self.left_leg_chain = KinematicChain(node, 'pelvis', 'l_foot', self.left_leg_intermediate_joints())
        
        """ 
        self.q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, -0.312, 0.678, -0.366, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, -0.312, 0.678, -0.366, 0.0]).reshape(-1, 1) # Shallow squat
        """
        self.q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, -1.126, 1.630, -0.504, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, -1.126, 1.630, -0.504, 0.0]).reshape(-1, 1) # Deep squat
        self.qdot_max = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                  0, 0, 0, 0, 0, 0, 
                                  0, 0, 0, 0, 0, 0, 0, 0, 
                                  1, 1, 1, 1, 1, 1])

        self.p0 = np.array([-0.0000010414, -0.233, -0.00000000015678]).reshape(-1, 1) # Shallow squat W.R.T l_foot
        # self.p0 = np.array([-0.0000010421, -0.22301, -0.00000000027629]).reshape(-1, 1) # Deep squat W.R.T l_foot
        self.p_final = np.array([0.4628, -0.22304, 0.30902]).reshape(-1, 1) # W.R.T l_foot
        
        self.p_right = self.p0
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

        # End after one cycle.
        if t > 6:
            return None

        Rd = Reye()
        wd = np.zeros((3,1))

        qlast_left = (self.q[10:16, 0]).reshape(-1,1)
        qlast_right = (self.q[24:, 0]).reshape(-1, 1)
        
        qdot_max_diag = np.diag(self.qdot_max)

        (p_right, R_right, Jv_right, Jw_right) = self.right_leg_chain.fkin(qlast_right)
        (p_left, R_left, Jv_left, Jw_left) = self.left_leg_chain.fkin(qlast_left)

        p_rl = np.transpose(R_left) @ (p_right - p_left)
        R_rl = np.transpose(R_left) @ R_right

        self.p_right = p_rl

        def indexlist(start, num): return list(range(start, start+num))
        i_left = indexlist(10, 6)
        i_right = indexlist(24, 6)
        J_half = np.zeros((3,30))

        Jv_bar_left = J_half.copy()
        Jv_bar_left[:, i_left] = Jv_left

        Jw_bar_left = J_half.copy()
        Jw_bar_left[:, i_left] = Jw_left

        Jv_bar_right = J_half.copy()
        Jv_bar_right[:, i_right] = Jv_right

        Jw_bar_right = J_half.copy()
        Jw_bar_right[:, i_right] = Jw_right

        Jv_rl = np.transpose(R_left) @ (Jv_bar_right - Jv_bar_left + (crossmat(p_right - p_left) @ Jw_bar_left))
        Jw_rl = np.transpose(R_left) @ (Jw_bar_right - Jw_bar_left)

        J_rl = np.vstack((Jv_rl, Jw_rl))
        v_rl = np.vstack((vd, wd))
        e_rl = np.vstack((ep(pd, p_rl), eR(Rd, R_rl)))
        J_rl_pinv = np.linalg.pinv(J_rl @ qdot_max_diag)

        qdot = qdot_max_diag @ J_rl_pinv @ (v_rl + self.lam * e_rl)
        q = self.q + dt * qdot

        self.q = q
        
        return (q.flatten().tolist(), qdot.flatten().tolist())
    
#
# Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the soccer node for 100Hz updates, using the above
    # Trajectory class.
    soccer = SoccerNode('soccer', 100, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    soccer.spin()

    # Shudown the node and ROS.
    soccer.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()