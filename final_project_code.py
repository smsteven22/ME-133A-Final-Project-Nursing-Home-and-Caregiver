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
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.pub_marker = self.create_publisher(MarkerArray, '/visualization_marker_array', quality)

        # Initialize the ball position, velocity, set the acceleration.
        self.radius = 0.1

        self.aball = np.array([0.0, 0.0, -9.81]).reshape((3,1))
        self.vball = np.array([-15.0, 0.0,  14.62]).reshape((3,1))
        self.pball = np.array([self.vball[0,0] * -3, self.trajectory.p0[1,0], 0.0]).reshape((3,1))

        # Manual ball kinematic equations to generate pkick
        self.pkick = np.array([self.pball[0,0] - (3 * self.vball[0,0]), self.pball[1,0],  0.0]).reshape((3,1))
        nobouncedeltaz = (self.vball[2,0] * 3) - (0.5 * self.aball[2,0] * (3 ^ 2))
        if -nobouncedeltaz <= self.pball[2,0]:
            self.pkick[2,0] = self.pball[2,0] + nobouncedeltaz 
        else:
            # bounce1time eqn only holds for initial z position of 0
            bounce1time = (-2 * self.vball[2,0]) / self.aball[2,0]
            vbounce1 = self.vball[2,0] + (self.aball[2,0] * bounce1time)
            vup1 = -vbounce1
            zstatus1 = (vup1 * (3 - bounce1time)) + (0.5 * self.aball[2,0] * (3 - bounce1time))
            self.pkick[2,0] = zstatus1

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

        # Check for the ground
        if self.pball[2, 0] < self.radius:
            self.pball[2,0] = self.radius + (self.radius - self.pball[2,0])
            self.vball[2,0] *= -1.0

        # Check for right foot
        if (abs(self.pball[0,0] - self.trajectory.p_right[0,0]) < (2 * self.radius)) and (abs(self.pball[1,0] - self.trajectory.p_right[1,0]) < (2 * self.radius)) and (abs(self.pball[2,0] - self.trajectory.p_right[2,0]) < (2 * self.radius)):
            self.pball[0,0] = self.radius + (self.radius - self.pball[0,0])
            self.vball[0,0] *= -1.0

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
        if not (len(qdot) == len(q)):
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
        self.marker.header.stamp = now.to_msg()
        self.marker.pose.position = Point_from_p(self.pball)
        self.pub_marker.publish(self.mark)

class Trajectory():
    def __init__(self, node):
        self.right_leg_chain = KinematicChain(node, 'pelvis', 'r_foot', self.right_leg_intermediate_joints())
        self.left_leg_chain = KinematicChain(node, 'pelvis', 'l_foot', self.left_leg_intermediate_joints())
        
        self.q0 = np.array([0.0, 0.0, -0.312, 0.678, -0.366, 0.0, 0.0, 0.0, -0.312, 0.678, -0.366, 0.0]).reshape(-1, 1) # Shallow squat
        # self.q0 = np.array([0.0, 0.0, -1.126, 1.630, -0.504, 0.0, 0.0, 0.0, -1.126, 1.630, -0.504, 0.0]).reshape(-1, 1) # Deep squat
        self.qdot_max = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

        # self.p0 = np.array([-0.034242, -0.1115, -0.83125]).reshape(-1, 1) # W.R.T Pelvis
        # self.p_final = np.array([0.40741, -0.11154, -0.52203]).reshape(-1, 1) # W.R.T Pelvis

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
        self.p_right = new_p_right

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