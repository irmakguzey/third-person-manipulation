# Script to plot the trajectory and position of the base frame with respect to the camera
import os
import glob 
import time
import hydra 
import pickle
import numpy as np
import cv2, PIL
from cv2 import aruco
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from holobot.utils.network import ZMQCameraSubscriber
from PIL import Image as im
import torchvision.transforms as T
from torchvision.transforms.functional import crop
from scipy.spatial.transform import Rotation

from pathlib import Path
from tqdm import tqdm
from holobot_api.api import DeployAPI 
from holobot.robot.kinova import KinovaArm
from holobot.robot.allegro.allegro import AllegroHand
from holobot.robot.allegro.allegro_kdl import AllegroKDL
from third_person_man.utils import get_path_in_package
from third_person_man.kinematics import FingertipIKSolver
from third_person_man.utils import VideoRecorder


mpl.use('Agg')


class BaseframeReplay():
    def __init__(self, cfg): 
        self.camera_intrinsics = np.array(cfg.camera_intrinsics)
        self.distortion_coefficients = np.array(cfg.distortion_coefficients)
        self.end_effector_position = np.array(cfg.end_effector_position)
        self.end_effector_realignment_matrix = np.array(cfg.end_effector_realignment_matrix)
        self.translation_ratio = cfg.translation_ratio
        self.eef_to_hand = cfg.eef_to_hand
        self.ratio_list = []

        self.save_dir = cfg.save_dir
        self.replay_dir = cfg.replay_dir
        self.host = cfg.host
        self.port = cfg.port
        self.calibr_duration = cfg.calibr_duration
        self.test_duration = cfg.test_duration

        self.base_rvec = np.array(cfg.rvec_calibr)
        self.base_tvec = np.array(cfg.tvec_calibr) 

        self.arm = KinovaArm()
        self.hand = AllegroHand()
        self.hand_solver = AllegroKDL()
        self.solver = FingertipIKSolver(
            urdf_path = get_path_in_package('kinematics/assets/allegro_hand_right.urdf'),
            desired_finger_types= ['index']
        )

        required_data = {
            'rgb_idxs': [0],
            'depth_idxs': [0]
        }
        self.deploy_api = DeployAPI(
            host_address = '172.24.71.220',
            required_data = required_data
        )

        self.video_recorder = VideoRecorder(
            save_dir = Path(self.save_dir),
            fps = 20
        )
        time.sleep(1)

    def _get_curr_image(self):
        image_subscriber = ZMQCameraSubscriber( # TODO: Change this such that it will create a subscriber only once
            host = self.host,
            # port = port + self.view_num,
            port = self.port + 1, 
            topic_type = 'RGB'
        )
        # print("image subscriber get!")
        image, _ = image_subscriber.recv_rgb_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = im.fromarray(image)
        return image 
    

    def get_markers(self, image, plot_marker_axis):
        markers = []

        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners)
        frame_axis = frame_markers.copy()

        for i in range(len(corners)):
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.01,
                                                               self.camera_intrinsics,
                                                               np.zeros(5))
            markers.append([rvec, tvec])

            if plot_marker_axis == True:
                if i == 0:
                    frame_axis = cv2.drawFrameAxes(frame_markers.copy(), self.camera_intrinsics, self.distortion_coefficients, rvec, tvec, 0.01)
                else:
                    frame_axis = cv2.drawFrameAxes(frame_axis.copy(), self.camera_intrinsics, self.distortion_coefficients, rvec, tvec, 0.01)
        return markers, frame_axis


    def marker_to_camera_frame(self, rvec, tvec, point_in_marker_frame): 
        # Convert rotation vector to rotation matrix
        rotation_matrix_origin, _ = cv2.Rodrigues(rvec)
        # Construct transformation matrix from origin to camera
        T_origin_to_camera = np.hstack((rotation_matrix_origin, tvec.T[:,0]))
        T_origin_to_camera = np.vstack((T_origin_to_camera, np.array([0, 0, 0, 1])))
        point_homogeneous = np.concatenate((point_in_marker_frame.T, np.ones((1, 1))), axis=0)
        # Transform the point to the camera frame
        p_tvec = np.dot(T_origin_to_camera, point_homogeneous)[:3]
        return p_tvec


    def camera_to_marker_frame(self, rvec, tvec, point_in_camera_frame):
        # Convert rotation vector to rotation matrix
        rotation_matrix_origin, _ = cv2.Rodrigues(rvec)
        # Construct transformation matrix from origin to camera
        T_origin_to_camera = np.hstack((rotation_matrix_origin, tvec.T[:, 0]))
        T_origin_to_camera = np.vstack((T_origin_to_camera, np.array([0, 0, 0, 1])))
        # Take the inverse of the transformation matrix
        T_camera_to_origin = np.linalg.inv(T_origin_to_camera)
        # Convert the point in camera frame to homogeneous coordinates
        point_homogeneous = np.concatenate((point_in_camera_frame.T, np.ones((1, 1))), axis=0)
        # Transform the point back to the marker frame
        p_in_marker = np.dot(T_camera_to_origin, point_homogeneous)[:3]
        return p_in_marker
    
    def base_frame(self, end_rvec, end_tvec, end_trans, end_quat):

        # transform from base frame to end effector
        rotation_end_to_base = Rotation.from_quat(end_quat).as_matrix()
        rotation_end_to_base = np.hstack((rotation_end_to_base, end_trans.reshape(3,1)))
        rotation_end_to_base = np.vstack((rotation_end_to_base, np.array([0, 0, 0, 1])))
        rotation_base_to_end = np.linalg.inv(rotation_end_to_base)

        # transform from the end effector to camera
        rotation_end_to_camera, _ = cv2.Rodrigues(end_rvec)
        rotation_end_to_camera = np.hstack((rotation_end_to_camera, end_tvec))
        rotation_end_to_camera = np.vstack((rotation_end_to_camera, np.array([0, 0, 0, 1])))

        rotation_matrix_base_to_camera = np.dot(rotation_end_to_camera, rotation_base_to_end)
        base_rvec = Rotation.from_matrix(rotation_matrix_base_to_camera[:3,:3]).as_rotvec()
        base_tvec = rotation_matrix_base_to_camera[:3,3]
        return base_rvec, base_tvec
    
    def end_effector(self, base_rvec, base_tvec, end_trans, end_quat): # This is given base frame, calculate the end effector position
        # transform from base frame to end effector
        rotation_end_to_base = Rotation.from_quat(end_quat).as_matrix()
        rotation_end_to_base = np.hstack((rotation_end_to_base, end_trans.reshape(3,1)))
        rotation_end_to_base = np.vstack((rotation_end_to_base, np.array([0, 0, 0, 1])))

        # transform from the base frame to camera
        rotation_base_to_camera, _ = cv2.Rodrigues(base_rvec)
        rotation_base_to_camera = np.hstack((rotation_base_to_camera, base_tvec.reshape(3,1)))
        rotation_base_to_camera = np.vstack((rotation_base_to_camera, np.array([0, 0, 0, 1])))

        rotation_matrix_eef_to_camera = np.dot(rotation_base_to_camera, rotation_end_to_base)
        eef_rvec = Rotation.from_matrix(rotation_matrix_eef_to_camera[:3,:3]).as_rotvec()
        eef_tvec = rotation_matrix_eef_to_camera[:3,3]
        return eef_rvec, eef_tvec
    
    def pose(self, base_rvec, base_tvec, end_rvec, end_tvec):

        # transform from the base frame to camera
        rotation_base_to_camera, _ = cv2.Rodrigues(base_rvec)
        rotation_base_to_camera = np.hstack((rotation_base_to_camera, base_tvec.reshape(3,1)))
        rotation_base_to_camera = np.vstack((rotation_base_to_camera, np.array([0, 0, 0, 1])))
        rotation_camera_to_base = np.linalg.inv(rotation_base_to_camera)

        # transform from the end effector to camera
        rotation_end_to_camera, _ = cv2.Rodrigues(end_rvec)
        rotation_end_to_camera = np.hstack((rotation_end_to_camera, end_tvec))
        rotation_end_to_camera = np.vstack((rotation_end_to_camera, np.array([0, 0, 0, 1])))

        rotation_matrix_eef_to_base = np.dot(rotation_camera_to_base, rotation_end_to_camera)
        eef_quat = Rotation.from_matrix(rotation_matrix_eef_to_base[:3,:3]).as_quat().reshape(1,4)
        eef_tvec = rotation_matrix_eef_to_base[:3,3].reshape(1,3) / self.translation_ratio
        pose = np.concatenate((eef_tvec, eef_quat),axis=1)
        return pose
    

    def hand_position(self, end_rvec, end_tvec): # The orientation of hand (end effector) with respect to the camera
        
        # transform from the end effector ro the camera
        rotation_end_to_camera, _ = cv2.Rodrigues(end_rvec)
        rotation_end_to_camera = np.hstack((rotation_end_to_camera, end_tvec.reshape(3,1)))
        rotation_end_to_camera = np.vstack((rotation_end_to_camera, np.array([0, 0, 0, 1])))

        rotation_matrix_hand_to_camera = np.dot(rotation_end_to_camera, self.eef_to_hand)
        hand_rvec = Rotation.from_matrix(rotation_matrix_hand_to_camera[:3,:3]).as_rotvec()
        hand_tvec = rotation_matrix_hand_to_camera[:3,3]
        return hand_rvec, hand_tvec

    def finger_tips(self, hand_rvec, hand_tvec): # The orientation of fingertips with respect to the camera
        # Get fingertip orientation with respect to hand
        # transform from the hand to the camera
        rotation_hand_to_camera, _ = cv2.Rodrigues(hand_rvec)
        rotation_hand_to_camera = np.hstack((rotation_hand_to_camera, hand_tvec.reshape(3,1)))
        rotation_hand_to_camera = np.vstack((rotation_hand_to_camera, np.array([0, 0, 0, 1])))
        # Get hand positions
        features = self.hand.get_joint_position()

        fingertip_poses = []
        for i, finger_type in enumerate(['index', 'middle', 'ring', 'thumb']):
            finger_tvec, finger_rvec = self.hand_solver.finger_forward_kinematics(
                finger_type, features[i*4:(i+1)*4]
            )
            finger_tvec = finger_tvec * self.translation_ratio
            # Stack tvec and rvec
            fingertip_pose = np.hstack((finger_rvec, finger_tvec.reshape(3,1)))
            fingertip_pose = np.vstack((fingertip_pose, np.array([0, 0, 0, 1])))
            fingertip_poses.append(fingertip_pose)

        fingertip_poses = np.stack(fingertip_poses, axis=0)

        fingertip_poses_to_camera = []
        for H_F_E in fingertip_poses: # Homo to take fingertip pose frame to the end effector frame
            H_F_O = rotation_hand_to_camera @ H_F_E     # Homo to take fingertip pose frame to the origin 
            #convert this to rvec and tvec
            finger_rvec = Rotation.from_matrix(H_F_O[:3,:3]).as_rotvec()
            finger_tvec = H_F_O[:3,3]
            fingertip_poses_to_camera.append([finger_rvec,finger_tvec])

        return fingertip_poses_to_camera
    
    def finger_to_hand(self, finger_rvec, finger_tvec, hand_rvec, hand_tvec): # This is the reverse function of finger_tips
        # Given finger to camera and hand to camera, returns finger to hand H_F_E
        rotation_hand_to_camera, _ = cv2.Rodrigues(hand_rvec)
        rotation_hand_to_camera = np.hstack((rotation_hand_to_camera, hand_tvec.reshape(3,1)))
        rotation_hand_to_camera = np.vstack((rotation_hand_to_camera, np.array([0, 0, 0, 1])))
        rotation_camera_to_hand = np.linalg.inv(rotation_hand_to_camera)

        H_F_O, _ = cv2.Rodrigues(finger_rvec)
        H_F_O = np.hstack((H_F_O, finger_tvec.reshape(3,1)))
        H_F_O = np.vstack((H_F_O, np.array([0, 0, 0, 1])))

        H_F_E = np.dot(rotation_camera_to_hand, H_F_O)

        return H_F_E







    def get_base_frame(self, first_frame, save_individual_frame): # returns the plotted image of current frame
        image = self._get_curr_image()
        markers, frame_axis = self.get_markers(image, False)
        
        #get the current status of the KinovaArm
        pose = self.arm.get_cartesian_position()
        end_effector_in_base_frame = np.array(pose[0:3]) 
        end_effector_rotation_in_base_frame = np.array(pose[3:])

        if len(markers) == 0:
            base_rvec = base_tvec  = np.array([[0,0,0]])
            return first_frame, base_rvec, base_tvec, frame_axis, False
        
        #choose the marker to be reference frame
        rvec = markers[0][0]
        tvec = markers[0][1]

        end_tvec = self.marker_to_camera_frame(rvec, tvec, self.end_effector_position)
        rotation_marker_to_camera, _ = cv2.Rodrigues(rvec)
        rotation_end_to_camera = np.dot(rotation_marker_to_camera, self.end_effector_realignment_matrix)
        end_rvec = Rotation.from_matrix(rotation_end_to_camera).as_rotvec()

        frame_axis = cv2.drawFrameAxes(frame_axis.copy(), self.camera_intrinsics, self.distortion_coefficients, end_rvec, end_tvec, 0.01)

        if first_frame == True:
            base_rvec = base_tvec  = np.array([[0,0,0]])
            self.prev_end_tvec = end_tvec
            self.prev_end_in_base = end_effector_in_base_frame
            first_frame = False
            return first_frame, base_rvec, base_tvec, frame_axis, False

        #NOTE: we want to know the ratio between the translation in camera frame and end effector frame
        # change_in_tvec = (self.prev_end_tvec - end_tvec).reshape(1,-1)
        # change_in_arm = self.prev_end_in_base - end_effector_in_base_frame

        # translation_ratio = change_in_tvec / change_in_arm 
        # self.ratio_list.append(np.mean(translation_ratio))
        # final_ratio = np.mean(self.ratio_list)

        end_effector_in_base_frame_calibr = end_effector_in_base_frame * self.translation_ratio
        base_rvec, base_tvec = self.base_frame(end_rvec, end_tvec, end_effector_in_base_frame_calibr, end_effector_rotation_in_base_frame)
        frame_axis = cv2.drawFrameAxes(frame_axis.copy(), self.camera_intrinsics, self.distortion_coefficients, base_rvec, base_tvec, 0.01)

        os.makedirs(self.save_dir, exist_ok=True)
        img = Image.fromarray(frame_axis)
        if save_individual_frame == True:
            save_filename = "saved_image.png"  
            save_path = os.path.join(self.save_dir, save_filename)
            img.save(save_path)
        
        self.prev_end_tvec = end_tvec
        self.prev_end_in_base = end_effector_in_base_frame

        return False, base_rvec, base_tvec, frame_axis, True
        
    
    def save_baseframe_trajectory(self):

        start_time = time.time()
        time_step = 0
        first_frame = True
        base_rvec_list = []
        base_tvec_list = []
        while time.time() - start_time < self.calibr_duration:
            first_frame, base_rvec, base_tvec, img, success = self.get_base_frame(first_frame, False)
            if success:
                base_rvec_list.append(base_rvec)
                base_tvec_list.append(base_tvec)
                print("Calibrating base frame position!")
            if time_step == 0: 
                self.video_recorder.init(obs=img)
            time_step += 1
            self.video_recorder.record_realsense(img)

    
        self.base_rvec = np.mean(np.stack(base_rvec_list, axis=0), axis=0)
        self.base_tvec = np.mean(np.stack(base_tvec_list, axis=0), axis=0)
        base_rvec = np.array2string(self.base_rvec, precision=5, separator=', ', suppress_small=True)
        base_tvec = np.array2string(self.base_tvec, precision=5, separator=', ', suppress_small=True)
        print('Calibrated base_rvec: {}'.format(base_rvec))
        print('Calibrated base_tvec: {}'.format(base_tvec))

        self.video_recorder.save('base_frame_trajectory.mp4')
            

    def test_base_calibr(self): # Test the calibrated base frame by comparing cartesian position and end effector position
        if self.base_rvec is None or self.base_tvec is None:
            print('ERROR!!!Please Calibrate First!!!')
            return 
        
        ## NOTE: Plot the gt end_effector and calibrated base frame
        image = self._get_curr_image()
        markers, frame_axis = self.get_markers(image, False)
        
        #get the current status of the KinovaArm
        pose = self.arm.get_cartesian_position()
        end_effector_in_base_frame = np.array(pose[0:3])  * self.translation_ratio
        end_effector_rotation_in_base_frame = np.array(pose[3:])

        frame_axis = cv2.drawFrameAxes(frame_axis.copy(), self.camera_intrinsics, self.distortion_coefficients, self.base_rvec, self.base_tvec, 0.01)

        if len(markers) == 0:
            return frame_axis
        
        #choose the marker to be reference frame
        rvec = markers[0][0]
        tvec = markers[0][1]

        # This is the end effector position that can be seen as GT 
        end_tvec = self.marker_to_camera_frame(rvec, tvec, self.end_effector_position)
        rotation_marker_to_camera, _ = cv2.Rodrigues(rvec)
        rotation_end_to_camera = np.dot(rotation_marker_to_camera, self.end_effector_realignment_matrix)
        end_rvec = Rotation.from_matrix(rotation_end_to_camera).as_rotvec()

        # This is the predicted end effector position
        eef_rvec, eef_tvec = self.end_effector(self.base_rvec, self.base_tvec, end_effector_in_base_frame, end_effector_rotation_in_base_frame)

        error_rvec = end_rvec - eef_rvec
        error_tvec = end_tvec.reshape(1,3) - eef_tvec.reshape(1,3)
        print("End effector prediction error: ")
        print("Error in rvec: {}".format(error_rvec))
        print("Error in tvec: {}".format(error_tvec))

        # Another comparison is predicted cartesian movement vs actual cartesian movement
        # predicted pose is derived from calibrated base and actual end_rvec,  end_tvec
        eef_pose = self.pose(self.base_rvec, self.base_tvec, end_rvec, end_tvec)
        error_pose = pose - eef_pose

        print("Error in pose: {}\n".format(error_pose))

        frame_axis = cv2.drawFrameAxes(frame_axis.copy(), self.camera_intrinsics, self.distortion_coefficients, end_rvec, end_tvec, 0.01)
        frame_axis = cv2.drawFrameAxes(frame_axis.copy(), self.camera_intrinsics, self.distortion_coefficients, eef_rvec, eef_tvec, 0.01)

        eef_zero = np.array([[0.000000000001,0,0]])
        eef_in_image, _ = cv2.projectPoints(eef_zero, eef_rvec, eef_tvec, self.camera_intrinsics, self.distortion_coefficients)
        eef_in_image = eef_in_image.squeeze()
        center = (int(eef_in_image[0]), int(eef_in_image[1]))
        cv2.circle(frame_axis, center, 5, (255,255,0), -1) 
        return frame_axis

    
    def save_end_effector_trajectory(self):
        start_time = time.time()
        time_step = 0
        while time.time() - start_time < self.test_duration:
            img  = self.test_base_calibr()
            print('Getting end effector positions!')
            if time_step == 0: 
                self.video_recorder.init(obs=img)
            time_step += 1
            self.video_recorder.record_realsense(img)

        self.video_recorder.save('end_effector_trajectory.mp4')

    
    def test_fingertip_calibr(self):
        if self.base_rvec is None or self.base_tvec is None:
            print('ERROR!!!Please Calibrate First!!!')
            return 
        
        ## NOTE: Plot the gt end_effector and calibrated base frame
        image = self._get_curr_image()
        markers, frame_axis = self.get_markers(image, False)
        frame_axis = cv2.drawFrameAxes(frame_axis.copy(), self.camera_intrinsics, self.distortion_coefficients, self.base_rvec, self.base_tvec, 0.01)
        
        #get the current status of the KinovaArm
        pose = self.arm.get_cartesian_position()
        end_effector_in_base_frame = np.array(pose[0:3])  * self.translation_ratio
        end_effector_rotation_in_base_frame = np.array(pose[3:])
        
        # This is the predicted end effector position
        eef_rvec, eef_tvec = self.end_effector(self.base_rvec, self.base_tvec, end_effector_in_base_frame, end_effector_rotation_in_base_frame)
        frame_axis = cv2.drawFrameAxes(frame_axis.copy(), self.camera_intrinsics, self.distortion_coefficients, eef_rvec, eef_tvec, 0.01)
        # Get hand orietation
        hand_rvec, hand_tvec = self.hand_position(eef_rvec, eef_tvec)
        # Get the predicted fingertip orientation
        fingertip_poses = self.finger_tips(hand_rvec, hand_tvec)
        eef_pose = [eef_rvec, eef_tvec]
        
            
        return fingertip_poses, eef_pose, frame_axis

    def save_fingertip_trajectory(self):
        start_time = time.time()
        time_step = 0
        fingertip_frames = []
        eef_frames = []
        while time.time() - start_time < self.test_duration:
            fingertip_poses, eef_pose, frame_axis  = self.test_fingertip_calibr()
            fingertip_frames.append(fingertip_poses)
            eef_frames.append(eef_pose)
            for finger in fingertip_poses: 
                frame_axis = cv2.drawFrameAxes(frame_axis.copy(), self.camera_intrinsics, self.distortion_coefficients, finger[0], finger[1], 0.01)
            print('Getting fingertip positions!')
            if time_step == 0: 
                self.video_recorder.init(obs=frame_axis)
            time_step += 1
            self.video_recorder.record_realsense(frame_axis)
        
        frames = {}
        frames['fingertip'] = fingertip_frames
        frames['eef'] = eef_frames

        self.video_recorder.save('fingertip_trajectory.mp4')
        file_path = self.save_dir +'/fingertip_poses.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(frames, file)


    def get_pose(self, four_fingers):
        # Get the current hand joint positions (to calculate current position)
        features = self.hand.get_joint_position()
        self.solver.set_positions(
            joint_positions = features[:],
            endeff_position = np.zeros(7) # For now we are going to ignore the endeffector position 
        ) 
        hand_action, _, errors, _, _ = self.solver.move_to_pose(
                    poses = four_fingers,
                    demo_action = None)
        
        return hand_action



    def ik_replay_on_real_robot(self): # This tests the inverse kinematics of current setup
        fingertip_file = self.replay_dir + '/fingertip_poses.pkl'
        with open(fingertip_file, 'rb') as file:
            poses = pickle.load(file)
        fingertip_poses = poses['fingertip']
        eef_poses = poses['eef']
        print('Number of replay frames: {}'.format(len(fingertip_poses)))
        # print('eef poses: {}'.format(eef_poses))

        joint_commands = []
        desired_finger_position = []
        for frame in range(len(fingertip_poses)):
            four_fingers = []
            fingertips = fingertip_poses[frame]
            eef_rvec, eef_tvec = eef_poses[frame]
            # Get hand orientation
            hand_rvec, hand_tvec = self.hand_position(eef_rvec, eef_tvec)
            # Calculate the finger_to_hand_rvec, finger_to_hand_tvec
            for i, finger_type in enumerate(['index', 'middle', 'ring', 'thumb']):
                finger_to_hand_frame = self.finger_to_hand(fingertips[i][0], fingertips[i][1], hand_rvec, hand_tvec)
                four_fingers.append(finger_to_hand_frame)
            
            # Pass the pose into inverse kinematics
            joint_command = {}
            joint_command['allegro'] = self.get_pose(four_fingers)
            self.deploy_api.send_robot_action(joint_command)


            
            
            desired_finger_position.append(four_fingers)
            # joint_commands.append(hand_action)

            
            # Perform inverse kinematics, get and record joint commands





        
        

        return desired_finger_position, joint_commands




        # Then: (as a test) get the current cur_eef_rvec, cur_eef_tvec of the end effector


        # Apply change to current end effector to get gt_finger_poses (ground_truth)

