import os
import cv2
import time
import torch
import argparse
import numpy as np
import datetime

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

import rospy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import OccupancyGrid

from cv_bridge import CvBridge

from constants import *


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


class Detector():
    def __init__(self, device='cuda', # cuda or cpu
                 inp_dets=384, # input size of the detection (square) must be divisible by 32
                 pose_backbone="resnet50",  # backbone model for FastPose model
                 inp_pose=[224, 160], # should be div0sible by 32; input pose size
                 camera=None,
                 tracker_max_age=5,
                 visualize=True):
        """
        DETECTOR SETUP
        """
        self.visualize = visualize

        # detects the person in frame with tinyyolo v3
        self.detect_model = TinyYOLOv3_onecls(inp_dets, device=device, conf_thres=0.9)

        # then detect the pose from just the person frame
        self.pose_model = SPPE_FastPose(pose_backbone,
                                        inp_pose[0],
                                        inp_pose[1],
                                        device=device)

        # track multiple people / skeletons in the frame
        self.tracker = Tracker(max_age=tracker_max_age, n_init=1)

        # actual model for action detection
        self.action_model = TSSTG()

        self.resize_fn = ResizePadding(inp_dets, inp_dets)

        self.fps_time = 0
        self.f = 0

        """
        ROS NODE SETUP
        """
        self.rate = rospy.Rate(FREQUENCY)

        self.camera = camera
        if camera == None:
            # grab images from the rosbot camera
            self._img_sub = rospy.Subscriber(IMAGE_TOPIC,
                                             CompressedImage,
                                             self._image_callback,
                                             queue_size=1)
        else:
            # otherwise, use a webcam connected to the robot (by tape lol)
            self.cam = CamLoader(self.camera, preprocess=self.preproc).start()


        # constantly receive the occupancy grid and robot pose information
        # from slam so that you can send it to Daniel's node whenever
        # a fall is detected. We don't need to run image processing on this
        # all the time. Only when the fall is detected
        self._pose_sub = rospy.Subscriber(SLAM_POSE_TOPIC,
                                          PoseStamped,
                                          self._pose_callback,
                                          queue_size=1)
        self._map_sub = rospy.Subscriber(SLAM_MAP_TOPIC,
                                         OccupancyGrid,
                                         self._map_callback,
                                         queue_size=1)


        # publish the current action
        self._action_pub = rospy.Publisher(ACTION_PUB_TOPIC,
                                           String,
                                           queue_size=1)
        # publish the location of the person in the frame as well as their
        # pseudo distance from the frame
        self._location_pub = rospy.Publisher(LOCATION_PUB_TOPIC,
                                             Point,
                                             queue_size=1)
        # once the image of the fall has been saved, publish it so that
        # the web interface listener could send it to the cloud
        self._img_loc_pub = rospy.Publisher(IMG_LOC_PUB_TOPIC,
                                            String,
                                            queue_size=1)

        # Publish the saved map and pose from slam once a fall is detected
        self._pose_pub = rospy.Publisher(SLAM_POSE_PUB,
                                          Pose,
                                          queue_size=1)
        self._map_pub = rospy.Publisher(SLAM_MAP_PUB,
                                         OccupancyGrid,
                                         queue_size=1)

        """
        UTIL SETUP
        """
        # for image conversions from ros Image to opencv mat
        self.bridge = CvBridge()

        # track_id indicates the person of interest. The robot should first
        # track the first person it sees.
        self.dominant_track = 1

        self.current_pose = None
        self.current_map = None

    def preproc(self, image):
        """
        Given an opencv image, resize and convert the color scheme
        to RGB
        """
        image = self.resize_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def publish_current_loc(self):
        """
        publish the current map and pose to the image processor which
        turns the occupancy grid and pose into a jpeg and send it to the
        web interface for display
        """
        self._map_pub.publish(self.current_map)
        self._pose_pub.publish(self.current_pose)

    def _pose_callback(self, msg):
        self.current_pose = msg.pose
        
    def _map_callback(self, msg):
        self.current_map = msg

    def _image_callback(self, msg):
        """
        Parse the ros Image and convert it to openCV image format
        """
        np_arr = np.fromstring(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if not self.visualize:
            cv2.imshow('image', cv_image)
            cv2.waitKey(1)
        cv_image = self.preproc(cv_image)
        self.process(cv_image)

    def publish_location(self, pos, volume):
        """
        For the sake of simplicity, Point.x and Point.y represents
        the respective position of the center of the bounding box in
        the image frame. Point.z represents the subject's distance
        from the robot. This should be consumed by the PID controller node
        """
        vec = Point(np.float64(pos[0]), np.float64(pos[1]), volume)
        self._location_pub.publish(vec)

    def publish_action(self, action):
        """
        Possible action states:
        Standing Up
        Walking
        Sitting
        Fall Down
        Lying Down
        Pending (ST-GCN requires 30 previous timesteps for prediction)
        """
        string_msg = String()
        string_msg.data = action
        self._action_pub.publish(string_msg)

    def publish_img_location(self, filename):
        """
        Simply publishes the location of the fall image on the local
        filesystem whenever called
        """
        loc_msg = String()
        loc_msg.data = filename
        self._img_loc_pub.publish(filename)

    def calc_volume(self, x1, y1, x2, y2):
        """
        For the sake of latency and fps, we opted against using
        the depth image from the robot. Instead we use the area of
        the bounding box to determine the subject's distance from the robot
        """
        ydiff = abs(y2-y1)
        xdiff = abs(x2-x1)
        return ydiff * xdiff

    def process(self, frame):
        """
        Given an opencv image, detect the person, their pose, and their action.
        Send ultimate output to the respective topics
        """
        # detect people in the frame
        detected = self.detect_model.detect(frame, need_resize=False, expand_bb=10)
        # run Intersection over Union (IoU) match and kalman filter based tracker
        self.tracker.predict()

        for track in self.tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            # Predict skeleton pose of each bboxs.
            poses = self.pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]


        self.tracker.update(detections)

        # we prefer false positives over false negatives so if we see at least
        # one "Fall Down" state, we prioritize that over all other observations
        dominant_state = ""
        preferred_state = ""

        dominant_location = [0.0, 0.0]
        dominant_volume = 0.0

        # go through every single tracks
        for i, track in enumerate(self.tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)
            action = 'pending...'
            clr = (0, 255, 0)

            # if the pose has been detected at least 30 times (timeframe
            # in the "temporal" aspect of spatial temporal GCN)
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                
                # predict the action based on the temporal skeletal graph
                # going back 30 frames
                out = self.action_model.predict(pts, frame.shape[:2])
                action_name = self.action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)

                if track_id == self.dominant_track:
                    dominant_state = action_name

                if action_name == 'Fall Down':
                    clr = (255, 0, 0)
                    preferred_state = action_name if track_id == self.dominant_track else ""
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)
                    preferred_state = action_name if track_id == self.dominant_track else ""

            if track_id == self.dominant_track:
                dominant_location = [center[0], center[1]]
                dominant_volume = self.calc_volume(bbox[0], bbox[1], bbox[2], bbox[3])

            # visualization code
            if track.time_since_update == 0:
                if self.visualize:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (self.f, 1.0 / (time.time() - self.fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]
        self.fps_time = time.time()

        if self.visualize:
            cv2.imshow("detected frame", frame)
            cv2.waitKey(1)

        if preferred_state != "":
            dominant_state = preferred_state

        # when a fall is detected, save the current frame and save it to serve it on
        # the web interface later.
        if dominant_state.lower() == "fall down":
            fn = "imgs/" + str(datetime.datetime.now()) + ".jpg"
            cv2.imwrite(fn, frame)
            fn = "../FallDetection/" + fn
            self.publish_img_location(fn)
            self.publish_current_loc()


        all_ids = [t.track_id for t in self.tracker.tracks]
        if self.dominant_track not in all_ids and len(self.tracker.tracks) > 0:
            print("Dominant Track Lost")
            self.dominant_track = int(input("Input new dominant track: "))

        # publish the state and location of the person
        if self.dominant_track in all_ids:
            self.publish_action(dominant_state)
            self.publish_location(dominant_location, dominant_volume)
        else:
            self.publish_location([0, 0], 0)

    def run(self):
        # if we're using the webcam
        if self.camera != None:
            while not rospy.is_shutdown() and self.cam.grabbed():
                self.process(self.cam.getitem())
                self.rate.sleep()
            return
        # if we're using the built in camera
        while not rospy.is_shutdown():
            self.rate.sleep()



if __name__ == "__main__":
    rospy.init_node("Detector")
    rospy.sleep(2)

    det = Detector(camera=CAM_ID)

    try:
        det.run()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS NODE INTERRUPTED")
