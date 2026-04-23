#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson-side ROS node that detects surface defects with two TensorRT engines
(erosion seg + crack det), centres the gimbal on each hit, back-projects the
aligned D435 depth through the gimbal chain to produce a map-frame pose, and
publishes it as /defect_candidates. A snapshot is also written to
~/defect_records/.

Assumes: ROS Noetic sourced, ROS_MASTER_URI pointing at the AGV, SLAM
publishing map -> base_footprint, both engines present at the paths below,
and two STM32 gimbal boards on USB.
"""

import os
import sys
import math
import time
import ctypes
from datetime import datetime

import numpy as np
import cv2
import serial
import serial.tools.list_ports
import pyrealsense2 as rs

# ── ROS ───────────────────────────────────────────────────────────────────────
try:
    import rospy
    import tf
    import tf.transformations
    from geometry_msgs.msg import PoseStamped
except ImportError:
    print("ERROR: ROS Python packages not found.")
    print("Please run:  source /opt/ros/noetic/setup.bash")
    sys.exit(1)

# ── TensorRT ──────────────────────────────────────────────────────────────────
try:
    import tensorrt as trt
except ImportError:
    print("ERROR: tensorrt not found. Is it installed on this Jetson?")
    sys.exit(1)


# ── TensorRT engines ─────────────────────────────────────────────────────────
LIBCUDART_PATH = "/usr/local/cuda-10.2/targets/aarch64-linux/lib/libcudart.so.10.2"
ENGINE_SEG = "/home/fyp/robot_project/defect_detection/erosion_fp16.engine"
ENGINE_DET = "/home/fyp/robot_project/defect_detection/crack_fp16.engine"

INPUT_W    = 640
INPUT_H    = 640

CONF_DET    = 0.25
CONF_SEG    = 0.25
IOU_THRESH  = 0.50
MASK_THRESH = 0.50


# ── RealSense ────────────────────────────────────────────────────────────────
COLOR_W   = 640
COLOR_H   = 480
COLOR_FPS = 30

# Valid depth range. D435 readings under ~15 cm are noise, readings past
# 5 m lose accuracy.
DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 5.0

# Half-window for a median filter over the bbox-centre depth pixel.
DEPTH_SAMPLE_RADIUS = 5


# ── Visual servoing ──────────────────────────────────────────────────────────
# Proportional gain: pixel error -> per-frame angle increment (rad).
KP_YAW   = 0.0005
KP_PITCH = 0.0005

# Sign of the axis output. Flip if the gimbal drives away from the target
# instead of toward it.
DIR_YAW   = 1
DIR_PITCH = 1

# Measured mechanical travel (rad). Stay ~0.1 rad inside the hard stops to
# avoid cable wrap-up and motor stall damage.
MAX_YAW_ANGLE   =  2.0
MIN_YAW_ANGLE   = -1.4
MAX_PITCH_ANGLE =  2.8
MIN_PITCH_ANGLE =  2.4

# Working-zero angles the gimbal is commanded to after boot-up (boards
# power on at mechanical zero, not the working-zero).
YAW_INIT   = 0.3
PITCH_INIT = 2.8

DEADZONE = 30           # pixels; within this, trigger localization
PUBLISH_COOLDOWN = 5.0  # s; de-dupes records at the same spot

SAVE_DIR = os.path.expanduser("~/defect_records")


# ── Gimbal -> base_footprint extrinsics ─────────────────────────────────────
# base_footprint convention: X fwd, Y left, Z up.

# Gimbal mount position in base_footprint (m).
GIMBAL_MOUNT_X = 0.0
GIMBAL_MOUNT_Y = 0.0
GIMBAL_MOUNT_Z = 0.20

# Fixed rotation of the gimbal base relative to base_footprint (rad). Zero
# if the gimbal forward direction coincides with the AGV forward direction.
GIMBAL_MOUNT_YAW   = 0.0
GIMBAL_MOUNT_PITCH = 0.0
GIMBAL_MOUNT_ROLL  = 0.0

# Camera optical centre offset from the gimbal rotation centre, expressed
# in the gimbal-head frame (x fwd, y left, z up) at yaw=pitch=0.
CAM_OFFSET_X = 0.0
CAM_OFFSET_Y = 0.0
CAM_OFFSET_Z = 0.0

# Sign of the yaw command vs. ROS right-hand-rule about Z.
# +1 if a positive T command turns the camera CCW seen from above, else -1.
YAW_ROT_SIGN = -1

# +1 if a positive pitch command tilts the camera down, else -1.
PITCH_ROT_SIGN = 1

# Rotation axis assignment. Override if the mechanics don't follow the
# usual yaw-about-Z, pitch-about-Y convention.
YAW_AXIS   = 'Z'
PITCH_AXIS = 'Y'


# ── CUDA helpers ─────────────────────────────────────────────────────────────

_libcudart = ctypes.CDLL(LIBCUDART_PATH)
_cudaMemcpyHostToDevice = 1
_cudaMemcpyDeviceToHost = 2


def _cuda_check(status, msg):
    if status != 0:
        raise RuntimeError(f"{msg} (cudaError={status})")


def _cuda_malloc(nbytes):
    ptr = ctypes.c_void_p()
    _cuda_check(_libcudart.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
    return ptr.value


def _cuda_free(ptr):
    _cuda_check(_libcudart.cudaFree(ctypes.c_void_p(ptr)), "cudaFree")


def _cuda_memcpy_htod(dst, src_host):
    _cuda_check(
        _libcudart.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src_host.ctypes.data),
                              src_host.nbytes, _cudaMemcpyHostToDevice),
        "cudaMemcpy H2D")


def _cuda_memcpy_dtoh(dst_host, src):
    _cuda_check(
        _libcudart.cudaMemcpy(ctypes.c_void_p(dst_host.ctypes.data), ctypes.c_void_p(src),
                              dst_host.nbytes, _cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H")


# ── Image preprocessing ──────────────────────────────────────────────────────

def _letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = im.shape[:2]
    nh, nw = new_shape
    r = min(nh / h, nw / w)
    resized = (int(round(w * r)), int(round(h * r)))
    im2 = cv2.resize(im, resized, interpolation=cv2.INTER_LINEAR)
    padw = nw - resized[0]
    padh = nh - resized[1]
    left  = int(round(padw / 2 - 0.1))
    right = int(round(padw / 2 + 0.1))
    top   = int(round(padh / 2 - 0.1))
    bot   = int(round(padh / 2 + 0.1))
    return cv2.copyMakeBorder(im2, top, bot, left, right,
                               cv2.BORDER_CONSTANT, value=color), r, (left, top)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _xywh2xyxy(x):
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _box_iou(box1, box2):
    x1 = np.maximum(box1[0], box2[:, 0])
    y1 = np.maximum(box1[1], box2[:, 1])
    x2 = np.minimum(box1[2], box2[:, 2])
    y2 = np.minimum(box1[3], box2[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (a1 + a2 - inter + 1e-9)


def _nms(boxes, scores, iou_thres):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = _box_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep


# ── TRT inference runner ────────────────────────────────────────────────────

class TRTRunner:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.in_idxs, self.out_idxs = [], []
        for i in range(self.engine.num_bindings):
            (self.in_idxs if self.engine.binding_is_input(i) else self.out_idxs).append(i)

        self.in_idx  = self.in_idxs[0]
        self.in_name = self.engine.get_binding_name(self.in_idx)
        in_shape = tuple(self.context.get_binding_shape(self.in_idx))
        if -1 in in_shape:
            self.context.set_binding_shape(self.in_idx, (1, 3, INPUT_H, INPUT_W))

        self.h = {}
        self.d = {}
        self.bindings = [0] * self.engine.num_bindings
        for i in range(self.engine.num_bindings):
            name  = self.engine.get_binding_name(i)
            shape = tuple(self.context.get_binding_shape(i))
            host  = np.empty(shape, dtype=np.float32)
            dev   = _cuda_malloc(host.nbytes)
            self.h[name] = host
            self.d[name] = dev
            self.bindings[i] = dev
        self.out_names = [self.engine.get_binding_name(i) for i in self.out_idxs]

    def infer(self, inp):
        np.copyto(self.h[self.in_name], inp)
        _cuda_memcpy_htod(self.d[self.in_name], self.h[self.in_name])
        self.context.execute_v2(self.bindings)
        outs = {}
        for name in self.out_names:
            _cuda_memcpy_dtoh(self.h[name], self.d[name])
            outs[name] = self.h[name].copy()
        return outs

    def __del__(self):
        try:
            for ptr in self.d.values():
                _cuda_free(ptr)
        except Exception:
            pass


# ── Post-processing ─────────────────────────────────────────────────────────

def _postprocess_seg(outs, orig_shape, r, pad):
    det = proto = None
    for v in outs.values():
        if v.ndim == 3 and v.shape[0] == 1 and v.shape[1] >= 6 and v.shape[2] >= 1000:
            det = v
        elif v.ndim == 4 and v.shape[0] == 1:
            proto = v
    if det is None or proto is None:
        return [], [], [], []

    det    = det[0].transpose(1, 0)
    boxes  = det[:, 0:4]
    scores = det[:, 4]
    mcoefs = det[:, 5:37]

    keep = scores > CONF_SEG
    if not np.any(keep):
        return [], [], [], []
    boxes, scores, mcoefs = boxes[keep], scores[keep], mcoefs[keep]
    boxes = _xywh2xyxy(boxes)
    ki    = _nms(boxes, scores, IOU_THRESH)
    boxes, scores, mcoefs = boxes[ki], scores[ki], mcoefs[ki]

    proto    = proto[0]
    nm, mh, mw = proto.shape
    proto_flat = proto.reshape(nm, -1)
    padw, padh = pad
    h0, w0 = orig_shape

    boxes_out, masks_out = [], []
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        boxes_out.append([
            max(0, min(w0 - 1, (x1 - padw) / r)),
            max(0, min(h0 - 1, (y1 - padh) / r)),
            max(0, min(w0 - 1, (x2 - padw) / r)),
            max(0, min(h0 - 1, (y2 - padh) / r)),
        ])
        m = _sigmoid(mcoefs[i] @ proto_flat).reshape(mh, mw).astype(np.float32)
        m = cv2.resize(m, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        xs, ys = int(padw), int(padh)
        xe, ye = int(padw + r * w0), int(padh + r * h0)
        m_orig = cv2.resize(m[ys:ye, xs:xe], (w0, h0), interpolation=cv2.INTER_LINEAR)
        masks_out.append((m_orig > MASK_THRESH).astype(np.uint8))

    return boxes_out, scores.tolist(), [0] * len(scores), masks_out


def _postprocess_det(outs, orig_shape, r, pad):
    out = list(outs.values())[0]
    if out.ndim != 3 or out.shape[0] != 1:
        return [], [], []
    _, a, b = out.shape
    pred = out[0].transpose(1, 0) if b >= a else out[0]
    N    = pred.shape[0]

    boxes = pred[:, 0:4]
    rest  = pred[:, 4:]
    if rest.shape[1] >= 2:
        obj = rest[:, 0]
        cls = rest[:, 1:]
        cid = np.argmax(cls, axis=1)
        scores = obj * cls[np.arange(N), cid]
    else:
        cid    = np.zeros(N, dtype=np.int32)
        scores = rest[:, 0] if rest.shape[1] >= 1 else np.zeros(N)

    keep = scores > CONF_DET
    if not np.any(keep):
        return [], [], []
    boxes, scores, cid = _xywh2xyxy(boxes[keep]), scores[keep], cid[keep]

    padw, padh = pad
    h0, w0 = orig_shape
    final_boxes, final_scores, final_cls = [], [], []
    for c in np.unique(cid):
        idx = np.where(cid == c)[0]
        ki  = _nms(boxes[idx], scores[idx], IOU_THRESH)
        for k in ki:
            x1, y1, x2, y2 = boxes[idx[k]]
            final_boxes.append([
                max(0, min(w0 - 1, (x1 - padw) / r)),
                max(0, min(h0 - 1, (y1 - padh) / r)),
                max(0, min(w0 - 1, (x2 - padw) / r)),
                max(0, min(h0 - 1, (y2 - padh) / r)),
            ])
            final_scores.append(float(scores[idx[k]]))
            final_cls.append(int(c))
    return final_boxes, final_scores, final_cls


# ── Coordinate-transform helpers ────────────────────────────────────────────

def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)


def _rot_by_axis(axis, angle):
    return {'X': _rot_x, 'Y': _rot_y, 'Z': _rot_z}[axis](angle)


# RealSense optical (X right, Y down, Z forward) -> ROS (X fwd, Y left, Z up).
_R_OPT_TO_ROS = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)


def _build_mount_rotation():
    return (_rot_z(GIMBAL_MOUNT_YAW)
            @ _rot_y(GIMBAL_MOUNT_PITCH)
            @ _rot_x(GIMBAL_MOUNT_ROLL))


_R_MOUNT = _build_mount_rotation()


def transform_to_map(point_cam_optical, current_yaw, current_pitch, tf_listener):
    """
    Lift a 3D point from the RealSense optical frame to the map frame.

    Chain: optical -> ROS-camera -> gimbal head (+CAM_OFFSET) -> apply pitch
    (inner axis) -> apply yaw (outer axis) -> base_footprint (+mount rot &
    translation) -> map (TF).

    Returns the map-frame point as a shape-(3,) array, or None if TF fails.
    """
    P = np.array(point_cam_optical, dtype=np.float64)

    P = _R_OPT_TO_ROS @ P
    P = P + np.array([CAM_OFFSET_X, CAM_OFFSET_Y, CAM_OFFSET_Z])

    # Pitch axis is inboard of yaw, so pitch rotates first.
    R_pitch = _rot_by_axis(PITCH_AXIS, PITCH_ROT_SIGN * current_pitch)
    P = R_pitch @ P

    R_yaw = _rot_by_axis(YAW_AXIS, YAW_ROT_SIGN * current_yaw)
    P = R_yaw @ P

    P = _R_MOUNT @ P + np.array([GIMBAL_MOUNT_X, GIMBAL_MOUNT_Y, GIMBAL_MOUNT_Z])

    try:
        (trans, rot) = tf_listener.lookupTransform(
            "map", "base_footprint", rospy.Time(0))
        T = tf.transformations.quaternion_matrix(rot)
        return T[:3, :3] @ P + np.array(trans)
    except (tf.LookupException, tf.ConnectivityException,
            tf.ExtrapolationException) as e:
        rospy.logwarn("defect_localizer: TF map<-base_footprint failed: %s", e)
        return None


def get_robust_depth(depth_frame, u, v):
    """
    Median of valid depths in a (2*DEPTH_SAMPLE_RADIUS+1) square around (u, v).
    Returns None if no pixel in the window is inside [DEPTH_MIN_M, DEPTH_MAX_M].
    """
    h = depth_frame.get_height()
    w = depth_frame.get_width()
    r = DEPTH_SAMPLE_RADIUS

    depths = []
    for row in range(max(0, v - r), min(h, v + r + 1)):
        for col in range(max(0, u - r), min(w, u + r + 1)):
            d = depth_frame.get_distance(col, row)
            if DEPTH_MIN_M < d < DEPTH_MAX_M:
                depths.append(d)

    if not depths:
        return None
    return float(np.median(depths))


# ── Main node ────────────────────────────────────────────────────────────────

class DefectLocalizerNode:

    _BOX_COLORS = {
        "erosion": (0, 255, 0),
        "crack":   (0, 0, 255),
    }

    def __init__(self):
        rospy.init_node("defect_localizer", anonymous=False)
        rospy.loginfo("defect_localizer: initialising node...")

        os.makedirs(SAVE_DIR, exist_ok=True)
        rospy.loginfo("defect_localizer: saving images to %s", SAVE_DIR)

        self.tf_listener = tf.TransformListener()

        self.defect_pub = rospy.Publisher(
            "/defect_candidates", PoseStamped, queue_size=10)

        # Track commanded angles (closed-loop FOC: commanded ~ actual).
        self.current_yaw   = YAW_INIT
        self.current_pitch = PITCH_INIT
        self._last_publish_time = 0.0

        self._init_gimbal()
        self._init_trt()
        self._init_realsense()

        rospy.loginfo("defect_localizer: all components ready.")

    # ── Gimbal ──────────────────────────────────────────────────────────
    def _init_gimbal(self):
        rospy.loginfo("defect_localizer: scanning for STM32 serial ports...")
        ports = [p.device for p in serial.tools.list_ports.comports()
                 if "ACM" in p.device or "USB" in p.device]

        self.ser_yaw   = None
        self.ser_pitch = None

        if not ports:
            rospy.logwarn("defect_localizer: no STM32 found -- gimbal servo disabled.")
            return

        rospy.loginfo("defect_localizer: found %d port(s): %s", len(ports), ports)

        # Port enumeration order is not stable; swap the two lines below if the
        # axes are reversed after a reboot.
        if len(ports) >= 1:
            self.ser_yaw = serial.Serial(ports[0], 115200, timeout=0.1)
            rospy.loginfo("defect_localizer: yaw  port -> %s", ports[0])
        if len(ports) >= 2:
            self.ser_pitch = serial.Serial(ports[1], 115200, timeout=0.1)
            rospy.loginfo("defect_localizer: pitch port -> %s", ports[1])
        else:
            rospy.logwarn("defect_localizer: only 1 STM32 found -- pitch axis disabled.")

        rospy.loginfo("defect_localizer: waiting 5 s for FOC alignment...")
        rospy.sleep(5.0)
        if self.ser_yaw:   self.ser_yaw.reset_input_buffer()
        if self.ser_pitch: self.ser_pitch.reset_input_buffer()

        # Boards power up at mechanical zero; drive to the working zero.
        rospy.loginfo("defect_localizer: moving gimbal to working start position...")
        if self.ser_yaw:
            self.ser_yaw.write(f"T{YAW_INIT:.4f}\n".encode())
        if self.ser_pitch:
            self.ser_pitch.write(f"T{PITCH_INIT:.4f}\n".encode())
        rospy.sleep(2.0)
        rospy.loginfo("defect_localizer: gimbal ready.")

    def _send_gimbal(self, yaw_rad, pitch_rad):
        if self.ser_yaw:
            self.ser_yaw.write(f"T{yaw_rad:.4f}\n".encode())
        if self.ser_pitch:
            self.ser_pitch.write(f"T{pitch_rad:.4f}\n".encode())
        self.current_yaw   = yaw_rad
        self.current_pitch = pitch_rad

    def _return_to_zero(self):
        # Return to the working zero, not the mechanical zero.
        rospy.loginfo("defect_localizer: returning gimbal to start position...")
        self._send_gimbal(YAW_INIT, PITCH_INIT)
        rospy.sleep(1.0)

    # ------------------------------------------------ TRT initialization ------
    def _init_trt(self):
        rospy.loginfo("defect_localizer: loading TRT engines...")
        self.seg_runner = TRTRunner(ENGINE_SEG)
        self.det_runner = TRTRunner(ENGINE_DET)
        rospy.loginfo("defect_localizer: TRT engines loaded.")

    # ------------------------------------------------ RealSense initialization -
    def _init_realsense(self):
        rospy.loginfo("defect_localizer: starting RealSense D435...")
        self.pipeline = rs.pipeline()
        cfg = rs.config()

        # Color stream (used for TRT inference and visualization)
        cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)

        # Depth stream (new! used for 3D back-projection)
        # Match color resolution so rs.align is a 1:1 mapping.
        cfg.enable_stream(rs.stream.depth, COLOR_W, COLOR_H, rs.format.z16, COLOR_FPS)

        self.rs_profile = self.pipeline.start(cfg)

        self.align = rs.align(rs.stream.color)

        color_stream = self.rs_profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        rospy.loginfo(
            "defect_localizer: color intrinsics -- fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
            self.intrinsics.fx, self.intrinsics.fy,
            self.intrinsics.ppx, self.intrinsics.ppy)
        rospy.loginfo("defect_localizer: RealSense started.")

    # ── Inference ───────────────────────────────────────────────────────
    def _run_inference(self, color_image):
        """
        Run both engines; prefer a crack hit over an erosion hit.
        Returns ((x1,y1,x2,y2), confidence, class_name) or None.
        """
        h0, w0 = color_image.shape[:2]
        img_lb, r, pad = _letterbox(color_image, (INPUT_H, INPUT_W))
        inp = np.ascontiguousarray(
            np.transpose(
                cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
                (2, 0, 1))[None],
            dtype=np.float32)

        outs_seg = self.seg_runner.infer(inp)
        outs_det = self.det_runner.infer(inp)

        c_boxes, c_scores, _ = _postprocess_det(outs_det, (h0, w0), r, pad)
        e_boxes, e_scores, _, _ = _postprocess_seg(outs_seg, (h0, w0), r, pad)

        if c_boxes:
            b = c_boxes[0]
            return (b[0], b[1], b[2], b[3]), c_scores[0], "crack"
        if e_boxes:
            b = e_boxes[0]
            return (b[0], b[1], b[2], b[3]), e_scores[0], "erosion"
        return None

    # ── Publish defect ──────────────────────────────────────────────────
    def _publish_defect(self, map_xyz, class_name, confidence):
        msg = PoseStamped()
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(map_xyz[0])
        msg.pose.position.y = float(map_xyz[1])
        msg.pose.position.z = float(map_xyz[2])
        # Identity orientation — heading is unknown for a point defect.
        msg.pose.orientation.w = 1.0
        self.defect_pub.publish(msg)
        rospy.loginfo(
            "defect_localizer: PUBLISHED [%s conf=%.2f] map=(%.3f, %.3f, %.3f)",
            class_name, confidence, map_xyz[0], map_xyz[1], map_xyz[2])

    # ── Main loop ───────────────────────────────────────────────────────
    def run(self):
        CENTER_X = COLOR_W / 2.0
        CENTER_Y = COLOR_H / 2.0

        rospy.loginfo("defect_localizer: entering main loop. Press q or Ctrl-C to exit.")
        rospy.loginfo("defect_localizer: gimbal/mount parameters are placeholders; "
                      "verify before trusting published coordinates.")

        try:
            while not rospy.is_shutdown():
                frames   = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned  = self.align.process(frames)
                c_frame  = aligned.get_color_frame()
                d_frame  = aligned.get_depth_frame()
                if not c_frame or not d_frame:
                    continue

                color_image  = np.asanyarray(c_frame.get_data())
                display      = color_image.copy()
                depth_text   = ""

                result = self._run_inference(color_image)

                if result is not None:
                    (x1, y1, x2, y2), conf, cls_name = result
                    color = self._BOX_COLORS.get(cls_name, (0, 255, 255))

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    err_x = cx - CENTER_X
                    err_y = cy - CENTER_Y

                    if abs(err_x) > DEADZONE or abs(err_y) > DEADZONE:
                        # Not aligned — drive the gimbal toward the target.
                        new_yaw   = self.current_yaw   + err_x * KP_YAW   * DIR_YAW
                        new_pitch = self.current_pitch + err_y * KP_PITCH * DIR_PITCH
                        new_yaw   = max(MIN_YAW_ANGLE,   min(MAX_YAW_ANGLE,   new_yaw))
                        new_pitch = max(MIN_PITCH_ANGLE, min(MAX_PITCH_ANGLE, new_pitch))
                        self._send_gimbal(new_yaw, new_pitch)

                    else:
                        # Aligned — localize and publish, rate-limited by PUBLISH_COOLDOWN.
                        now = time.time()
                        if now - self._last_publish_time > PUBLISH_COOLDOWN:
                            u = int(round(cx))
                            v = int(round(cy))

                            depth_m = get_robust_depth(d_frame, u, v)

                            if depth_m is None:
                                rospy.logwarn(
                                    "defect_localizer: no valid depth at (%d,%d) "
                                    "-- skipping this detection.", u, v)
                            else:
                                depth_text = f"d={depth_m:.2f}m"

                                # rs2_deproject expects float pixel coords in the aligned color frame.
                                point_cam = rs.rs2_deproject_pixel_to_point(
                                    self.intrinsics, [float(u), float(v)], depth_m)
                                rospy.logdebug(
                                    "defect_localizer: camera frame P=[%.3f, %.3f, %.3f] m",
                                    *point_cam)

                                map_xyz = transform_to_map(
                                    point_cam,
                                    self.current_yaw,
                                    self.current_pitch,
                                    self.tf_listener)

                                if map_xyz is not None:
                                    self._publish_defect(map_xyz, cls_name, conf)

                                    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    img = os.path.join(SAVE_DIR,
                                                        f"{cls_name}_{ts}.jpg")
                                    cv2.imwrite(img, color_image)
                                    rospy.loginfo(
                                        "defect_localizer: image saved -> %s", img)

                                    self._last_publish_time = now

                    cv2.rectangle(display,
                                  (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{cls_name} {conf:.2f} {depth_text}"
                    cv2.putText(display, label,
                                (int(x1), max(int(y1) - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.circle(display, (int(cx), int(cy)), 5, color, -1)

                cxi, cyi = int(CENTER_X), int(CENTER_Y)
                cv2.line(display, (cxi - 20, cyi), (cxi + 20, cyi), (0, 255, 0), 2)
                cv2.line(display, (cxi, cyi - 20), (cxi, cyi + 20), (0, 255, 0), 2)

                cv2.putText(display,
                            f"yaw={math.degrees(self.current_yaw):.1f}  "
                            f"pitch={math.degrees(self.current_pitch):.1f}",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 0), 1)

                cv2.imshow("Defect Localizer", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rospy.loginfo("defect_localizer: q pressed, exiting.")
                    break

        except KeyboardInterrupt:
            rospy.loginfo("defect_localizer: Ctrl-C received.")
        finally:
            self._return_to_zero()
            self.pipeline.stop()
            if self.ser_yaw   and self.ser_yaw.is_open:   self.ser_yaw.close()
            if self.ser_pitch and self.ser_pitch.is_open: self.ser_pitch.close()
            cv2.destroyAllWindows()
            rospy.loginfo("defect_localizer: shutdown complete.")


if __name__ == "__main__":
    try:
        DefectLocalizerNode().run()
    except rospy.ROSInterruptException:
        pass