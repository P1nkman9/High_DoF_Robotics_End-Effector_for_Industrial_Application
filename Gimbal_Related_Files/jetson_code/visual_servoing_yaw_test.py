#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive single/dual-axis visual servoing on Jetson. Runs the erosion-seg
and crack-det TRT engines per frame, centres the highest-priority target
in the image, and saves the best of up to 3 photos taken while locked.

The first serial port found is assumed to be the yaw axis; if a second is
present, it drives pitch.
"""

import serial
import serial.tools.list_ports
import time
import os
import ctypes
from datetime import datetime
import pyrealsense2 as rs
import numpy as np
import cv2
import tensorrt as trt

# ================= Tuning =================
KP_YAW = 0.0005
KP_PITCH = 0.0005

DIR_YAW = 1
DIR_PITCH = 1

# Measured yaw travel: front=0.3, left=-1.4, right=2.0 (rad).
YAW_MIN = -1.4
YAW_MAX = 2.0
YAW_INIT = 0.3

# Measured pitch travel: level=2.8, max upward tilt=2.4 (rad).
PITCH_MIN = 2.4
PITCH_MAX = 3.2
PITCH_INIT = 2.8

DEADZONE      = 30    # pixels from image centre counted as "locked"
UNLOCK_FRAMES = 5     # frames outside the deadzone before a lock ends

SAVE_DIR = "erosion_crack_records"
MAX_PHOTOS_PER_LOCK = 3
PHOTO_INTERVAL    = 1.0

# BGR colours used to annotate detections.
CLASS_COLORS = {
    "erosion": (0, 255, 0),
    "crack": (0, 0, 255),
}

# ================= TensorRT Model =================
LIBCUDART_PATH = "/usr/local/cuda-10.2/targets/aarch64-linux/lib/libcudart.so.10.2"
ENGINE_SEG = "/home/fyp/robot_project/defect_detection/erosion_fp16.engine"   # erosion seg
ENGINE_DET = "/home/fyp/robot_project/defect_detection/crack_fp16.engine"    # crack det

INPUT_W = 640
INPUT_H = 640

CONF_DET = 0.25
CONF_SEG = 0.25
IOU_THRESH = 0.50
MASK_THRESH = 0.50
# =====================================================

# ---------- CUDA Runtime (cudart) ----------
_libcudart = ctypes.CDLL(LIBCUDART_PATH)
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

def _check(status, msg):
    if status != 0:
        raise RuntimeError(f"{msg} (cudaError={status})")

def cuda_malloc(nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    _check(_libcudart.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc failed")
    return ptr.value

def cuda_free(ptr: int):
    _check(_libcudart.cudaFree(ctypes.c_void_p(ptr)), "cudaFree failed")

def cuda_memcpy(dst_ptr: int, src_ptr: int, nbytes: int, kind: int):
    _check(_libcudart.cudaMemcpy(ctypes.c_void_p(dst_ptr), ctypes.c_void_p(src_ptr), nbytes, kind),
           "cudaMemcpy failed")

def cuda_memcpy_htod(dst_dev: int, src_host: np.ndarray):
    cuda_memcpy(dst_dev, src_host.ctypes.data, src_host.nbytes, cudaMemcpyHostToDevice)

def cuda_memcpy_dtoh(dst_host: np.ndarray, src_dev: int):
    cuda_memcpy(dst_host.ctypes.data, src_dev, dst_host.nbytes, cudaMemcpyDeviceToHost)

# ---------- Utils ----------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = im.shape[:2]
    nh, nw = new_shape
    r = min(nh / h, nw / w)
    resized = (int(round(w * r)), int(round(h * r)))
    im2 = cv2.resize(im, resized, interpolation=cv2.INTER_LINEAR)
    padw = nw - resized[0]
    padh = nh - resized[1]
    left = int(round(padw / 2 - 0.1))
    right = int(round(padw / 2 + 0.1))
    top = int(round(padh / 2 - 0.1))
    bottom = int(round(padh / 2 + 0.1))
    im3 = cv2.copyMakeBorder(im2, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im3, r, (left, top)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def box_iou(box1, box2):
    x1 = np.maximum(box1[0], box2[:, 0])
    y1 = np.maximum(box1[1], box2[:, 1])
    x2 = np.minimum(box1[2], box2[:, 2])
    y2 = np.minimum(box1[3], box2[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1 + area2 - inter + 1e-9)

def nms(boxes, scores, iou_thres):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = box_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep

# ---------- Generic TRT runner ----------
class TRTRunner:
    def __init__(self, engine_path: str):
        print(f"[INFO] Loading TRT engine: {engine_path}")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.in_idxs, self.out_idxs = [], []
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.in_idxs.append(i)
            else:
                self.out_idxs.append(i)
        
        self.in_idx = self.in_idxs[0]
        self.in_name = self.engine.get_binding_name(self.in_idx)

        in_shape = tuple(self.context.get_binding_shape(self.in_idx))
        if -1 in in_shape:
            self.context.set_binding_shape(self.in_idx, (1, 3, INPUT_H, INPUT_W))

        self.h = {}          
        self.d = {}          
        self.bindings = [0] * self.engine.num_bindings

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.context.get_binding_shape(i))
            host = np.empty(shape, dtype=np.float32)
            dev = cuda_malloc(host.nbytes)
            self.h[name] = host
            self.d[name] = dev
            self.bindings[i] = dev

        self.out_names = [self.engine.get_binding_name(i) for i in self.out_idxs]

    def infer(self, input_chw: np.ndarray):
        np.copyto(self.h[self.in_name], input_chw)
        cuda_memcpy_htod(self.d[self.in_name], self.h[self.in_name])
        ok = self.context.execute_v2(self.bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 failed")
        outs = {}
        for name in self.out_names:
            cuda_memcpy_dtoh(self.h[name], self.d[name])
            outs[name] = self.h[name].copy()
        return outs

    def __del__(self):
        try:
            for ptr in self.d.values():
                cuda_free(ptr)
        except Exception:
            pass

# ---------- Postprocess Functions ----------
def postprocess_seg(outs, orig_shape, r, pad):
    det = None
    proto = None
    for k, v in outs.items():
        if v.ndim == 3 and v.shape[0] == 1 and v.shape[1] >= 6 and v.shape[2] >= 1000:
            det = v
        elif v.ndim == 4 and v.shape[0] == 1:
            proto = v
    if det is None or proto is None:
        return [], [], [], []

    det = det[0].transpose(1, 0)
    boxes_xywh = det[:, 0:4]
    cls_score = det[:, 4]
    mask_coef = det[:, 5:5+32]

    scores = cls_score
    keep = scores > CONF_SEG
    if not np.any(keep):
        return [], [], [], []

    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    mask_coef = mask_coef[keep]

    boxes = xywh2xyxy(boxes_xywh)
    keep_idx = nms(boxes, scores, IOU_THRESH)
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    mask_coef = mask_coef[keep_idx]

    proto = proto[0]
    nm, mh, mw = proto.shape
    proto_flat = proto.reshape(nm, -1)

    masks_640 = []
    for i in range(boxes.shape[0]):
        m = sigmoid(mask_coef[i] @ proto_flat).reshape(mh, mw).astype(np.float32)
        m = cv2.resize(m, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        masks_640.append(m)

    padw, padh = pad
    h0, w0 = orig_shape

    boxes_out, masks_out = [], []
    for i, m in enumerate(masks_640):
        x1, y1, x2, y2 = boxes[i]
        x1 = max(0, min(w0 - 1, (x1 - padw) / r))
        y1 = max(0, min(h0 - 1, (y1 - padh) / r))
        x2 = max(0, min(w0 - 1, (x2 - padw) / r))
        y2 = max(0, min(h0 - 1, (y2 - padh) / r))
        boxes_out.append([x1, y1, x2, y2])

        x_start, y_start = int(padw), int(padh)
        x_end, y_end = int(padw + r * w0), int(padh + r * h0)
        m_crop = m[y_start:y_end, x_start:x_end]
        m_orig = cv2.resize(m_crop, (w0, h0), interpolation=cv2.INTER_LINEAR)
        masks_out.append((m_orig > MASK_THRESH).astype(np.uint8))

    cls_ids = [0] * len(scores)
    return boxes_out, scores.tolist(), cls_ids, masks_out

def parse_det_output(out: np.ndarray):
    if out.ndim != 3 or out.shape[0] != 1:
        raise ValueError(f"Unexpected det output shape: {out.shape}")
    _, a, b = out.shape
    if b >= a:
        N, C = b, a
        pred = out[0].transpose(1, 0)
    else:
        N, C = a, b
        pred = out[0]

    boxes = pred[:, 0:4]
    rest = pred[:, 4:]

    if rest.shape[1] >= 2:
        obj = rest[:, 0]
        cls_scores = rest[:, 1:]
        if cls_scores.shape[1] >= 1:
            cid = np.argmax(cls_scores, axis=1)
            cls = cls_scores[np.arange(N), cid]
            scores = obj * cls
            return boxes, scores, cid
            
    cid = np.zeros((N,), dtype=np.int32)
    scores = rest[:, 0] if rest.shape[1] >= 1 else np.zeros((N,), dtype=np.float32)
    return boxes, scores, cid

def postprocess_det(outs, orig_shape, r, pad):
    out = list(outs.values())[0]
    boxes_xywh, scores, cls_ids = parse_det_output(out)

    keep = scores > CONF_DET
    if not np.any(keep):
        return [], [], []

    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    boxes = xywh2xyxy(boxes_xywh)
    final_boxes, final_scores, final_cls = [], [], []
    for cid in np.unique(cls_ids):
        idx = np.where(cls_ids == cid)[0]
        b = boxes[idx]
        s = scores[idx]
        keep_idx = nms(b, s, IOU_THRESH)
        for k in keep_idx:
            final_boxes.append(b[k])
            final_scores.append(float(s[k]))
            final_cls.append(int(cid))

    padw, padh = pad
    h0, w0 = orig_shape
    mapped = []
    for x1, y1, x2, y2 in final_boxes:
        x1 = max(0, min(w0 - 1, (x1 - padw) / r))
        y1 = max(0, min(h0 - 1, (y1 - padh) / r))
        x2 = max(0, min(w0 - 1, (x2 - padw) / r))
        y2 = max(0, min(h0 - 1, (y2 - padh) / r))
        mapped.append([x1, y1, x2, y2])

    return mapped, final_scores, final_cls

# ================= Core tracking program =================

def find_stm32_ports():
    print("Scanning for STM32 devices...")
    ports = [p.device for p in serial.tools.list_ports.comports() if 'ACM' in p.device or 'USB' in p.device]
    return ports

def run_yolo_inference(color_image, seg_runner, det_runner):
    """Run both engines and return the highest-priority hit (crack > erosion)."""
    h0, w0 = color_image.shape[:2]

    img_lb, r, pad = letterbox(color_image, (INPUT_H, INPUT_W))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.transpose(img_rgb, (2, 0, 1))[None, ...]
    inp = np.ascontiguousarray(inp, dtype=np.float32)

    outs_seg = seg_runner.infer(inp)
    outs_det = det_runner.infer(inp)

    e_boxes, e_scores, _, e_masks = postprocess_seg(outs_seg, (h0, w0), r, pad)
    c_boxes, c_scores, _ = postprocess_det(outs_det, (h0, w0), r, pad)

    if len(c_boxes) > 0:
        box = c_boxes[0]
        return (box[0], box[1], box[2], box[3]), c_scores[0], "crack"

    if len(e_boxes) > 0:
        box = e_boxes[0]
        return (box[0], box[1], box[2], box[3]), e_scores[0], "erosion"

    return None

def _save_best(candidates: list, save_dir: str):
    """Write the highest-confidence frame from one lock session to disk."""
    best_img, best_conf, best_class = max(candidates, key=lambda x: x[1])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
    filename  = os.path.join(save_dir, f"{best_class}_{timestamp}.jpg")
    cv2.imwrite(filename, best_img)
    confs = [f"{c[1]:.3f}" for c in candidates]
    print(f"[SAVE] [{best_class.upper()}] best of {len(candidates)} (confs: {confs}) -> conf={best_conf:.3f} | {filename}")


def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"[INFO] Created save directory: {SAVE_DIR}/")

    print("\n[INFO] Initialising TensorRT engines...")
    try:
        seg_runner = TRTRunner(ENGINE_SEG)
        det_runner = TRTRunner(ENGINE_DET)
        print("[INFO] Both TensorRT engines loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load TensorRT engine: {e}")
        return

    ports = find_stm32_ports()
    if len(ports) == 0:
        print("[ERROR] No STM32 device found. Please connect at least one.")
        return

    print(f"[INFO] Found {len(ports)} serial port(s): {ports}")

    ser_yaw = None
    ser_pitch = None

    try:
        if len(ports) >= 1:
            ser_yaw = serial.Serial(ports[0], 115200, timeout=0.1)
            print(f"--> [YAW  motor] connected to {ports[0]}")

        if len(ports) >= 2:
            ser_pitch = serial.Serial(ports[1], 115200, timeout=0.1)
            print(f"--> [PITCH motor] connected to {ports[1]}")
        else:
            print("--> [INFO] Single-motor mode — yaw tracking only.")

        print("[INFO] Waiting for FOC calibration (5 s)...")
        time.sleep(5)

        if ser_yaw:   ser_yaw.reset_input_buffer()
        if ser_pitch: ser_pitch.reset_input_buffer()

        print("[INFO] Moving gimbal to working start position...")
        if ser_yaw:
            ser_yaw.write(f"T{YAW_INIT:.4f}\n".encode('utf-8'))
        if ser_pitch:
            ser_pitch.write(f"T{PITCH_INIT:.4f}\n".encode('utf-8'))
        time.sleep(2)

    except Exception as e:
        print(f"[ERROR] Failed to open serial port: {e}")
        return

    current_yaw   = YAW_INIT
    current_pitch = PITCH_INIT

    print("[INFO] Starting D435 colour stream...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"[ERROR] Camera failed to start: {e}")
        return

    CENTER_X = 640 / 2
    CENTER_Y = 480 / 2

    # Candidate buffer for the current lock session.
    photo_candidates   = []    # (img_copy, conf, class_name)
    last_capture_time  = 0.0
    was_locked         = False
    unlock_frames      = 0

    print("\n[INFO] System ready. (single/dual motor adaptive)")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            raw_image    = np.asanyarray(color_frame.get_data())
            display_image = raw_image.copy()

            detection_result = run_yolo_inference(raw_image, seg_runner, det_runner)

            if detection_result is not None:
                (x_min, y_min, x_max, y_max), conf, class_name = detection_result

                box_color = CLASS_COLORS.get(class_name, (0, 255, 255))

                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2

                err_x = cx - CENTER_X
                err_y = cy - CENTER_Y

                # Draw annotations on display_image so the saved frame has them too.
                cv2.rectangle(display_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), box_color, 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(display_image, label, (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                cv2.circle(display_image, (int(cx), int(cy)), 5, box_color, -1)

                if abs(err_x) > DEADZONE or abs(err_y) > DEADZONE:
                    unlock_frames += 1
                    if was_locked and unlock_frames >= UNLOCK_FRAMES:
                        if photo_candidates:
                            _save_best(photo_candidates, SAVE_DIR)
                            photo_candidates = []
                        was_locked    = False
                        unlock_frames = 0

                    delta_yaw   = err_x * KP_YAW   * DIR_YAW
                    delta_pitch = err_y * KP_PITCH  * DIR_PITCH

                    current_yaw   = max(YAW_MIN,   min(YAW_MAX,   current_yaw   + delta_yaw))
                    current_pitch = max(PITCH_MIN,  min(PITCH_MAX, current_pitch + delta_pitch))

                    if ser_yaw:
                        ser_yaw.write(f"T{current_yaw:.4f}\n".encode('utf-8'))
                    if ser_pitch:
                        ser_pitch.write(f"T{current_pitch:.4f}\n".encode('utf-8'))

                else:
                    # Locked on target: capture at most MAX_PHOTOS_PER_LOCK frames.
                    unlock_frames = 0
                    was_locked    = True
                    current_time  = time.time()
                    if (len(photo_candidates) < MAX_PHOTOS_PER_LOCK and
                            current_time - last_capture_time > PHOTO_INTERVAL):
                        photo_candidates.append((display_image.copy(), conf, class_name))
                        last_capture_time = current_time
                        print(f"[CANDIDATE] {len(photo_candidates)}/{MAX_PHOTOS_PER_LOCK} "
                              f"buffered (conf={conf:.3f})")
                        if len(photo_candidates) == MAX_PHOTOS_PER_LOCK:
                            _save_best(photo_candidates, SAVE_DIR)
                            photo_candidates = []
                            was_locked = False

            cv2.line(display_image, (int(CENTER_X)-20, int(CENTER_Y)), (int(CENTER_X)+20, int(CENTER_Y)), (0, 255, 0), 2)
            cv2.line(display_image, (int(CENTER_X), int(CENTER_Y)-20), (int(CENTER_X), int(CENTER_Y)+20), (0, 255, 0), 2)

            cv2.imshow("TensorRT YOLO Servoing", display_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        pipeline.stop()
        try:
            if ser_yaw and ser_yaw.is_open:
                ser_yaw.write(f"T{YAW_INIT:.4f}\n".encode('utf-8'))
                time.sleep(0.1)
                ser_yaw.close()
            if ser_pitch and ser_pitch.is_open:
                ser_pitch.write(f"T{PITCH_INIT:.4f}\n".encode('utf-8'))
                time.sleep(0.1)
                ser_pitch.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[INFO] System shut down safely.")

if __name__ == "__main__":
    main()