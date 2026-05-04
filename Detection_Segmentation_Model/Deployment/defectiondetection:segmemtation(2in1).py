#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import ctypes
import numpy as np
import cv2
import pyrealsense2 as rs
import tensorrt as trt

# ========= 你要改这里 =========
LIBCUDART_PATH = "/usr/local/cuda-10.2/targets/aarch64-linux/lib/libcudart.so.10.2"

ENGINE_SEG = "/home/fyp/robot_project/defect_detection/erosion_fp16.engine"   # erosion seg
ENGINE_DET = "/home/fyp/robot_project/defect_detection/crack_fp16.engine"    # crack det 

# 类名（显示用）
NAME_EROSION = "erosion"   
NAME_CRACK = "crack"
# ==============================

INPUT_W = 640
INPUT_H = 640

RS_W, RS_H, RS_FPS = 640, 480, 30

CONF_DET = 0.25
CONF_SEG = 0.25
IOU_THRESH = 0.50

MASK_THRESH = 0.50
MASK_ALPHA = 0.45


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
    resized = (int(round(w * r)), int(round(h * r)))  # (new_w, new_h)
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


# ---------- Generic TRT runner (no pycuda) ----------
class TRTRunner:
    """
    Supports 1 input, N outputs (N>=1), fp32 IO.
    Allocates host/device buffers and runs execute_v2.
    """
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # Find bindings
        self.in_idxs = []
        self.out_idxs = []
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.in_idxs.append(i)
            else:
                self.out_idxs.append(i)
        if len(self.in_idxs) != 1:
            raise RuntimeError(f"Expected 1 input, got {len(self.in_idxs)}")
        self.in_idx = self.in_idxs[0]

        # Set dynamic shape if needed
        in_shape = tuple(self.context.get_binding_shape(self.in_idx))
        if -1 in in_shape:
            self.context.set_binding_shape(self.in_idx, (1, 3, INPUT_H, INPUT_W))
            in_shape = tuple(self.context.get_binding_shape(self.in_idx))

        # Allocate host/device per binding
        self.h = {}
        self.d = {}
        self.bindings = [0] * self.engine.num_bindings

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.context.get_binding_shape(i))
            if -1 in shape:
                # after setting input, output should be known; if not, fail fast
                shape = tuple(self.context.get_binding_shape(i))
                if -1 in shape:
                    raise RuntimeError(f"Dynamic shape still unresolved for binding {name}: {shape}")

            # Use fp32 host buffers (most TRT exports keep fp32 IO even in fp16 engine)
            host = np.empty(shape, dtype=np.float32)
            dev = cuda_malloc(host.nbytes)
            self.h[name] = host
            self.d[name] = dev
            self.bindings[i] = dev

        self.in_name = self.engine.get_binding_name(self.in_idx)
        self.out_names = [self.engine.get_binding_name(i) for i in self.out_idxs]

        print(f"\n=== Engine: {engine_path} ===")
        print("IN :", self.in_name, self.h[self.in_name].shape, self.h[self.in_name].dtype)
        for n in self.out_names:
            print("OUT:", n, self.h[n].shape, self.h[n].dtype)

    def infer(self, input_chw: np.ndarray):
        # input_chw must match input shape
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


# ---------- Postprocess: SEG (expects output0 det + output1 proto like your erosion engine) ----------
def postprocess_seg(outs, orig_shape, r, pad):
    # identify by shape:
    det = None
    proto = None
    for k, v in outs.items():
        if v.ndim == 3 and v.shape[0] == 1 and v.shape[1] >= 6 and v.shape[2] >= 1000:
            det = v
        elif v.ndim == 4 and v.shape[0] == 1:
            proto = v
    if det is None or proto is None:
        return [], [], [], []

    # Your engine: det (1,37,8400) and proto (1,32,160,160)
    det = det[0].transpose(1, 0)  # (N, C)
    boxes_xywh = det[:, 0:4]
    cls_score = det[:, 4]         # single class prob/logit depending export
    mask_coef = det[:, 5:5+32]    # (N,32)

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

    proto = proto[0]  # (32,160,160)
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
        x1 = (x1 - padw) / r
        y1 = (y1 - padh) / r
        x2 = (x2 - padw) / r
        y2 = (y2 - padh) / r
        x1 = max(0, min(w0 - 1, x1))
        y1 = max(0, min(h0 - 1, y1))
        x2 = max(0, min(w0 - 1, x2))
        y2 = max(0, min(h0 - 1, y2))
        boxes_out.append([x1, y1, x2, y2])

        x_start = int(padw)
        y_start = int(padh)
        x_end = int(padw + r * w0)
        y_end = int(padh + r * h0)
        m_crop = m[y_start:y_end, x_start:x_end]
        m_orig = cv2.resize(m_crop, (w0, h0), interpolation=cv2.INTER_LINEAR)
        masks_out.append((m_orig > MASK_THRESH).astype(np.uint8))

    cls_ids = [0] * len(scores)
    return boxes_out, scores.tolist(), cls_ids, masks_out


# ---------- Postprocess: DET (generic YOLO head, try (1,C,N) or (1,N,C)) ----------
def parse_det_output(out: np.ndarray):
    if out.ndim != 3 or out.shape[0] != 1:
        raise ValueError(f"Unexpected det output shape: {out.shape}")

    _, a, b = out.shape
    # decide layout
    if b >= a:
        # (1,C,N)
        C, N = a, b
        pred = out[0].transpose(1, 0)  # (N,C)
    else:
        # (1,N,C)
        N, C = a, b
        pred = out[0]                  # (N,C)

    boxes = pred[:, 0:4]
    rest = pred[:, 4:]

    # common: (x,y,w,h,obj,cls...)
    if rest.shape[1] >= 2:
        obj = rest[:, 0]
        cls_scores = rest[:, 1:]
        if cls_scores.shape[1] >= 1:
            cid = np.argmax(cls_scores, axis=1)
            cls = cls_scores[np.arange(N), cid]
            scores = obj * cls
            return boxes, scores, cid
    # fallback: (x,y,w,h,cls...) single-class or multi-class without obj
    cid = np.zeros((N,), dtype=np.int32)
    scores = rest[:, 0] if rest.shape[1] >= 1 else np.zeros((N,), dtype=np.float32)
    return boxes, scores, cid


def postprocess_det(outs, orig_shape, r, pad):
    # pick first output
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
        x1 = (x1 - padw) / r
        y1 = (y1 - padh) / r
        x2 = (x2 - padw) / r
        y2 = (y2 - padh) / r
        x1 = max(0, min(w0 - 1, x1))
        y1 = max(0, min(h0 - 1, y1))
        x2 = max(0, min(w0 - 1, x2))
        y2 = max(0, min(h0 - 1, y2))
        mapped.append([x1, y1, x2, y2])

    return mapped, final_scores, final_cls


# ---------- Drawing ----------
def overlay_erosion(frame, boxes, scores, masks):
    out = frame.copy()
    color = (0, 255, 0)
    for box, score, mask in zip(boxes, scores, masks):
        m = mask.astype(bool)
        layer = np.zeros_like(out, dtype=np.uint8)
        layer[:] = color
        out[m] = cv2.addWeighted(out, 1 - MASK_ALPHA, layer, MASK_ALPHA, 0)[m]

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{NAME_EROSION} {score:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out

def overlay_crack(frame, boxes, scores):
    out = frame
    color = (0, 0, 255)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{NAME_CRACK} {score:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def main():
    print("[INFO] Loading engines...")
    seg_runner = TRTRunner(ENGINE_SEG)
    det_runner = TRTRunner(ENGINE_DET)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
    pipeline.start(config)
    for _ in range(5):
        pipeline.wait_for_frames()

    last = time.time()
    fps = 0.0
    print("\n[INFO] Running... press 'q' or ESC to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            c = frames.get_color_frame()
            if not c:
                continue
            frame = np.asanyarray(c.get_data())
            h0, w0 = frame.shape[:2]

            # preprocess once
            img_lb, r, pad = letterbox(frame, (INPUT_H, INPUT_W))
            img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = np.transpose(img_rgb, (2, 0, 1))[None, ...]
            inp = np.ascontiguousarray(inp, dtype=np.float32)

            # run both engines
            outs_seg = seg_runner.infer(inp)
            outs_det = det_runner.infer(inp)

            # postprocess
            e_boxes, e_scores, _, e_masks = postprocess_seg(outs_seg, (h0, w0), r, pad)
            c_boxes, c_scores, _ = postprocess_det(outs_det, (h0, w0), r, pad)

            # overlay: erosion mask first, then crack boxes
            vis = overlay_erosion(frame, e_boxes, e_scores, e_masks)
            vis = overlay_crack(vis, c_boxes, c_scores)

            now = time.time()
            dt = now - last
            last = now
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("D435i + TRT (crack det + erosion seg)", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or k == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()