import os
import cv2
import sys
import io
import gc
import re
import ssl
import time
import json
import math
import uuid
import base64
import queue
import signal
import hashlib
import sqlite3
import argparse
import threading
import tempfile
import itertools
import traceback
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict, deque, namedtuple
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
try:
    from openalpr import Alpr
except Exception:
    Alpr = None
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except Exception:
    FastAPI = None
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdfcanvas
except Exception:
    pdfcanvas = None

Ticket = namedtuple("Ticket", ["id", "timestamp", "plate", "plate_conf", "speed", "limit", "vehicle_class", "image_path", "meta"])

class AtomicBool:
    def __init__(self, value=False):
        self.v = value
        self.m = threading.Lock()
    def set(self, value: bool):
        with self.m:
            self.v = bool(value)
    def get(self):
        with self.m:
            return self.v

class AsyncPool:
    def __init__(self, workers=2):
        self.q = queue.Queue()
        self.ws = []
        self.alive = AtomicBool(True)
        for _ in range(max(1, workers)):
            t = threading.Thread(target=self._loop, daemon=True)
            t.start()
            self.ws.append(t)
    def _loop(self):
        while self.alive.get():
            try:
                fn, args, kwargs = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                fn(*args, **kwargs)
            except Exception:
                pass
            finally:
                self.q.task_done()
    def submit(self, fn, *args, **kwargs):
        self.q.put((fn, args, kwargs))
    def join(self):
        self.q.join()
    def stop(self):
        self.alive.set(False)
        for _ in self.ws:
            self.q.put((lambda: None, (), {}))

class RollingStats:
    def __init__(self, size=120):
        self.buf = deque(maxlen=size)
    def push(self, x):
        self.buf.append(float(x))
    def mean(self):
        if not self.buf:
            return 0.0
        return float(sum(self.buf) / len(self.buf))
    def last(self):
        if not self.buf:
            return 0.0
        return float(self.buf[-1])

class Config:
    def __init__(self, d):
        self.d = d
    def get(self, k, default=None):
        return self.d.get(k, default)
    @staticmethod
    def from_args(args):
        return Config({
            "model": args.model,
            "device": args.device,
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "classes": args.classes,
            "line_a": json.loads(args.line_a),
            "line_b": json.loads(args.line_b),
            "line_distance_m": args.line_distance_m,
            "speed_limit_kmh": args.speed_limit_kmh,
            "fps": args.fps,
            "output": args.output,
            "db": args.db,
            "display": args.display,
            "alpr_country": args.alpr_country,
            "alpr_config": args.alpr_config,
            "alpr_runtime": args.alpr_runtime,
            "plate_roi_ratio": args.plate_roi_ratio,
            "tracker_max_age": args.tracker_max_age,
            "tracker_max_iou": args.tracker_max_iou,
            "tracker_n_init": args.tracker_n_init,
            "violation_cooldown": args.violation_cooldown,
            "rest": args.rest,
            "rest_host": args.rest_host,
            "rest_port": args.rest_port
        })

class Detector:
    def __init__(self, model_path, device=None, conf=0.25, iou=0.45, classes=None, imgsz=640):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.imgsz = imgsz
        self.names = self.model.names if hasattr(self.model, "names") else {}
    def detect(self, frame):
        res = self.model.predict(source=frame, device=self.device, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
        out = []
        if not res:
            return out
        r = res[0]
        boxes = r.boxes
        if boxes is None:
            return out
        for b in boxes:
            c = int(b.cls)
            if self.classes is not None and c not in self.classes:
                continue
            xyxy = b.xyxy[0].cpu().numpy()
            conf = float(b.conf[0].cpu().numpy())
            out.append((xyxy, c, conf))
        return out

class Tracker:
    def __init__(self, max_age=60, max_iou_distance=0.7, n_init=3):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, max_iou_distance=max_iou_distance)
    def update(self, detections, frame):
        dets = []
        for xyxy, c, conf in detections:
            x1, y1, x2, y2 = xyxy
            dets.append(([x1, y1, x2 - x1, y2 - y1], conf, c))
        tracks = self.tracker.update_tracks(dets, frame=frame)
        out = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            ltrb = t.to_ltrb()
            out.append((tid, np.array(ltrb, dtype=np.float32)))
        return out

class PerspectiveMapper:
    def __init__(self, src_pts=None, dst_scale=1.0):
        self.src = None if src_pts is None else np.array(src_pts, dtype=np.float32)
        self.dst = None
        self.M = None
        self.dst_scale = float(dst_scale)
    def calibrate_from_lines(self, line_a, line_b, real_distance_m):
        a1, a2 = line_a
        b1, b2 = line_b
        p1 = np.array([(a1[0]+a2[0])/2.0, (a1[1]+a2[1])/2.0], dtype=np.float32)
        p2 = np.array([(b1[0]+b2[0])/2.0, (b1[1]+b2[1])/2.0], dtype=np.float32)
        px_dist = float(np.linalg.norm(p2 - p1))
        ppm = px_dist / max(1e-6, real_distance_m)
        self.M = None
        return ppm
    def pixels_to_meters(self, pixels, ppm):
        return float(pixels) / max(1e-6, ppm)

class AlprEngine:
    def __init__(self, country="us", config="/etc/openalpr/openalpr.conf", runtime_data="/usr/share/openalpr/runtime_data"):
        self.enabled = False
        self.engine = None
        if Alpr is not None:
            try:
                a = Alpr(country, config, runtime_data)
                if a.is_loaded():
                    self.engine = a
                    self.enabled = True
            except Exception:
                self.engine = None
        self.lock = threading.Lock()
    def recognize(self, bgr):
        if not self.enabled:
            return None, 0.0
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self.lock:
            r = self.engine.recognize_array(rgb.tobytes(), w, h, 3)
        plate = None
        conf = 0.0
        if r and "results" in r and len(r["results"]) > 0:
            best = None
            best_c = -1
            for rr in r["results"]:
                for c in rr.get("candidates", []):
                    if c.get("confidence", 0) > best_c:
                        best = c.get("plate", None)
                        best_c = c.get("confidence", 0)
            plate = best
            conf = float(best_c)
        return plate, conf
    def close(self):
        if self.engine is not None:
            try:
                self.engine.unload()
            except Exception:
                pass

class OCRFallback:
    def __init__(self):
        self.ok = pytesseract is not None
    def recognize(self, bgr):
        if not self.ok:
            return None, 0.0
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        g = cv2.bilateralFilter(g, 9, 75, 75)
        text = pytesseract.image_to_string(g, config="--oem 3 --psm 7")
        t = re.sub(r"[^A-Z0-9]", "", text.upper())
        if not t:
            return None, 0.0
        return t, 50.0

def lines_intersect(a1, a2, b1, b2):
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)

def box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (float((x1+x2)/2.0), float((y1+y2)/2.0))

class SpeedEstimator:
    def __init__(self, line_a, line_b, line_distance_meters, fps):
        self.line_a = line_a
        self.line_b = line_b
        self.real_d = float(line_distance_meters)
        self.fps = float(fps)
        self.t_enter = {}
        self.t_exit = {}
        self.state = {}
        self.last_center = {}
        self.ppm = None
        self.mapper = PerspectiveMapper()
        self.v_hist = defaultdict(lambda: RollingStats(30))
    def calibrate(self):
        self.ppm = self.mapper.calibrate_from_lines(self.line_a, self.line_b, self.real_d)
    def update(self, tid, prev_center, curr_center):
        a1, a2 = self.line_a
        b1, b2 = self.line_b
        crossed_a = lines_intersect(prev_center, curr_center, a1, a2)
        crossed_b = lines_intersect(prev_center, curr_center, b1, b2)
        now = time.time()
        if crossed_a:
            self.t_enter[tid] = now
            self.state[tid] = True
        if crossed_b and self.state.get(tid, False):
            self.t_exit[tid] = now
        if tid in self.t_enter and tid in self.t_exit:
            t1 = self.t_enter.pop(tid)
            t2 = self.t_exit.pop(tid)
            self.state.pop(tid, None)
            dt = max(1e-6, t2 - t1)
            if self.ppm is None:
                self.calibrate()
            v_ms = self.real_d / dt
            v_kmh = v_ms * 3.6
            self.v_hist[tid].push(v_kmh)
            return v_kmh
        return None

class VehicleClassifier:
    def __init__(self, names):
        self.names = names
    def name(self, cls_id):
        if self.names is None:
            return "vehicle"
        if isinstance(self.names, dict):
            return self.names.get(cls_id, "vehicle")
        if isinstance(self.names, (list, tuple)) and 0 <= int(cls_id) < len(self.names):
            return self.names[int(cls_id)]
        return "vehicle"

class ROIPlateExtractor:
    def __init__(self, ratio=0.35):
        self.ratio = ratio
    def extract(self, frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        y_top = y1 + int((1.0 - self.ratio) * h)
        roi = frame[y_top:y2, x1:x2]
        if roi.size == 0:
            return frame[y1:y2, x1:x2]
        return roi

class TicketDB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("CREATE TABLE IF NOT EXISTS tickets(id TEXT PRIMARY KEY, timestamp TEXT, plate TEXT, conf REAL, speed REAL, limit REAL, class TEXT, image_path TEXT, meta TEXT)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON tickets(timestamp)")
        self.conn.commit()
    def insert(self, t: Ticket):
        with self.lock:
            self.conn.execute("INSERT INTO tickets(id, timestamp, plate, conf, speed, limit, class, image_path, meta) VALUES(?,?,?,?,?,?,?,?,?)", (
                t.id, t.timestamp, t.plate, t.plate_conf, t.speed, t.limit, t.vehicle_class, t.image_path, json.dumps(t.meta or {})
            ))
            self.conn.commit()
    def list_recent(self, n=50):
        with self.lock:
            cur = self.conn.execute("SELECT id,timestamp,plate,conf,speed,limit,class,image_path,meta FROM tickets ORDER BY timestamp DESC LIMIT ?", (int(n),))
            rows = cur.fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r[0],
                "timestamp": r[1],
                "plate": r[2],
                "conf": r[3],
                "speed": r[4],
                "limit": r[5],
                "class": r[6],
                "image_path": r[7],
                "meta": json.loads(r[8] or "{}")
            })
        return out

class TicketWriter:
    def __init__(self, out_dir, db_path):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.db = TicketDB(db_path)
        self.pool = AsyncPool(3)
    def _compose(self, frame, bbox, info):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].copy()
        img = frame.copy()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        tlines = [
            f"Violation ID: {info['id']}",
            f"Timestamp: {info['timestamp']}",
            f"Plate: {info['plate']} ({info['plate_conf']:.1f})",
            f"Speed: {info['speed']:.2f} km/h",
            f"Limit: {info['speed_limit']:.2f} km/h",
            f"Class: {info['vehicle_class']}"
        ]
        y0 = max(16, y1 - 8 - 18 * len(tlines))
        for i, t in enumerate(tlines):
            cv2.putText(img, t, (x1, y0 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return img, crop
    def _pdf(self, info, img_path, pdf_path):
        if pdfcanvas is None:
            return
        c = pdfcanvas.Canvas(pdf_path, pagesize=A4)
        w, h = A4
        try:
            c.drawImage(img_path, 36, h/2 - 50, width=w-72, height=h/2, preserveAspectRatio=True, anchor='n')
        except Exception:
            pass
        y = h/2 - 70
        lines = [
            f"Violation ID: {info['id']}",
            f"Timestamp: {info['timestamp']}",
            f"Plate: {info['plate']} ({info['plate_conf']:.1f})",
            f"Speed: {info['speed']:.2f} km/h",
            f"Limit: {info['speed_limit']:.2f} km/h",
            f"Class: {info['vehicle_class']}"
        ]
        for s in lines:
            c.drawString(48, y, s)
            y -= 16
        c.showPage()
        c.save()
    def write(self, frame, bbox, vehicle_class, plate, plate_conf, speed, speed_limit, meta=None):
        vid = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        info = {"id": vid, "timestamp": ts, "plate": plate or "UNKNOWN", "plate_conf": float(plate_conf or 0.0), "speed": float(speed), "speed_limit": float(speed_limit), "vehicle_class": vehicle_class or "vehicle", "meta": meta or {}}
        img, crop = self._compose(frame, bbox, info)
        img_path = os.path.join(self.out_dir, f"{vid}.jpg")
        crop_path = os.path.join(self.out_dir, f"{vid}_crop.jpg")
        pdf_path = os.path.join(self.out_dir, f"{vid}.pdf")
        t = Ticket(vid, ts, info["plate"], info["plate_conf"], info["speed"], info["speed_limit"], info["vehicle_class"], img_path, info["meta"])
        def task():
            try:
                cv2.imwrite(img_path, img)
                cv2.imwrite(crop_path, crop)
            except Exception:
                pass
            try:
                self.db.insert(t)
            except Exception:
                pass
            try:
                with open(os.path.join(self.out_dir, f"{vid}.json"), "w") as f:
                    json.dump(info, f, indent=2)
            except Exception:
                pass
            try:
                self._pdf(info, img_path, pdf_path)
            except Exception:
                pass
        self.pool.submit(task)
        return info

class Visualizer:
    def __init__(self):
        self.fps_stats = RollingStats(60)
    def draw_base(self, frame, line_a, line_b, limit):
        a1, a2 = line_a
        b1, b2 = line_b
        cv2.line(frame, (int(a1[0]), int(a1[1])), (int(a2[0]), int(a2[1])), (0, 255, 0), 2)
        cv2.line(frame, (int(b1[0]), int(b1[1])), (int(b2[0]), int(b2[1])), (0, 255, 255), 2)
        cv2.putText(frame, f"Limit {limit:.0f} km/h", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    def draw_track(self, frame, tid, bbox, extra):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {tid} {extra}", (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    def draw_fps(self, frame, dt):
        fps = 1.0 / max(1e-6, dt)
        self.fps_stats.push(fps)
        cv2.putText(frame, f"FPS {self.fps_stats.mean():.1f}", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

class PlatePipeline:
    def __init__(self, alpr: AlprEngine, ocr: OCRFallback, extractor: ROIPlateExtractor):
        self.alpr = alpr
        self.ocr = ocr
        self.extractor = extractor
    def run(self, frame, bbox):
        roi = self.extractor.extract(frame, bbox)
        plate, conf = self.alpr.recognize(roi)
        if plate is None and self.ocr is not None:
            p2, c2 = self.ocr.recognize(roi)
            if c2 > conf:
                return p2, c2
        return plate, conf

class GracefulKiller:
    def __init__(self):
        self.kill_now = AtomicBool(False)
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    def exit_gracefully(self, *args):
        self.kill_now.set(True)

class RESTServer:
    def __init__(self, db: TicketDB, host="0.0.0.0", port=8000):
        self.db = db
        self.host = host
        self.port = int(port)
        self.thread = None
        self.app = None
        if FastAPI is not None:
            app = FastAPI()
            app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
            @app.get("/tickets")
            def tickets(n: int = 50):
                return self.db.list_recent(n)
            self.app = app
    def start(self):
        if self.app is None:
            return
        def run():
            try:
                uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")
            except Exception:
                pass
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

class VideoSource:
    def __init__(self, source, desired_fps=0.0):
        self.cap = cv2.VideoCapture(source)
        self.ok = self.cap.isOpened()
        self.fps = desired_fps if desired_fps > 0 else float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    def read(self):
        return self.cap.read()
    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass

class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.detector = Detector(cfg.get("model"), device=cfg.get("device"), conf=cfg.get("conf", 0.25), iou=cfg.get("iou", 0.45), classes=cfg.get("classes"), imgsz=cfg.get("imgsz", 640))
        self.tracker = Tracker(max_age=cfg.get("tracker_max_age", 60), max_iou_distance=cfg.get("tracker_max_iou", 0.7), n_init=cfg.get("tracker_n_init", 3))
        self.alpr = AlprEngine(cfg.get("alpr_country", "us"), cfg.get("alpr_config", "/etc/openalpr/openalpr.conf"), cfg.get("alpr_runtime", "/usr/share/openalpr/runtime_data"))
        self.ocr = OCRFallback()
        self.plate = PlatePipeline(self.alpr, self.ocr, ROIPlateExtractor(cfg.get("plate_roi_ratio", 0.35)))
        self.tickets = TicketWriter(cfg.get("output", "violations"), cfg.get("db", "violations.sqlite"))
        self.visual = Visualizer()
        self.vehicle_names = VehicleClassifier(self.detector.names)
        fps = cfg.get("fps", 0.0)
        self.source = None
        self.speed_limit = float(cfg.get("speed_limit_kmh", 60.0))
        la = cfg.get("line_a")
        lb = cfg.get("line_b")
        self.speed = SpeedEstimator((tuple(la[0]), tuple(la[1])), (tuple(lb[0]), tuple(lb[1])), cfg.get("line_distance_m"), fps if fps and fps > 0 else 30.0)
        self.prev_centers = {}
        self.cooldown = {}
        self.violation_cooldown = float(cfg.get("violation_cooldown", 5.0))
        self.rest = None
        if cfg.get("rest", False):
            self.rest = RESTServer(self.tickets.db, host=cfg.get("rest_host", "0.0.0.0"), port=cfg.get("rest_port", 8000))
            self.rest.start()
    def _allowed(self, tid):
        t = self.cooldown.get(tid, 0)
        now = time.time()
        if now < t:
            return False
        self.cooldown[tid] = now + self.violation_cooldown
        return True
    def _match_class(self, bbox, detections):
        for d_xyxy, d_cls, d_conf in detections:
            dx1, dy1, dx2, dy2 = d_xyxy
            x1, y1, x2, y2 = bbox
            iou = self._iou((dx1, dy1, dx2, dy2), (x1, y1, x2, y2))
            if iou > 0.5:
                return d_cls
        return None
    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union
    def open(self, src):
        self.source = VideoSource(src, desired_fps=self.cfg.get("fps", 0.0))
        if not self.source.ok:
            raise RuntimeError("cannot open source")
        if not self.cfg.get("fps", 0.0):
            self.speed.fps = self.source.fps
    def loop(self):
        killer = GracefulKiller()
        prev_t = time.time()
        while not killer.kill_now.get():
            ok, frame = self.source.read()
            if not ok:
                break
            t0 = time.time()
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections, frame)
            vis = frame.copy()
            self.visual.draw_base(vis, self.speed.line_a, self.speed.line_b, self.speed_limit)
            for tid, bbox in tracks:
                x1, y1, x2, y2 = bbox
                c = box_center(bbox)
                pc = self.prev_centers.get(tid, c)
                v = self.speed.update(tid, pc, c)
                self.prev_centers[tid] = c
                extra = ""
                if v is not None:
                    extra = f"{v:.1f} km/h"
                self.visual.draw_track(vis, tid, bbox, extra)
                if v is not None and v > self.speed_limit and self._allowed(tid):
                    plate, pconf = self.plate.run(frame, bbox)
                    cls_id = self._match_class(bbox, detections)
                    vehicle_class = self.vehicle_names.name(cls_id if cls_id is not None else -1)
                    self.tickets.write(frame, bbox, vehicle_class, plate, pconf, v, self.speed_limit, meta={"track_id": tid})
            if self.cfg.get("display", False):
                dt = time.time() - prev_t
                prev_t = time.time()
                self.visual.draw_fps(vis, dt)
                cv2.imshow("Speed Enforcement", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            dt_loop = time.time() - t0
            if self.source.fps > 0:
                target = 1.0 / self.source.fps
                if dt_loop < target:
                    time.sleep(target - dt_loop)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.alpr.close()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--source", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes", type=int, nargs="*", default=[2,3,5,7])
    ap.add_argument("--line_a", type=str, required=True)
    ap.add_argument("--line_b", type=str, required=True)
    ap.add_argument("--line_distance_m", type=float, required=True)
    ap.add_argument("--speed_limit_kmh", type=float, required=True)
    ap.add_argument("--fps", type=float, default=0)
    ap.add_argument("--output", type=str, default="violations")
    ap.add_argument("--db", type=str, default="violations.sqlite")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--alpr_country", type=str, default="us")
    ap.add_argument("--alpr_config", type=str, default="/etc/openalpr/openalpr.conf")
    ap.add_argument("--alpr_runtime", type=str, default="/usr/share/openalpr/runtime_data")
    ap.add_argument("--plate_roi_ratio", type=float, default=0.35)
    ap.add_argument("--tracker_max_age", type=int, default=60)
    ap.add_argument("--tracker_max_iou", type=float, default=0.7)
    ap.add_argument("--tracker_n_init", type=int, default=3)
    ap.add_argument("--violation_cooldown", type=float, default=5.0)
    ap.add_argument("--rest", action="store_true")
    ap.add_argument("--rest_host", type=str, default="0.0.0.0")
    ap.add_argument("--rest_port", type=int, default=8000)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = Config.from_args(args)
    pl = Pipeline(cfg)
    pl.open(args.source)
    pl.loop()

if __name__ == "__main__":
    main()
