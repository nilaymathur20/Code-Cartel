"""
Indian Chauraha Traffic Counter v4
- Fixed two-wheeler detection (motorcycle/bicycle/scooter)
- Pandas analytics and Excel export
- Per-class confidence thresholds
- Gemini AI (optional)
- Full-coverage ROI zones
"""

from __future__ import annotations
import os
import cv2
import csv
import json
import time
import math
import random
import queue
import threading
import importlib
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from collections import defaultdict, deque
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageTk
from ultralytics import YOLO

# ═══════════════════════════════════════
# OPTIONAL DEPENDENCIES
# ═══════════════════════════════════════
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except (ImportError, Exception):
    PANDAS_AVAILABLE = False
    print("[WARN] Pandas not available. Analytics features will be disabled.")

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# ═══════════════════════════════════════
# OPTIONAL AI IMPORT
# ═══════════════════════════════════════
GEMINI_AVAILABLE = False
genai = None

def _try_import_gemini():
    global GEMINI_AVAILABLE, genai
    try:
        # Try new SDK first
        genai = importlib.import_module("google.genai")
        GEMINI_AVAILABLE = True
        print("[OK] Google GenAI (new) available")
    except (ImportError, ModuleNotFoundError):
        try:
            # Fallback to legacy SDK
            genai = importlib.import_module("google.generativeai")
            GEMINI_AVAILABLE = True
            print("[OK] Google GenerativeAI (legacy) available")
        except (ImportError, ModuleNotFoundError):
            GEMINI_AVAILABLE = False
            genai = None
            print("[INFO] Gemini AI not available (optional)")

_try_import_gemini()


# ═══════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════
DISPLAY_W = 960
DISPLAY_H = 640

# Extended vehicle classes for better Indian traffic detection
# Key fix: include person(0) to help detect riders
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Classes we COUNT as vehicles
# Default COCO IDs + placeholders for custom Indian models
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    80: "auto-rickshaw"  # Typical ID in many custom Indian models
}

# Two-wheeler class IDs (need special handling)
TWO_WHEELER_IDS = {1, 3}  # bicycle, motorcycle

# Four-wheeler and heavy class IDs
FOUR_WHEELER_IDS = {2, 5, 7}  # car, bus, truck

ZONE_NAMES = ["North", "South", "East", "West", "Center"]

ZONE_COLORS_BGR = {
    "North":  (80, 80, 255),
    "South":  (80, 255, 80),
    "East":   (255, 180, 80),
    "West":   (80, 255, 255),
    "Center": (255, 0, 255)
}

CLASS_COLORS = {
    "person":     (255, 200, 200),
    "bicycle":    (0, 255, 255),
    "car":        (0, 255, 0),
    "motorcycle": (255, 0, 255),
    "bus":        (0, 165, 255),
    "truck":      (0, 0, 255),
    "unknown":    (200, 200, 200)
}

# Indian vehicle category mapping
INDIAN_CATEGORY = {
    "bicycle":    "Two-Wheeler",
    "motorcycle": "Two-Wheeler",
    "car":        "Four-Wheeler",
    "bus":        "Heavy Vehicle",
    "truck":      "Heavy Vehicle",
    "auto-rickshaw": "Auto-Rickshaw",
    "person":     "Pedestrian",
    "unknown":    "Unknown"
}


# ═══════════════════════════════════════
# ROI MANAGER
# ═══════════════════════════════════════
class ROIManager:
    def __init__(self):
        self.polygons = {z: [] for z in ZONE_NAMES}
        self.frame_w = 960
        self.frame_h = 540

    def set_frame_size(self, w, h):
        self.frame_w = w
        self.frame_h = h

    def clear_zone(self, zone):
        self.polygons[zone] = []

    def clear_all(self):
        for z in ZONE_NAMES:
            self.polygons[z] = []

    def add_point(self, zone, pt):
        self.polygons[zone].append((int(pt[0]), int(pt[1])))

    def pop_point(self, zone):
        if self.polygons[zone]:
            self.polygons[zone].pop()

    def has_polygons(self):
        return any(len(pts) >= 3 for pts in self.polygons.values())

    def generate_default(self, w, h):
        self.frame_w = w
        self.frame_h = h
        left = int(w * 0.30)
        right = int(w * 0.70)
        top = int(h * 0.30)
        bottom = int(h * 0.70)

        self.polygons["North"] = [(0, 0), (w, 0), (w, top), (0, top)]
        self.polygons["South"] = [(0, bottom), (w, bottom), (w, h), (0, h)]
        self.polygons["West"] = [(0, top), (left, top), (left, bottom), (0, bottom)]
        self.polygons["East"] = [(right, top), (w, top), (w, bottom), (right, bottom)]
        self.polygons["Center"] = [(left, top), (right, top), (right, bottom), (left, bottom)]

    def point_zone(self, point):
        x, y = int(point[0]), int(point[1])
        for zone in ["Center", "North", "South", "East", "West"]:
            pts = self.polygons.get(zone, [])
            if len(pts) >= 3:
                contour = np.array(pts, dtype=np.int32)
                if cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0:
                    return zone

        w, h = self.frame_w, self.frame_h
        if w <= 0 or h <= 0:
            return "Center"
        nx, ny = x / w, y / h
        if 0.30 <= nx <= 0.70 and 0.30 <= ny <= 0.70:
            return "Center"
        if ny < 0.30:
            return "North"
        elif ny > 0.70:
            return "South"
        elif nx < 0.30:
            return "West"
        elif nx > 0.70:
            return "East"
        return "North" if ny < 0.50 else "South"

    def draw(self, frame, alpha=0.20):
        overlay = frame.copy()
        for zone, pts in self.polygons.items():
            color = ZONE_COLORS_BGR.get(zone, (255, 255, 255))
            if len(pts) >= 3:
                contour = np.array(pts, dtype=np.int32)
                cv2.fillPoly(overlay, [contour], color)
                cv2.polylines(frame, [contour], True, color, 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = pts[0]
                (tw, th), _ = cv2.getTextSize(zone, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.putText(frame, zone, (cx - tw // 2, cy + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            elif len(pts) >= 1:
                for i, p in enumerate(pts):
                    cv2.circle(frame, p, 5, color, -1)
                    if i > 0:
                        cv2.line(frame, pts[i - 1], pts[i], color, 2)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def save_json(self, path, source_size=None):
        data = {"source_size": source_size, "polygons": self.polygons}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        polys = data.get("polygons", {})
        for z in ZONE_NAMES:
            self.polygons[z] = [(int(p[0]), int(p[1])) for p in polys.get(z, [])]
        sz = data.get("source_size")
        if sz:
            self.frame_w = sz.get("width", self.frame_w)
            self.frame_h = sz.get("height", self.frame_h)
        return sz


# ═══════════════════════════════════════
# TRACK STATE
# ═══════════════════════════════════════
class TrackState:
    def __init__(self, track_id, cls_name):
        self.track_id = track_id
        self.cls_name = cls_name
        self.centroids = deque(maxlen=120)
        self.zone_history = []
        self.entry_zone = None
        self.exit_zone = None
        self.entered_center = False
        self.counted = False
        self.last_seen = time.time()
        self.first_seen = time.time()
        self.conf = 0.0
        self.bbox = None
        self.indian_category = INDIAN_CATEGORY.get(cls_name, "Unknown")

    def update(self, centroid, cls_name=None, conf=None, bbox=None):
        self.centroids.append(centroid)
        if cls_name:
            self.cls_name = cls_name
            self.indian_category = INDIAN_CATEGORY.get(cls_name, "Unknown")
        if conf is not None:
            self.conf = conf
        if bbox is not None:
            self.bbox = bbox
        self.last_seen = time.time()

    def add_zone(self, zone):
        if not self.zone_history or self.zone_history[-1] != zone:
            self.zone_history.append(zone)


# ═══════════════════════════════════════
# TWO-WHEELER ENHANCER
# ═══════════════════════════════════════
class TwoWheelerEnhancer:
    """
    Improves two-wheeler detection by:
    1. Associating nearby person detections with motorcycles
    2. Boosting small object detection
    3. Re-classifying ambiguous detections
    """

    @staticmethod
    def associate_riders(detections):
        """
        If a person bbox overlaps significantly with a
        motorcycle/bicycle bbox, merge them as one two-wheeler.
        Also: lone small persons moving fast are likely riders
        on undetected two-wheelers.

        detections: list of (x1,y1,x2,y2, conf, cls_id, track_id)
        Returns: filtered detections with improved classifications
        """
        persons = []
        vehicles = []
        others = []

        for det in detections:
            x1, y1, x2, y2, conf, cls_id, tid = det
            if cls_id == 0:  # person
                persons.append(det)
            elif cls_id in VEHICLE_CLASSES:
                vehicles.append(det)
            else:
                others.append(det)

        matched_person_indices = set()

        # For each person, check if it overlaps a motorcycle/bicycle
        for pi, pdet in enumerate(persons):
            px1, py1, px2, py2 = pdet[0], pdet[1], pdet[2], pdet[3]
            p_area = (px2 - px1) * (py2 - py1)

            best_iou = 0
            best_vi = -1

            for vi, vdet in enumerate(vehicles):
                vx1, vy1, vx2, vy2, vconf, vcls, vtid = vdet

                # Only check two-wheelers
                if vcls not in TWO_WHEELER_IDS:
                    continue

                # Calculate IoU
                ix1 = max(px1, vx1)
                iy1 = max(py1, vy1)
                ix2 = min(px2, vx2)
                iy2 = min(py2, vy2)

                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    v_area = (vx2 - vx1) * (vy2 - vy1)
                    union = p_area + v_area - inter
                    iou = inter / union if union > 0 else 0

                    # Also check containment
                    containment = inter / p_area if p_area > 0 else 0

                    score = max(iou, containment * 0.8)
                    if score > best_iou:
                        best_iou = score
                        best_vi = vi

            if best_iou > 0.15:
                # Person is likely a rider — mark as matched
                matched_person_indices.add(pi)

                # Expand the vehicle bbox to include rider
                vdet = vehicles[best_vi]
                vx1 = min(pdet[0], vdet[0])
                vy1 = min(pdet[1], vdet[1])
                vx2 = max(pdet[2], vdet[2])
                vy2 = max(pdet[3], vdet[3])

                # Boost confidence
                new_conf = min(0.95, max(vdet[4], pdet[4]) + 0.1)
                vehicles[best_vi] = (vx1, vy1, vx2, vy2,
                                    new_conf, vdet[5], vdet[6])

        # Unmatched small persons in traffic might be riders
        # on undetected two-wheelers
        for pi, pdet in enumerate(persons):
            if pi in matched_person_indices:
                continue

            px1, py1, px2, py2, pconf, pcls, ptid = pdet
            pw = px2 - px1
            ph = py2 - py1
            aspect = ph / pw if pw > 0 else 0

            # Heuristic: small upright person in traffic area
            # is likely a two-wheeler rider
            if pw < 100 and ph < 200 and aspect > 1.2 and ptid is not None:
                # Re-classify as motorcycle
                new_det = (px1, py1, px2, py2, pconf * 0.7, 3, ptid)
                vehicles.append(new_det)
                matched_person_indices.add(pi)

        return vehicles


# ═══════════════════════════════════════
# COUNTER ENGINE
# ═══════════════════════════════════════
class CounterEngine:
    APPROACH = {"North", "South", "East", "West"}

    def __init__(self):
        self.reset()

    def reset(self):
        self.tracks = {}
        self.direction_counts = defaultdict(int)
        self.class_counts = defaultdict(int)
        self.category_counts = defaultdict(int)
        self.events = []
        self.emergency_events = []
        self.alerted_emergency_ids = set()
        self.total = 0
        self.detection_log = []  # every frame detection

    def cleanup(self, timeout=5.0):
        now = time.time()
        dead = [tid for tid, s in self.tracks.items()
                if now - s.last_seen > timeout]
        for tid in dead:
            del self.tracks[tid]

    def process(self, track_id, cls_name, centroid, zone,
                conf=None, bbox=None, require_center=False):
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(track_id, cls_name)

        state = self.tracks[track_id]
        state.update(centroid, cls_name, conf, bbox)
        state.add_zone(zone)

        # Log every detection for pandas analysis
        self.detection_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "track_id": track_id,
            "class": cls_name,
            "category": INDIAN_CATEGORY.get(cls_name, "Unknown"),
            "cx": centroid[0],
            "cy": centroid[1],
            "zone": zone,
            "confidence": round(conf, 3) if conf else 0.0
        })

        # Keep log manageable
        if len(self.detection_log) > 50000:
            self.detection_log = self.detection_log[-30000:]

        if state.entry_zone is None and zone in self.APPROACH:
            state.entry_zone = zone

        if zone == "Center":
            state.entered_center = True

        if (
            state.entry_zone is not None
            and zone in self.APPROACH
            and zone != state.entry_zone
            and not state.counted
        ):
            if require_center and not state.entered_center:
                return None

            state.exit_zone = zone
            state.counted = True

            direction = f"{state.entry_zone}->{state.exit_zone}"
            self.direction_counts[direction] += 1
            self.class_counts[cls_name] += 1
            category = INDIAN_CATEGORY.get(cls_name, "Unknown")
            self.category_counts[category] += 1
            self.total += 1

            event = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "track_id": track_id,
                "vehicle_class": cls_name,
                "category": category,
                "entry_zone": state.entry_zone,
                "exit_zone": state.exit_zone,
                "direction": direction,
                "confidence": round(conf, 3) if conf else "",
                "duration_sec": round(time.time() - state.first_seen, 1)
            }
            self.events.append(event)
            return event
        return None

    def get_summary(self):
        return {
            "total_vehicles": self.total,
            "active_tracks": len(self.tracks),
            "direction_counts": dict(self.direction_counts),
            "class_counts": dict(self.class_counts),
            "category_counts": dict(self.category_counts),
            "total_events": len(self.events),
            "total_detections": len(self.detection_log)
        }


# ═══════════════════════════════════════
# PANDAS ANALYTICS ENGINE
# ═══════════════════════════════════════
class PandasAnalytics:
    """
    Provides pandas-based traffic analytics:
    - DataFrames for events and detections
    - Time-series analysis
    - Flow rate calculations
    - Direction flow matrix
    - Vehicle distribution
    - Peak hour detection
    - Excel/CSV export with multiple sheets
    """

    @staticmethod
    def events_to_df(events):
        if not PANDAS_AVAILABLE or not events:
            return pd.DataFrame() if PANDAS_AVAILABLE else None
        df = pd.DataFrame(events)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    @staticmethod
    def detections_to_df(detection_log):
        if not PANDAS_AVAILABLE or not detection_log:
            return pd.DataFrame() if PANDAS_AVAILABLE else None
        df = pd.DataFrame(detection_log)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    @staticmethod
    def direction_matrix(events_df):
        """Create a from->to matrix of vehicle counts."""
        if not PANDAS_AVAILABLE or events_df is None or events_df.empty:
            return pd.DataFrame() if PANDAS_AVAILABLE else None

        zones = ["North", "South", "East", "West"]
        matrix = pd.DataFrame(0, index=zones, columns=zones)

        for _, row in events_df.iterrows():
            entry = row.get("entry_zone", "")
            exit_z = row.get("exit_zone", "")
            if entry in zones and exit_z in zones:
                matrix.loc[entry, exit_z] += 1

        return matrix

    @staticmethod
    def flow_rate(events_df, interval="1min"):
        """Calculate vehicle flow rate over time intervals."""
        if not PANDAS_AVAILABLE or events_df is None or events_df.empty or "timestamp" not in events_df.columns:
            return pd.DataFrame() if PANDAS_AVAILABLE else None

        df = events_df.set_index("timestamp")
        rate = df.resample(interval).size().reset_index()
        rate.columns = ["time_interval", "vehicle_count"]
        return rate

    @staticmethod
    def class_distribution(events_df):
        """Vehicle class distribution with percentages."""
        if not PANDAS_AVAILABLE or events_df is None or events_df.empty:
            return pd.DataFrame() if PANDAS_AVAILABLE else None

        dist = events_df["vehicle_class"].value_counts().reset_index()
        dist.columns = ["vehicle_class", "count"]
        total = dist["count"].sum()
        dist["percentage"] = (dist["count"] / total * 100).round(1)
        dist["category"] = dist["vehicle_class"].map(INDIAN_CATEGORY)
        return dist

    @staticmethod
    def category_distribution(events_df):
        """Indian category distribution (Two-Wheeler, Four-Wheeler, Heavy)."""
        if not PANDAS_AVAILABLE or events_df is None or events_df.empty:
            return pd.DataFrame() if PANDAS_AVAILABLE else None

        if "category" not in events_df.columns:
            events_df["category"] = events_df["vehicle_class"].map(INDIAN_CATEGORY)

        dist = events_df["category"].value_counts().reset_index()
        dist.columns = ["category", "count"]
        total = dist["count"].sum()
        dist["percentage"] = (dist["count"] / total * 100).round(1)
        return dist

    @staticmethod
    def peak_analysis(events_df, interval="5min"):
        """Find peak traffic periods."""
        if not PANDAS_AVAILABLE or events_df is None or events_df.empty or "timestamp" not in events_df.columns:
            return pd.DataFrame() if PANDAS_AVAILABLE else None

        df = events_df.set_index("timestamp")
        rate = df.resample(interval).agg(
            total_vehicles=("track_id", "count"),
            two_wheelers=("category", lambda x: (x == "Two-Wheeler").sum()),
            four_wheelers=("category", lambda x: (x == "Four-Wheeler").sum()),
            heavy_vehicles=("category", lambda x: (x == "Heavy Vehicle").sum()),
            directions=("direction", lambda x: x.nunique())
        ).reset_index()

        rate.columns = ["time_period", "total", "two_wheelers",
                        "four_wheelers", "heavy", "unique_directions"]
        return rate.sort_values("total", ascending=False)

    @staticmethod
    def direction_by_class(events_df):
        """Direction counts broken down by vehicle class."""
        if not PANDAS_AVAILABLE or events_df is None or events_df.empty:
            return pd.DataFrame() if PANDAS_AVAILABLE else None

        pivot = pd.crosstab(
            events_df["direction"],
            events_df["vehicle_class"],
            margins=True,
            margins_name="Total"
        )
        return pivot

    @staticmethod
    def zone_dwell_time(events_df):
        """Average time vehicles spend crossing the intersection."""
        if not PANDAS_AVAILABLE or events_df is None or events_df.empty or "duration_sec" not in events_df.columns:
            return pd.DataFrame() if PANDAS_AVAILABLE else None

        dwell = events_df.groupby("vehicle_class")["duration_sec"].agg(
            ["mean", "median", "min", "max", "count"]
        ).round(1).reset_index()
        dwell.columns = ["vehicle_class", "avg_sec", "median_sec",
                        "min_sec", "max_sec", "count"]
        return dwell

    @staticmethod
    def generate_full_report(events, detection_log, duration_sec):
        """Generate comprehensive text report using pandas."""
        if not PANDAS_AVAILABLE:
            return "Pandas is not installed or failed to load. Analytics report is unavailable."

        lines = []
        lines.append("=" * 60)
        lines.append("  INDIAN CHAURAHA TRAFFIC ANALYSIS REPORT")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Monitoring Duration: {duration_sec / 60:.1f} minutes")
        lines.append("=" * 60)

        events_df = PandasAnalytics.events_to_df(events)
        det_df = PandasAnalytics.detections_to_df(detection_log)

        if events_df.empty:
            lines.append("\nNo counted events to analyze.")
            return "\n".join(lines)

        total = len(events_df)
        rate = total / (duration_sec / 60) if duration_sec > 0 else 0

        lines.append(f"\n1. OVERVIEW")
        lines.append(f"   Total vehicles counted: {total}")
        lines.append(f"   Flow rate: {rate:.1f} vehicles/minute")
        lines.append(f"   Total detections logged: {len(det_df)}")

        # Density
        if rate < 5:
            density = "LOW"
        elif rate < 15:
            density = "MEDIUM"
        elif rate < 30:
            density = "HIGH"
        else:
            density = "CONGESTED"
        lines.append(f"   Traffic density: {density}")

        # Category breakdown
        lines.append(f"\n2. VEHICLE CATEGORIES (Indian Classification)")
        cat_df = PandasAnalytics.category_distribution(events_df)
        if not cat_df.empty:
            for _, row in cat_df.iterrows():
                bar = "█" * int(row["percentage"] / 2)
                lines.append(f"   {row['category']:15s}: {row['count']:4d} "
                            f"({row['percentage']:5.1f}%) {bar}")

        # Class breakdown
        lines.append(f"\n3. VEHICLE TYPES")
        cls_df = PandasAnalytics.class_distribution(events_df)
        if not cls_df.empty:
            for _, row in cls_df.iterrows():
                bar = "█" * int(row["percentage"] / 2)
                lines.append(f"   {row['vehicle_class']:12s}: {row['count']:4d} "
                            f"({row['percentage']:5.1f}%) {bar}")

        # Two-wheeler stats
        two_w = events_df[events_df["vehicle_class"].isin(["motorcycle", "bicycle"])]
        four_w = events_df[events_df["vehicle_class"].isin(["car"])]
        heavy = events_df[events_df["vehicle_class"].isin(["bus", "truck"])]

        lines.append(f"\n4. TWO-WHEELER ANALYSIS")
        lines.append(f"   Total two-wheelers: {len(two_w)}")
        if total > 0:
            lines.append(f"   Two-wheeler share: {len(two_w)/total*100:.1f}%")
        if len(two_w) > 0:
            tw_dirs = two_w["direction"].value_counts()
            lines.append(f"   Top two-wheeler directions:")
            for d, c in tw_dirs.head(3).items():
                lines.append(f"     {d}: {c}")

        # Direction matrix
        lines.append(f"\n5. DIRECTION FLOW MATRIX")
        matrix = PandasAnalytics.direction_matrix(events_df)
        if not matrix.empty:
            lines.append(f"   {'':8s} " + " ".join(f"{c:>8s}" for c in matrix.columns))
            for idx in matrix.index:
                vals = " ".join(f"{matrix.loc[idx, c]:8d}" for c in matrix.columns)
                lines.append(f"   {idx:8s} {vals}")

        # Direction counts ranked
        lines.append(f"\n6. DIRECTION RANKING")
        dir_counts = events_df["direction"].value_counts()
        for i, (d, c) in enumerate(dir_counts.items(), 1):
            pct = c / total * 100
            bar = "█" * int(pct / 2)
            lines.append(f"   {i}. {d:20s}: {c:4d} ({pct:5.1f}%) {bar}")

        # Crossing time
        lines.append(f"\n7. CROSSING DURATION")
        dwell = PandasAnalytics.zone_dwell_time(events_df)
        if not dwell.empty:
            for _, row in dwell.iterrows():
                lines.append(f"   {row['vehicle_class']:12s}: "
                            f"avg={row['avg_sec']:.1f}s "
                            f"median={row['median_sec']:.1f}s "
                            f"(n={row['count']})")

        # Confidence stats
        lines.append(f"\n8. DETECTION CONFIDENCE")
        if "confidence" in events_df.columns:
            conf_vals = pd.to_numeric(events_df["confidence"], errors="coerce").dropna()
            if len(conf_vals) > 0:
                lines.append(f"   Mean confidence: {conf_vals.mean():.3f}")
                lines.append(f"   Min confidence:  {conf_vals.min():.3f}")
                lines.append(f"   Max confidence:  {conf_vals.max():.3f}")

        # Recommendations
        lines.append(f"\n9. BASIC RECOMMENDATIONS")
        if dir_counts.size > 0:
            top_dir = dir_counts.index[0]
            entry = top_dir.split("->")[0]
            lines.append(f"   • Highest flow: {top_dir} — prioritize {entry} green phase")

        if len(two_w) > len(four_w):
            lines.append("   • Two-wheeler dominant — consider dedicated bike lanes")
        if len(heavy) > 0:
            lines.append("   • Heavy vehicle presence — consider weight/time restrictions")
        if rate > 20:
            lines.append("   • High traffic density — consider signal optimization")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    @staticmethod
    def export_excel(events, detection_log, emergency_events, duration_sec, filepath):
        """Export comprehensive Excel report with multiple sheets."""
        if not PANDAS_AVAILABLE:
            return False
        
        events_df = PandasAnalytics.events_to_df(events)
        det_df = PandasAnalytics.detections_to_df(detection_log)
        emerg_df = pd.DataFrame(emergency_events) if PANDAS_AVAILABLE and emergency_events else pd.DataFrame()

        engine = "openpyxl" if OPENPYXL_AVAILABLE else None
        
        with pd.ExcelWriter(filepath, engine=engine) as writer:
            # Sheet 1: All counted events
            if not events_df.empty:
                events_df.to_excel(writer, sheet_name="Counted Events", index=False)
            
            # Sheet 1b: Emergency Events
            if not emerg_df.empty:
                emerg_df.to_excel(writer, sheet_name="Emergency Events", index=False)

            # Sheet 2: Direction matrix
            matrix = PandasAnalytics.direction_matrix(events_df)
            if not matrix.empty:
                matrix.to_excel(writer, sheet_name="Direction Matrix")

            # Sheet 3: Class distribution
            cls_dist = PandasAnalytics.class_distribution(events_df)
            if not cls_dist.empty:
                cls_dist.to_excel(writer, sheet_name="Class Distribution",
                                index=False)

            # Sheet 4: Category distribution
            cat_dist = PandasAnalytics.category_distribution(events_df)
            if not cat_dist.empty:
                cat_dist.to_excel(writer, sheet_name="Category Distribution",
                                index=False)

            # Sheet 5: Direction by class
            dir_cls = PandasAnalytics.direction_by_class(events_df)
            if not dir_cls.empty:
                dir_cls.to_excel(writer, sheet_name="Direction by Class")

            # Sheet 6: Crossing time analysis
            dwell = PandasAnalytics.zone_dwell_time(events_df)
            if not dwell.empty:
                dwell.to_excel(writer, sheet_name="Crossing Duration",
                            index=False)

            # Sheet 7: Flow rate
            flow = PandasAnalytics.flow_rate(events_df, "1min")
            if not flow.empty:
                flow.to_excel(writer, sheet_name="Flow Rate 1min",
                            index=False)

            # Sheet 8: Summary stats
            summary_data = {
                "Metric": [
                    "Total Vehicles", "Duration (min)",
                    "Flow Rate (veh/min)", "Total Detections",
                    "Two-Wheelers", "Four-Wheelers", "Heavy Vehicles",
                    "Unique Directions", "Emergency Vehicles Detected"
                ],
                "Value": [
                    len(events_df),
                    round(duration_sec / 60, 1),
                    round(len(events_df) / (duration_sec / 60), 1) if duration_sec > 0 else 0,
                    len(det_df),
                    len(events_df[events_df["vehicle_class"].isin(["motorcycle", "bicycle"])]) if not events_df.empty else 0,
                    len(events_df[events_df["vehicle_class"] == "car"]) if not events_df.empty else 0,
                    len(events_df[events_df["vehicle_class"].isin(["bus", "truck"])]) if not events_df.empty else 0,
                    events_df["direction"].nunique() if not events_df.empty else 0,
                    len(emerg_df)
                ]
            }
            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name="Summary", index=False)

            # Sheet 9: Recent detections sample
            if not det_df.empty:
                det_df.tail(1000).to_excel(
                    writer, sheet_name="Detection Log (last 1000)",
                    index=False)
        return True


# ═══════════════════════════════════════
# AI ANALYZER
# ═══════════════════════════════════════
class AIAnalyzer:
    def __init__(self):
        self.configured = False
        self.model = None
        self.vision_model = None
        self.use_new_sdk = False

    def configure(self, api_key):
        if not GEMINI_AVAILABLE or genai is None:
            return False
        try:
            if hasattr(genai, "Client"): # new SDK google-genai
                self.model = genai.Client(api_key=api_key)
                self.use_new_sdk = True
            else: # legacy SDK google-generativeai
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel("gemini-2.0-flash")
                self.vision_model = genai.GenerativeModel("gemini-2.0-flash")
            self.configured = True
            return True
        except Exception as e:
            print(f"Gemini error: {e}")
            return False

    def analyze_traffic(self, summary, duration_sec=0):
        if not self.configured:
            return "AI not configured. Enter Gemini API key."
        prompt = f"""Expert Indian traffic analyst. Analyze:
- Total: {summary['total_vehicles']}
- Duration: {duration_sec:.0f}s
- Directions: {json.dumps(summary['direction_counts'])}
- Classes: {json.dumps(summary['class_counts'])}
- Categories: {json.dumps(summary.get('category_counts', {}))}

Provide: density, dominant flow, two-wheeler analysis,
congestion risk, signal timing, anomalies.
Indian context. Concise."""
        try:
            if self.use_new_sdk:
                response = self.model.models.generate_content(
                    model="gemini-2.0-flash", contents=prompt)
                return response.text
            else:
                return self.model.generate_content(prompt).text
        except Exception as e:
            return f"Error: {e}"

    def analyze_scene(self, frame):
        if not self.configured:
            return "AI not configured."
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil.thumbnail((800, 600))
            prompt = """Analyze this Indian intersection traffic camera.
Describe: vehicle types (bikes/scooters/autos/cars/buses/trucks),
density, road condition, violations, weather, flow. Indian context."""
            if self.use_new_sdk:
                # new SDK handles PIL directly in contents list
                response = self.model.models.generate_content(
                    model="gemini-2.0-flash", contents=[prompt, pil])
                return response.text
            else:
                return self.vision_model.generate_content([prompt, pil]).text
        except Exception as e:
            return f"Error: {e}"

    def get_recommendations(self, summary):
        if not self.configured:
            return "AI not configured."
        prompt = f"""Indian intersection data:
{json.dumps(summary)}
Provide: signal timing, management, infrastructure,
safety, peak strategy. Practical for Indian municipality."""
        try:
            if self.use_new_sdk:
                response = self.model.models.generate_content(
                    model="gemini-2.0-flash", contents=prompt)
                return response.text
            else:
                return self.model.generate_content(prompt).text
        except Exception as e:
            return f"Error: {e}"


def is_ambulance(cls_name, conf, bbox):
    """
    Heuristic ambulance detection
    (since YOLO doesn't have ambulance class)
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # Rule 1: Tightened to bus only to reduce false positives
    if cls_name != "bus":
        return False

    # Rule 2: Increased confidence requirement
    if conf < 0.75:
        return False

    # Rule 3: Tightened aspect ratio range
    aspect_ratio = w / h if h > 0 else 0
    if 1.5 < aspect_ratio < 2.2:
        return True

    return False


# ═══════════════════════════════════════
# TRAFFIC SIMULATOR (PYGAME)
# ═══════════════════════════════════════
import pygame as pg

class SimTrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0
        self.vehicleCount = 0  # Track vehicles for adaptive timing

class SimVehicle(pg.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn, simulator):
        pg.sprite.Sprite.__init__(self)
        self.sim = simulator
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = self.sim.speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = self.sim.spawn_x[direction][lane]
        self.y = self.sim.spawn_y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        self.sim.vehicles[direction][lane].append(self)
        self.index = len(self.sim.vehicles[direction][lane]) - 1
        
        # Image loading with proper path
        path = os.path.join(self.sim.img_dir, direction, f"{vehicleClass}.png")
        if not os.path.exists(path):
            path = os.path.join(self.sim.img_dir, direction, "car.png")
        
        self.originalImage = pg.image.load(path)
        self.currentImage = pg.image.load(path)

        # Initialize Stop Position
        self._set_stop_pos()
        self.sim.simulation_group.add(self)

    def _set_stop_pos(self):
        direction = self.direction
        lane = self.lane
        gap = self.sim.gap
        
        if direction == 'right':
            if self.index > 0 and self.sim.vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = self.sim.vehicles[direction][lane][self.index-1].stop - self.sim.vehicles[direction][lane][self.index-1].currentImage.get_rect().width - gap
            else:
                self.stop = self.sim.defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            self.sim.spawn_x[direction][lane] -= temp
            self.sim.stops[direction][lane] -= temp
            
        elif direction == 'left':
            if self.index > 0 and self.sim.vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = self.sim.vehicles[direction][lane][self.index-1].stop + self.sim.vehicles[direction][lane][self.index-1].currentImage.get_rect().width + gap
            else:
                self.stop = self.sim.defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            self.sim.spawn_x[direction][lane] += temp
            self.sim.stops[direction][lane] += temp
            
        elif direction == 'down':
            if self.index > 0 and self.sim.vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = self.sim.vehicles[direction][lane][self.index-1].stop - self.sim.vehicles[direction][lane][self.index-1].currentImage.get_rect().height - gap
            else:
                self.stop = self.sim.defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            self.sim.spawn_y[direction][lane] -= temp
            self.sim.stops[direction][lane] -= temp
            
        elif direction == 'up':
            if self.index > 0 and self.sim.vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = self.sim.vehicles[direction][lane][self.index-1].stop + self.sim.vehicles[direction][lane][self.index-1].currentImage.get_rect().height + gap
            else:
                self.stop = self.sim.defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            self.sim.spawn_y[direction][lane] += temp
            self.sim.stops[direction][lane] += temp

    def move(self):
        gap2 = self.sim.gap2
        rotationAngle = self.sim.rotationAngle
        mid = self.sim.mid
        stopLines = self.sim.stopLines
        
        # --- Emergency Vehicle (Ambulance) ---
        if self.vehicleClass == 'ambulance':
            self._move_ambulance()
            return

        # --- Normal Vehicle ---
        if self.direction == 'right':
            if self.crossed == 0 and self.x + self.currentImage.get_rect().width > stopLines[self.direction]:
                self.crossed = 1
                self.sim.vehicles[self.direction]['crossed'] += 1
                self.sim.signals[0].vehicleCount += 1
            
            if self.willTurn:
                if self.crossed == 0 or self.x + self.currentImage.get_rect().width < mid[self.direction]['x']:
                    if (self.x + self.currentImage.get_rect().width <= self.stop or (self.sim.currentGreen == 0 and self.sim.currentYellow == 0) or self.crossed == 1) and \
                       (self.index == 0 or self.x + self.currentImage.get_rect().width < (self.sim.vehicles[self.direction][self.lane][self.index-1].x - gap2) or self.sim.vehicles[self.direction][self.lane][self.index-1].turned == 1):
                        self.x += self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pg.transform.rotate(self.originalImage, -self.rotateAngle)
                        # Fix Bus touching on turns: Smoother radius
                        offset_x = 2 if self.vehicleClass != 'bus' else 2.5
                        offset_y = 1.8 if self.vehicleClass != 'bus' else 2.2
                        self.x += offset_x
                        self.y += offset_y
                        if self.rotateAngle >= 90: self.turned = 1
                    else:
                        if self.index == 0 or self.y + self.currentImage.get_rect().height < (self.sim.vehicles[self.direction][self.lane][self.index-1].y - gap2):
                            self.y += self.speed
            else:
                if (self.x + self.currentImage.get_rect().width <= self.stop or self.crossed == 1 or (self.sim.currentGreen == 0 and self.sim.currentYellow == 0)) and \
                   (self.index == 0 or self.x + self.currentImage.get_rect().width < (self.sim.vehicles[self.direction][self.lane][self.index-1].x - gap2) or self.sim.vehicles[self.direction][self.lane][self.index-1].turned == 1):
                    self.x += self.speed

        elif self.direction == 'down':
            if self.crossed == 0 and self.y + self.currentImage.get_rect().height > stopLines[self.direction]:
                self.crossed = 1
                self.sim.vehicles[self.direction]['crossed'] += 1
                self.sim.signals[1].vehicleCount += 1
            
            if self.willTurn:
                if self.crossed == 0 or self.y + self.currentImage.get_rect().height < mid[self.direction]['y']:
                    if (self.y + self.currentImage.get_rect().height <= self.stop or (self.sim.currentGreen == 1 and self.sim.currentYellow == 0) or self.crossed == 1) and \
                       (self.index == 0 or self.y + self.currentImage.get_rect().height < (self.sim.vehicles[self.direction][self.lane][self.index-1].y - gap2) or self.sim.vehicles[self.direction][self.lane][self.index-1].turned == 1):
                        self.y += self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pg.transform.rotate(self.originalImage, -self.rotateAngle)
                        offset_x = -2.5 if self.vehicleClass != 'bus' else -3.0
                        offset_y = 2 if self.vehicleClass != 'bus' else 2.5
                        self.x += offset_x
                        self.y += offset_y
                        if self.rotateAngle >= 90: self.turned = 1
                    else:
                        if self.index == 0 or self.x > (self.sim.vehicles[self.direction][self.lane][self.index-1].x + self.sim.vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2):
                            self.x -= self.speed
            else:
                if (self.y + self.currentImage.get_rect().height <= self.stop or self.crossed == 1 or (self.sim.currentGreen == 1 and self.sim.currentYellow == 0)) and \
                   (self.index == 0 or self.y + self.currentImage.get_rect().height < (self.sim.vehicles[self.direction][self.lane][self.index-1].y - gap2) or self.sim.vehicles[self.direction][self.lane][self.index-1].turned == 1):
                    self.y += self.speed

        elif self.direction == 'left':
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
                self.sim.vehicles[self.direction]['crossed'] += 1
                self.sim.signals[2].vehicleCount += 1
            
            if self.willTurn:
                if self.crossed == 0 or self.x > mid[self.direction]['x']:
                    if (self.x >= self.stop or (self.sim.currentGreen == 2 and self.sim.currentYellow == 0) or self.crossed == 1) and \
                       (self.index == 0 or self.x > (self.sim.vehicles[self.direction][self.lane][self.index-1].x + self.sim.vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or self.sim.vehicles[self.direction][self.lane][self.index-1].turned == 1):
                        self.x -= self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pg.transform.rotate(self.originalImage, -self.rotateAngle)
                        offset_x = -1.8 if self.vehicleClass != 'bus' else -2.2
                        offset_y = -2.5 if self.vehicleClass != 'bus' else -3.0
                        self.x += offset_x
                        self.y += offset_y
                        if self.rotateAngle >= 90: self.turned = 1
                    else:
                        if self.index == 0 or self.y > (self.sim.vehicles[self.direction][self.lane][self.index-1].y + self.sim.vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2):
                            self.y -= self.speed
            else:
                if (self.x >= self.stop or self.crossed == 1 or (self.sim.currentGreen == 2 and self.sim.currentYellow == 0)) and \
                   (self.index == 0 or self.x > (self.sim.vehicles[self.direction][self.lane][self.index-1].x + self.sim.vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or self.sim.vehicles[self.direction][self.lane][self.index-1].turned == 1):
                    self.x -= self.speed

        elif self.direction == 'up':
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
                self.sim.vehicles[self.direction]['crossed'] += 1
                self.sim.signals[3].vehicleCount += 1
            
            if self.willTurn:
                if self.crossed == 0 or self.y > mid[self.direction]['y']:
                    if (self.y >= self.stop or (self.sim.currentGreen == 3 and self.sim.currentYellow == 0) or self.crossed == 1) and \
                       (self.index == 0 or self.y > (self.sim.vehicles[self.direction][self.lane][self.index-1].y + self.sim.vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or self.sim.vehicles[self.direction][self.lane][self.index-1].turned == 1):
                        self.y -= self.speed
                else:
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pg.transform.rotate(self.originalImage, -self.rotateAngle)
                        offset_x = 1 if self.vehicleClass != 'bus' else 1.5
                        offset_y = -1 if self.vehicleClass != 'bus' else -1.5
                        self.x += offset_x
                        self.y += offset_y
                        if self.rotateAngle >= 90: self.turned = 1
                    else:
                        if self.index == 0 or self.x < (self.sim.vehicles[self.direction][self.lane][self.index-1].x - self.sim.vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width - gap2):
                            self.x += self.speed
            else:
                if (self.y >= self.stop or self.crossed == 1 or (self.sim.currentGreen == 3 and self.sim.currentYellow == 0)) and \
                   (self.index == 0 or self.y > (self.sim.vehicles[self.direction][self.lane][self.index-1].y + self.sim.vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or self.sim.vehicles[self.direction][self.lane][self.index-1].turned == 1):
                    self.y -= self.speed

    def _move_ambulance(self):
        # Ambulances ignore signals but respect gaps
        gap2 = self.sim.gap2
        stopLines = self.sim.stopLines
        mid = self.sim.mid
        rotationAngle = self.sim.rotationAngle
        
        if self.direction == 'right':
            if self.crossed == 0 and self.x + self.currentImage.get_rect().width > stopLines[self.direction]:
                self.crossed = 1
                self.sim.vehicles[self.direction]['crossed'] += 1
            if (self.index == 0) or (self.x + self.currentImage.get_rect().width < (self.sim.vehicles[self.direction][self.lane][self.index-1].x - gap2)):
                if self.willTurn and self.crossed == 1 and (self.x + self.currentImage.get_rect().width >= mid[self.direction]['x']):
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pg.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2; self.y += 1.8
                        if self.rotateAngle >= 90: self.turned = 1
                    else:
                        if self.index == 0 or self.y + self.currentImage.get_rect().height < (self.sim.vehicles[self.direction][self.lane][self.index-1].y - gap2):
                            self.y += self.speed
                else: self.x += self.speed
        elif self.direction == 'down':
            if self.crossed == 0 and self.y + self.currentImage.get_rect().height > stopLines[self.direction]:
                self.crossed = 1
                self.sim.vehicles[self.direction]['crossed'] += 1
            if (self.index == 0) or (self.y + self.currentImage.get_rect().height < (self.sim.vehicles[self.direction][self.lane][self.index-1].y - gap2)):
                if self.willTurn and self.crossed == 1 and (self.y + self.currentImage.get_rect().height >= mid[self.direction]['y']):
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pg.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5; self.y += 2
                        if self.rotateAngle >= 90: self.turned = 1
                    else:
                        if self.index == 0 or self.x > (self.sim.vehicles[self.direction][self.lane][self.index-1].x + self.sim.vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2):
                            self.x -= self.speed
                else: self.y += self.speed
        elif self.direction == 'left':
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
                self.sim.vehicles[self.direction]['crossed'] += 1
            if (self.index == 0) or (self.x > (self.sim.vehicles[self.direction][self.lane][self.index-1].x + self.sim.vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2)):
                if self.willTurn and self.crossed == 1 and (self.x <= mid[self.direction]['x']):
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pg.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8; self.y -= 2.5
                        if self.rotateAngle >= 90: self.turned = 1
                    else:
                        if self.index == 0 or self.y > (self.sim.vehicles[self.direction][self.lane][self.index-1].y + self.sim.vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2):
                            self.y -= self.speed
                else: self.x -= self.speed
        elif self.direction == 'up':
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
                self.sim.vehicles[self.direction]['crossed'] += 1
            if (self.index == 0) or (self.y > (self.sim.vehicles[self.direction][self.lane][self.index-1].y + self.sim.vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2)):
                if self.willTurn and self.crossed == 1 and (self.y <= mid[self.direction]['y']):
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pg.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1; self.y -= 1
                        if self.rotateAngle >= 90: self.turned = 1
                    else:
                        if self.index == 0 or self.x < (self.sim.vehicles[self.direction][self.lane][self.index-1].x - self.sim.vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width - gap2):
                            self.x += self.speed
                else: self.y -= self.speed

class TrafficSimulator:
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.running = True
        self.clock = pg.time.Clock() # Fix LAG: Clock for frame rate regulation
        
        self.defaultRed = 150
        self.defaultYellow = 5
        self.defaultGreen = 20
        self.defaultMinimum = 10
        self.defaultMaximum = 60
        
        self.signals = []
        self.noOfSignals = 4
        self.simTime = 300
        self.timeElapsed = 0
        self.currentGreen = 0
        self.nextGreen = 1
        self.currentYellow = 0
        
        self.vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'rickshaw', 4:'bike', 5:'ambulance'}
        self.speeds = {'car':2.25, 'bus':1.8, 'truck':1.8, 'rickshaw':2, 'bike':2.5, 'ambulance':3.5}
        
        # Spawn points and state
        self.spawn_x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}
        self.spawn_y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}
        
        self.vehicles = {
            'right': {0:[], 1:[], 2:[], 'crossed':0},
            'down':  {0:[], 1:[], 2:[], 'crossed':0},
            'left':  {0:[], 1:[], 2:[], 'crossed':0},
            'up':    {0:[], 1:[], 2:[], 'crossed':0}
        }
        self.directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}
        self.stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
        self.defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
        self.stops = {'right': [580,580,580], 'down': [320,320,320], 'left': [810,810,810], 'up': [545,545,545]}
        self.mid = {'right': {'x':705, 'y':445}, 'down': {'x':695, 'y':450}, 'left': {'x':695, 'y':425}, 'up': {'x':695, 'y':400}}
        self.rotationAngle = 3
        self.gap = 15
        self.gap2 = 15
        
        self.simulation_group = pg.sprite.Group()

    def run(self):
        pg.init()
        screen = pg.display.set_mode((1400, 800))
        pg.display.set_caption("Traffic Simulator Integrated")
        
        # Load static assets
        try:
            bg = pg.image.load(os.path.join(self.img_dir, "mod_int.png"))
            red_img = pg.image.load(os.path.join(self.img_dir, "signals", "red.png"))
            yel_img = pg.image.load(os.path.join(self.img_dir, "signals", "yellow.png"))
            grn_img = pg.image.load(os.path.join(self.img_dir, "signals", "green.png"))
            font = pg.font.Font(None, 30)
        except Exception as e:
            print(f"Asset Load Error: {e}")
            return

        # Initialize signals
        self.signals.append(SimTrafficSignal(0, self.defaultYellow, self.defaultGreen, self.defaultMinimum, self.defaultMaximum))
        self.signals.append(SimTrafficSignal(self.signals[0].red + self.defaultYellow + self.defaultGreen, self.defaultYellow, self.defaultGreen, self.defaultMinimum, self.defaultMaximum))
        self.signals.append(SimTrafficSignal(self.defaultRed, self.defaultYellow, self.defaultGreen, self.defaultMinimum, self.defaultMaximum))
        self.signals.append(SimTrafficSignal(self.defaultRed, self.defaultYellow, self.defaultGreen, self.defaultMinimum, self.defaultMaximum))

        # Start background threads
        threading.Thread(target=self._update_signals, daemon=True).start()
        threading.Thread(target=self._spawn_loop, daemon=True).start()
        threading.Thread(target=self._sim_timer, daemon=True).start()

        signalCoods = [(530,230),(810,230),(810,570),(530,570)]
        timerCoods = [(530,210),(810,210),(810,550),(530,550)]
        countCoods = [(480,210),(880,210),(880,550),(480,550)]

        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
            
            screen.blit(bg, (0, 0))
            
            # Draw Signals
            for i in range(4):
                if i == self.currentGreen:
                    if self.currentYellow == 1:
                        screen.blit(yel_img, signalCoods[i])
                        txt = "STOP" if self.signals[i].yellow == 0 else str(self.signals[i].yellow)
                    else:
                        screen.blit(grn_img, signalCoods[i])
                        txt = "SLOW" if self.signals[i].green == 0 else str(self.signals[i].green)
                else:
                    screen.blit(red_img, signalCoods[i])
                    txt = str(self.signals[i].red) if self.signals[i].red <= 10 else "---"
                
                # Render timers
                t_surf = font.render(txt, True, (255,255,255), (0,0,0))
                screen.blit(t_surf, timerCoods[i])
                
                # Render counts
                c_surf = font.render(str(self.vehicles[self.directionNumbers[i]]['crossed']), True, (0,0,0), (255,255,255))
                screen.blit(c_surf, countCoods[i])

            # Draw & Move Vehicles
            for v in self.simulation_group:
                screen.blit(v.currentImage, (v.x, v.y))
                v.move()

            pg.display.update()
            self.clock.tick(60) # 60 FPS cap to reduce LAG

        pg.quit()

    def _update_signals(self):
        while self.running:
            while self.signals[self.currentGreen].green > 0:
                self._tick_timers()
                if self.signals[(self.currentGreen + 1) % 4].red == 5:
                    self._adaptive_timing()
                time.sleep(1)
            
            self.currentYellow = 1
            while self.signals[self.currentGreen].yellow > 0:
                self._tick_timers()
                time.sleep(1)
            
            self.currentYellow = 0
            # Reset current
            self.signals[self.currentGreen].green = self.defaultGreen
            self.signals[self.currentGreen].yellow = self.defaultYellow
            self.signals[self.currentGreen].red = self.defaultRed
            
            # Transition
            self.currentGreen = self.nextGreen
            self.nextGreen = (self.currentGreen + 1) % 4
            self.signals[self.nextGreen].red = self.signals[self.currentGreen].yellow + self.signals[self.currentGreen].green

    def _tick_timers(self):
        for i in range(4):
            if i == self.currentGreen:
                if self.currentYellow == 0:
                    self.signals[i].green -= 1
                    self.signals[i].totalGreenTime += 1
                else:
                    self.signals[i].yellow -= 1
            else:
                self.signals[i].red -= 1

    def _adaptive_timing(self):
        # Adaptive logic using vehicleCount (NO. OF VEHICLES IN SIGNAL CLASS)
        target = (self.currentGreen + 1) % 4
        # Just a simple heuristic: count waiting vehicles in the next direction
        count = 0
        dir_name = self.directionNumbers[target]
        for lane in range(3):
            for v in self.vehicles[dir_name][lane]:
                if v.crossed == 0: count += 1
        
        greenTime = math.ceil(count * 2.5 / 2) # Average 2.5s per vehicle, 2 lanes
        self.signals[target].green = max(self.defaultMinimum, min(self.defaultMaximum, greenTime))

    def _spawn_loop(self):
        while self.running:
            # Weighted Distribution using simple random logic
            v_type_idx = random.choices(range(6), weights=[40, 10, 10, 15, 20, 1])[0]
            v_type = self.vehicleTypes[v_type_idx]
            
            # Lane selection (Distribution)
            if v_type_idx == 4: lane = 0 # bikes in lane 0
            elif v_type_idx == 5: lane = random.randint(0, 1) # ambulance
            else: lane = random.randint(0, 1) + 1 # heavy/cars in lanes 1, 2
            
            dir_idx = random.randint(0, 3)
            will_turn = 1 if lane == 2 and random.random() < 0.6 else 0
            
            SimVehicle(lane, v_type, dir_idx, self.directionNumbers[dir_idx], will_turn, self)
            
            # Preemption for ambulance
            if v_type_idx == 5:
                self._preempt(dir_idx)
                time.sleep(0.5)
            else:
                time.sleep(0.8)

    def _preempt(self, direction):
        self.signals[self.currentGreen].green = 0
        self.signals[self.currentGreen].yellow = 0
        self.currentGreen = direction
        self.nextGreen = (direction + 1) % 4
        self.signals[self.currentGreen].green = 15
        self.currentYellow = 0

    def _sim_timer(self):
        while self.running:
            time.sleep(1)
            self.timeElapsed += 1
            if self.timeElapsed >= self.simTime:
                # self.running = False
                pass

# ═══════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════
class TrafficCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Indian Chauraha Traffic Counter v4")
        self.root.geometry("1450x920")
        self.root.configure(bg="#1a1a2e")
        self.root.minsize(1200, 700)

        self.yolo_model = None
        self.cap = None
        self.running = False
        self.paused = False
        self.worker = None
        self.ui_queue = queue.Queue(maxsize=5)

        self.roi = ROIManager()
        self.counter = CounterEngine()
        self.ai = AIAnalyzer()
        self.enhancer = TwoWheelerEnhancer()

        self.last_emergency_zone = None
        self.last_emergency_time = 0

        self.source_var = tk.StringVar(value="0")
        self.model_var = tk.StringVar(value="yolov8s.pt")
        self.conf_var = tk.DoubleVar(value=0.25)
        self.twowheeler_conf_var = tk.DoubleVar(value=0.15)
        self.imgsz_var = tk.IntVar(value=960)
        self.skip_var = tk.IntVar(value=1)
        self.status_var = tk.StringVar(value="Ready")
        self.require_center = tk.BooleanVar(value=False)
        self.record_var = tk.BooleanVar(value=False)
        self.debug_var = tk.BooleanVar(value=True)
        self.enhance_2w_var = tk.BooleanVar(value=True)
        self.gemini_key_var = tk.StringVar(value="")

        self.raw_frame = None
        self.last_annotated = None
        self.frame_idx = 0
        self.fps = 0.0
        self.prev_time = time.time()
        self.start_time = time.time()
        self.video_writer = None

        self.roi_edit = False
        self.edit_zone = tk.StringVar(value="North")
        self.d_scale = 1.0
        self.d_pad_x = 0
        self.d_pad_y = 0

        self._build_gui()
        self._load_yolo()
        self.root.after(25, self._poll_queue)

    def _build_gui(self):
        style = ttk.Style()
        style.theme_use("clam")

        # ═══ TOP ═══
        top = tk.Frame(self.root, bg="#16213e")
        top.pack(fill=tk.X, padx=6, pady=4)

        r1 = tk.Frame(top, bg="#16213e")
        r1.pack(fill=tk.X, pady=2)

        ttk.Label(r1, text="Source:").pack(side=tk.LEFT, padx=4)
        ttk.Entry(r1, textvariable=self.source_var, width=22).pack(side=tk.LEFT, padx=2)
        ttk.Button(r1, text="Browse", command=self._browse).pack(side=tk.LEFT, padx=2)

        ttk.Label(r1, text="Model:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Combobox(r1, textvariable=self.model_var, width=12, state="readonly",
                    values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                            "yolov8l.pt"]).pack(side=tk.LEFT, padx=2)
        ttk.Button(r1, text="Load", command=self._load_yolo).pack(side=tk.LEFT, padx=2)

        ttk.Label(r1, text="Conf:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Scale(r1, from_=0.10, to=0.80, variable=self.conf_var,
                orient=tk.HORIZONTAL, length=80).pack(side=tk.LEFT, padx=2)

        ttk.Label(r1, text="2W Conf:").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Scale(r1, from_=0.05, to=0.50, variable=self.twowheeler_conf_var,
                orient=tk.HORIZONTAL, length=80).pack(side=tk.LEFT, padx=2)

        ttk.Label(r1, text="ImgSz:").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Spinbox(r1, from_=320, to=1280, increment=32,
                    textvariable=self.imgsz_var, width=5).pack(side=tk.LEFT, padx=2)

        ttk.Label(r1, text="Skip:").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Spinbox(r1, from_=1, to=5, textvariable=self.skip_var,
                    width=3).pack(side=tk.LEFT, padx=2)

        r1b = tk.Frame(top, bg="#16213e")
        r1b.pack(fill=tk.X, pady=1)

        ttk.Checkbutton(r1b, text="Enhance 2-Wheeler",
                        variable=self.enhance_2w_var).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(r1b, text="Center Req",
                        variable=self.require_center).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(r1b, text="Debug",
                        variable=self.debug_var).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(r1b, text="Record",
                        variable=self.record_var).pack(side=tk.LEFT, padx=4)

        ttk.Label(r1b, text="Gemini Key:").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(r1b, textvariable=self.gemini_key_var, width=16,
                show="*").pack(side=tk.LEFT, padx=2)
        ttk.Button(r1b, text="Set", command=self._set_key).pack(side=tk.LEFT, padx=2)

        ttk.Label(r1b, textvariable=self.status_var).pack(side=tk.RIGHT, padx=6)

        r2 = tk.Frame(top, bg="#16213e")
        r2.pack(fill=tk.X, pady=2)

        buttons = [
            ("▶ Start", self.start), ("■ Stop", self.stop),
            ("⏸ Pause", self.toggle_pause), ("Reset", self.reset_counts),
            ("📷 Snap", self.save_snap), ("CSV", self.export_csv),
            ("Excel Report", self.export_excel),
            ("📊 Pandas Report", self.pandas_report),
            ("🤖 AI Analyze", self.ai_analyze),
            ("🔍 AI Scene", self.ai_scene),
            ("💡 AI Tips", self.ai_recommend),
            ("🚀 Simulation", self.launch_simulation),
        ]
        for txt, cmd in buttons:
            ttk.Button(r2, text=txt, command=cmd).pack(side=tk.LEFT, padx=3)

        # ═══ BODY ═══
        body = tk.Frame(self.root, bg="#1a1a2e")
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg="#1a1a2e")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.video_lbl = tk.Label(left, bg="black")
        self.video_lbl.pack(fill=tk.BOTH, expand=True)
        self.video_lbl.bind("<Button-1>", self._on_left_click)
        self.video_lbl.bind("<Button-3>", self._on_right_click)

        roi_bar = tk.Frame(left, bg="#0f3460")
        roi_bar.pack(fill=tk.X, pady=(4, 0))

        self.roi_btn = ttk.Button(roi_bar, text="ROI: OFF",
                                command=self._toggle_roi)
        self.roi_btn.pack(side=tk.LEFT, padx=4)
        ttk.Label(roi_bar, text="Zone:").pack(side=tk.LEFT, padx=2)
        ttk.Combobox(roi_bar, textvariable=self.edit_zone,
                    values=ZONE_NAMES, width=8,
                    state="readonly").pack(side=tk.LEFT, padx=2)
        for txt, cmd in [("Undo", self._undo_pt), ("Clear Zone", self._clear_zone),
                        ("Clear All", self._clear_all), ("Auto ROI", self._auto_roi),
                        ("Save ROI", self._save_roi), ("Load ROI", self._load_roi)]:
            ttk.Button(roi_bar, text=txt, command=cmd).pack(side=tk.LEFT, padx=3)

        # ═══ RIGHT PANEL ═══
        right = tk.Frame(body, bg="#16213e", width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)
        right.pack_propagate(False)

        canvas = tk.Canvas(right, bg="#16213e", highlightthickness=0)
        sb = ttk.Scrollbar(right, orient="vertical", command=canvas.yview)
        self.af = tk.Frame(canvas, bg="#16213e")
        self.af.bind("<Configure>",
                    lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.af, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(self.af, text="📊 Analytics", font=("Arial", 14, "bold"),
                bg="#16213e", fg="white").pack(pady=8)

        self.fps_lbl = tk.Label(self.af, text="FPS: 0.0",
                                font=("Consolas", 11, "bold"),
                                bg="#16213e", fg="#00ffcc")
        self.fps_lbl.pack(anchor="w", padx=10)

        self.tracks_lbl = tk.Label(self.af, text="Active: 0",
                                font=("Consolas", 11),
                                bg="#16213e", fg="white")
        self.tracks_lbl.pack(anchor="w", padx=10)

        self.total_lbl = tk.Label(self.af, text="Total: 0",
                                font=("Arial", 14, "bold"),
                                bg="#16213e", fg="#ffd700")
        self.total_lbl.pack(anchor="w", padx=10, pady=(4, 4))

        # Category counts (Indian)
        tk.Label(self.af, text="Vehicle Categories",
                font=("Arial", 11, "bold"),
                bg="#16213e", fg="white").pack(anchor="w", padx=10)

        self.cat_labels = {}
        for cat in ["Two-Wheeler", "Auto-Rickshaw", "Four-Wheeler", "Heavy Vehicle"]:
            lbl = tk.Label(self.af, text=f"  {cat}: 0",
                        font=("Consolas", 10),
                        bg="#16213e", fg="#ff79c6", anchor="w")
            lbl.pack(fill=tk.X, padx=10)
            self.cat_labels[cat] = lbl

        # Direction counts
        tk.Label(self.af, text="Direction Counts",
                font=("Arial", 11, "bold"),
                bg="#16213e", fg="white").pack(anchor="w", padx=10, pady=(8, 0))

        self.dir_labels = {}
        all_dirs = [
            "North->South", "North->East", "North->West",
            "South->North", "South->East", "South->West",
            "East->West", "East->North", "East->South",
            "West->East", "West->North", "West->South"
        ]
        for d in all_dirs:
            lbl = tk.Label(self.af, text=f"  {d}: 0",
                        font=("Consolas", 9),
                        bg="#16213e", fg="#64dfdf", anchor="w")
            lbl.pack(fill=tk.X, padx=10)
            self.dir_labels[d] = lbl

        # Class counts
        tk.Label(self.af, text="Vehicle Types",
                font=("Arial", 11, "bold"),
                bg="#16213e", fg="white").pack(anchor="w", padx=10, pady=(8, 0))

        self.cls_labels = {}
        for name in ["bicycle", "car", "motorcycle", "bus", "truck"]:
            emoji = {"bicycle": "🚲", "car": "🚗", "motorcycle": "🏍️",
                    "bus": "🚌", "truck": "🚛"}.get(name, "")
            lbl = tk.Label(self.af, text=f"  {emoji} {name}: 0",
                        font=("Consolas", 10),
                        bg="#16213e", fg="#f38181", anchor="w")
            lbl.pack(fill=tk.X, padx=10)
            self.cls_labels[name] = lbl

        self.roi_lbl = tk.Label(self.af, text="ROI: Not set",
                                font=("Arial", 10),
                                bg="#16213e", fg="#ff6b6b")
        self.roi_lbl.pack(anchor="w", padx=10, pady=(10, 2))

        self.ai_lbl = tk.Label(self.af, text="AI: Not configured",
                            font=("Arial", 10),
                            bg="#16213e", fg="#ff6b6b")
        self.ai_lbl.pack(anchor="w", padx=10, pady=2)

        self.emergency_lbl = tk.Label(self.af, text="🚨 Emergency: 0",
                                    font=("Arial", 11, "bold"),
                                    bg="#16213e", fg="#ff4b2b")
        self.emergency_lbl.pack(anchor="w", padx=10, pady=4)

        tk.Label(self.af, text=(
            "━━ Two-Wheeler Tips ━━\n"
            "• Lower 2W Conf slider\n"
            "• Enable Enhance 2-Wheeler\n"
            "• Use yolov8s or yolov8m\n"
            "• Use ImgSz 960+"
        ), font=("Arial", 9), bg="#16213e", fg="#888",
            justify="left").pack(anchor="w", padx=10, pady=8)
        
    # ── Model ──
    def _load_yolo(self):
        name = self.model_var.get()
        try:
            self.status_var.set(f"Loading {name}...")
            self.root.update_idletasks()
            self.yolo_model = YOLO(name)
            self.status_var.set(f"{name} loaded ✓")
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Error", str(e))

    def _set_key(self):
        key = self.gemini_key_var.get().strip()
        if not key:
            messagebox.showwarning("Key", "Enter Gemini key")
            return
        if not GEMINI_AVAILABLE:
            messagebox.showerror("Error", "Gemini SDK not found")
            return
        if self.ai.configure(key):
            self.ai_lbl.config(text="AI: Connected ✓", fg="#00ff99")
        else:
            self.ai_lbl.config(text="AI: Failed", fg="#ff6b6b")

    def ai_analyze(self):
        if not self.ai.configured:
            messagebox.showinfo("AI", "Set Gemini key first")
            return
        s = self.counter.get_summary()
        d = time.time() - self.start_time
        self.status_var.set("AI analyzing...")
        def run():
            r = self.ai.analyze_traffic(s, d)
            self.root.after(0, lambda: self._show_text("AI Traffic Analysis", r))
        threading.Thread(target=run, daemon=True).start()

    def ai_scene(self):
        if not self.ai.configured:
            messagebox.showinfo("AI", "Set Gemini key first")
            return
        frame = self.last_annotated or self.raw_frame
        if frame is None:
            return
        fc = frame.copy()
        self.status_var.set("AI scene analysis...")
        def run():
            r = self.ai.analyze_scene(fc)
            self.root.after(0, lambda: self._show_text("AI Scene Analysis", r))
        threading.Thread(target=run, daemon=True).start()

    def ai_recommend(self):
        if not self.ai.configured:
            messagebox.showinfo("AI", "Set Gemini key first")
            return
        s = self.counter.get_summary()
        def run():
            r = self.ai.get_recommendations(s)
            self.root.after(0, lambda: self._show_text("AI Recommendations", r))
        threading.Thread(target=run, daemon=True).start()

    def launch_simulation(self):
        """Launches the Pygame-based traffic simulator in a separate thread."""
        self.status_var.set("Launching Simulator...")
        # Check if images directory exists
        img_path = os.path.join(os.getcwd(), "Smart-Traffic-Management-main", "images")
        if not os.path.exists(img_path):
             # Fallback to local images
             img_path = os.path.join(os.getcwd(), "images")
             if not os.path.exists(img_path):
                 messagebox.showerror("Error", f"Images not found in {img_path}")
                 return

        def start_sim():
            try:
                sim = TrafficSimulator(img_path)
                sim.run()
            except Exception as e:
                print(f"Simulator Error: {e}")
                self.root.after(0, lambda: messagebox.showerror("Simulator Error", str(e)))

        t = threading.Thread(target=start_sim, daemon=True)
        t.start()

    def pandas_report(self):
        if not PANDAS_AVAILABLE:
            messagebox.showwarning("Pandas", "Pandas is not installed or broken. Analytics report is unavailable.")
            return
        dur = time.time() - self.start_time
        report = PandasAnalytics.generate_full_report(
            self.counter.events, self.counter.detection_log, dur)
        self._show_text("Pandas Traffic Report", report)

    def _show_text(self, title, text):
        self.status_var.set("Report ready")
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("750x550")
        win.configure(bg="#1a1a2e")
        tk.Label(win, text=title, font=("Arial", 14, "bold"),
                bg="#1a1a2e", fg="white").pack(pady=8)
        t = scrolledtext.ScrolledText(win, wrap=tk.WORD,
                                    font=("Consolas", 10),
                                    bg="#0f3460", fg="#e0e0e0")
        t.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        t.insert(tk.END, text)
        t.config(state=tk.DISABLED)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)

    # ── Source ──
    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        if path:
            self.source_var.set(path)

    def _open_source(self):
        src = self.source_var.get().strip()
        return cv2.VideoCapture(int(src) if src.isdigit() else src)

    # ── ROI ──
    def _toggle_roi(self):
        self.roi_edit = not self.roi_edit
        self.roi_btn.config(text=f"ROI: {'ON' if self.roi_edit else 'OFF'}")
        self._refresh_display()

    def _screen_to_frame(self, sx, sy):
        if self.raw_frame is None:
            return None
        fx = int((sx - self.d_pad_x) / self.d_scale)
        fy = int((sy - self.d_pad_y) / self.d_scale)
        h, w = self.raw_frame.shape[:2]
        if 0 <= fx < w and 0 <= fy < h:
            return (fx, fy)
        return None

    def _on_left_click(self, e):
        if not self.roi_edit:
            return
        pt = self._screen_to_frame(e.x, e.y)
        if pt:
            self.roi.add_point(self.edit_zone.get(), pt)
            self._update_roi_lbl()
            self._refresh_display()

    def _on_right_click(self, e):
        if not self.roi_edit:
            return
        self.roi.pop_point(self.edit_zone.get())
        self._update_roi_lbl()
        self._refresh_display()

    def _undo_pt(self):
        self.roi.pop_point(self.edit_zone.get())
        self._update_roi_lbl()
        self._refresh_display()

    def _clear_zone(self):
        self.roi.clear_zone(self.edit_zone.get())
        self._update_roi_lbl()
        self._refresh_display()

    def _clear_all(self):
        self.roi.clear_all()
        self._update_roi_lbl()
        self._refresh_display()

    def _auto_roi(self):
        if self.raw_frame is not None:
            h, w = self.raw_frame.shape[:2]
        else:
            w, h = 960, 540
        self.roi.generate_default(w, h)
        self._update_roi_lbl()
        self._refresh_display()

    def _save_roi(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        sz = None
        if self.raw_frame is not None:
            h, w = self.raw_frame.shape[:2]
            sz = {"width": w, "height": h}
        self.roi.save_json(path, sz)

    def _load_roi(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        self.roi.load_json(path)
        self._update_roi_lbl()
        self._refresh_display()

    def _update_roi_lbl(self):
        ok = [z for z in ZONE_NAMES if len(self.roi.polygons.get(z, [])) >= 3]
        self.roi_lbl.config(
            text=f"ROI: {', '.join(ok)}" if ok else "ROI: Not set",
            fg="#00ff99" if ok else "#ff6b6b")

    # ── Controls ──
    def start(self):
        if self.running:
            return
        if not self.yolo_model:
            messagebox.showerror("Error", "Load model first")
            return
        self.cap = self._open_source()
        if not self.cap.isOpened():
            self.status_var.set("Failed to open source")
            return

        ret, first = self.cap.read()
        if ret:
            self.raw_frame = first.copy()
            h, w = first.shape[:2]
            self.roi.set_frame_size(w, h)
            if not self.roi.has_polygons():
                self.roi.generate_default(w, h)
                self._update_roi_lbl()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.running = True
        self.paused = False
        self.frame_idx = 0
        self.fps = 0.0
        self.prev_time = time.time()
        self.start_time = time.time()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
        self.status_var.set("▶ Running")

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.status_var.set("■ Stopped")

    def toggle_pause(self):
        if self.running:
            self.paused = not self.paused
            self.status_var.set("⏸ Paused" if self.paused else "▶ Running")

    def reset_counts(self):
        self.counter.reset()
        self.start_time = time.time()
        self._update_analytics()

    def save_snap(self):
        frame = self.last_annotated or self.raw_frame
        if frame is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if path:
            cv2.imwrite(path, frame)

    def export_csv(self):
        if not PANDAS_AVAILABLE:
            messagebox.showwarning("Pandas", "Pandas is not installed or broken. CSV export is unavailable.")
            return
        if not self.counter.events:
            messagebox.showinfo("CSV", "No events")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        df = PandasAnalytics.events_to_df(self.counter.events)
        df.to_csv(path, index=False)
        self.status_var.set("CSV exported ✓")

    def export_excel(self):
        if not PANDAS_AVAILABLE:
            messagebox.showwarning("Pandas", "Pandas is not installed or broken. Excel export is unavailable.")
            return
        if not OPENPYXL_AVAILABLE:
            messagebox.showwarning("OpenPyXL", "openpyxl is not installed. Excel export is unavailable.")
            return
        if not self.counter.events:
            messagebox.showinfo("Excel", "No events")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        try:
            dur = time.time() - self.start_time
            if PandasAnalytics.export_excel(self.counter.events, self.counter.detection_log, 
                                            self.counter.emergency_events, dur, path):
                self.status_var.set("Excel exported ✓")
            else:
                messagebox.showerror("Error", "Failed to export Excel (Pandas error)")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ── Video Loop ──
    def _loop(self):
        try:
            while self.running and self.cap:
                if self.paused:
                    time.sleep(0.05)
                    continue

                # Retry loop for frame capture
                ret, frame = None, None
                for attempt in range(3):
                    ret, frame = self.cap.read()
                    if ret:
                        break
                    time.sleep(1.0)
                
                if not ret:
                    self._q({"type": "status", "text": "Source ended"})
                    break

                self.raw_frame = frame.copy()
                self.frame_idx += 1
                
                # Memory cleanup once every 30 frames
                if self.frame_idx % 30 == 0:
                    self.counter.cleanup(timeout=5.0)

                skip = max(1, self.skip_var.get())
                do_infer = (self.frame_idx % skip == 0)

                if do_infer:
                    annotated = self._process(frame)
                    self.last_annotated = annotated.copy()
                else:
                    annotated = (self.last_annotated
                                if self.last_annotated is not None
                                else frame.copy())

                now = time.time()
                dt = now - self.prev_time
                if dt > 0:
                    ifps = 1.0 / dt
                    self.fps = (0.85 * self.fps + 0.15 * ifps
                                if self.fps > 0 else ifps)
                self.prev_time = now

                if self.record_var.get() and do_infer:
                    if self.video_writer is None:
                        h, w = annotated.shape[:2]
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        self.video_writer = cv2.VideoWriter(
                            f"output_{ts}.mp4",
                            cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
                    self.video_writer.write(annotated)

                self._q({"type": "frame", "frame": annotated})
                if do_infer:
                    self._q({"type": "analytics"})

        except Exception as e:
            self._q({"type": "status", "text": f"Error: {e}"})
        finally:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

    def _process(self, frame):
        imgsz = int(self.imgsz_var.get())
        main_conf = float(self.conf_var.get())
        tw_conf = float(self.twowheeler_conf_var.get())
        enhance_2w = self.enhance_2w_var.get()

        # Class filter: include person(0) if enhancing 2W
        # Auto-update classes from model if available
        if self.yolo_model and hasattr(self.yolo_model, 'names'):
            model_names = self.yolo_model.names
            # Sync our VEHICLE_CLASSES with model's actual name-to-id mapping
            for idx, name in model_names.items():
                clean_name = name.lower().replace("-", " ").strip()
                if any(k in clean_name for k in ["car", "bus", "truck", "motorcycle", "bicycle", "auto", "rickshaw", "scooter"]):
                    if "rickshaw" in clean_name or "auto" in clean_name:
                        VEHICLE_CLASSES[idx] = "auto-rickshaw"
                    else:
                        # Map to nearest COCO name for consistency
                        for standard in ["car", "bus", "truck", "motorcycle", "bicycle"]:
                            if standard in clean_name:
                                VEHICLE_CLASSES[idx] = standard
                                break

        classes = list(VEHICLE_CLASSES.keys())
        if enhance_2w:
            classes.append(0)

        try:
            results = self.yolo_model.track(
                source=frame,
                persist=True,
                conf=min(main_conf, tw_conf),
                imgsz=imgsz,
                classes=classes,
                verbose=False
            )
        except Exception:
            return frame.copy()

        out = frame.copy()
        if not results or not results[0].boxes:
            # Still draw ROI and overlay even if no detections
            self.roi.draw(out)
            self._draw_overlay(out)
            return out

        # 1. Collect raw detections
        raw_dets = []
        for box in results[0].boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cid = int(box.cls[0])
                cnf = float(box.conf[0])
                tid = int(box.id[0]) if box.id is not None else None
                
                # Apply per-class confidence filter
                thresh = tw_conf if cid in TWO_WHEELER_IDS or cid == 0 else main_conf
                if cnf >= thresh:
                    raw_dets.append((x1, y1, x2, y2, cnf, cid, tid))
            except:
                continue

        # 2. Enhance 2-wheelers if requested
        if enhance_2w:
            final_dets = self.enhancer.associate_riders(raw_dets)
        else:
            final_dets = [d for d in raw_dets if d[5] in VEHICLE_CLASSES]

        # 3. Process and Draw
        for det in final_dets:
            x1, y1, x2, y2, cnf, cid, tid = det
            cls_name = VEHICLE_CLASSES.get(cid, "unknown")
            
            # Centroid calculation
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Zone assignment and Counting
            zone = self.roi.point_zone((cx, cy))
            if tid is not None:
                self.counter.process(tid, cls_name, (cx, cy), zone, 
                                     conf=cnf, bbox=(x1, y1, x2, y2),
                                     require_center=self.require_center.get())

            # Rendering
            is_emergency = is_ambulance(cls_name, cnf, (x1, y1, x2, y2))
            if is_emergency:
                self.last_emergency_zone = zone
                self.last_emergency_time = time.time()
                
                # Log emergency event
                self.counter.emergency_events.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "track_id": tid,
                    "vehicle_class": cls_name,
                    "zone": zone,
                    "confidence": round(cnf, 3),
                    "bbox": (x1, y1, x2, y2)
                })

                # Show alert if first time for this ID
                if tid is not None and tid not in self.counter.alerted_emergency_ids:
                    self.counter.alerted_emergency_ids.add(tid)
                    self.root.after(0, lambda z=zone: self._show_emergency_alert(z))

            color = (0, 0, 255) if is_emergency else (0, 255, 0)
            
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            
            label = f"{cls_name} {cnf:.2f}"
            if is_emergency:
                label = "AMBULANCE/EMERGENCY"
            
            # Debug info
            if self.debug_var.get():
                cv2.circle(out, (cx, cy), 4, (255, 0, 255), -1)
                if tid is not None:
                    label += f" ID:{tid}"
                if zone:
                    label += f" ({zone})"

            cv2.putText(out, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 4. Final Scene Decoration
        self.roi.draw(out)
        self._draw_overlay(out)
        return out

    def _show_emergency_alert(self, zone):
        """Show a self-destroying emergency alert popup."""
        win = tk.Toplevel(self.root)
        win.title("🚨 EMERGENCY ALERT")
        win.geometry("400x150")
        win.configure(bg="#ff4b2b")
        win.attributes("-topmost", True)
        
        tk.Label(win, text="🚨 EMERGENCY VEHICLE DETECTED",
                 font=("Arial", 14, "bold"),
                 bg="#ff4b2b", fg="white").pack(expand=True, pady=(20, 0))
        
        tk.Label(win, text=f"ZONE: {zone}",
                 font=("Arial", 16, "bold"),
                 bg="#ff4b2b", fg="white").pack(expand=True, pady=(0, 20))
        
        win.after(5000, win.destroy)

    def _draw_overlay(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (300, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"FPS: {self.fps:.1f}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Total: {self.counter.total}", (15, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show category summary
        two_w = self.counter.category_counts.get("Two-Wheeler", 0)
        four_w = self.counter.category_counts.get("Four-Wheeler", 0)
        heavy = self.counter.category_counts.get("Heavy Vehicle", 0)

        cv2.putText(frame, f"2W: {two_w}  4W: {four_w}  Heavy: {heavy}",
                    (15, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 200, 200), 2)

        cv2.putText(frame, f"Active: {len(self.counter.tracks)}",
                    (15, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)

        y = 118
        for d, c in sorted(self.counter.direction_counts.items(),
                            key=lambda x: x[1], reverse=True)[:4]:
            if c > 0:
                cv2.putText(frame, f"{d}: {c}", (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (100, 255, 200), 1)
                y += 18

        # Signal preemption flashing banner
        if time.time() - self.last_emergency_time < 3:
            if int(time.time() * 2) % 2 == 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                msg = f"🚨 CLEAR ZONE: {self.last_emergency_zone} - EMERGENCY VEHICLE"
                cv2.putText(frame, msg, (int(frame.shape[1]*0.1), 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

    # ── Queue ──
    def _q(self, item):
        try:
            self.ui_queue.put_nowait(item)
        except queue.Full:
            try:
                self.ui_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.ui_queue.put_nowait(item)
            except queue.Full:
                pass

    def _poll_queue(self):
        try:
            for _ in range(5):
                item = self.ui_queue.get_nowait()
                t = item.get("type")

                if t == "frame":
                    self._show(item["frame"])

                elif t == "analytics":
                    self._update_analytics()

                elif t == "status":
                    self.status_var.set(item["text"])

        except queue.Empty:
            pass

        self.root.after(25, self._poll_queue)

    # ── Display ──
    def _show(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            fw, fh = pil.size
            scale = min(DISPLAY_W / fw, DISPLAY_H / fh)
            nw, nh = int(fw * scale), int(fh * scale)
            self.d_scale = scale
            self.d_pad_x = (DISPLAY_W - nw) // 2
            self.d_pad_y = (DISPLAY_H - nh) // 2
            resized = pil.resize((nw, nh), Image.LANCZOS)
            canvas = Image.new("RGB", (DISPLAY_W, DISPLAY_H), (0, 0, 0))
            canvas.paste(resized, (self.d_pad_x, self.d_pad_y))
            imgtk = ImageTk.PhotoImage(image=canvas)
            self.video_lbl.imgtk = imgtk
            self.video_lbl.configure(image=imgtk)
        except Exception:
            pass

    def _refresh_display(self):
        if self.raw_frame is not None:
            temp = self.raw_frame.copy()
            self.roi.draw(temp)
            self._show(temp)

    def _update_analytics(self):
        try:
            self.fps_lbl.config(text=f"FPS: {self.fps:.1f}")
            self.tracks_lbl.config(text=f"Active: {len(self.counter.tracks)}")
            self.total_lbl.config(text=f"Total: {self.counter.total}")

            for cat, lbl in self.cat_labels.items():
                c = self.counter.category_counts.get(cat, 0)
                lbl.config(text=f"  {cat}: {c}",
                            fg="#ffd700" if c > 0 else "#ff79c6")

            for d, lbl in self.dir_labels.items():
                c = self.counter.direction_counts.get(d, 0)
                lbl.config(text=f"  {d}: {c}",
                            fg="#00ff99" if c > 0 else "#64dfdf")

            for name, lbl in self.cls_labels.items():
                c = self.counter.class_counts.get(name, 0)
                emoji = {"bicycle": "🚲", "car": "🚗", "motorcycle": "🏍️",
                            "bus": "🚌", "truck": "🚛"}.get(name, "")
                lbl.config(text=f"  {emoji} {name}: {c}",
                            fg="#ffd700" if c > 0 else "#f38181")

            self.emergency_lbl.config(text=f"🚨 Emergency: {len(self.counter.emergency_events)}")
        except Exception:
            pass

    def on_close(self):
        self.stop()
        self.root.destroy()


# ═══════════════════════════════════════
# RUN
# ═══════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()