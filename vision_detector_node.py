#!/usr/bin/env python3
"""
ROS 2 Node: Enhanced Vision Block Detector
Detects blocks with state tracking (on_table, stacked, etc.) and smart selection

Publishes:
    /blocks/status (std_msgs/String): Current status
    /blocks/waypoints_pixel (std_msgs/Float32MultiArray): Pixel waypoints
    /blocks/info (std_msgs/String): Detection info with block states
    /blocks/block_states (std_msgs/String): JSON of block states

Subscribes:
    /blocks/command (std_msgs/String): Commands from Claude
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import time
import csv
import json
from datetime import datetime
from pathlib import Path
import threading
from typing import Dict, List, Optional, Tuple

# ========================= USER CONSTANTS =========================
CAM_INDEX = 2
CAP_WIDTH = 640
CAP_HEIGHT = 480
TARGET_PREVIEW_FPS = 6
FRAME_DELAY_MS = max(1, int(1000 / max(0.5, TARGET_PREVIEW_FPS)))

# Output directories
MAIN_OUTPUT_DIR = Path("block_captures")
ARCHIVE_DIR = Path("block_archive")
CSV_OUTPUT_DIR = Path("block_captures")
CSV_OUTPUT_FILENAME = "waypoints_pixel.csv"

# Stability settings
MARKER_HOLD_SEC = 1.5
PIXEL_STABILITY_TOL = 20.0
DETECTION_SCORE_THRESHOLD = 0.5

# HSV Color Ranges
COLOR_RANGES = {
    'red': [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([160, 100, 100]), np.array([180, 255, 255]))
    ],
    'blue': [(np.array([100, 100, 100]), np.array([130, 255, 255]))],
    'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
    'yellow': [(np.array([20, 100, 100]), np.array([35, 255, 255]))]
}

# Detection thresholds - MATCHED TO WORKING CODE!
MIN_CONTOUR_AREA = 800      # Changed from 500 to match working code
MAX_CONTOUR_AREA = 15000    # Changed from 10000 to match working code
MIN_ASPECT_RATIO = 0.5      # NEW - from working code
MAX_ASPECT_RATIO = 2.0      # NEW - from working code
MIN_SOLIDITY = 0.7          # NEW - from working code

# Stacking detection threshold (pixels)
STACK_PROXIMITY_PX = 30  # If blocks within 30px, consider stacked

# =================================================================


class BlockState:
    """Track state of individual block"""
    def __init__(self, color: str, idx: int, x: int, y: int, area: int, rotation: float = 0.0):
        self.color = color
        self.idx = idx  # 0, 1, 2... for multiple blocks of same color
        self.x = x
        self.y = y
        self.area = area
        self.rotation = rotation  # Rotation angle in degrees
        self.state = "on_table"  # "on_table", "stacked_on_X", "unknown"
        self.stacked_on = None  # Color of block below
        self.has_block_on_top = False
    
    def to_dict(self):
        return {
            'color': self.color,
            'idx': self.idx,
            'id': f"{self.color}_{self.idx}",
            'x': self.x,
            'y': self.y,
            'area': self.area,
            'rotation': self.rotation,
            'state': self.state,
            'stacked_on': self.stacked_on,
            'has_block_on_top': self.has_block_on_top
        }


class StabilityLock:
    """Tracks detection stability over time"""
    def __init__(self, hold_sec=1.5, px_tol=20.0):
        self.hold_sec = hold_sec
        self.px_tol = px_tol
        self._last_centroids = None
        self._since = None
    
    def update(self, centroids):
        now = time.time()
        if centroids is None or len(centroids) == 0:
            self._last_centroids = None
            self._since = None
            return 0.0
        
        if self._last_centroids is None or len(self._last_centroids) != len(centroids):
            self._last_centroids = centroids
            self._since = now
            return 0.0
        
        centroids_arr = np.array(centroids)
        last_arr = np.array(self._last_centroids)
        distances = np.linalg.norm(centroids_arr - last_arr, axis=1)
        avg_dist = np.mean(distances)
        
        if avg_dist < self.px_tol:
            return now - (self._since or now)
        else:
            self._last_centroids = centroids
            self._since = now
            return 0.0
    
    def reset(self):
        self._last_centroids = None
        self._since = None


def detect_colored_blocks(frame) -> Tuple[Dict, Dict, float, List[BlockState]]:
    """
    Detect blocks and determine their states
    
    Returns:
        detected_blocks: Dict[color] = list of block dicts
        masks_debug: Debug masks
        score: Detection confidence
        block_states: List of BlockState objects with state info
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_blocks = {}
    masks_debug = {}
    total_blocks = 0
    all_block_states = []
    
    # FIX: Track color indices properly
    color_counters = {'red': 0, 'blue': 0, 'green': 0, 'yellow': 0}
    
    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            mask |= cv2.inRange(hsv, lower, upper)
        
        # Noise reduction - EXACTLY from working code lines 157-160
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        masks_debug[color_name] = mask.copy()
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # CRITICAL FIX: Filter by aspect ratio (from working code lines 96-99)
                aspect_ratio = float(w) / h if h > 0 else 0
                if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                    continue
                
                # CRITICAL FIX: Filter by solidity (from working code lines 101-105)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                if solidity < MIN_SOLIDITY:
                    continue
                
                cx = x + w // 2
                cy = y + h // 2
                
                # Calculate rotation angle
                rotation_angle = 0.0
                if len(cnt) >= 5:
                    # Use minAreaRect for initial rotation
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # For better square detection, calculate angle from corners
                    # Get the first edge vector
                    edge = box[1] - box[0]
                    rotation_angle = np.degrees(np.arctan2(edge[1], edge[0]))
                    
                    # Normalize to 0-360 range
                    if rotation_angle < 0:
                        rotation_angle += 360
                    
                    # For display purposes, keep in 0-90 range
                    # (since squares have 4-fold symmetry)
                    rotation_angle = rotation_angle % 90
                
                # FIX: Use color counter for proper indexing
                block_idx = color_counters[color_name]
                color_counters[color_name] += 1
                
                block_dict = {
                    'x': int(cx),
                    'y': int(cy),
                    'width': int(w),
                    'height': int(h),
                    'area': int(area),
                    'color': color_name,
                    'idx': block_idx,
                    'rotation': float(rotation_angle)
                }
                blocks.append(block_dict)
                
                # Create BlockState with rotation
                block_state = BlockState(color_name, block_idx, int(cx), int(cy), int(area), float(rotation_angle))
                all_block_states.append(block_state)
                
                total_blocks += 1
        
        if blocks:
            detected_blocks[color_name] = blocks
    
    # Determine which blocks are stacked
    all_block_states = analyze_stacking(all_block_states)
    
    score = min(1.0, total_blocks / 5.0) if total_blocks > 0 else 0.0
    
    return detected_blocks, masks_debug, score, all_block_states


def analyze_stacking(block_states: List[BlockState]) -> List[BlockState]:
    """
    Analyze which blocks are stacked on others
    Based on proximity and relative positions
    """
    if len(block_states) <= 1:
        return block_states
    
    # Check each pair of blocks
    for i, block_a in enumerate(block_states):
        for j, block_b in enumerate(block_states):
            if i == j:
                continue
            
            # Calculate distance
            dist = np.sqrt((block_a.x - block_b.x)**2 + (block_a.y - block_b.y)**2)
            
            # If blocks are close enough, they might be stacked
            if dist < STACK_PROXIMITY_PX:
                # Assume smaller area = on top (camera perspective)
                if block_a.area < block_b.area:
                    # block_a is on top of block_b
                    block_a.state = f"stacked_on_{block_b.color}"
                    block_a.stacked_on = block_b.color
                    block_b.has_block_on_top = True
                elif block_b.area < block_a.area:
                    # block_b is on top of block_a
                    block_b.state = f"stacked_on_{block_a.color}"
                    block_b.stacked_on = block_a.color
                    block_a.has_block_on_top = True
    
    return block_states


class VisionDetectorNode(Node):
    
    def __init__(self):
        super().__init__('vision_detector_node')
        
        self.declare_parameter('output_csv', 'block_captures/waypoints_pixel.csv')
        self.declare_parameter('auto_detect_on_startup', False)
        self.declare_parameter('camera_index', CAM_INDEX)
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/blocks/status', 10)
        self.waypoints_pub = self.create_publisher(Float32MultiArray, '/blocks/waypoints_pixel', 10)
        self.info_pub = self.create_publisher(String, '/blocks/info', 10)
        self.block_states_pub = self.create_publisher(String, '/blocks/block_states', 10)
        self.command_pub = self.create_publisher(String, '/blocks/command', 10)  # CRITICAL FIX: Forward command to executor!
        
        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/blocks/command', self.command_callback, 10
        )
        self.executor_status_sub = self.create_subscription(
            String, '/blocks/executor_status', self.executor_status_callback, 10
        )
        
        self.bridge = CvBridge()
        
        # State
        self.camera = None
        self.running = False
        self.auto_capture_enabled = False
        self.captured_result = None
        self.detection_thread = None
        
        # Enhanced state tracking
        self.current_command = None
        self.current_block_states = []  # List of BlockState objects
        self.needs_clarification = False
        self.clarification_message = ""
        
        # Command queue for sequential commands
        self.command_queue = []
        self.processing_command = False
        
        # Color filtering
        self.target_colors = None
        
        # Stability lock
        self.lock = StabilityLock(MARKER_HOLD_SEC, PIXEL_STABILITY_TOL)
        
        # Create output directories
        MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.get_logger().info('='*60)
        self.get_logger().info('Enhanced Vision Block Detector initialized')
        self.get_logger().info(f'Output CSV: {self.get_parameter("output_csv").value}')
        self.get_logger().info(f'Camera index: {self.get_parameter("camera_index").value}')
        self.get_logger().info('Features: Block state tracking, smart selection')
        self.get_logger().info('='*60)
        self.publish_status('idle')
    
    def publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'Vision status: {status}')
    
    def executor_status_callback(self, msg):
        """Handle executor status updates - cancel vision if executor is using memory"""
        status = msg.data
        
        # If executor is executing from memory, vision should stop waiting
        if status == 'executing' and self.processing_command and self.auto_capture_enabled:
            self.get_logger().info('✓ Executor using memory - canceling vision detection')
            self.auto_capture_enabled = False
            self.processing_command = False
            self.publish_status('idle')
    
    def command_callback(self, msg):
        """Handle commands - queue if busy, process immediately if idle"""
        cmd = msg.data.strip().lower()
        
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info(f'📥 COMMAND RECEIVED: "{cmd}"')
        
        # SPECIAL: rescan/restart commands bypass queue and reset everything
        if 'rescan' in cmd or 'restart' in cmd or 'reset' in cmd:
            self.get_logger().info('🔄 SPECIAL COMMAND: Resetting system...')
            self.get_logger().info('='*60)
            
            # Clear queue
            self.command_queue.clear()
            
            # Force reset state
            self.processing_command = False
            self.auto_capture_enabled = False
            
            # Process this command immediately
            self.current_command = cmd
            self.parse_command(cmd)
            
            if 'stop' in cmd:
                self.stop_detection()
            else:
                self.processing_command = True
                self.start_detection()
            return
        
        # Check if currently processing
        if self.processing_command:
            self.get_logger().info('⏳ System is BUSY - adding to queue')
            self.command_queue.append(cmd)
            self.get_logger().info(f'📋 Queue size: {len(self.command_queue)}')
            self.get_logger().info('='*60)
            return
        
        # Process immediately if idle
        self.get_logger().info('='*60)
        
        self.current_command = cmd
        self.parse_command(cmd)
        
        if 'stop' in cmd:
            self.stop_detection()
        else:
            self.processing_command = True
            self.start_detection()
    
    def parse_command(self, cmd: str):
        """Parse command to determine target colors"""
        colors_in_command = []
        
        # Handle RESET command
        if 'reset' in cmd or 'clear' in cmd or 'restart' in cmd:
            self.target_colors = None
            self.current_block_states = []
            self.get_logger().info('🔄 RESET command detected - will clear detection history')
            return
        
        # Handle colon format: "stack:green:blue"
        if ':' in cmd:
            parts = cmd.split(':')
            for part in parts:
                part = part.strip()
                if part in ['red', 'blue', 'green', 'yellow']:
                    colors_in_command.append(part)
        else:
            # Handle natural language
            for color in ['red', 'blue', 'green', 'yellow']:
                if color in cmd:
                    colors_in_command.append(color)
        
        # Special commands
        if 'all' in cmd:
            self.target_colors = None  # Detect all
            self.get_logger().info('🎯 Target: ALL blocks')
        elif colors_in_command:
            self.target_colors = colors_in_command
            self.get_logger().info(f'🎯 Target blocks: {", ".join(colors_in_command)}')
        else:
            self.target_colors = None
            self.get_logger().info('🎯 No color filter - will detect all blocks')
    
    def start_detection(self):
        """Start camera or reset for new detection"""
        
        if self.running:
            self.get_logger().info('📹 Camera already running - RE-ENABLING auto-capture for new command')
            self.auto_capture_enabled = True  # ✅ RE-ENABLE auto-capture!
            self.lock.reset()
            self.close_result_windows()
            self.publish_status('detecting')  # ✅ FIX: Set status to detecting!
            self.get_logger().info('✓ Auto-capture ENABLED - waiting for stability...')
            return
        
        self.publish_status('starting')
        
        try:
            cam_idx = self.get_parameter('camera_index').value
            self.camera = cv2.VideoCapture(cam_idx)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
            
            if not self.camera.isOpened():
                raise Exception(f"Failed to open camera {cam_idx}")
            
            self.running = True
            self.auto_capture_enabled = True
            self.lock.reset()
            
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            self.publish_status('detecting')
            self.get_logger().info('='*60)
            self.get_logger().info('✓ CAMERA STARTED - Live Preview Active')
            self.get_logger().info('='*60)
            
        except Exception as e:
            self.get_logger().error(f'Failed to start camera: {e}')
            self.publish_status('error')
            self.running = False
    
    def close_result_windows(self):
        """Close result windows but keep live camera"""
        try:
            cv2.destroyWindow("1. Original Frame")
            cv2.destroyWindow("2. Detected Blocks (Annotated)")
            cv2.destroyWindow("3. Detection Statistics")
        except:
            pass
    
    def _detection_loop(self):
        """Main detection loop"""
        
        cv2.namedWindow("ROS Block Detector - Live Camera", cv2.WINDOW_NORMAL)
        
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # CRITICAL FIX: Rotate frame 180° to match Dobot perspective!
            # This matches working code line 73: frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            display_frame = frame.copy()
            
            detected_blocks, masks_debug, score, block_states = detect_colored_blocks(frame)
            
            # Store block states
            self.current_block_states = block_states
            
            centroids = []
            for block_state in block_states:
                centroids.append((block_state.x, block_state.y))
            
            # Draw detection overlay
            self.draw_block_overlay(display_frame, block_states)
            
            hold_time = self.lock.update(centroids)
            
            # Log stability
            total_blocks = len(block_states)
            if total_blocks > 0 and hold_time > 0:
                self.get_logger().info(
                    f'Stability: {hold_time:.2f}s / {MARKER_HOLD_SEC:.1f}s, Auto: {self.auto_capture_enabled}',
                    throttle_duration_sec=1.0
                )
            
            status_text = f"Blocks: {total_blocks}"
            cv2.putText(display_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Stability indicator
            if total_blocks > 0 and score >= DETECTION_SCORE_THRESHOLD:
                if hold_time > 0:
                    progress = min(1.0, hold_time / MARKER_HOLD_SEC)
                    bar_width = int(300 * progress)
                    
                    cv2.rectangle(display_frame, (10, 60), (310, 90), (100, 100, 100), -1)
                    cv2.rectangle(display_frame, (10, 60), (10 + bar_width, 90), (0, 255, 0), -1)
                    
                    lock_text = f"STABLE: {hold_time:.1f}s / {MARKER_HOLD_SEC:.1f}s"
                    cv2.putText(display_frame, lock_text, (15, 82),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    if not self.auto_capture_enabled:
                        cv2.putText(display_frame, "Auto-capture DISABLED", (10, 110),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA)
                    
                    if hold_time >= MARKER_HOLD_SEC and self.auto_capture_enabled:
                        self.get_logger().info('🔒 BLOCKS LOCKED - Auto-capturing!')
                        self.capture_and_analyze(frame, detected_blocks, masks_debug, block_states)
                        self.lock.reset()
                        self.auto_capture_enabled = False
                else:
                    cv2.putText(display_frame, "Hold steady... (or press SPACE)", (10, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow("ROS Block Detector - Live Camera", display_frame)
            
            key = cv2.waitKey(FRAME_DELAY_MS)
            
            if key == ord(' ') or key == 32:
                self.get_logger().info('⌨️  SPACEBAR - Manual capture!')
                self.capture_and_analyze(frame, detected_blocks, masks_debug, block_states)
                self.lock.reset()
                self.auto_capture_enabled = False
            
            if key == ord('q') or key == 27:
                break
        
        cv2.destroyWindow("ROS Block Detector - Live Camera")
    
    def draw_block_overlay(self, frame: np.ndarray, block_states: List[BlockState]):
        """Draw blocks with state info and rotation indicators"""
        color_bgr = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255)
        }
        
        for block in block_states:
            x, y = block.x, block.y
            bgr = color_bgr.get(block.color, (255, 255, 255))
            
            # Draw circle
            cv2.circle(frame, (x, y), 8, bgr, -1, cv2.LINE_AA)
            
            # Draw rotation indicator line
            angle_rad = np.radians(block.rotation)
            end_x = int(x + 25 * np.cos(angle_rad))
            end_y = int(y + 25 * np.sin(angle_rad))
            cv2.line(frame, (x, y), (end_x, end_y), bgr, 2, cv2.LINE_AA)
            cv2.circle(frame, (end_x, end_y), 4, bgr, -1, cv2.LINE_AA)
            
            # Draw ID
            block_id = f"{block.color}_{block.idx}"
            cv2.putText(frame, block_id, (x - 30, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2, cv2.LINE_AA)
            
            # Draw rotation angle
            rotation_text = f"{block.rotation:.1f}°"
            cv2.putText(frame, rotation_text, (x - 20, y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
            
            # Draw state
            if block.state != "on_table":
                cv2.putText(frame, block.state, (x - 30, y + 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    
    def capture_and_analyze(self, frame, detected_blocks, masks_debug, block_states):
        """Capture and analyze - NO FILTERING, always publish all blocks"""
        
        self.publish_status('analyzing')
        
        # NO AMBIGUITY CHECK - Executor handles it with ID support!
        # Generic "blue" → executor uses blue_0
        # Specific "blue_2" → executor uses blue_2
        
        # NO FILTERING - Always publish ALL blocks
        # Executor will handle filtering based on command
        
        original_count = len(block_states)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info(f'📸 SNAPSHOT @ {ts}')
        self.get_logger().info(f'  Total blocks detected: {original_count}')
        for block in block_states:
            self.get_logger().info(f'    • {block.color}_{block.idx}: {block.state}')
        
        if self.target_colors:
            self.get_logger().info(f'  🎯 Command targets: {", ".join(self.target_colors)}')
            self.get_logger().info(f'  📋 Sending ALL {original_count} blocks to converter')
            self.get_logger().info(f'  ⚙️  Executor will filter to: {", ".join(self.target_colors)}')
        else:
            self.get_logger().info(f'  📋 Sending ALL {original_count} blocks')
        
        self.get_logger().info('='*60)
        
        # Save and publish ALL blocks (no filtering)
        self.save_and_publish(frame, detected_blocks, masks_debug, block_states, ts)
        
        self.publish_status('idle')
        self.get_logger().info('📹 Camera in PREVIEW MODE - waiting for next command...')
    
    def check_ambiguity(self, block_states: List[BlockState]) -> Optional[str]:
        """
        Check if command is ambiguous
        Returns clarification message if ambiguous, None if clear
        """
        cmd = self.current_command
        if not cmd:
            return None
        
        # Parse what user wants to do
        if 'place' in cmd or 'stack' in cmd:
            # Extract source and target colors
            colors = []
            for color in ['red', 'blue', 'green', 'yellow']:
                if color in cmd:
                    colors.append(color)
            
            if len(colors) >= 1:
                source_color = colors[0]
                
                # Count blocks of source color that are on_table
                source_blocks = [b for b in block_states 
                               if b.color == source_color and b.state == "on_table"]
                
                # If multiple source blocks on table, ambiguous!
                if len(source_blocks) > 1:
                    block_ids = [f"{b.color}_{b.idx}" for b in source_blocks]
                    return f"Multiple {source_color} blocks on table: {', '.join(block_ids)}. Which one?"
        
        return None
    
    def filter_blocks_smart(self, block_states: List[BlockState]) -> List[BlockState]:
        """
        Smart block selection:
        - For stacking: pick source block on_table, not already stacked
        - Include target block
        """
        cmd = self.current_command
        
        if 'place' in cmd or 'stack' in cmd:
            # Extract source and target
            colors = []
            for color in ['red', 'blue', 'green', 'yellow']:
                if color in cmd:
                    colors.append(color)
            
            if len(colors) >= 2:
                source_color = colors[0]
                target_color = colors[1]
                
                # Find source block (prefer on_table)
                source_blocks = [b for b in block_states 
                               if b.color == source_color and b.state == "on_table"]
                
                if not source_blocks:
                    # No blocks on table, pick any
                    source_blocks = [b for b in block_states if b.color == source_color]
                
                source_block = source_blocks[0] if source_blocks else None
                
                # Find target block
                target_blocks = [b for b in block_states if b.color == target_color]
                target_block = target_blocks[0] if target_blocks else None
                
                selected = []
                if source_block:
                    selected.append(source_block)
                if target_block:
                    selected.append(target_block)
                
                return selected
        
        # Default: filter by colors mentioned
        if self.target_colors:
            return [b for b in block_states if b.color in self.target_colors]
        
        return block_states
    
    def save_and_publish(self, frame, detected_blocks, masks_debug, all_block_states, ts):
        """Save files and publish waypoints for ALL blocks"""
        
        # Create waypoint list from ALL blocks
        waypoints = [(b.x, b.y) for b in all_block_states]
        
        # Save CSV with ALL blocks
        csv_path = Path(self.get_parameter('output_csv').value)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "color", "rotation"])  # Add rotation!
            for block in all_block_states:
                writer.writerow([block.x, block.y, block.color, block.rotation])
        
        self.get_logger().info(f'✓ Saved CSV with ALL {len(waypoints)} blocks: {csv_path}')
        
        # Create annotated image with ALL blocks
        annotated_img = self.create_annotated_image(frame, all_block_states)
        masks_img = self.create_masks_image(masks_debug)
        
        # Save to main folder (overwrite)
        main_paths = {
            "detected": MAIN_OUTPUT_DIR / "detected.png",
            "annotated": MAIN_OUTPUT_DIR / "annotated.png",
            "masks": MAIN_OUTPUT_DIR / "masks.png"
        }
        
        # Save to archive folder (timestamped)
        archive_paths = {
            "detected": ARCHIVE_DIR / f"detected_{ts}.png",
            "annotated": ARCHIVE_DIR / f"annotated_{ts}.png",
            "masks": ARCHIVE_DIR / f"masks_{ts}.png",
        }
        
        # Write images
        main_ok = []
        main_ok.append(("detected.png", cv2.imwrite(str(main_paths["detected"]), frame)))
        main_ok.append(("annotated.png", cv2.imwrite(str(main_paths["annotated"]), annotated_img)))
        main_ok.append(("masks.png", cv2.imwrite(str(main_paths["masks"]), masks_img)))
        
        archive_ok = []
        archive_ok.append(("detected", cv2.imwrite(str(archive_paths["detected"]), frame)))
        archive_ok.append(("annotated", cv2.imwrite(str(archive_paths["annotated"]), annotated_img)))
        archive_ok.append(("masks", cv2.imwrite(str(archive_paths["masks"]), masks_img)))
        
        # Log saved files
        self.get_logger().info('-'*60)
        self.get_logger().info('MAIN FOLDER:')
        self.get_logger().info(f'  {MAIN_OUTPUT_DIR.resolve()}')
        for key, success in main_ok:
            self.get_logger().info(f'  • {key:15s}: {"OK" if success else "FAILED"}')
        
        self.get_logger().info('-'*60)
        self.get_logger().info('ARCHIVE FOLDER:')
        self.get_logger().info(f'  {ARCHIVE_DIR.resolve()}')
        for key, success in archive_ok:
            self.get_logger().info(f'  • {key}_{ts}.png: {"OK" if success else "FAILED"}')
        self.get_logger().info('-'*60)
        
        # Display result windows
        self.display_results(frame, annotated_img)
        
        # Publish waypoints for ALL blocks with color info
        msg = Float32MultiArray()
        flattened = []
        for block in all_block_states:
            flattened.append(float(block.x))
            flattened.append(float(block.y))
        msg.data = flattened
        self.waypoints_pub.publish(msg)
        self.get_logger().info(f'✓ Published ALL {len(all_block_states)} waypoints to converter')
        
        # Publish block states with ALL blocks
        states_json = json.dumps([b.to_dict() for b in all_block_states])
        msg = String()
        msg.data = states_json
        self.block_states_pub.publish(msg)
        self.get_logger().info(f'✓ Published ALL {len(all_block_states)} block states to executor')
        
        # Publish command context for executor
        if self.target_colors:
            # DON'T forward command - LLM parser already published it to /blocks/command!
            # Vision's job is just to detect blocks, not forward commands
            # cmd_msg = String()
            # cmd_msg.data = self.current_command
            # self.command_pub.publish(cmd_msg)
            self.get_logger().info(f'✓ Detection complete for command: {self.current_command}')
        
        # CRITICAL: Disable auto-capture immediately after capture
        self.auto_capture_enabled = False
        self.processing_command = False
        self.publish_status('idle')
        
        # Process next command in queue if any
        if self.command_queue:
            next_cmd = self.command_queue.pop(0)
            self.get_logger().info('\n' + '='*60)
            self.get_logger().info('📹 Camera in PREVIEW MODE')
            self.get_logger().info('  Auto-capture DISABLED (waiting for next command)')
            self.get_logger().info('  Press SPACEBAR for manual capture')
            self.get_logger().info('='*60)
            self.get_logger().info(f'\n🔄 Processing queued command: "{next_cmd}"')
            self.get_logger().info(f'📋 Remaining in queue: {len(self.command_queue)}')
            
            # Process next command after a short delay
            time.sleep(1.0)
            self.current_command = next_cmd
            self.parse_command(next_cmd)
            self.processing_command = True
            self.start_detection()  # This will set auto_capture = True
        else:
            # No more queued commands - stay in preview
            self.get_logger().info('\n' + '='*60)
            self.get_logger().info('📹 Camera in PREVIEW MODE')
            self.get_logger().info('  Auto-capture DISABLED (waiting for next command)')
            self.get_logger().info('  Press SPACEBAR for manual capture')
            self.get_logger().info('='*60)
            self.get_logger().info('\n✓ Queue empty - ready for new commands')
            self.get_logger().info('📹 Camera in PREVIEW MODE - waiting for next command...')
    
    def create_annotated_image(self, frame: np.ndarray, block_states: List[BlockState]) -> np.ndarray:
        """Create annotated image with numbered blocks and rotation angles"""
        annotated = frame.copy()
        
        color_bgr = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255)
        }
        
        for i, block in enumerate(block_states):
            x, y = block.x, block.y
            bgr = color_bgr.get(block.color, (255, 255, 255))
            
            # Draw circle at center
            cv2.circle(annotated, (x, y), 8, bgr, -1, cv2.LINE_AA)
            
            # Draw rotation indicator line (30px long)
            angle_rad = np.radians(block.rotation)
            end_x = int(x + 30 * np.cos(angle_rad))
            end_y = int(y + 30 * np.sin(angle_rad))
            cv2.line(annotated, (x, y), (end_x, end_y), bgr, 3, cv2.LINE_AA)
            
            # Draw arrow head
            cv2.circle(annotated, (end_x, end_y), 5, bgr, -1, cv2.LINE_AA)
            
            # Draw block ID
            block_id = f"{block.color}_{block.idx}"
            cv2.putText(annotated, block_id, (x - 30, y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2, cv2.LINE_AA)
            
            # Draw rotation angle
            rotation_text = f"{block.rotation:.1f}°"
            cv2.putText(annotated, rotation_text, (x - 25, y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Draw state
            if block.state != "on_table":
                cv2.putText(annotated, block.state, (x - 30, y + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Total count
        total = len(block_states)
        cv2.putText(annotated, f"Detected: {total} blocks (with rotation)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        return annotated
    
    def create_masks_image(self, masks: Dict) -> np.ndarray:
        """Combine color masks into 2x2 grid"""
        h, w = CAP_HEIGHT, CAP_WIDTH
        combined = np.zeros((h * 2, w * 2), dtype=np.uint8)
        
        positions = {
            'red': (0, 0),
            'blue': (w, 0),
            'green': (0, h),
            'yellow': (w, h)
        }
        
        for color, mask in masks.items():
            if color in positions:
                px, py = positions[color]
                combined[py:py+h, px:px+w] = mask
        
        return combined
    
    def display_results(self, frame: np.ndarray, annotated_img: np.ndarray):
        """Display result windows that stay open"""
        
        cv2.namedWindow("1. Original Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("1. Original Frame", frame)
        
        cv2.namedWindow("2. Detected Blocks (Annotated)", cv2.WINDOW_NORMAL)
        cv2.imshow("2. Detected Blocks (Annotated)", annotated_img)
        
        cv2.waitKey(1)
        
        self.get_logger().info('✓ Result windows displayed')
    
    def stop_detection(self):
        """Stop camera"""
        self.running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        self.publish_status('idle')
    
    def reset(self):
        """Reset state"""
        self.stop_detection()
        self.lock.reset()
        self.current_block_states = []
        self.needs_clarification = False


def main(args=None):
    rclpy.init(args=args)
    node = VisionDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_detection()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()