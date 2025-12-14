#!/usr/bin/env python3
"""
ROS 2 Node: Enhanced Block Executor
Handles pick & place with dynamic Z-height adjustment and multi-block operations

Publishes:
    /blocks/executor_status (std_msgs/String): Current status
    /blocks/execution_progress (std_msgs/Float32): Progress (0.0-1.0)
    
Subscribes:
    /blocks/waypoints_dobot (std_msgs/Float32MultiArray): Dobot waypoints
    /blocks/block_states (std_msgs/String): Block states JSON
    /blocks/command (std_msgs/String): Command for operation type
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Float32MultiArray
import numpy as np
import time
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional

# ========================= CONSTANTS =========================

# Physical constants
TABLE_SURFACE_Z = -48.65  # mm - table surface height
BLOCK_HEIGHT = 11.0  # mm - height of one block (1.1cm)
HOVER_HEIGHT = 15.0  # mm - hover above highest point
TRAVEL_HEIGHT_OFFSET = 20.0  # mm - travel height above blocks

# Pick and place timing
PICK_DELAY = 0.2  # seconds
PLACE_DELAY = 0.2  # seconds
MOVE_DELAY = 0.1  # seconds

# Home position
HOME_X = 245.68
HOME_Y = 9.18
HOME_Z = 143.61
HOME_R = 2.14

# Speed settings
SPEED_LIN = 50  # mm/s linear speed
SPEED_ROT = 50  # degrees/s rotation speed

# Multi-block operation settings
ROW_SPACING = 30.0  # mm - spacing for "arrange in row"
SIDE_BY_SIDE_OFFSET = 30.0  # mm - offset for side-by-side placement (left/right/above/below)

# Rotation settings - ADDED FROM WORKING CODE!
APPLY_ROTATION = True
ROTATION_THRESHOLD = 1.0  # degrees - only rotate if difference > this value

# =================================================================

try:
    from pydobot import Dobot
    PYDOBOT_AVAILABLE = True
except ImportError:
    PYDOBOT_AVAILABLE = False


class BlockExecutorNode(Node):
    
    def __init__(self):
        super().__init__('executor_node')
        
        # Parameters
        self.declare_parameter('dobot_port', '/dev/ttyACM0')
        self.declare_parameter('simulation_mode', not PYDOBOT_AVAILABLE)
        self.declare_parameter('enable_pick_place', True)
        self.declare_parameter('auto_home_on_startup', False)  # DISABLED - home blocks ROS callbacks!
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/blocks/executor_status', 10)
        self.progress_pub = self.create_publisher(Float32, '/blocks/execution_progress', 10)
        self.block_states_pub = self.create_publisher(String, '/blocks/block_states', 10)  # CRITICAL!
        
        # Subscribers
        self.waypoints_sub = self.create_subscription(
            Float32MultiArray, '/blocks/waypoints_dobot', 
            self.waypoints_callback, 10
        )
        self.block_states_sub = self.create_subscription(
            String, '/blocks/block_states',
            self.block_states_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, '/blocks/command',
            self.command_callback, 10
        )
        
        # State
        self.device = None
        self.simulation_mode = self.get_parameter('simulation_mode').value
        self.enable_pick_place = self.get_parameter('enable_pick_place').value
        self.executing = False
        self.should_stop = False
        
        # Enhanced state tracking
        self.current_command = ""
        self.block_states = []  # List of block state dicts
        self.stack_heights = {}  # Dict[color] = current height in mm
        self.rotation_states = {}  # Dict[color] = current rotation
        
        # NEW: Block memory system for tracking positions across operations
        self.block_memory = {}  # Dict[block_id] = {'x': x, 'y': y, 'z': z, 'color': color}
        self.memory_enabled = True  # Flag to enable/disable memory
        
        # NEW: Spatial awareness system
        self.spatial_map = {}  # Dict[block_id] = spatial relationships
        self.distance_matrix = {}  # Dict[(id1, id2)] = distance
        
        # Command queue for sequential execution
        self.command_queue = []
        self.queued_waypoints = None
        
        # Track executed commands to prevent double execution
        self.last_executed_command = ""
        self.execution_timestamp = 0.0
        
        # CSV data cache - READ ROTATIONS FROM CSV!
        self.csv_rotations = {}  # Dict[index] = rotation angle
        self.csv_colors = {}  # Dict[index] = color
        self.declare_parameter('dobot_csv', 'block_captures/waypoints_dobot.csv')
        
        # Initialize stack heights (all on table initially)
        for color in ['red', 'blue', 'green', 'yellow']:
            self.stack_heights[color] = TABLE_SURFACE_Z
            self.rotation_states[color] = 0.0
        
        # Connect to Dobot
        if PYDOBOT_AVAILABLE:
            self.get_logger().info('✓ pydobot imported successfully!')
        
        if not self.simulation_mode:
            try:
                port = self.get_parameter('dobot_port').value
                self.get_logger().info(f'Connecting to Dobot on {port}...')
                self.device = Dobot(port=port)
                self.device.speed(SPEED_LIN, SPEED_ROT)
                time.sleep(0.5)
                self.get_logger().info('✓ Dobot connected successfully!')
            except Exception as e:
                self.get_logger().warn(f'Could not connect to Dobot: {e}')
                self.get_logger().info('🎮 Switching to SIMULATION mode')
                self.simulation_mode = True
        
        self.get_logger().info('='*60)
        self.get_logger().info('Enhanced Block Executor initialized')
        self.get_logger().info(f'Table surface Z: {TABLE_SURFACE_Z} mm')
        self.get_logger().info(f'Block height: {BLOCK_HEIGHT} mm')
        self.get_logger().info(f'Rotation threshold: {ROTATION_THRESHOLD}°')
        self.get_logger().info(f'Mode: {"SIMULATION" if self.simulation_mode else "REAL"}')
        self.get_logger().info('Features: Dynamic Z-height, rotation matching, multi-block operations')
        self.get_logger().info('='*60)
        self.publish_status('idle')
        
        # Auto-home
        if self.get_parameter('auto_home_on_startup').value:
            self.go_home()
    
    def publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'Dobot status: {status}')
    
    def publish_progress(self, progress: float):
        msg = Float32()
        msg.data = float(progress)
        self.progress_pub.publish(msg)
    
    def command_callback(self, msg):
        """
        Queue commands for sequential execution
        Smart execution: Use memory if blocks already detected, otherwise wait for vision
        """
        # DEBUG: Log that callback was called
        self.get_logger().info('🔔 COMMAND_CALLBACK TRIGGERED!')
        self.get_logger().info(f'🔔 Received message: "{msg.data}"')
        
        cmd = msg.data.strip().lower()
        self.get_logger().info(f'🔔 Parsed command: "{cmd}"')
        
        # 🆕 CRITICAL FIX: Check if this command was just executed (within last 5 seconds)
        # This prevents double execution when LLM sends parsed command after waypoints already executed
        current_time = time.time()
        if cmd == self.last_executed_command and (current_time - self.execution_timestamp) < 5.0:
            self.get_logger().info(f'⏭️  SKIPPING: Command already executed {(current_time - self.execution_timestamp):.1f}s ago')
            self.get_logger().info(f'   (Preventing duplicate execution from LLM parser)')
            return
        
        # Handle RESET/RESCAN commands - clear memory for fresh start
        if 'reset' in cmd or 'clear' in cmd or 'restart' in cmd or 'rescan' in cmd:
            self.reset_memory()
            # For rescan, still queue command so vision runs
            if 'rescan' in cmd:
                self.command_queue.append(cmd)
                self.get_logger().info(f'📝 Command queued ({len(self.command_queue)} in queue): "{cmd}"')
                self.get_logger().info('👁️  Waiting for vision detection...')
            return
        
        # Store as current command (will be used when waypoints arrive)
        self.current_command = cmd
        
        # 🚀 NEW: Check if we can execute from memory directly
        if self.can_execute_from_memory(cmd):
            self.get_logger().info(f'📝 Command: "{cmd}"')
            self.get_logger().info('⚡ SMART EXECUTION: Using memory - no vision needed!')
            self.execute_from_memory(cmd)
        else:
            # Add to queue only if waiting for vision
            self.command_queue.append(cmd)
            self.get_logger().info(f'📝 Command queued ({len(self.command_queue)} in queue): "{cmd}"')
            self.get_logger().info('👁️  Waiting for vision detection...')
    
    def reset_memory(self):
        """Reset all block memory and stack heights"""
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('🔄 RESETTING MEMORY')
        self.get_logger().info('='*60)
        
        self.block_memory.clear()
        for color in ['red', 'blue', 'green', 'yellow']:
            self.stack_heights[color] = TABLE_SURFACE_Z
            self.rotation_states[color] = 0.0
        
        self.get_logger().info('✓ Block memory cleared')
        self.get_logger().info('✓ Stack heights reset to table surface')
        self.get_logger().info('='*60 + '\n')
        self.publish_status('idle')
    
    def extract_colors_in_order(self, cmd: str) -> list:
        """
        Extract colors OR specific block IDs from command in ORDER they appear
        NOW SUPPORTS ORDINAL NUMBERS: "second red", "first blue", etc.
        
        Examples:
            "place green on red" → ['green', 'red']
            "place red_1 on blue_0" → ['red_1', 'blue_0']
            "pick second red" → ['red_1']
            "first blue" → ['blue_0']
            "third yellow" → ['yellow_2']
        """
        import re
        
        color_list = ['red', 'blue', 'green', 'yellow']
        found_items = []
        
        # Map ordinal words to numbers
        ordinal_map = {
            'first': 0, '1st': 0,
            'second': 1, '2nd': 1,
            'third': 2, '3rd': 2,
            'fourth': 3, '4th': 3,
            'fifth': 4, '5th': 4,
            'sixth': 5, '6th': 5,
            'seventh': 6, '7th': 6,
            'eighth': 7, '8th': 7
        }
        
        # Find each color/ID and its position in the command
        item_positions = []
        
        for color in color_list:
            # First check for ordinal + color (e.g., "second red", "first blue")
            for ordinal_word, index in ordinal_map.items():
                # Pattern: ordinal word + optional "the" + color + optional "block"
                pattern = rf'\b{ordinal_word}\s+(?:the\s+)?{color}(?:\s+block)?\b'
                match = re.search(pattern, cmd, re.IGNORECASE)
                if match:
                    pos = match.start()
                    block_id = f"{color}_{index}"  # e.g., "red_1" for "second red"
                    item_positions.append((pos, block_id))
                    break  # Only match once per color
            
            # Then check for specific IDs (e.g., "red_0", "red_1", etc.)
            pattern = rf'\b{color}_(\d+)\b'
            matches = re.finditer(pattern, cmd)
            for match in matches:
                pos = match.start()
                block_id = match.group(0)  # e.g., "red_1"
                # Check if THIS EXACT block_id was already added (not just any block of this color!)
                already_found = any(item == block_id for _, item in item_positions)
                if not already_found:
                    item_positions.append((pos, block_id))
            
            # Finally check for just the color name (defaults to first/index 0)
            pos = cmd.find(color)
            if pos != -1:
                # Make sure we didn't already find an ID at this position
                already_found = any(p == pos or item.startswith(color + '_') for p, item in item_positions)
                if not already_found:
                    item_positions.append((pos, color))
        
        # Sort by position (earliest first)
        item_positions.sort(key=lambda x: x[0])
        
        # Extract just the items in order
        found_items = [item for pos, item in item_positions]
        
        return found_items

    
    def can_execute_from_memory(self, cmd: str) -> bool:
        """
        Check if command can be executed using existing memory
        Returns True if all required blocks are in memory
        """
        # If memory is empty, need vision
        if not self.block_memory:
            return False
        
        # If currently executing, can't start another
        if self.executing:
            return False
        
        # Extract colors from command IN ORDER
        colors_in_cmd = self.extract_colors_in_order(cmd)
        
        # NEW: If no colors mentioned, check if it's a spatial command
        if not colors_in_cmd:
            # Check for spatial keywords
            spatial_keywords = ['nearest', 'farthest', 'closest', 'between']
            has_spatial = any(keyword in cmd for keyword in spatial_keywords)
            
            if has_spatial:
                # Spatial command - can execute if memory has any blocks
                if self.block_memory:
                    self.get_logger().info(f'  ✓ Spatial command - memory available')
                    return True
                else:
                    self.get_logger().info(f'  ⚠️  Spatial command but no blocks in memory - need vision')
                    return False
            else:
                # No colors and no spatial keywords - need vision
                return False
        
        # Check if all required blocks are in memory
        for item in colors_in_cmd:
            # Check if it's a specific ID (e.g., "red_1") or just a color (e.g., "red")
            if '_' in item:
                # Specific ID requested - check if exact block exists
                if item not in self.block_memory:
                    self.get_logger().info(f'  ⚠️  {item} not in memory - need vision')
                    return False
            else:
                # Just color - find any block with this color
                found = False
                for block_id, block_data in self.block_memory.items():
                    if block_data.get('color') == item:
                        found = True
                        break
                
                if not found:
                    self.get_logger().info(f'  ⚠️  {item} not in memory - need vision')
                    return False
        
        self.get_logger().info(f'  ✓ All blocks found in memory: {colors_in_cmd}')
        return True
    
    def execute_from_memory(self, cmd: str):
        """
        Execute command using block positions from memory
        Now supports SPATIAL COMMANDS (no colors needed!)
        """
        # CRITICAL: Publish status immediately so vision cancels detection!
        self.publish_status('executing')
        
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('🧠 EXECUTING FROM MEMORY')
        self.get_logger().info('='*60)
        
        # 🆕 NEW: Try spatial parsing first
        spatial_result = self.parse_spatial_command(cmd)
        
        if spatial_result['type'] == 'place_between':
            # Execute between operation
            self.get_logger().info('  🎯 Spatial command: PLACE BETWEEN')
            source_id = spatial_result['source_id']
            ref1_id = spatial_result['ref1_id']
            ref2_id = spatial_result['ref2_id']
            
            if source_id and ref1_id and ref2_id:
                self.get_logger().info(f'  📍 {source_id} between {ref1_id} and {ref2_id}')
                self.execute_place_between(source_id, ref1_id, ref2_id)
            else:
                self.get_logger().error('Failed to resolve block IDs for between operation')
            
            # 🆕 Track execution
            self.last_executed_command = cmd
            self.execution_timestamp = time.time()
            
            # Clear command
            if self.command_queue and self.command_queue[0] == cmd:
                self.command_queue.pop(0)
            self.current_command = ""
            return
        
        elif spatial_result['type'] in ['stack', 'place_side']:
            # Execute stack or side placement using resolved IDs
            source_id = spatial_result['source_id']
            target_id = spatial_result['target_id']
            
            if not source_id or not target_id:
                self.get_logger().error('Failed to resolve block IDs')
                self.current_command = ""
                return
            
            self.get_logger().info(f'  🎯 Spatial command: {spatial_result["type"].upper()}')
            self.get_logger().info(f'  📍 {source_id} → {target_id}')
            
            # Get positions from memory
            source_mem = self.get_block_from_memory(source_id)
            target_mem = self.get_block_from_memory(target_id)
            
            if not source_mem or not target_mem:
                self.get_logger().error('Blocks not found in memory')
                self.current_command = ""
                return
            
            waypoints = np.array([
                [source_mem['x'], source_mem['y']],
                [target_mem['x'], target_mem['y']]
            ])
            
            if spatial_result['type'] == 'stack':
                self.execute_stack(waypoints)
            else:  # place_side
                direction = spatial_result.get('direction', 'beside')
                self.execute_place_side(waypoints, direction, SIDE_BY_SIDE_OFFSET)
            
            # 🆕 Track execution
            self.last_executed_command = cmd
            self.execution_timestamp = time.time()
            
            # Clear command
            if self.command_queue and self.command_queue[0] == cmd:
                self.command_queue.pop(0)
            self.current_command = ""
            return
        
        # Fallback: Old color-based parsing
        self.get_logger().info('  📝 Using color-based parsing')
        
        # ✅ FIX: Extract colors IN THE ORDER they appear in command
        colors_in_cmd = self.extract_colors_in_order(cmd)
        
        self.get_logger().info(f'  📝 Command: "{cmd}"')
        self.get_logger().info(f'  🎨 Colors in order: {colors_in_cmd}')
        
        # Build waypoints array from memory IN COMMAND ORDER
        waypoints = []
        block_ids = []
        
        for item in colors_in_cmd:
            # Check if it's a specific ID (e.g., "red_1") or just a color (e.g., "red")
            if '_' in item:
                # Specific ID requested - find exact match
                if item in self.block_memory:
                    block_data = self.block_memory[item]
                    waypoints.append([block_data['x'], block_data['y']])
                    block_ids.append(item)
                    self.get_logger().info(f'  📍 {item}: ({block_data["x"]:.2f}, {block_data["y"]:.2f}, Z={block_data["z"]:.2f})')
                else:
                    self.get_logger().warn(f'  ⚠️  {item} not found in memory')
            else:
                # Just color - find first block with this color
                for block_id, block_data in self.block_memory.items():
                    if block_data.get('color') == item:
                        waypoints.append([block_data['x'], block_data['y']])
                        block_ids.append(block_id)
                        self.get_logger().info(f'  📍 {block_id}: ({block_data["x"]:.2f}, {block_data["y"]:.2f}, Z={block_data["z"]:.2f})')
                        break
        
        if not waypoints:
            self.get_logger().error('Failed to build waypoints from memory!')
            return
        
        waypoints = np.array(waypoints)
        
        # Detect operation type
        operation = self.detect_operation(cmd, len(waypoints))
        self.get_logger().info(f'🎯 Operation: {operation["type"]}')
        
        # Execute!
        self.execute_operation(operation, waypoints)
        
        # 🆕 Track this execution to prevent duplicates
        self.last_executed_command = cmd
        self.execution_timestamp = time.time()
        
        # ✅ FIX: Remove current command from queue BEFORE processing next
        if self.command_queue and self.command_queue[0] == cmd:
            self.command_queue.pop(0)
        
        # Clear current command
        self.current_command = ""


    
    def update_block_memory(self, block_id: str, x: float, y: float, z: float, color: str = None, rotation: float = None):
        """
        Update block position in memory after movement
        CRITICAL for dynamic Z-height tracking!
        """
        if not self.memory_enabled:
            return
        
        # Keep existing rotation if not provided
        existing = self.block_memory.get(block_id, {})
        if rotation is None:
            rotation = existing.get('rotation', 0.0)
        
        self.block_memory[block_id] = {
            'x': x,
            'y': y,
            'z': z,
            'color': color or block_id.split('_')[0],
            'rotation': rotation,  # Store rotation!
            'last_updated': time.time()
        }
        
        self.get_logger().info(f'📍 Memory updated: {block_id} → ({x:.2f}, {y:.2f}, Z={z:.2f}, R={rotation:.1f}°)')
        
        # CRITICAL: Publish updated block states for LLM parser!
        self.publish_block_states_from_memory()
    
    def publish_block_states_from_memory(self):
        """
        Publish current block states from memory to /blocks/block_states
        This keeps LLM parser updated with current positions and stack states.
        Uses XY + Z logic so stacked blocks share (x, y) with their base.
        """
        block_states = []

        # How close in XY (in mm) we consider "same position" for a stack
        XY_TOLERANCE_MM = 5.0
        XY_TOLERANCE_SQ = XY_TOLERANCE_MM * XY_TOLERANCE_MM

        for block_id, block_data in self.block_memory.items():
            x = block_data['x']
            y = block_data['y']
            z = block_data['z']

            state = "on_table"
            stacked_on = None
            has_block_on_top = False

            # ---------- Who am I stacked on? (am I the TOP block?) ----------
            # If this block is elevated above the table, it might be on top of another block.
            if z > TABLE_SURFACE_Z + 1.0:
                best_base_id = None
                best_dist2 = None

                for other_id, other_data in self.block_memory.items():
                    if other_id == block_id:
                        continue

                    base_x = other_data['x']
                    base_y = other_data['y']
                    base_z = other_data['z']

                    # Check Z relationship: my z should be ~ base_z + BLOCK_HEIGHT
                    dz = z - (base_z + BLOCK_HEIGHT)
                    if abs(dz) > 2.0:
                        continue

                    # Check XY closeness
                    dx = x - base_x
                    dy = y - base_y
                    dist2 = dx * dx + dy * dy
                    if dist2 > XY_TOLERANCE_SQ:
                        continue

                    # Choose the closest base in XY if multiple candidates exist
                    if best_dist2 is None or dist2 < best_dist2:
                        best_dist2 = dist2
                        best_base_id = other_id

                if best_base_id is not None:
                    base_data = self.block_memory[best_base_id]
                    stacked_on = best_base_id
                    state = f"stacked_on_{base_data['color']}"
                else:
                    # Elevated but no clear base found directly under it
                    state = "stacked"

            # ---------- Do I have a block on top of me? (am I the BASE block?) ----------
            for other_id, other_data in self.block_memory.items():
                if other_id == block_id:
                    continue

                top_x = other_data['x']
                top_y = other_data['y']
                top_z = other_data['z']

                # Top block should be ~ one BLOCK_HEIGHT above this block
                dz_top = top_z - (z + BLOCK_HEIGHT)
                if abs(dz_top) > 2.0:
                    continue

                dx = top_x - x
                dy = top_y - y
                dist2 = dx * dx + dy * dy
                if dist2 <= XY_TOLERANCE_SQ:
                    has_block_on_top = True
                    break  # one is enough

            block_state = {
                'id': block_id,
                'color': block_data['color'],
                'x': int(x),
                'y': int(y),
                'area': 1000,  # Dummy value
                'rotation': block_data.get('rotation', 0.0),
                'state': state,
                'stacked_on': stacked_on,
                'has_block_on_top': has_block_on_top
            }
            block_states.append(block_state)

        # Publish as JSON
        import json
        msg = String()
        msg.data = json.dumps(block_states)
        self.block_states_pub.publish(msg)
        self.get_logger().debug(f'📡 Published {len(block_states)} block states from memory')
    
    # =================== SPATIAL AWARENESS SYSTEM ===================
    
    def calculate_distance(self, block1_id: str, block2_id: str) -> float:
        """Calculate Euclidean distance between two blocks"""
        b1 = self.block_memory.get(block1_id)
        b2 = self.block_memory.get(block2_id)
        
        if not b1 or not b2:
            return float('inf')
        
        dx = b1['x'] - b2['x']
        dy = b1['y'] - b2['y']
        return np.sqrt(dx*dx + dy*dy)
    
    def build_distance_matrix(self):
        """Build distance matrix for all blocks"""
        self.distance_matrix = {}
        block_ids = list(self.block_memory.keys())
        
        for i, id1 in enumerate(block_ids):
            for id2 in block_ids[i+1:]:
                dist = self.calculate_distance(id1, id2)
                self.distance_matrix[(id1, id2)] = dist
                self.distance_matrix[(id2, id1)] = dist  # Symmetric
    
    def find_nearest_block(self, reference_point: str = "camera") -> Optional[str]:
        """
        Find nearest block to reference point
        
        Args:
            reference_point: "camera" (lowest Y) or block_id
        """
        if not self.block_memory:
            return None
        
        if reference_point == "camera":
            # Nearest to camera = lowest Y coordinate
            nearest_id = min(self.block_memory.items(), 
                           key=lambda x: x[1]['y'])[0]
            return nearest_id
        else:
            # Nearest to specific block
            if reference_point not in self.block_memory:
                return None
            
            min_dist = float('inf')
            nearest_id = None
            
            for block_id in self.block_memory:
                if block_id != reference_point:
                    dist = self.calculate_distance(reference_point, block_id)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_id = block_id
            
            return nearest_id
    
    def find_farthest_block(self, reference_point: str = "camera") -> Optional[str]:
        """
        Find farthest block from reference point
        
        Args:
            reference_point: "camera" (highest Y) or block_id
        """
        if not self.block_memory:
            return None
        
        if reference_point == "camera":
            # Farthest from camera = highest Y coordinate
            farthest_id = max(self.block_memory.items(), 
                            key=lambda x: x[1]['y'])[0]
            return farthest_id
        else:
            # Farthest from specific block
            if reference_point not in self.block_memory:
                return None
            
            max_dist = 0
            farthest_id = None
            
            for block_id in self.block_memory:
                if block_id != reference_point:
                    dist = self.calculate_distance(reference_point, block_id)
                    if dist > max_dist:
                        max_dist = dist
                        farthest_id = block_id
            
            return farthest_id
    
    def find_block_between(self, block1_id: str, block2_id: str) -> Optional[str]:
        """
        Find block closest to midpoint between block1 and block2
        
        Returns block_id or None
        """
        b1 = self.block_memory.get(block1_id)
        b2 = self.block_memory.get(block2_id)
        
        if not b1 or not b2:
            return None
        
        # Calculate midpoint
        mid_x = (b1['x'] + b2['x']) / 2
        mid_y = (b1['y'] + b2['y']) / 2
        
        # Find closest block to midpoint (excluding b1 and b2)
        min_dist = float('inf')
        closest_id = None
        
        for block_id, block_data in self.block_memory.items():
            if block_id not in [block1_id, block2_id]:
                dx = block_data['x'] - mid_x
                dy = block_data['y'] - mid_y
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_id = block_id
        
        return closest_id
    
    def calculate_midpoint(self, block1_id: str, block2_id: str) -> Optional[tuple]:
        """
        Calculate midpoint between two blocks
        
        Returns (x, y) or None
        """
        b1 = self.block_memory.get(block1_id)
        b2 = self.block_memory.get(block2_id)
        
        if not b1 or not b2:
            return None
        
        mid_x = (b1['x'] + b2['x']) / 2
        mid_y = (b1['y'] + b2['y']) / 2
        
        return (mid_x, mid_y)
    
    def find_block_with_criteria(self, criteria: str, reference: str = None) -> Optional[str]:
        """
        Universal block finder based on spatial criteria
        
        Args:
            criteria: "nearest", "farthest", "between_X_Y", "closest_to_X"
            reference: block_id or "camera" (default: "camera")
        
        Examples:
            find_block_with_criteria("nearest") → nearest to camera
            find_block_with_criteria("farthest") → farthest from camera
            find_block_with_criteria("nearest", "red_0") → nearest to red_0
            find_block_with_criteria("between_red_0_blue_0") → block between red and blue
        """
        if not self.block_memory:
            return None
        
        reference = reference or "camera"
        
        if criteria == "nearest":
            return self.find_nearest_block(reference)
        elif criteria == "farthest":
            return self.find_farthest_block(reference)
        elif criteria.startswith("between_"):
            # Parse "between_red_0_blue_0"
            parts = criteria.split('_')
            if len(parts) >= 4:
                block1 = f"{parts[1]}_{parts[2]}"
                block2 = f"{parts[3]}_{parts[4]}"
                return self.find_block_between(block1, block2)
        elif criteria.startswith("closest_to_"):
            # Parse "closest_to_red_0"
            block_id = criteria.replace("closest_to_", "")
            return self.find_nearest_block(block_id)
        
        return None
    
    def get_spatial_description(self) -> str:
        """
        Generate human-readable spatial description of all blocks
        Used for LLM context
        """
        if not self.block_memory:
            return "No blocks detected"
        
        # Build distance matrix
        self.build_distance_matrix()
        
        # Find spatial relationships
        nearest_to_camera = self.find_nearest_block("camera")
        farthest_from_camera = self.find_farthest_block("camera")
        
        desc = "SPATIAL LAYOUT:\n"
        desc += f"  📍 Nearest to camera: {nearest_to_camera}\n"
        desc += f"  📍 Farthest from camera: {farthest_from_camera}\n\n"
        
        desc += "BLOCKS AND POSITIONS:\n"
        for block_id, data in sorted(self.block_memory.items()):
            x, y, z = data['x'], data['y'], data['z']
            color = data.get('color', 'unknown')
            
            # Calculate distances to other blocks
            neighbors = []
            for other_id in self.block_memory:
                if other_id != block_id:
                    dist = self.calculate_distance(block_id, other_id)
                    neighbors.append((other_id, dist))
            
            # Sort by distance
            neighbors.sort(key=lambda x: x[1])
            
            desc += f"  • {block_id} ({color}) at ({x:.1f}, {y:.1f}, Z={z:.1f})\n"
            if neighbors:
                closest = neighbors[0]
                desc += f"    → Closest to: {closest[0]} ({closest[1]:.1f}mm away)\n"
        
        return desc
    
    def resolve_spatial_reference(self, reference: str) -> Optional[str]:
        """
        Resolve spatial reference to actual block_id
        
        Args:
            reference: "nearest", "farthest", "red_0", "closest_to_blue", etc.
        
        Returns:
            block_id or None
        """
        reference = reference.strip().lower()
        
        # If it's already a block ID (color_number), return it
        if '_' in reference and any(color in reference for color in ['red', 'blue', 'green', 'yellow']):
            return reference
        
        # Spatial references
        if reference in ['nearest', 'closest']:
            return self.find_nearest_block("camera")
        elif reference == 'farthest':
            return self.find_farthest_block("camera")
        elif reference.startswith('nearest_to_') or reference.startswith('closest_to_'):
            # "nearest_to_red_0" or "closest_to_blue"
            target = reference.replace('nearest_to_', '').replace('closest_to_', '')
            # Try to find the target block
            target_id = self.resolve_spatial_reference(target)
            if target_id:
                return self.find_nearest_block(target_id)
        elif reference.startswith('farthest_from_'):
            # "farthest_from_red_0"
            target = reference.replace('farthest_from_', '')
            target_id = self.resolve_spatial_reference(target)
            if target_id:
                return self.find_farthest_block(target_id)
        
        # Color reference (just "red", "blue", etc.) - find first of that color
        for color in ['red', 'blue', 'green', 'yellow']:
            if reference == color:
                for block_id in self.block_memory:
                    if block_id.startswith(color + '_'):
                        return block_id
        
        return None
    
    def parse_spatial_command(self, cmd: str) -> Dict[str, any]:
        """
        Parse command that may contain spatial references
        
        Returns dict with:
            - type: 'stack', 'place_side', 'place_between'
            - source_id: resolved block ID
            - target_id: resolved block ID (for stack/place_side)
            - ref1_id, ref2_id: resolved IDs (for place_between)
            - direction: direction for place_side
        """
        cmd = cmd.lower().strip()
        
        result = {'type': None, 'source_id': None, 'target_id': None}
        
        # Check for "between" operation
        if 'between' in cmd:
            # Format: "place [source] between [ref1] and [ref2]"
            # Example: "place blue_0 between red_0 and yellow_0"
            parts = cmd.split('between')
            if len(parts) == 2:
                # Get source
                source_part = parts[0].replace('place', '').strip()
                source_id = self.resolve_spatial_reference(source_part)
                
                # Get ref1 and ref2
                refs_part = parts[1].strip()
                if ' and ' in refs_part:
                    ref_parts = refs_part.split(' and ')
                    ref1_id = self.resolve_spatial_reference(ref_parts[0].strip())
                    ref2_id = self.resolve_spatial_reference(ref_parts[1].strip())
                    
                    result['type'] = 'place_between'
                    result['source_id'] = source_id
                    result['ref1_id'] = ref1_id
                    result['ref2_id'] = ref2_id
                    return result
        
        # Check for side-by-side operations
        for direction in ['left of', 'right of', 'beside', 'next to']:
            if direction in cmd:
                parts = cmd.split(direction)
                if len(parts) == 2:
                    source_part = parts[0].replace('place', '').strip()
                    target_part = parts[1].strip()
                    
                    source_id = self.resolve_spatial_reference(source_part)
                    target_id = self.resolve_spatial_reference(target_part)
                    
                    # Normalize direction
                    if 'left' in direction:
                        dir_key = 'left'
                    elif 'right' in direction:
                        dir_key = 'right'
                    else:
                        dir_key = 'beside'
                    
                    result['type'] = 'place_side'
                    result['source_id'] = source_id
                    result['target_id'] = target_id
                    result['direction'] = dir_key
                    return result
        
        # Check for stacking operation (on/on top of)
        if ' on ' in cmd:
            parts = cmd.split(' on ')
            if len(parts) == 2:
                source_part = parts[0].replace('place', '').replace('stack', '').strip()
                target_part = parts[1].replace('top of', '').replace('top', '').strip()
                
                source_id = self.resolve_spatial_reference(source_part)
                target_id = self.resolve_spatial_reference(target_part)
                
                result['type'] = 'stack'
                result['source_id'] = source_id
                result['target_id'] = target_id
                return result
        
        return result
    
    
    # =================== END SPATIAL AWARENESS ===================
    
    
    def get_block_from_memory(self, block_id: str) -> Optional[Dict]:
        """Retrieve block position from memory"""
        return self.block_memory.get(block_id, None)
    
    def initialize_block_memory_from_detection(self, waypoints: np.ndarray):
        """
        Initialize memory with detected blocks (first detection only)
        All blocks start at table surface Z
        """
        if not self.memory_enabled:
            return
        
        # Only initialize if memory is empty (first detection)
        if self.block_memory:
            self.get_logger().info('📝 Memory already contains blocks - keeping existing positions')
            return
        
        self.get_logger().info('\n🔷 Initializing block memory from first detection:')
        
        # FIX: Track color counts to assign proper indices
        color_counters = {'red': 0, 'blue': 0, 'green': 0, 'yellow': 0}
        
        for i, (x, y) in enumerate(waypoints):
            color = self.csv_colors.get(i, 'unknown')
            rotation = self.csv_rotations.get(i, 0.0)  # Get rotation from CSV
            
            # FIX: Use color counter instead of loop index
            color_idx = color_counters.get(color, 0)
            color_counters[color] = color_idx + 1
            
            block_id = f"{color}_{color_idx}"
            
            # Initialize at table surface WITH ROTATION
            self.block_memory[block_id] = {
                'x': x,
                'y': y,
                'z': TABLE_SURFACE_Z,
                'color': color,
                'rotation': rotation,  # Store rotation!
                'last_updated': time.time()
            }
            
            self.get_logger().info(f'  • {block_id}: ({x:.2f}, {y:.2f}, Z={TABLE_SURFACE_Z}, R={rotation:.1f}°)')
        
        self.get_logger().info('✓ Block memory initialized\n')
    
    def block_states_callback(self, msg):
        """Receive block states from vision"""
        try:
            self.block_states = json.loads(msg.data)
            self.get_logger().info(f'Received {len(self.block_states)} block states')
            
            # DEBUG: Log rotation values
            for block in self.block_states:
                block_id = block.get('id', 'unknown')
                rotation = block.get('rotation', 'NOT_FOUND')
                self.get_logger().info(f'  Block {block_id}: rotation={rotation}')
        except Exception as e:
            self.get_logger().error(f'Failed to parse block states: {e}')
            pass
    
    def read_rotations_from_csv(self):
        """Read rotation angles from dobot CSV file"""
        csv_path = Path(self.get_parameter('dobot_csv').value)
        
        if not csv_path.exists():
            self.get_logger().warn(f'CSV file not found: {csv_path}')
            return
        
        try:
            import csv
            self.csv_rotations = {}
            self.csv_colors = {}
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    # Read color
                    if 'color' in row:
                        self.csv_colors[i] = row['color']
                    
                    # Read rotation
                    if 'rotation' in row:
                        try:
                            self.csv_rotations[i] = float(row['rotation'])
                        except:
                            self.csv_rotations[i] = 0.0
                    else:
                        self.csv_rotations[i] = 0.0
            
            self.get_logger().info(f'✓ Read rotations from CSV: {csv_path}')
            for idx, rot in self.csv_rotations.items():
                color = self.csv_colors.get(idx, 'unknown')
                self.get_logger().info(f'  [{idx}] {color}: {rot:.1f}°')
        
        except Exception as e:
            self.get_logger().error(f'Failed to read CSV rotations: {e}')
    
    def waypoints_callback(self, msg):
        """
        Receive Dobot waypoints and execute based on current command
        Skip if already executed from memory
        """
        if self.executing:
            self.get_logger().warn('Already executing - ignoring waypoints')
            return
        
        # Skip if no current command (means it was already executed from memory)
        if not self.current_command:
            self.get_logger().info('ℹ️  No pending command - waypoints received after memory execution')
            # Still initialize/update memory for future use
            waypoints = np.array(msg.data).reshape(-1, 2)
            self.read_rotations_from_csv()
            self.initialize_block_memory_from_detection(waypoints)
            return
        
        # Read rotations from CSV
        self.read_rotations_from_csv()
        
        # Convert to numpy array
        waypoints = np.array(msg.data).reshape(-1, 2)
        num_waypoints = len(waypoints)
        
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info(f'📥 RECEIVED {num_waypoints} DOBOT WAYPOINTS')
        self.get_logger().info('='*60)
        
        for i, (x, y) in enumerate(waypoints):
            rotation = self.csv_rotations.get(i, 0.0)
            color = self.csv_colors.get(i, 'unknown')
            self.get_logger().info(f'  {i+1}. {color:7s} at ({x:7.2f}, {y:7.2f}) @ {rotation:.1f}°')
        
        # Initialize block memory if empty (first detection)
        self.initialize_block_memory_from_detection(waypoints)
        
        # Filter waypoints based on command
        filtered_waypoints = self.filter_waypoints_by_command(self.current_command, waypoints)
        
        if filtered_waypoints is None or len(filtered_waypoints) == 0:
            self.get_logger().error('No waypoints after filtering!')
            return
        
        # Detect operation type
        operation = self.detect_operation(self.current_command, len(filtered_waypoints))
        
        # Execute
        self.execute_operation(operation, filtered_waypoints)
        
        # 🆕 Track this execution to prevent duplicates
        self.last_executed_command = self.current_command
        self.execution_timestamp = time.time()
        
        # Clear command after execution
        self.current_command = ""
    
    def filter_waypoints_by_command(self, cmd: str, all_waypoints: np.ndarray) -> Optional[np.ndarray]:
        """
        Filter waypoints based on colors mentioned in command
        RESPECTS the order colors appear in the command!
        
        Returns filtered waypoints or all if no specific colors mentioned
        """
        cmd = cmd.lower()
        
        self.get_logger().info(f'\n🔍 Filtering waypoints based on command: "{cmd}"')
        
        # ✅ FIX: Extract colors IN ORDER from command
        colors_in_cmd = self.extract_colors_in_order(cmd)
        
        # If no specific colors mentioned or "all" in command, use all waypoints
        if not colors_in_cmd or 'all' in cmd:
            self.get_logger().info(f'  Command mentions: ALL blocks')
            return all_waypoints
        
        self.get_logger().info(f'  Command mentions colors IN ORDER: {colors_in_cmd}')
        
        # Build waypoints in COMMAND ORDER
        filtered_waypoints = []
        filtered_ids = []
        
        for item in colors_in_cmd:
            # Check if it's a specific ID (e.g., "red_1") or just a color (e.g., "red")
            if '_' in item:
                # Specific ID requested - find exact match
                for i, block in enumerate(self.block_states):
                    if block.get('id') == item and i < len(all_waypoints):
                        filtered_waypoints.append(all_waypoints[i])
                        filtered_ids.append(block.get("id"))
                        self.get_logger().info(f'  Including waypoint {i}: {block.get("id")} at ({all_waypoints[i][0]:.2f}, {all_waypoints[i][1]:.2f})')
                        break
            else:
                # Just color - find first block with this color
                for i, block in enumerate(self.block_states):
                    if block.get('color') == item and i < len(all_waypoints):
                        filtered_waypoints.append(all_waypoints[i])
                        filtered_ids.append(block.get("id"))
                        self.get_logger().info(f'  Including waypoint {i}: {block.get("id")} at ({all_waypoints[i][0]:.2f}, {all_waypoints[i][1]:.2f})')
                        break
        
        if not filtered_waypoints:
            self.get_logger().warn(f'  No blocks found for colors: {colors_in_cmd}')
            return None
        
        self.get_logger().info(f'  Matched {len(filtered_waypoints)} blocks: {filtered_ids}')
        
        return np.array(filtered_waypoints)
    
    def detect_operation(self, cmd: str, num_waypoints: int) -> Dict:
        """
        Detect what operation to perform
        
        Returns dict with:
            type: 'stack', 'stack_all', 'arrange_row', 'place_side', 'single', 'multiple'
            params: operation-specific parameters
        """
        # Stack all blocks
        if 'stack all' in cmd or 'stack them all' in cmd:
            return {'type': 'stack_all', 'order': 'auto'}
        
        # Arrange in row
        if 'arrange' in cmd and 'row' in cmd:
            return {'type': 'arrange_row', 'spacing': ROW_SPACING}
        
        # Side-by-side placement (NEW!)
        if num_waypoints == 2:
            # Check for directional placement
            if 'left of' in cmd or 'left to' in cmd:
                return {'type': 'place_side', 'direction': 'left', 'offset': SIDE_BY_SIDE_OFFSET}
            elif 'right of' in cmd or 'right to' in cmd:
                return {'type': 'place_side', 'direction': 'right', 'offset': SIDE_BY_SIDE_OFFSET}
            elif 'above' in cmd:
                return {'type': 'place_side', 'direction': 'above', 'offset': SIDE_BY_SIDE_OFFSET}
            elif 'below' in cmd:
                return {'type': 'place_side', 'direction': 'below', 'offset': SIDE_BY_SIDE_OFFSET}
            # Smart defaults for ambiguous words
            elif 'beside' in cmd or 'next to' in cmd or 'near' in cmd:
                # Intelligent direction detection
                if 'left' in cmd:
                    direction = 'left'
                elif 'right' in cmd:
                    direction = 'right'
                else:
                    direction = 'right'  # Default to right
                self.get_logger().info(f'  📍 Ambiguous placement detected: using {direction.upper()} as default')
                return {'type': 'place_side', 'direction': direction, 'offset': SIDE_BY_SIDE_OFFSET}
        
        # Stack two blocks
        if num_waypoints == 2:
            # Check for rotation
            rotation = 0
            if 'rotate' in cmd:
                # Extract rotation angle
                import re
                match = re.search(r'(\d+)\s*deg', cmd)
                if match:
                    rotation = int(match.group(1))
            
            return {'type': 'stack', 'rotation': rotation}
        
        # Single block
        if num_waypoints == 1:
            return {'type': 'single'}
        
        # Multiple separate operations
        return {'type': 'multiple'}
    
    def execute_operation(self, operation: Dict, waypoints: np.ndarray) -> bool:
        """Execute based on operation type"""
        
        op_type = operation['type']
        
        self.get_logger().info(f'🎯 Operation: {op_type}')
        
        if op_type == 'stack':
            return self.execute_stack(waypoints, operation.get('rotation', 0))
        elif op_type == 'stack_all':
            return self.execute_stack_all(waypoints)
        elif op_type == 'arrange_row':
            return self.execute_arrange_row(waypoints, operation.get('spacing', ROW_SPACING))
        elif op_type == 'place_side':
            return self.execute_place_side(waypoints, operation.get('direction', 'left'), operation.get('offset', SIDE_BY_SIDE_OFFSET))
        elif op_type == 'single':
            return self.execute_single(waypoints[0])
        else:  # multiple
            return self.execute_multiple(waypoints)
    
    def execute_stack(self, waypoints: np.ndarray, rotation: int = 0) -> bool:
        """
        Stack operation with dynamic Z-height and rotation matching
        
        Args:
            waypoints: [source, target]
            rotation: Rotation angle in degrees (optional override)
        """
        if len(waypoints) != 2:
            self.get_logger().error('Stack requires exactly 2 waypoints')
            return False
        
        source_xy = waypoints[0]
        target_xy = waypoints[1]
        
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('📦 STACK OPERATION')
        self.get_logger().info(f'  Source: ({source_xy[0]:.2f}, {source_xy[1]:.2f})')
        self.get_logger().info(f'  Target: ({target_xy[0]:.2f}, {target_xy[1]:.2f})')
        if rotation:
            self.get_logger().info(f'  Manual Rotation Override: {rotation}°')
        self.get_logger().info('='*60)
        
        self.publish_status('executing')
        self.executing = True
        
        try:
            # Get source and target colors and IDs from command
            cmd = self.current_command.lower()
            colors_in_cmd = self.extract_colors_in_order(cmd)
            
            if len(colors_in_cmd) < 2:
                self.get_logger().error('Could not parse colors from command')
                return False
            
            source_color = colors_in_cmd[0]
            target_color = colors_in_cmd[1]
            
            self.get_logger().info(f'  🎨 Source: {source_color}, Target: {target_color}')
            
            # Find block IDs from filtered waypoints (respects command order!)
            # The filtered waypoints are already in the order we want
            if len(colors_in_cmd) < 2:
                self.get_logger().error('Need at least 2 colors for stack operation')
                return False
            
            # Get the actual block IDs from the filtered/detected blocks
            # Find which blocks match our source and target colors
            source_candidates = []
            target_candidates = []
            
            for i, block in enumerate(self.block_states):
                block_color = block.get('color')
                block_id = block.get('id')
                
                # Match source color (could be "red" or "red_1")
                if '_' in source_color:
                    # Specific ID requested
                    if block_id == source_color:
                        source_candidates.append(block_id)
                else:
                    # Just color - collect all matching
                    if block_color == source_color:
                        source_candidates.append(block_id)
                
                # Match target color
                if '_' in target_color:
                    if block_id == target_color:
                        target_candidates.append(block_id)
                else:
                    if block_color == target_color:
                        target_candidates.append(block_id)
            
            # Use first match for each (unless specific ID requested)
            if not source_candidates or not target_candidates:
                self.get_logger().error(f'Could not find blocks: source={source_color}, target={target_color}')
                self.get_logger().error(f'  Available: {[b.get("id") for b in self.block_states]}')
                return False
            
            source_id = source_candidates[0]
            target_id = target_candidates[0]
            
            self.get_logger().info(f'  🆔 Block IDs: {source_id} → {target_id}')
            
            # DYNAMIC Z-HEIGHT: Get current positions from memory
            source_mem = self.get_block_from_memory(source_id)
            target_mem = self.get_block_from_memory(target_id)
            
            if source_mem and target_mem:
                # Use memory positions (blocks may have been moved)
                source_x, source_y, source_z = source_mem['x'], source_mem['y'], source_mem['z']
                target_x, target_y, target_z = target_mem['x'], target_mem['y'], target_mem['z']
                
                self.get_logger().info(f'  📍 Using MEMORY positions:')
                self.get_logger().info(f'     {source_id}: ({source_x:.2f}, {source_y:.2f}, Z={source_z:.2f})')
                self.get_logger().info(f'     {target_id}: ({target_x:.2f}, {target_y:.2f}, Z={target_z:.2f})')
            else:
                # Fall back to detection positions (first time)
                source_x, source_y = source_xy[0], source_xy[1]
                target_x, target_y = target_xy[0], target_xy[1]
                source_z = TABLE_SURFACE_Z
                target_z = TABLE_SURFACE_Z
                
                self.get_logger().info(f'  📍 Using DETECTION positions (first time)')
            
            # Calculate placement Z: on top of target block
            place_z = target_z + BLOCK_HEIGHT
            travel_z = place_z + TRAVEL_HEIGHT_OFFSET
            
            # Get rotations from MEMORY (not CSV indices!)
            source_rotation = source_mem.get('rotation', 0.0) if source_mem else 0.0
            target_rotation = target_mem.get('rotation', 0.0) if target_mem else 0.0
            rotation_diff = target_rotation - source_rotation
            
            self.get_logger().info(f'📏 Heights:')
            self.get_logger().info(f'  Source pick Z: {source_z:.2f} mm (current position)')
            self.get_logger().info(f'  Target current Z: {target_z:.2f} mm')
            self.get_logger().info(f'  Place Z: {place_z:.2f} mm (target + {BLOCK_HEIGHT}mm)')
            self.get_logger().info(f'  Travel Z: {travel_z:.2f} mm')
            self.get_logger().info(f'📐 Rotations:')
            self.get_logger().info(f'  Source: {source_rotation:.1f}°')
            self.get_logger().info(f'  Target: {target_rotation:.1f}°')
            self.get_logger().info(f'  Difference: {rotation_diff:.1f}° (will rotate to match)')
            
            # Execute sequence - MATCHES WORKING CODE EXACTLY!
            # 1. Move to source at travel height
            self.get_logger().info(f'\n  ➜ 1. Moving above source block...')
            if not self.move_to(source_x, source_y, travel_z, -rotation_diff):
                return False
            
            # 2. Lower to pick
            self.get_logger().info(f'  ⬇ 2. Lowering to pick...')
            if not self.move_to(source_x, source_y, source_z, -rotation_diff):
                return False
            
            # 3. Pick
            self.get_logger().info(f'  🧲 3. Activating suction...')
            if not self.activate_suction(True):
                return False
            time.sleep(PICK_DELAY)
            self.get_logger().info(f'     (Waiting for suction to grip...)')
            
            # 4. Raise with block
            self.get_logger().info(f'  ⬆ 4. Raising with block...')
            if not self.move_to(source_x, source_y, travel_z):
                return False
            
            # 5. ROTATION STEP - Match target block orientation (CRITICAL!)
            if APPLY_ROTATION and abs(rotation_diff) > ROTATION_THRESHOLD:
                self.get_logger().info(f'  🔄 5. Rotating {rotation_diff:.1f}° to match target orientation...')
                if not self.rotate_relative(rotation_diff):
                    self.get_logger().warn('     Rotation failed, continuing without rotation')
                else:
                    self.get_logger().info(f'     (Block rotation complete)')
            else:
                self.get_logger().info(f'  ⏭  5. Skipping rotation (diff={rotation_diff:.1f}° < threshold={ROTATION_THRESHOLD}°)')
            
            # 6. Move to target
            self.get_logger().info(f'  ➜ 6. Moving to target location...')
            if not self.move_to(target_x, target_y, travel_z):
                return False
            
            # 7. Lower to place
            self.get_logger().info(f'  ⬇ 7. Lowering to place...')
            if not self.move_to(target_x, target_y, place_z):
                return False
            
            # 8. Release
            self.get_logger().info(f'  🔓 8. Releasing suction...')
            if not self.activate_suction(False):
                return False
            time.sleep(PLACE_DELAY)
            self.get_logger().info(f'     (Waiting for release...)')
            
            # 9. Raise clear
            self.get_logger().info(f'  ⬆ 9. Raising clear...')
            if not self.move_to(target_x, target_y, travel_z):
                return False
            
            # CRITICAL: Update block memory with new positions AND rotation!
            # Source block now has target's rotation (we rotated it to match)
            self.update_block_memory(source_id, target_x, target_y, place_z, source_color, target_rotation)
            # Target block stays at its current Z (it's now supporting source)
            if target_mem:
                self.get_logger().info(f'  📍 {target_id} remains at Z={target_z:.2f} (supporting {source_id})')
            
            self.get_logger().info('\n✓ STACK OPERATION COMPLETE!')
            self.get_logger().info(f'  {source_id} is now on {target_id} at Z={place_z:.2f}mm')
            
            # Home
            self.go_home()
            
            self.publish_status('idle')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Stack operation failed: {e}')
            self.publish_status('error')
            return False
        finally:
            self.executing = False
    
    def execute_stack_all(self, waypoints: np.ndarray) -> bool:
        """
        Stack all blocks in auto order (alphabetical)
        Bottom to top: blue → green → red → yellow
        """
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('🏗️  STACK ALL BLOCKS')
        self.get_logger().info('  Order: blue → green → red → yellow')
        self.get_logger().info('='*60)
        
        # Auto-determine stacking order
        order = ['blue', 'green', 'red', 'yellow']
        
        # Find blocks for each color
        blocks_by_color = {}
        for block in self.block_states:
            color = block.get('color')
            if color and block.get('state') == 'on_table':
                if color not in blocks_by_color:
                    blocks_by_color[color] = []
                blocks_by_color[color].append(block)
        
        # Stack them
        base_block = None
        for i, color in enumerate(order):
            if color not in blocks_by_color:
                continue
            
            block = blocks_by_color[color][0]
            
            if i == 0:
                # First block is base
                base_block = block
                self.get_logger().info(f'  Base: {color}')
            else:
                # Stack on previous
                if base_block:
                    source_wp = np.array([block['x'], block['y']])
                    target_wp = np.array([base_block['x'], base_block['y']])
                    
                    self.get_logger().info(f'\n  Stacking {color} on base...')
                    if not self.execute_stack(np.array([source_wp, target_wp])):
                        return False
        
        self.get_logger().info('\n✓ STACK ALL COMPLETE!')
        return True
    
    def execute_arrange_row(self, waypoints: np.ndarray, spacing: float) -> bool:
        """
        Arrange blocks in a horizontal row with specified spacing
        Blocks arranged alphabetically: blue, green, red, yellow
        """
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info(f'📏 ARRANGE IN ROW (spacing: {spacing}mm)')
        self.get_logger().info('='*60)
        
        self.publish_status('executing')
        self.executing = True
        
        try:
            # Sort blocks alphabetically
            sorted_blocks = []
            for block in self.block_states:
                sorted_blocks.append(block)
            sorted_blocks.sort(key=lambda b: b.get('color', ''))
            
            # Calculate row positions
            num_blocks = len(sorted_blocks)
            start_x = 250.0  # Starting X position
            y_pos = 0.0      # Y position (centered)
            
            self.get_logger().info(f'Arranging {num_blocks} blocks in row')
            self.get_logger().info(f'Order: {", ".join([b.get("color") for b in sorted_blocks])}')
            
            for i, block in enumerate(sorted_blocks):
                target_x = start_x + (i * spacing)
                target_y = y_pos
                
                color = block.get('color')
                source_x = block.get('x')
                source_y = block.get('y')
                
                self.get_logger().info(f'\n[{i+1}/{num_blocks}] Moving {color} block')
                self.get_logger().info(f'  From: ({source_x:.2f}, {source_y:.2f})')
                self.get_logger().info(f'  To:   ({target_x:.2f}, {target_y:.2f})')
                
                # Pick and place sequence
                travel_z = TABLE_SURFACE_Z + BLOCK_HEIGHT + TRAVEL_HEIGHT_OFFSET
                
                # 1. Move to source
                if not self.move_to(source_x, source_y, travel_z):
                    return False
                
                # 2. Lower to pick
                if not self.move_to(source_x, source_y, TABLE_SURFACE_Z):
                    return False
                
                # 3. Pick
                if not self.activate_suction(True):
                    return False
                time.sleep(PICK_DELAY)
                
                # 4. Raise
                if not self.move_to(source_x, source_y, travel_z):
                    return False
                
                # 5. Move to target
                if not self.move_to(target_x, target_y, travel_z):
                    return False
                
                # 6. Lower to place
                if not self.move_to(target_x, target_y, TABLE_SURFACE_Z):
                    return False
                
                # 7. Release
                if not self.activate_suction(False):
                    return False
                time.sleep(PLACE_DELAY)
                
                # 8. Raise
                if not self.move_to(target_x, target_y, travel_z):
                    return False
                
                self.get_logger().info(f'✓ {color} placed at position {i+1}')
            
            # Home
            self.go_home()
            
            self.get_logger().info('\n✓ ARRANGE ROW COMPLETE!')
            self.publish_status('idle')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Arrange row failed: {e}')
            self.publish_status('error')
            return False
        finally:
            self.executing = False
    
    def execute_place_side(self, waypoints: np.ndarray, direction: str, offset: float) -> bool:
        """
        Place source block next to target block (side-by-side)
        
        Args:
            waypoints: [source, target]
            direction: 'left', 'right', 'above', 'below'
            offset: Distance in mm from target block
        """
        if len(waypoints) != 2:
            self.get_logger().error('Side-by-side placement requires exactly 2 waypoints')
            return False
        
        source_xy = waypoints[0]
        target_xy = waypoints[1]
        
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info(f'↔️  SIDE-BY-SIDE PLACEMENT')
        self.get_logger().info(f'  Direction: {direction.upper()}')
        self.get_logger().info(f'  Offset: {offset}mm')
        self.get_logger().info(f'  Source: ({source_xy[0]:.2f}, {source_xy[1]:.2f})')
        self.get_logger().info(f'  Target: ({target_xy[0]:.2f}, {target_xy[1]:.2f})')
        self.get_logger().info('='*60)
        
        self.publish_status('executing')
        self.executing = True
        
        try:
            # Calculate target position based on direction
            if direction == 'left':
                # Place to the left (negative Y)
                place_x = target_xy[0]
                place_y = target_xy[1] - offset
            elif direction == 'right':
                # Place to the right (positive Y)
                place_x = target_xy[0]
                place_y = target_xy[1] + offset
            elif direction == 'beside':
                # 'beside' is ambiguous - default to right placement
                place_x = target_xy[0]
                place_y = target_xy[1] + offset
            elif direction == 'above':
                # Place above (positive X in robot coords, assuming camera is rotated)
                place_x = target_xy[0] + offset
                place_y = target_xy[1]
            elif direction == 'below':
                # Place below (negative X)
                place_x = target_xy[0] - offset
                place_y = target_xy[1]
            else:
                self.get_logger().error(f'Unknown direction: {direction}')
                return False
            
            self.get_logger().info(f'  Calculated placement: ({place_x:.2f}, {place_y:.2f})')
            
            # Get colors from command
            cmd = self.current_command.lower()
            colors_in_cmd = self.extract_colors_in_order(cmd)
            
            if len(colors_in_cmd) < 2:
                self.get_logger().error('Could not parse colors from command')
                return False
            
            source_color = colors_in_cmd[0]
            target_color = colors_in_cmd[1]
            
            # Find block IDs from detected blocks (SAME LOGIC AS execute_stack!)
            source_candidates = []
            target_candidates = []
            
            for i, block in enumerate(self.block_states):
                block_color = block.get('color')
                block_id = block.get('id')
                
                # Match source color
                if '_' in source_color:
                    if block_id == source_color:
                        source_candidates.append(block_id)
                else:
                    if block_color == source_color:
                        source_candidates.append(block_id)
                
                # Match target color
                if '_' in target_color:
                    if block_id == target_color:
                        target_candidates.append(block_id)
                else:
                    if block_color == target_color:
                        target_candidates.append(block_id)
            
            if not source_candidates or not target_candidates:
                self.get_logger().error(f'Could not find blocks: source={source_color}, target={target_color}')
                return False
            
            source_id = source_candidates[0]
            target_id = target_candidates[0]
            
            # Extract base color
            source_base_color = source_color.split('_')[0] if '_' in source_color else source_color
            target_base_color = target_color.split('_')[0] if '_' in target_color else target_color
            
            self.get_logger().info(f'  🆔 Block IDs: {source_id} → beside {target_id}')
            
            source_block = {'color': source_base_color, 'id': source_id}
            target_block = {'color': target_base_color, 'id': target_id}
            
            if source_block:
                self.get_logger().info(f'  🎨 Moving: {source_block["color"]} {direction} of {target_block.get("color") if target_block else "target"}')
            
            # Calculate Z-heights (both on table)
            travel_z = TABLE_SURFACE_Z + BLOCK_HEIGHT + TRAVEL_HEIGHT_OFFSET
            
            # Execute pick and place sequence
            # 1. Move to source at travel height
            self.get_logger().info(f'\n  ➜ 1. Moving to SOURCE...')
            if not self.move_to(source_xy[0], source_xy[1], travel_z):
                return False
            
            # 2. Lower to pick
            self.get_logger().info(f'  ⬇ 2. Lowering to pick...')
            if not self.move_to(source_xy[0], source_xy[1], TABLE_SURFACE_Z):
                return False
            
            # 3. Pick
            self.get_logger().info(f'  🧲 3. Activating suction...')
            if not self.activate_suction(True):
                return False
            time.sleep(PICK_DELAY)
            
            # 4. Raise with block
            self.get_logger().info(f'  ⬆ 4. Raising with block...')
            if not self.move_to(source_xy[0], source_xy[1], travel_z):
                return False
            
            # 5. Move to placement position
            self.get_logger().info(f'  ➜ 5. Moving to placement position ({direction})...')
            if not self.move_to(place_x, place_y, travel_z):
                return False
            
            # 6. Lower to place
            self.get_logger().info(f'  ⬇ 6. Lowering to place...')
            if not self.move_to(place_x, place_y, TABLE_SURFACE_Z):
                return False
            
            # 7. Release
            self.get_logger().info(f'  🔓 7. Releasing suction...')
            if not self.activate_suction(False):
                return False
            time.sleep(PLACE_DELAY)
            
            # 8. Raise clear
            self.get_logger().info(f'  ⬆ 8. Raising clear...')
            if not self.move_to(place_x, place_y, travel_z):
                return False
            
            # UPDATE MEMORY: Source block is now at new position
            if source_block:
                if source_id in self.block_memory:
                    old_pos = self.block_memory[source_id]
                    self.block_memory[source_id] = {
                        'x': place_x,
                        'y': place_y,
                        'z': TABLE_SURFACE_Z,
                        'color': source_base_color,
                        'rotation': old_pos.get('rotation', 0.0),  # Keep rotation!
                        'last_updated': time.time()
                    }
                    self.get_logger().info(f'📍 Memory updated: {source_id} → ({place_x:.2f}, {place_y:.2f}, Z={TABLE_SURFACE_Z:.2f})')
                    self.get_logger().info(f'  (was at: {old_pos["x"]:.2f}, {old_pos["y"]:.2f}, Z={old_pos["z"]:.2f})')
                    
                    # CRITICAL FIX: Publish updated memory to LLM!
                    self.publish_block_states_from_memory()
            
            # Home
            self.go_home()
            
            self.get_logger().info(f'\n✓ SIDE-BY-SIDE PLACEMENT COMPLETE!')
            if source_block and target_block:
                self.get_logger().info(f'  {source_block.get("color")} is now {direction} of {target_block.get("color")}')
            
            self.publish_status('idle')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Side-by-side placement failed: {e}')
            self.publish_status('error')
            return False
        finally:
            self.executing = False
    
    def execute_place_between(self, source_id: str, ref1_id: str, ref2_id: str) -> bool:
        """
        Place source block at midpoint between ref1 and ref2
        
        Args:
            source_id: Block to move
            ref1_id: First reference block
            ref2_id: Second reference block
        """
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('🎯 PLACE BETWEEN OPERATION')
        self.get_logger().info(f'  Source: {source_id}')
        self.get_logger().info(f'  Between: {ref1_id} and {ref2_id}')
        self.get_logger().info('='*60)
        
        self.publish_status('executing')
        self.executing = True
        
        try:
            # Get positions from memory
            source_mem = self.get_block_from_memory(source_id)
            ref1_mem = self.get_block_from_memory(ref1_id)
            ref2_mem = self.get_block_from_memory(ref2_id)
            
            if not source_mem or not ref1_mem or not ref2_mem:
                self.get_logger().error('One or more blocks not found in memory!')
                return False
            
            source_x, source_y, source_z = source_mem['x'], source_mem['y'], source_mem['z']
            
            # Calculate midpoint
            midpoint = self.calculate_midpoint(ref1_id, ref2_id)
            if not midpoint:
                self.get_logger().error('Failed to calculate midpoint')
                return False
            
            place_x, place_y = midpoint
            
            self.get_logger().info(f'📍 Positions:')
            self.get_logger().info(f'  {source_id}: ({source_x:.2f}, {source_y:.2f})')
            self.get_logger().info(f'  {ref1_id}: ({ref1_mem["x"]:.2f}, {ref1_mem["y"]:.2f})')
            self.get_logger().info(f'  {ref2_id}: ({ref2_mem["x"]:.2f}, {ref2_mem["y"]:.2f})')
            self.get_logger().info(f'  Midpoint: ({place_x:.2f}, {place_y:.2f})')
            
            # Calculate Z-heights
            travel_z = TABLE_SURFACE_Z + BLOCK_HEIGHT + TRAVEL_HEIGHT_OFFSET
            
            # Execute pick and place sequence
            self.get_logger().info(f'\n  ➜ 1. Moving to SOURCE...')
            if not self.move_to(source_x, source_y, travel_z):
                return False
            
            self.get_logger().info(f'  ⬇ 2. Lowering to pick...')
            if not self.move_to(source_x, source_y, source_z):
                return False
            
            self.get_logger().info(f'  🧲 3. Activating suction...')
            if not self.activate_suction(True):
                return False
            time.sleep(PICK_DELAY)
            
            self.get_logger().info(f'  ⬆ 4. Raising with block...')
            if not self.move_to(source_x, source_y, travel_z):
                return False
            
            self.get_logger().info(f'  ➜ 5. Moving to MIDPOINT...')
            if not self.move_to(place_x, place_y, travel_z):
                return False
            
            self.get_logger().info(f'  ⬇ 6. Lowering to place...')
            if not self.move_to(place_x, place_y, TABLE_SURFACE_Z):
                return False
            
            self.get_logger().info(f'  🔓 7. Releasing suction...')
            if not self.activate_suction(False):
                return False
            time.sleep(PLACE_DELAY)
            
            self.get_logger().info(f'  ⬆ 8. Raising clear...')
            if not self.move_to(place_x, place_y, travel_z):
                return False
            
            # Update memory
            if source_id in self.block_memory:
                source_color = source_mem.get('color', source_id.split('_')[0])
                old_pos = self.block_memory[source_id]
                self.block_memory[source_id] = {
                    'x': place_x,
                    'y': place_y,
                    'z': TABLE_SURFACE_Z,
                    'color': source_color,
                    'rotation': old_pos.get('rotation', 0.0),
                    'last_updated': time.time()
                }
                self.get_logger().info(f'📍 Memory updated: {source_id} → ({place_x:.2f}, {place_y:.2f})')
                self.publish_block_states_from_memory()
            
            # Home
            self.go_home()
            
            self.get_logger().info(f'\n✓ PLACE BETWEEN COMPLETE!')
            self.get_logger().info(f'  {source_id} is now between {ref1_id} and {ref2_id}')
            
            self.publish_status('idle')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Place between failed: {e}')
            self.publish_status('error')
            return False
        finally:
            self.executing = False
    
    
    def execute_single(self, waypoint: np.ndarray) -> bool:
        """Execute single block operation"""
        self.get_logger().info(f'\n📦 SINGLE BLOCK: ({waypoint[0]:.2f}, {waypoint[1]:.2f})')
        # Implementation similar to stack but with offset placement
        return True
    
    def execute_multiple(self, waypoints: np.ndarray) -> bool:
        """Execute multiple separate operations"""
        for i, wp in enumerate(waypoints):
            self.get_logger().info(f'\n[{i+1}/{len(waypoints)}] Processing block')
            if not self.execute_single(wp):
                return False
        return True
    
    def find_block_at(self, xy: np.ndarray) -> Optional[Dict]:
        """Find block state at given position"""
        for block in self.block_states:
            # Convert pixel to robot coords (would need converter)
            # For now, use index matching
            pass
        return None
    
    def get_pick_height(self, block: Optional[Dict]) -> float:
        """Calculate pick height based on block state"""
        if block:
            color = block.get('color')
            if color and color in self.stack_heights:
                # If block is on a stack, pick from current height
                return self.stack_heights[color]
        
        # Default: table surface
        return TABLE_SURFACE_Z
    
    def get_place_height(self, block: Optional[Dict]) -> float:
        """Calculate place height based on target block state"""
        if block:
            color = block.get('color')
            if color and color in self.stack_heights:
                # Place on top of current stack
                return self.stack_heights[color] + BLOCK_HEIGHT
        
        # Default: on table
        return TABLE_SURFACE_Z + BLOCK_HEIGHT + 2.0  # +2mm clearance
    
    def move_to(self, x: float, y: float, z: float, r=None) -> bool:
        """Move to position"""
        if r is None:
            r=0
        if self.simulation_mode:
            self.get_logger().info(f'    [Sim] Move to: X={x:7.2f}, Y={y:7.2f}, Z={z:7.2f}')
            time.sleep(MOVE_DELAY)
            return True
        else:
            try:
                qid = self.device.move_to(x, y, z, r)  # Get queue ID
                self.device.wait_for_cmd(qid)  # Wait for completion
                return True
            except Exception as e:
                self.get_logger().error(f'Move failed: {e}')
                return False
    
    def activate_suction(self, enable: bool) -> bool:
        """Activate/deactivate suction"""
        if self.simulation_mode:
            self.get_logger().info(f'  [Sim] Suction {"ON" if enable else "OFF"}')
            return True
        else:
            try:
                self.device.suck(enable)
                return True
            except Exception as e:
                self.get_logger().error(f'Suction control failed: {e}')
                return False
    
    def rotate(self, angle: int) -> bool:
        """Rotate end effector to absolute angle"""
        if self.simulation_mode:
            self.get_logger().info(f'  [Sim] Rotate to {angle}°')
            time.sleep(0.2)
            return True
        else:
            try:
                # Rotate to absolute angle
                x, y, z, r = self.device.pose()
                qid = self.device.move_to(x, y, z, angle)  # Get queue ID
                self.device.wait_for_cmd(qid)  # Wait for completion
                return True
            except Exception as e:
                self.get_logger().error(f'Rotation failed: {e}')
                return False
    
    def rotate_relative(self, angle_diff: float) -> bool:
        """Rotate end effector by relative angle"""
        if self.simulation_mode:
            self.get_logger().info(f'  [Sim] Rotate by {angle_diff:.1f}°')
            time.sleep(0.2)
            return True
        else:
            try:
                # Rotate relative to current position
                x, y, z, r = self.device.pose()
                new_r = (r + angle_diff) % 360
                qid = self.device.move_to(x, y, z, new_r)  # Get queue ID
                self.device.wait_for_cmd(qid)  # Wait for completion
                return True
            except Exception as e:
                self.get_logger().error(f'Relative rotation failed: {e}')
                return False
    
    def go_home(self) -> bool:
        """Return to home position"""
        self.get_logger().info('\n🏠 Returning to home...')
        return self.move_to(HOME_X, HOME_Y, HOME_Z, HOME_R)


def main(args=None):
    rclpy.init(args=args)
    node = BlockExecutorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()