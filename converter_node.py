#!/usr/bin/env python3
"""
ROS 2 Node: Coordinate Converter for Blocks (FULL-FEATURED)
Converts pixel coordinates to Dobot coordinates with comprehensive logging

Provides Service:
    /blocks/convert_coordinates (maze_solver_msgs/ConvertCoordinates)

Publishes:
    /blocks/conversion_status (std_msgs/String): Current status
    /blocks/waypoints_dobot (std_msgs/Float32MultiArray): Converted waypoints

Subscribes:
    /blocks/waypoints_pixel (std_msgs/Float32MultiArray): Input waypoints
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

# ========================= HOMOGRAPHY FUNCTIONS =========================

def homography_from_4pt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute H (3x3) s.t. [X,Y,1]^T ~ H @ [x,y,1]^T using DLT (SVD)."""
    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([-x, -y, -1,  0,  0,  0, x*X, y*X, X])
        A.append([ 0,  0,  0, -x, -y, -1, x*Y, y*Y, Y])
    A = np.asarray(A, dtype=float)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply homography transformation to points"""
    n = pts.shape[0]
    homo = np.hstack([pts, np.ones((n, 1))])      # Nx3
    mapped = (H @ homo.T).T                        # Nx3
    w = mapped[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return mapped[:, :2] / w


def read_pixels_csv(csv_path: Path) -> tuple:
    """
    Read pixel coordinates from CSV file
    Returns: (coordinates_array, colors_list, rotations_list)
    """
    xs, ys, colors, rotations = [], [], [], []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        field_map = {k.lower().strip(): k for k in (reader.fieldnames or [])}
        
        if "x" not in field_map or "y" not in field_map:
            raise ValueError("Input CSV must have headers 'x' and 'y'.")
        
        kx, ky = field_map["x"], field_map["y"]
        kcolor = field_map.get("color", None)  # Color is optional
        krotation = field_map.get("rotation", None)  # Rotation is optional
        
        for row in reader:
            xs.append(float(row[kx]))
            ys.append(float(row[ky]))
            
            if kcolor and kcolor in row:
                colors.append(row[kcolor])
            else:
                colors.append(None)
            
            if krotation and krotation in row:
                try:
                    rotations.append(float(row[krotation]))
                except:
                    rotations.append(0.0)
            else:
                rotations.append(0.0)
    
    coords = np.column_stack([xs, ys])
    return coords, colors, rotations


def write_dobot_csv(csv_path: Path, dobot_pts: np.ndarray, colors: list = None, rotations: list = None) -> None:
    """Write Dobot coordinates to CSV file with optional colors and rotations"""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write header with color and rotation if available
        if colors and any(colors):
            if rotations and any(rotations):
                writer.writerow(["x", "y", "color", "rotation"])
                for i, (X, Y) in enumerate(dobot_pts):
                    color = colors[i] if i < len(colors) else None
                    rotation = rotations[i] if i < len(rotations) else 0.0
                    writer.writerow([f"{X:.6f}", f"{Y:.6f}", color if color else "", f"{rotation:.2f}"])
            else:
                writer.writerow(["x", "y", "color"])
                for i, (X, Y) in enumerate(dobot_pts):
                    color = colors[i] if i < len(colors) else None
                    writer.writerow([f"{X:.6f}", f"{Y:.6f}", color if color else ""])
        else:
            writer.writerow(["x", "y"])
            for X, Y in dobot_pts:
                writer.writerow([f"{X:.6f}", f"{Y:.6f}"])


# ========================= DOBOT CALIBRATION =========================
# CRITICAL: These corners define the transformation from camera to robot space
# Calibrate these values for your specific setup!
DOBOT_CORNERS = [
    (209, -115),   # Top-left corner of camera view → robot coordinates
    (209,  164),   # Top-right corner
    (415,  164),   # Bottom-right corner
    (415, -115),   # Bottom-left corner
]

# =================================================================


class ConverterNode(Node):
    """ROS 2 Node for converting pixel coordinates to Dobot coordinates"""
    
    def __init__(self):
        super().__init__('converter_node')
        
        # Declare parameters
        self.declare_parameter('input_csv', 'block_captures/waypoints_pixel.csv')
        self.declare_parameter('output_csv', 'block_captures/waypoints_dobot.csv')
        self.declare_parameter('auto_convert_on_startup', False)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/blocks/conversion_status', 10)
        self.waypoints_pub = self.create_publisher(Float32MultiArray, '/blocks/waypoints_dobot', 10)
        
        # Subscribers
        self.pixel_sub = self.create_subscription(
            Float32MultiArray, '/blocks/waypoints_pixel', self.pixel_waypoints_callback, 10
        )
        
        # State
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        
        self.get_logger().info('='*60)
        self.get_logger().info('Block Coordinate Converter Node initialized')
        self.get_logger().info(f'Input CSV: {self.get_parameter("input_csv").value}')
        self.get_logger().info(f'Output CSV: {self.get_parameter("output_csv").value}')
        self.get_logger().info(f'Image size: {self.image_width}x{self.image_height}')
        self.get_logger().info('='*60)
        self.publish_status('idle')
        
        # Auto-convert on startup if parameter is set
        if self.get_parameter('auto_convert_on_startup').value:
            self.convert_from_csv()
    
    def publish_status(self, status: str):
        """Publish current status"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'Conversion status: {status}')
    
    def pixel_waypoints_callback(self, msg):
        """
        Automatically convert when pixel waypoints are received via topic
        This also saves to CSV file
        """
        num_waypoints = len(msg.data) // 2
        
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info(f'📥 RECEIVED {num_waypoints} pixel waypoints via topic')
        self.get_logger().info('='*60)
        
        self.publish_status('converting')
        
        # Convert Float32MultiArray to numpy array
        pixel_array = np.array(msg.data).reshape(-1, 2)
        
        # Try to get colors and rotations from pixel CSV if it exists
        input_csv = Path(self.get_parameter('input_csv').value)
        colors = None
        rotations = None
        if input_csv.exists():
            try:
                _, colors, rotations = read_pixels_csv(input_csv)
                self.get_logger().info(f'✓ Read {len(colors)} color labels from CSV')
                if rotations:
                    self.get_logger().info(f'✓ Read {len(rotations)} rotation angles from CSV')
            except:
                pass
        
        self.get_logger().info('🔄 Converting pixel coordinates to robot coordinates...')
        
        # Convert using homography
        dobot_array = self.convert_pixels_to_dobot(
            pixel_array,
            self.image_width,
            self.image_height
        )
        
        if dobot_array is not None:
            # Save to CSV file with colors and rotations
            output_csv = Path(self.get_parameter('output_csv').value)
            try:
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                write_dobot_csv(output_csv, dobot_array, colors, rotations)
                self.get_logger().info(f'✓ Saved Dobot waypoints to: {output_csv}')
                if colors:
                    self.get_logger().info(f'  With color labels: {colors}')
                if rotations:
                    self.get_logger().info(f'  With rotations: {rotations}')
            except Exception as e:
                self.get_logger().error(f'Failed to save CSV: {e}')
            
            # Log conversion summary
            self.log_conversion_summary(pixel_array, dobot_array, colors, rotations)
            
            # Publish converted waypoints to topic
            self.publish_dobot_waypoints(dobot_array)
            
            self.publish_status('done')
            
            self.get_logger().info('='*60)
            self.get_logger().info('✓ CONVERSION COMPLETE')
            self.get_logger().info(f'  📊 {num_waypoints} waypoints converted')
            self.get_logger().info(f'  📁 Saved to: {output_csv}')
            self.get_logger().info(f'  🔄 Published to /blocks/waypoints_dobot')
            self.get_logger().info('='*60 + '\n')
        else:
            self.publish_status('error')
    
    def convert_from_csv(self) -> bool:
        """
        Read pixel coordinates from CSV, convert, save to CSV, and publish
        """
        input_csv = Path(self.get_parameter('input_csv').value)
        output_csv = Path(self.get_parameter('output_csv').value)
        
        if not input_csv.exists():
            self.get_logger().error(f'Input CSV not found: {input_csv}')
            self.publish_status('error')
            return False
        
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info(f'📂 Reading pixel waypoints from: {input_csv}')
        self.get_logger().info('='*60)
        self.publish_status('converting')
        
        try:
            # Read pixel coordinates, colors, and rotations from CSV
            pixel_array, colors, rotations = read_pixels_csv(input_csv)
            self.get_logger().info(f'✓ Loaded {len(pixel_array)} pixel waypoints from CSV')
            if colors:
                self.get_logger().info(f'✓ With color labels: {colors}')
            if rotations:
                self.get_logger().info(f'✓ With rotations: {rotations}')
            
            # Convert using homography
            self.get_logger().info('🔄 Converting pixel coordinates to robot coordinates...')
            dobot_array = self.convert_pixels_to_dobot(
                pixel_array,
                self.image_width,
                self.image_height
            )
            
            if dobot_array is not None:
                # Save to CSV file with colors and rotations
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                write_dobot_csv(output_csv, dobot_array, colors, rotations)
                self.get_logger().info(f'✓ Saved Dobot waypoints to: {output_csv}')
                
                # Log conversion summary
                self.log_conversion_summary(pixel_array, dobot_array, colors, rotations)
                
                # Publish converted waypoints to topic
                self.publish_dobot_waypoints(dobot_array)
                
                self.publish_status('done')
                
                self.get_logger().info('='*60)
                self.get_logger().info('✓ CONVERSION COMPLETE')
                self.get_logger().info(f'  📊 {len(pixel_array)} waypoints converted')
                self.get_logger().info(f'  📁 Saved to: {output_csv}')
                self.get_logger().info(f'  🔄 Published to /blocks/waypoints_dobot')
                self.get_logger().info('='*60 + '\n')
                
                return True
            else:
                self.publish_status('error')
                return False
        
        except Exception as e:
            self.get_logger().error(f'CSV conversion failed: {e}')
            self.publish_status('error')
            return False
    
    def convert_pixels_to_dobot(self, pixel_pts: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        """
        Convert pixel coordinates to Dobot coordinates using homography
        
        Args:
            pixel_pts: Nx2 array of pixel coordinates
            img_w: Image width
            img_h: Image height
        
        Returns:
            Nx2 array of Dobot coordinates or None on error
        """
        try:
            # Define pixel-space corners (TL, TR, BR, BL)
            px_corners = np.array([
                [0.0,                 0.0],
                [float(img_w - 1),     0.0],
                [float(img_w - 1),     float(img_h - 1)],
                [0.0,                 float(img_h - 1)],
            ], dtype=float)
            
            # Dobot corners (calibrated physical workspace)
            dobot_corners = np.asarray(DOBOT_CORNERS, dtype=float)
            
            self.get_logger().info('  Computing homography matrix...')
            # Compute homography matrix
            H = homography_from_4pt(px_corners, dobot_corners)
            
            self.get_logger().info('  Applying transformation...')
            # Apply transformation to all waypoints
            dobot_pts = apply_homography(H, pixel_pts)
            
            self.get_logger().info(f'✓ Converted {len(pixel_pts)} points successfully')
            return dobot_pts
        
        except Exception as e:
            self.get_logger().error(f'Conversion failed: {e}')
            return None
    
    def log_conversion_summary(self, pixel_pts: np.ndarray, dobot_pts: np.ndarray, colors: list = None, rotations: list = None):
        """Log summary of conversion with sample points"""
        
        self.get_logger().info('-'*60)
        self.get_logger().info('CONVERSION SUMMARY:')
        
        # Show first 5 points as examples
        num_to_show = min(5, len(pixel_pts))
        for i in range(num_to_show):
            px, py = pixel_pts[i]
            dx, dy = dobot_pts[i]
            color_label = f" ({colors[i]})" if colors and i < len(colors) and colors[i] else ""
            rotation_label = f" @ {rotations[i]:.1f}°" if rotations and i < len(rotations) else ""
            self.get_logger().info(
                f'  Point {i+1}{color_label}{rotation_label}: ({px:6.1f}, {py:6.1f}) px → ({dx:7.2f}, {dy:7.2f}) mm'
            )
        
        if len(pixel_pts) > 5:
            self.get_logger().info(f'  ... and {len(pixel_pts)-5} more points')
        
        # Show coordinate ranges
        self.get_logger().info('-'*60)
        self.get_logger().info('COORDINATE RANGES:')
        self.get_logger().info(f'  Pixel X: {pixel_pts[:, 0].min():.1f} to {pixel_pts[:, 0].max():.1f}')
        self.get_logger().info(f'  Pixel Y: {pixel_pts[:, 1].min():.1f} to {pixel_pts[:, 1].max():.1f}')
        self.get_logger().info(f'  Robot X: {dobot_pts[:, 0].min():.2f} to {dobot_pts[:, 0].max():.2f} mm')
        self.get_logger().info(f'  Robot Y: {dobot_pts[:, 1].min():.2f} to {dobot_pts[:, 1].max():.2f} mm')
        if rotations:
            rotation_vals = [r for r in rotations if r is not None]
            if rotation_vals:
                self.get_logger().info(f'  Rotations: {min(rotation_vals):.1f}° to {max(rotation_vals):.1f}°')
        
        # Safety warnings
        self.get_logger().info('-'*60)
        self.get_logger().info('SAFETY CHECK:')
        
        safe_x_min, safe_x_max = 100, 500
        safe_y_min, safe_y_max = -250, 250
        
        out_of_bounds = False
        
        if dobot_pts[:, 0].min() < safe_x_min or dobot_pts[:, 0].max() > safe_x_max:
            self.get_logger().warn(f'⚠️  X coordinates may be outside safe range ({safe_x_min}-{safe_x_max}mm)')
            out_of_bounds = True
        
        if dobot_pts[:, 1].min() < safe_y_min or dobot_pts[:, 1].max() > safe_y_max:
            self.get_logger().warn(f'⚠️  Y coordinates may be outside safe range ({safe_y_min} to {safe_y_max}mm)')
            out_of_bounds = True
        
        if not out_of_bounds:
            self.get_logger().info('✓ All coordinates within safe workspace bounds')
        
        self.get_logger().info('-'*60)
    
    def publish_dobot_waypoints(self, dobot_pts: np.ndarray):
        """Publish Dobot waypoints as Float32MultiArray"""
        msg = Float32MultiArray()
        
        flattened = []
        for X, Y in dobot_pts:
            flattened.append(float(X))
            flattened.append(float(Y))
        
        msg.data = flattened
        self.waypoints_pub.publish(msg)
        self.get_logger().info(f'✓ Published {len(dobot_pts)} Dobot waypoints to /blocks/waypoints_dobot')


def main(args=None):
    rclpy.init(args=args)
    node = ConverterNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()