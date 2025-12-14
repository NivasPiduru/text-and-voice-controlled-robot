#!/usr/bin/env python3
"""
ROS 2 Node: LLM Command Parser
Parses natural language commands into structured robot commands using Google Gemini

Subscribes:
    /user/command (std_msgs/String): Natural language commands from user
    /blocks/block_states (std_msgs/String): Current block states from vision
    /blocks/executor_status (std_msgs/String): Executor status for memory state

Publishes:
    /blocks/command (std_msgs/String): Structured command for executor
    /llm/status (std_msgs/String): Parser status
    /llm/debug (std_msgs/String): Debug info (LLM reasoning)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import os
import numpy as np
import google.generativeai as genai
from typing import Dict, List, Optional

# ========================= CONFIGURATION =========================

# Get API key from environment variable
# Set it with: export GEMINI_API_KEY="your-key-here"
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')

# Model configuration
MODEL_NAME = "gemini-2.5-flash"  # Latest and best model!
TEMPERATURE = 0.1  # Low temperature for consistent, deterministic outputs
MAX_TOKENS = 8192

# =================================================================


class LLMCommandParser(Node):
    
    def __init__(self):
        super().__init__('llm_command_parser')
        
        # Publishers
        self.command_pub = self.create_publisher(String, '/blocks/command', 10)
        self.status_pub = self.create_publisher(String, '/llm/status', 10)
        self.debug_pub = self.create_publisher(String, '/llm/debug', 10)
        
        # Subscribers
        self.user_command_sub = self.create_subscription(
            String, '/user/command',
            self.user_command_callback, 10
        )
        self.block_states_sub = self.create_subscription(
            String, '/blocks/block_states',
            self.block_states_callback, 10
        )
        self.executor_status_sub = self.create_subscription(
            String, '/blocks/executor_status',
            self.executor_status_callback, 10
        )
        
        # State
        self.current_block_states = []  # List of block state dicts
        self.executor_status = "idle"
        self.processing = False
        self.pending_command = None  # Command waiting for block states
        
        # Initialize Gemini
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Disable safety filters - robot commands are harmless!
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            self.model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                generation_config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_TOKENS,
                },
                safety_settings=safety_settings
            )
            self.get_logger().info(f'✓ Gemini API initialized ({MODEL_NAME})')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize Gemini: {e}')
            self.get_logger().error('Set GEMINI_API_KEY environment variable!')
            self.model = None
        
        self.get_logger().info('='*60)
        self.get_logger().info('LLM Command Parser initialized')
        self.get_logger().info(f'Model: {MODEL_NAME}')
        self.get_logger().info('Waiting for block states and user commands...')
        self.get_logger().info('='*60)
        self.publish_status('idle')
    
    def publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'LLM status: {status}')
    
    def block_states_callback(self, msg):
        """Receive current block states from vision"""
        try:
            self.current_block_states = json.loads(msg.data)
            self.get_logger().debug(f'Updated block states: {len(self.current_block_states)} blocks')
            
            # If we have a pending command, process it now!
            if self.pending_command and not self.processing:
                self.get_logger().info(f'✓ Block states received! Processing pending command...')
                cmd_to_process = self.pending_command
                self.pending_command = None
                
                # Create a fake message to reuse the callback logic
                fake_msg = String()
                fake_msg.data = cmd_to_process
                self.user_command_callback(fake_msg)
                
        except Exception as e:
            self.get_logger().error(f'Failed to parse block states: {e}')
    
    def executor_status_callback(self, msg):
        """Track executor status"""
        self.executor_status = msg.data
    
    def user_command_callback(self, msg):
        """
        Receive natural language command from user
        Parse it and send structured command to executor
        """
        if self.processing:
            self.get_logger().warn('Already processing a command, please wait...')
            return
        
        if not self.model:
            self.get_logger().error('Gemini not initialized! Cannot parse command.')
            return
        
        user_command = msg.data.strip()
        
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info(f'📥 USER COMMAND: "{user_command}"')
        self.get_logger().info('='*60)
        
        # Check if we have block states
        if not self.current_block_states:
            self.get_logger().warn('⚠️  No block states available yet!')
            self.get_logger().info('🔄 Auto-triggering vision detection...')
            
            # Trigger vision WITHOUT sending to executor (use special prefix)
            trigger_msg = String()
            trigger_msg.data = f"rescan"  # This triggers vision but executor ignores during rescan
            self.command_pub.publish(trigger_msg)
            
            self.get_logger().info('👁️  Waiting for vision to detect blocks...')
            self.get_logger().info('   (Will auto-parse once block states arrive)')
            
            # Store command for later processing
            self.pending_command = user_command
            self.processing = False
            return
        
        self.processing = True
        self.publish_status('parsing')
        
        try:
            # Generate prompt with current scene context
            prompt = self.create_prompt(user_command, self.current_block_states)
            
            # Call Gemini API
            self.get_logger().info('🤖 Calling Gemini API...')
            response = self.model.generate_content(prompt)
            
            # Parse response
            structured_command = self.parse_gemini_response(response.text)
            
            if structured_command:
                # Publish to executor
                cmd_msg = String()
                cmd_msg.data = structured_command
                self.command_pub.publish(cmd_msg)
                
                self.get_logger().info('✓ Structured command sent to executor:')
                self.get_logger().info(f'  "{structured_command}"')
                
                # Publish debug info
                debug_msg = String()
                debug_msg.data = f"User: {user_command}\nLLM: {response.text}\nCommand: {structured_command}"
                self.debug_pub.publish(debug_msg)
                
                self.publish_status('success')
            else:
                self.get_logger().error('❌ Failed to parse LLM response')
                self.publish_status('error')
        
        except Exception as e:
            self.get_logger().error(f'LLM parsing failed: {e}')
            self.publish_status('error')
        
        finally:
            self.processing = False
    
    def create_prompt(self, user_command: str, block_states: List[Dict]) -> str:
        """
        Create prompt for Gemini with scene context
        """
        # Build scene description with SPATIAL AWARENESS
        scene_desc = "=" * 60 + "\n"
        scene_desc += "CURRENT WORKSPACE STATE:\n"
        scene_desc += "=" * 60 + "\n\n"
        
        # Group blocks by state
        on_table = []
        stacked = []
        
        # Calculate distances and spatial relationships
        block_positions = {}
        for block in block_states:
            block_id = block.get('id', 'unknown')
            x = block.get('x', 0)
            y = block.get('y', 0)
            block_positions[block_id] = (x, y)
        
        # Find nearest and farthest from camera (Y coordinate)
        if block_positions:
            nearest_id = min(block_positions.items(), key=lambda y: y[1][1])[0]
            farthest_id = max(block_positions.items(), key=lambda y: y[1][1])[0]
        else:
            nearest_id = None
            farthest_id = None
        
        # Calculate distance between any two blocks
        def calc_distance(id1, id2):
            if id1 not in block_positions or id2 not in block_positions:
                return float('inf')
            x1, y1 = block_positions[id1]
            x2, y2 = block_positions[id2]
            return np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Build block descriptions with spatial context
        for block in block_states:
            block_id = block.get('id', 'unknown')
            color = block.get('color', 'unknown')
            state = block.get('state', 'on_table')
            x = block.get('x', 0)
            y = block.get('y', 0)
            has_block_on_top = block.get('has_block_on_top', False)
            
            # Add spatial tags
            spatial_tags = []
            if block_id == nearest_id:
                spatial_tags.append("⭐ NEAREST TO CAMERA")
            if block_id == farthest_id:
                spatial_tags.append("⭐ FARTHEST FROM CAMERA")
            
            # Find closest neighbor
            min_dist = float('inf')
            closest_neighbor = None
            for other_id in block_positions:
                if other_id != block_id:
                    dist = calc_distance(block_id, other_id)
                    if dist < min_dist:
                        min_dist = dist
                        closest_neighbor = other_id
            
            if state == 'on_table':
                desc = f"  • {block_id} ({color}) at position ({x}, {y})"
                if spatial_tags:
                    desc += f" {' '.join(spatial_tags)}"
                if has_block_on_top:
                    desc += " [SUPPORTING A STACK]"
                if closest_neighbor:
                    desc += f"\n    → Closest to: {closest_neighbor} ({min_dist:.1f}mm away)"
                on_table.append(desc)
            else:
                stacked_on = block.get('stacked_on', 'unknown')
                desc = f"  • {block_id} ({color}) is stacked on {stacked_on}"
                if spatial_tags:
                    desc += f" {' '.join(spatial_tags)}"
                stacked.append(desc)
        
        if on_table:
            scene_desc += "📦 BLOCKS ON TABLE:\n" + "\n".join(on_table) + "\n\n"
        
        if stacked:
            scene_desc += "🏗️  STACKED BLOCKS:\n" + "\n".join(stacked) + "\n\n"
        
        # Add distance matrix for reference
        if len(block_positions) > 1:
            scene_desc += "📏 DISTANCES BETWEEN BLOCKS:\n"
            block_ids = list(block_positions.keys())
            for i, id1 in enumerate(block_ids):
                for id2 in block_ids[i+1:]:
                    dist = calc_distance(id1, id2)
                    scene_desc += f"  • {id1} ↔ {id2}: {dist:.1f}mm\n"
            scene_desc += "\n"
        
        scene_desc += "=" * 60 + "\n\n"
        
        # Create full prompt with enhanced rules
        prompt = f"""You are a robot command parser with FULL SPATIAL AWARENESS. Convert natural language commands into structured robot commands.

{scene_desc}

USER COMMAND: "{user_command}"

TASK: Parse the command and return a simple, structured command that the robot can execute.

IMPORTANT CAPABILITIES:
✅ You can identify blocks WITHOUT colors mentioned!
✅ You understand spatial relationships: nearest, farthest, between, closest to
✅ You can calculate distances and find blocks by position

COMMAND FORMATS:
1. Stacking: "place [source_id] on [target_id]"
2. Side-by-side: "place [source_id] [direction] of [target_id]"
   Directions: left, right, beside
3. In-between placement: "place [source_id] between [ref1_id] and [ref2_id]"

SPATIAL RULES:
1. "nearest block" or "closest block" → Use block marked "⭐ NEAREST TO CAMERA"
2. "farthest block" → Use block marked "⭐ FARTHEST FROM CAMERA"
3. "block nearest to X" → Use the block with smallest distance to X
4. "block farthest from X" → Use the block with largest distance to X
5. "block between X and Y" → Find block closest to midpoint of X and Y (calculate from positions)
6. "next to X" or "beside X" → Use block with smallest distance to X
7. "the stack" → Any block with "[SUPPORTING A STACK]" marker (this is the BASE of the stack)
8. **"next to the stack" or "beside the stack"** → Place beside the block with "[SUPPORTING A STACK]" marker (the bottom block holding the stack)
9. If color mentioned (e.g., "red"), use that color's block (e.g., red_0)
10. If NO color mentioned, use SPATIAL criteria to identify blocks
11. **STACK-PROTECTION RULE (VERY IMPORTANT):** By default you MUST NOT pick up or move any block that is part of an existing stack (either a base block marked "[SUPPORTING A STACK]" or a block whose state is stacked on another block) when the user uses generic phrases like "the blue block", "the green block", etc.
12. When a color has multiple blocks (e.g., green_0 and green_1), and some of them are in stacks, the DEFAULT choice for "the green block" is a green block that is **free on the table** (not supporting a stack and not stacked on anything). Only if **no** free block of that color exists may you consider a stacked one, and even then prefer the one that causes minimal disturbance to existing stacks.
13. The ONLY time you are allowed to use a block that is part of a stack as the source block is when the user explicitly refers to the stack, using phrases like:
    - "the blue block in the stack"
    - "the block on top of the stack"
    - "the block at the bottom of the stack"
    - "the stacked blue block"
    - "the blue block from the stack"
    In these cases, you MUST pick from the stack exactly as requested.
14. If the user says "place on top of the green block" and there are multiple green blocks, you MUST:
    - First prefer a green block that is **not** part of any stack (free on the table).
    - Only use a green block that is part of a stack if the user clearly says "green block in the stack" or "green block at the bottom of the stack".

CRITICAL: When command says "next to THE STACK", find the block marked "[SUPPORTING A STACK]" - this is the BASE block that has another block on top of it!

CALCULATION EXAMPLES:
- "nearest block" → Look for "⭐ NEAREST TO CAMERA" marker
- "farthest block" → Look for "⭐ FARTHEST FROM CAMERA" marker
- "block between red_0 and blue_0" → Calculate midpoint: ((red.x+blue.x)/2, (red.y+blue.y)/2), find closest block
- "block closest to yellow_0" → Check distances, pick block with minimum distance to yellow_0

COMMAND EXAMPLES:
✅ WITH COLORS:
- "place red on blue" → "place red_0 on blue_0"
- "place yellow beside green" → "place yellow_0 beside green_0"

✅ WITHOUT COLORS (spatial only):
- "pick nearest block and place on farthest block" → "place red_0 on yellow_0" (if red is nearest, yellow is farthest)
- "place nearest block beside farthest block" → "place blue_0 beside green_0"
- "stack closest block on the block next to red" → "place green_0 on blue_0" (green closest, blue nearest to red)

✅ MIXED (colors + spatial):
- "place nearest red on farthest blue" → "place red_0 on blue_2" (if multiple reds/blues)
- "place green on nearest block" → "place green_0 on red_0" (red is nearest)

✅ COMPLEX SPATIAL:
- "place blue between red and yellow" → "place blue_0 between red_0 and yellow_0"
- "pick block closest to the stack and place on farthest block" → "place green_0 on yellow_0"
- **"place blue next to the stack"** → "place blue_0 beside green_0" (if green_0 has [SUPPORTING A STACK] marker)
- **"pick blue and place beside the stack"** → "place blue_0 beside red_0" (if red_0 is marked [SUPPORTING A STACK])

✅ STACK-PROTECTION BEHAVIOR (YOUR SPECIFIC REQUIREMENT):
- If blue_0 is stacked on green_0, and green_1 is a free green block on the table, then:
  - "pick up the red block and place on top of the green block"
  → MUST be interpreted as: "place red_0 on green_1" (use a green block **not** involved in any stack).
- If the user explicitly says:
  - "pick up the blue block in the stack and place on top of the yellow block"
  → You MUST use the blue block that is part of the stack as the source (e.g., "place blue_0 on yellow_0").

CRITICAL: Always use the ACTUAL block IDs from the scene above (e.g., red_0, blue_0, etc.). Use spatial markers and distance data to identify blocks when no color is mentioned. By DEFAULT you must preserve all existing stacks and never disturb them unless the user clearly and explicitly asks for a block "in the stack" or similar wording.

Return ONLY the structured command, nothing else. No explanation, no JSON, just the command string.
"""
        
        return prompt
    
    def parse_gemini_response(self, response_text: str) -> Optional[str]:
        """
        Parse Gemini's response to extract structured command
        """
        # Clean up response
        command = response_text.strip()
        
        # Remove quotes if present
        command = command.strip('"\'')
        
        # Remove any explanation text (take only first line)
        if '\n' in command:
            command = command.split('\n')[0].strip()
        
        # Validate it looks like a command
        valid_patterns = ['place ', 'stack ', 'move ', 'between ']
        if any(pattern in command.lower() for pattern in valid_patterns):
            return command.lower()
        
        self.get_logger().warn(f'Unexpected LLM response: {response_text}')
        return None


def main(args=None):
    rclpy.init(args=args)
    node = LLMCommandParser()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    