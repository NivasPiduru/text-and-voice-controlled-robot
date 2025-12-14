# text-and-voice-controlled-robot
AI-powered robotic manipulation system with natural language voice commands using ROS2, OpenCV, and Google Gemini LLM

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AI-powered robotic manipulation system that responds to natural language voice commands, integrating computer vision, LLM-based parsing, and real-time robot control.

## 🎥 Demo

[**Watch Video Demo**](https://youtu.be/byuuGB1L9Vg) | [**website**](https://nivaspiduru.github.io/portfolio/portfolio-1-voice-robot/)

## 📋 Overview

This system enables non-technical users to control complex robotic tasks through simple voice commands like "stack the red block on the blue one." The robot sees its environment, understands natural language, and executes precise manipulations autonomously.

### Key Features

- **Wake Word Detection**: Hands-free activation using "hey robot" trigger phrase
- **Vision-Based Perception**: Real-time object detection with OpenCV for 4DOF pose estimation
- **Natural Language Understanding**: Google Gemini 2.5 Flash LLM parses complex multi-step commands
- **Intelligent Control**: Dynamic Z-height adjustment, rotation matching, and collision avoidance
- **Modular Architecture**: Five independent ROS2 nodes for scalable system design

## 🏗️ System Architecture
```
┌─────────────────┐
│  Voice Command  │ ──► Speech recognition + wake word detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Parser    │ ──► Google Gemini 2.5 Flash (natural language → robot actions)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vision Detector │ ──► OpenCV object detection + 4DOF pose estimation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Converter    │ ──► Homography-based pixel-to-robot coordinate transformation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Executor     │ ──► Motion planning + Dobot Magician control
└─────────────────┘
```

## 📦 Installation

### Prerequisites

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10+
- Dobot Magician Lite with suction

### Dependencies
```bash
pip install google-generativeai speechrecognition pyaudio opencv-python numpy
```

### ROS2 Packages
```bash
sudo apt install ros-humble-cv-bridge ros-humble-image-transport
```

## 🚀 Usage

### 1. Launch all nodes
```bash
# Terminal 1: Vision Detector
ros2 run voice_robot vision_detector_node

# Terminal 2: LLM Parser
ros2 run voice_robot llm_command_parser

# Terminal 3: Coordinate Converter
ros2 run voice_robot converter_node

# Terminal 4: Executor
ros2 run voice_robot executor_node

# Terminal 5: Voice Command
ros2 run voice_robot voice_command_node_gui
```

### 2. Issue voice commands

Say **"hey robot"** then your command:
- "Stack the red block on top of the blue block"
- "Move all blocks to the left side"
- "Pick up the green block and place it between the red and blue blocks"

### 3. Stop

Press **ESC** or say **"stop listening"**

## 🎯 Example Commands

| Command | Result |
|---------|--------|
| "Stack the red block on the blue one" | Picks red, places on blue with height adjustment |
| "Clear the workspace" | Moves all detected blocks to designated area |
| "Rotate the blue block 90 degrees" | Picks and rotates block before placing |
| "Make a tower with red, blue, green" | Stacks blocks in specified order |


## 🔬 Performance

- **Object Detection Success**: 95% under controlled lighting
- **Command Understanding**: 100% for clear speech
- **Positioning Accuracy**: ±3mm
- **Total Execution Time**: 8-12 seconds per pick-and-place operation
- **Stack Success Rate**: 98% for 2-level stacks, 92% for 3-level


## 🎓 Academic Context

**Course**: RAS 545 - Robotics and Autonomous Systems (Final Project)  
**Institution**: Arizona State University  

## 👨‍💻 Author

**Nivas Piduru**  
MS Robotics and Autonomous Systems, Arizona State University  
📧 nivaspiduru@gmail.com  
🔗 [Portfolio](https://nivaspiduru.github.io) | [LinkedIn](https://linkedin.com/in/nivas-piduru)

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments
- RAS 545 course staff
- Google Gemini API for LLM capabilities
- ROS2 community for excellent documentation
