#!/usr/bin/env python3
"""
Voice Command Node with Keyboard Control
- Wake word: "hey robot"
- Press ESC to stop
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import speech_recognition as sr
import threading
import time
import os
import sys

# For keyboard input
import termios
import tty
import select

class VoiceNode(Node):
    
    def __init__(self):
        super().__init__('voice_command_node')
        
        # Publisher
        self.command_pub = self.create_publisher(String, '/user/command', 10)
        
        # Speech recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 3500
        self.recognizer.pause_threshold = 1.0
        self.recognizer.dynamic_energy_threshold = True
        
        # State
        self.running = False
        self.waiting_for_wake = True
        
        # Settings
        self.declare_parameter('language', 'en-US')
        self.language = self.get_parameter('language').value
        self.wake_words = ['hey robot', 'robot']
        
        print('\n' + '='*60)
        print('🎤 VOICE COMMAND NODE')
        print('='*60)
        print(f'Language: {self.language}')
        print(f'Wake words: {self.wake_words}')
        print('='*60)
        print('⚠️  Press ESC or Ctrl+C to stop')
        print('='*60 + '\n')
        
        # Initialize microphone
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                print('🎧 Calibrating...')
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print('✓ Ready!\n')
        except Exception as e:
            print(f'❌ Microphone error: {e}')
            return
        
        # Start
        self.start()
    
    def start(self):
        """Start listening"""
        self.running = True
        
        # Start voice thread
        threading.Thread(target=self._voice_loop, daemon=True).start()
        
        # Start keyboard monitor thread
        threading.Thread(target=self._keyboard_monitor, daemon=True).start()
        
        print('🎤 Listening for "hey robot"...\n')
    
    def stop(self):
        """Stop listening"""
        self.running = False
        print('\n✓ Stopped\n')
    
    def _keyboard_monitor(self):
        """Monitor keyboard for ESC key"""
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            tty.setcbreak(sys.stdin.fileno())
            
            while self.running:
                # Check if key pressed
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    
                    # ESC key (ASCII 27) or 'q'
                    if ord(key) == 27 or key == 'q':
                        print('\n🛑 ESC pressed - stopping...')
                        self.running = False
                        break
        
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def _voice_loop(self):
        """Voice listening loop"""
        
        def callback(recognizer, audio):
            """Process audio"""
            if not self.running:
                return
            
            try:
                text = recognizer.recognize_google(audio, language=self.language)
                text_lower = text.lower()
                
                if self.waiting_for_wake:
                    # Check wake word
                    if any(w in text_lower for w in self.wake_words):
                        print('\n✓ Wake word detected!')
                        print('🎤 Say your command...\n')
                        self.waiting_for_wake = False
                    
                    elif 'stop listening' in text_lower:
                        print('\n🛑 Voice stop command')
                        self.running = False
                
                else:
                    # Process command
                    print(f'📤 Command: "{text}"')
                    
                    if 'stop listening' in text_lower:
                        print('\n🛑 Voice stop command')
                        self.running = False
                        return
                    
                    # Publish
                    msg = String()
                    msg.data = text
                    self.command_pub.publish(msg)
                    
                    print('✓ Sent to LLM\n')
                    print('🎤 Listening for "hey robot"...\n')
                    
                    # Reset
                    self.waiting_for_wake = True
            
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f'❌ Error: {e}')
        
        # Start background listening
        stop_listening = self.recognizer.listen_in_background(
            self.microphone, callback, phrase_time_limit=10
        )
        
        # Wait while running
        try:
            while self.running:
                time.sleep(0.1)
        finally:
            stop_listening(wait_for_stop=False)


def main():
    import signal
    
    # Redirect stderr to suppress ALSA
    try:
        stderr_fd = sys.stderr.fileno()
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, stderr_fd)
        os.close(null_fd)
    except:
        pass
    
    rclpy.init()
    node = None
    
    def shutdown(sig=None, frame=None):
        """Shutdown handler"""
        print('\n🛑 Shutting down...')
        if node:
            node.stop()
        
        # Force exit after 1 second
        time.sleep(1)
        os._exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    try:
        node = VoiceNode()
        
        # Keep running until stopped
        while node.running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        shutdown()


if __name__ == '__main__':
    main()