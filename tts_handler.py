#!/usr/bin/env python3

import os
import sys
import time
import subprocess
import platform
import getpass
from vibration_handler import VibrationHandler

# Initialize vibration handler
vibration = None
try:
    vibration = VibrationHandler()
except Exception as e:
    print(f"Warning: Vibration disabled: {e}")

def get_current_user():
    """Get the current user, supporting sudo and cross-platform"""
    try:
        if platform.system() == "Windows":
            return getpass.getuser(), None  # UID not used on Windows
        else:
            import pwd
            real_uid = int(os.environ.get('SUDO_UID', os.getuid()))
            return pwd.getpwuid(real_uid).pw_name, real_uid
    except Exception:
        return os.getenv('USER', os.getenv('USERNAME', '')), None

def speak(message, voice=None, speed=None, pitch=None):
    """Speak a message using espeak with optional parameters"""
    # Vibrate before speaking if available
    if vibration and vibration.bus:
        try:
            vibration.notification_vibrate()
            time.sleep(0.3)
        except Exception as e:
            print(f"Vibration error: {e}")
    
    current_user, real_uid = get_current_user()

    # Build espeak command
    espeak_cmd = ['espeak']
    if voice:
        espeak_cmd.extend(['-v', voice])
    if speed:
        espeak_cmd.extend(['-s', str(speed)])
    if pitch:
        espeak_cmd.extend(['-p', str(pitch)])
    espeak_cmd.append(f'"{message}"')

    try:
        # Common environment config
        display_env = f"DISPLAY={os.environ.get('DISPLAY', ':0')}"
        xauth_env = f"XAUTHORITY=/home/{current_user}/.Xauthority"
        pulse_env = f"PULSE_COOKIE=/home/{current_user}/.config/pulse/cookie"
        pulse_server_env = (
            f"PULSE_SERVER=unix:/run/user/{real_uid}/pulse/native"
            if real_uid is not None else ""
        )
        pulse_client_env = f"PULSE_CLIENTCONFIG=/home/{current_user}/.config/pulse/client.conf"

        env_cmd = f"{display_env} {xauth_env} {pulse_env} {pulse_server_env} {pulse_client_env} espeak {' '.join(espeak_cmd[1:])}"

        if os.geteuid() == 0 and platform.system() != "Windows":
            cmd = ['sudo', '-u', current_user, 'bash', '-c', env_cmd]
        else:
            cmd = ['bash', '-c', env_cmd]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Estimate speaking duration
        words = len(message.split())
        speaking_time = (words / 150) * 60

        try:
            stdout, stderr = process.communicate(timeout=speaking_time + 5)
            if process.returncode != 0:
                print(f"espeak error: {stderr}")
                return False
        except subprocess.TimeoutExpired:
            process.terminate()
            return False

        return True
    except Exception as e:
        print(f"Speak error: {e}")
        return False
