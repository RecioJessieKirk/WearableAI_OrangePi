import os
import platform
import wiringpi

print("Loading configuration...")

# Environment settings
DEBUG_MODE = False
HARDWARE_ENABLED = not DEBUG_MODE

# GPIO Settings using physical pin numbers
POWER_BTN_PIN = 11  # PH2 physical pin 11
AI_TRIGGER_PIN = 13  # PH3 physical pin 13

# I2C Settings
I2C_DEVICE_PRIMARY = "/dev/i2c-4"  # Primary device for ADS1115
I2C_DEVICE_SECONDARY = "/dev/i2c-0"
I2C_DEVICE_TERTIARY = "/dev/i2c-1"
I2C_DEVICE_QUATERNARY = "/dev/i2c-2"
I2C_MIC_ADDRESS = 0x4A
I2C_BATTERY_ADDRESS = 0x48  # ADS1115 address
I2C_VIBRATION_ADDRESS = 0x5A  # DRV2605L address

# Audio Settings - Using standard rates
SAMPLE_RATE = 16000  # Standard rate for speech
CHANNELS = 1  # Mono audio
FORMAT = "S16_LE"  # 16-bit signed little-endian
AUDIO_VOLUME = 100  # Percent (0-100)

# Mock GPIO implementation for development
class MockGPIO:
    INPUT = wiringpi.INPUT
    OUTPUT = wiringpi.OUTPUT
    PUD_UP = wiringpi.PUD_UP
    LOW = wiringpi.LOW
    HIGH = wiringpi.HIGH

    @staticmethod
    def wiringPiSetupPhys():
        print("Mock GPIO: Setting up physical pin mode")
        return 0
        
    @staticmethod
    def pinMode(pin, mode):
        print(f"Mock GPIO: Setting pin {pin} mode")
        
    @staticmethod
    def pullUpDnControl(pin, pud):
        print(f"Mock GPIO: Setting pin {pin} pull up/down")
        
    @staticmethod
    def digitalRead(pin):
        print(f"Mock: digitalRead(pin={pin})")
        return 1
        
    @staticmethod
    def wiringPiISR(pin, edge, callback):
        print(f"Mock GPIO: Setting up interrupt for pin {pin}")

# Use mock GPIO in debug mode
GPIO = MockGPIO if DEBUG_MODE else wiringpi

print("Configuration loaded successfully")

# Import tts_handler after configuration is loaded to avoid circular imports
from tts_handler import speak

# Now we can use speak
if DEBUG_MODE:
    speak("System starting in debug mode")
else:
    speak("System starting in hardware mode")
speak("Configuration loaded successfully")