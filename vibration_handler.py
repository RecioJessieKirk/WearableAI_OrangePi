import time
from smbus2 import SMBus
from config import I2C_DEVICE_PRIMARY, I2C_VIBRATION_ADDRESS

print("Loading Vibration Handler...")

class VibrationHandler:
    # DRV2605L Registers
    REG_STATUS = 0x00
    REG_MODE = 0x01
    REG_RTP_INPUT = 0x02
    REG_LIBRARY_SELECTION = 0x03
    REG_WAVEFORM_SEQUENCER_1 = 0x04
    REG_WAVEFORM_SEQUENCER_2 = 0x05
    REG_WAVEFORM_SEQUENCER_3 = 0x06
    REG_WAVEFORM_SEQUENCER_4 = 0x07
    REG_WAVEFORM_SEQUENCER_5 = 0x08
    REG_WAVEFORM_SEQUENCER_6 = 0x09
    REG_WAVEFORM_SEQUENCER_7 = 0x0A
    REG_WAVEFORM_SEQUENCER_8 = 0x0B
    REG_GO = 0x0C
    REG_OVERDRIVE_TIME_OFFSET = 0x0D
    REG_SUSTAIN_TIME_OFFSET_POS = 0x0E
    REG_SUSTAIN_TIME_OFFSET_NEG = 0x0F
    REG_BRAKE_TIME_OFFSET_POS = 0x10
    REG_AUDIO_TO_VIBE_CTRL = 0x11
    REG_AUDIO_TO_VIBE_MIN_INPUT = 0x12
    REG_AUDIO_TO_VIBE_MAX_INPUT = 0x13
    REG_AUDIO_TO_VIBE_MIN_OUTPUT = 0x14
    REG_AUDIO_TO_VIBE_MAX_OUTPUT = 0x15
    REG_RATED_VOLTAGE = 0x16
    REG_OD_CLAMP_VOLTAGE = 0x17
    REG_CAL_COMP = 0x18
    REG_CAL_BACK_EMF = 0x19
    REG_FEEDBACK_CTRL = 0x1A
    REG_CONTROL1 = 0x1B
    REG_CONTROL2 = 0x1C
    REG_CONTROL3 = 0x1D
    REG_CONTROL4 = 0x1E
    REG_CONTROL5 = 0x1F
    REG_LRA_OPEN_LOOP_PERIOD = 0x20
    REG_VBAT_MONITOR = 0x21
    REG_LRA_RESONANCE_PERIOD = 0x22

    # Mode register values
    MODE_INTERNAL_TRIGGER = 0x00
    MODE_EXTERNAL_TRIGGER_EDGE = 0x01
    MODE_EXTERNAL_TRIGGER_LEVEL = 0x02
    MODE_PWM_ANALOG_INPUT = 0x03
    MODE_AUDIO_TO_VIBE = 0x04
    MODE_REAL_TIME_PLAYBACK = 0x05
    MODE_DIAGNOSTICS = 0x06
    MODE_AUTO_CALIBRATION = 0x07

    # Library selection values
    LIBRARY_EMPTY = 0x00
    LIBRARY_TS2200_A = 0x01
    LIBRARY_TS2200_B = 0x02
    LIBRARY_TS2200_C = 0x03
    LIBRARY_TS2200_D = 0x04
    LIBRARY_TS2200_E = 0x05
    LIBRARY_LRA = 0x06

    def __init__(self):
        self.bus = None
        self.retry_count = 0
        self.MAX_RETRIES = 3
        # Import config here to avoid circular imports
        from config import I2C_DEVICE_PRIMARY, I2C_VIBRATION_ADDRESS
        self.i2c_device = I2C_DEVICE_PRIMARY
        self.i2c_address = I2C_VIBRATION_ADDRESS
        self._initialize_i2c()

    def _initialize_i2c(self):
        try:
            # Extract bus number from device path (e.g., "/dev/i2c-3" -> 3)
            bus_number = int(self.i2c_device.split('-')[-1])
            self.bus = SMBus(bus_number)
            print(f"Initialized SMBus on bus {bus_number}")
            
            # Configure the device
            self._configure_device()
            return True

        except Exception as e:
            print(f"Failed to initialize I2C: {e}")
            if self.bus:
                try:
                    self.bus.close()
                except:
                    pass
            self.bus = None
            return False

    def _configure_device(self):
        """Configure the DRV2605L for basic operation"""
        try:
            print("Starting DRV2605L configuration...")
            
            # Set to internal trigger mode
            self.write_register(self.REG_MODE, self.MODE_INTERNAL_TRIGGER)
            
            # Select ERM library (changed from LRA to ERM based on working test)
            self.write_register(self.REG_LIBRARY_SELECTION, self.LIBRARY_TS2200_A)
            
            # Set rated voltage (3.6V)
            self.write_register(self.REG_RATED_VOLTAGE, 0x90)
            
            # Set overdrive clamp voltage (4.2V)
            self.write_register(self.REG_OD_CLAMP_VOLTAGE, 0xA8)
            
            # Set feedback control for maximum strength
            self.write_register(self.REG_FEEDBACK_CTRL, 0x36)
            
            # Set control registers for maximum strength
            self.write_register(self.REG_CONTROL1, 0x93)  # Enable feedback, set drive time
            self.write_register(self.REG_CONTROL2, 0xF5)  # Set sample time and blanking time
            self.write_register(self.REG_CONTROL3, 0x80)  # Set ERM/LRA mode
            self.write_register(self.REG_CONTROL4, 0x20)  # Set auto-calibration
            self.write_register(self.REG_CONTROL5, 0x80)  # Set LRA open loop
            
            print("DRV2605L configured successfully")
            return True
            
        except Exception as e:
            print(f"Error configuring DRV2605L: {e}")
            return False

    def write_register(self, register, value):
        """Write a value to a register"""
        if self.bus is None:
            return False

        try:
            self.bus.write_byte_data(self.i2c_address, register, value)
            return True
        except Exception as e:
            print(f"Error writing register 0x{register:02X}: {e}")
            return False

    def read_register(self, register):
        """Read a value from a register"""
        if self.bus is None:
            return 0

        try:
            return self.bus.read_byte_data(self.i2c_address, register)
        except Exception as e:
            print(f"Error reading register 0x{register:02X}: {e}")
            return 0

    def play_effect(self, effect_id):
        """Play a specific haptic effect"""
        if self.bus is None:
            print("I2C not initialized")
            return False

        try:
            # Set the main effect
            self.write_register(self.REG_WAVEFORM_SEQUENCER_1, effect_id)
            # Add a strong buzz after the main effect
            self.write_register(self.REG_WAVEFORM_SEQUENCER_2, 15)
            # Clear remaining sequencer slots
            for reg in range(self.REG_WAVEFORM_SEQUENCER_3, self.REG_WAVEFORM_SEQUENCER_8 + 1):
                self.write_register(reg, 0)
            # Trigger the effect
            self.write_register(self.REG_GO, 1)
            return True
        except Exception as e:
            print(f"Error playing effect: {e}")
            return False

    def notification_vibrate(self):
        """Play a strong notification vibration"""
        if self.bus is None:
            return False

        try:
            # Set a sequence of strong effects
            # Effect 14: Strong Click
            self.write_register(self.REG_WAVEFORM_SEQUENCER_1, 14)
            # Effect 15: Strong Buzz
            self.write_register(self.REG_WAVEFORM_SEQUENCER_2, 15)
            # Effect 16: Strong Double Click
            self.write_register(self.REG_WAVEFORM_SEQUENCER_3, 16)
            # Effect 17: Strong Triple Click
            self.write_register(self.REG_WAVEFORM_SEQUENCER_4, 17)
            
            # Clear remaining sequencer slots
            for reg in range(self.REG_WAVEFORM_SEQUENCER_5, self.REG_WAVEFORM_SEQUENCER_8 + 1):
                self.write_register(reg, 0)
                
            # Trigger the effect
            self.write_register(self.REG_GO, 1)
            return True
        except Exception as e:
            print(f"Error playing effect: {e}")
            return False

    def __del__(self):
        if self.bus:
            try:
                self.bus.close()
            except:
                pass 