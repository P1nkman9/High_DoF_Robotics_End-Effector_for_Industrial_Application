#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jetson-side test for a single STM32 FOC axis: finds the serial port,
waits for SimpleFOC alignment, then sends target-angle commands.
"""

import serial
import serial.tools.list_ports
import time
import sys

def find_stm32_port():
    """Scan and return the first ACM or USB serial port found."""
    print("Scanning for available serial devices...")
    ports = [p.device for p in serial.tools.list_ports.comports() if 'ACM' in p.device or 'USB' in p.device]

    if not ports:
        print("No serial devices found! Please check USB connections.")
        return None

    print(f"Device found: {ports[0]}")
    return ports[0]

def main():
    target_port = find_stm32_port()
    if not target_port:
        return

    try:
        ser = serial.Serial(target_port, 115200, timeout=0.5)

        print("\nWaiting for SimpleFOC chip reset and motor calibration...")
        print("Note: the motor should produce a soft current whine and rotate slightly to align.")

        time.sleep(5)

        while ser.in_waiting:
            msg = ser.readline().decode('utf-8', errors='ignore').strip()
            if msg: print(f"[STM32 Init] {msg}")

        ser.reset_input_buffer()
        print("\nMotor calibration complete; control link is open!")
        print("-" * 40)
        print("Please choose a test mode:")
        print("1: Manually enter target angle")
        print("2: Auto sweep test (rotate between -3 and 3 radians automatically)")
        print("q: Quit")
        print("-" * 40)

        mode = input("Enter choice (1 / 2 / q): ").strip()

        if mode == '1':
            print("Manual mode entered. Enter target angle (radians), e.g. 1.57, -3.14. Type 'q' to quit.")
            while True:
                val = input("Enter target angle: ").strip()
                if val.lower() == 'q':
                    break
                try:
                    angle = float(val)
                    # SimpleFOC Commander parses the float after 'T' (e.g. T1.57).
                    cmd = f"T{angle}\n"
                    ser.write(cmd.encode('utf-8'))
                    print(f"  [Jetson] Sent command: {cmd.strip()}")

                    time.sleep(0.05)
                    while ser.in_waiting:
                        resp = ser.readline().decode('utf-8', errors='ignore').strip()
                        if resp: print(f"  [STM32] {resp}")

                except ValueError:
                    print("Please enter a valid number!")

        elif mode == '2':
            print("Auto sweep mode entered. Press Ctrl+C to stop.")
            target_angles = [0.0, 3.14, 0.0, -3.14]
            idx = 0
            while True:
                angle = target_angles[idx % len(target_angles)]
                cmd = f"T{angle}\n"
                ser.write(cmd.encode('utf-8'))
                print(f"  [Jetson] Sent sweep command: {cmd.strip()}")
                idx += 1
                time.sleep(2)

        elif mode.lower() == 'q':
            print("Exiting test.")

    except KeyboardInterrupt:
        print("\nInterrupt received; stopping program.")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            # Send the axis back to 0 rad on exit so the motor doesn't hold torque.
            ser.write(b"T0\n")
            time.sleep(0.1)
            ser.close()
            print("Serial port closed safely.")

if __name__ == "__main__":
    main()
