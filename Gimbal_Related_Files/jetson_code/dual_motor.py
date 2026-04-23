#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jetson-side dual-axis FOC driver. Two Arduinos run independent SimpleFOC
closed loops; each Commander listens for 'T<angle>'. This script opens both
USB serial links and dispatches per-axis targets.
"""

import serial
import serial.tools.list_ports
import time

BAUD_RATE = 115200
INIT_WAIT = 5  # seconds to allow for SimpleFOC alignment


def find_arduino_ports():
    """Return a list of ACM/USB serial devices currently visible."""
    print("Scanning for available serial devices...")
    ports = [p.device for p in serial.tools.list_ports.comports()
             if 'ACM' in p.device or 'USB' in p.device]

    if len(ports) == 0:
        print("No serial devices found! Please check USB connections.")
    elif len(ports) == 1:
        print(f"Warning: only one serial port {ports[0]} found; dual-motor control requires two.")
    else:
        print(f"Found serial ports: {ports}")

    return ports


def open_serial(port):
    """Open `port` at BAUD_RATE; return None on failure."""
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.5)
        print(f"  Opened serial port: {port}")
        return ser
    except Exception as e:
        print(f"  Failed to open serial port {port}: {e}")
        return None


def wait_for_init(ser, label):
    """Block for INIT_WAIT seconds, then drain and print any bootup logs."""
    print(f"\n[{label}] Waiting for SimpleFOC calibration (~{INIT_WAIT} seconds)...")
    time.sleep(INIT_WAIT)
    while ser.in_waiting:
        msg = ser.readline().decode('utf-8', errors='ignore').strip()
        if msg:
            print(f"  [{label} Init] {msg}")
    ser.reset_input_buffer()
    print(f"  [{label}] Calibration complete.")


def send_angle(ser, label, angle):
    """Write `T<angle>\\n` to the axis and echo the reply, if any."""
    cmd = f"T{angle}\n"
    ser.write(cmd.encode('utf-8'))
    print(f"  [{label}] Sent: {cmd.strip()}")
    time.sleep(0.05)
    while ser.in_waiting:
        resp = ser.readline().decode('utf-8', errors='ignore').strip()
        if resp:
            print(f"  [{label} Echo] {resp}")


def close_all(ser_a, ser_b):
    """Send each axis to 0 rad and close both ports."""
    for ser, label in [(ser_a, "Motor A"), (ser_b, "Motor B")]:
        if ser and ser.is_open:
            ser.write(b"T0\n")
    time.sleep(0.1)
    for ser, label in [(ser_a, "Motor A"), (ser_b, "Motor B")]:
        if ser and ser.is_open:
            ser.close()
            print(f"  [{label}] Serial port closed.")


def main():
    ports = find_arduino_ports()

    if len(ports) < 2:
        print("At least two serial ports are required to drive dual motors; exiting.")
        return

    # Port ordering is not stable across reboots, so confirm the mapping.
    print("\nDetected the following serial ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p}")

    print("\nDefault: serial port 0 -> Motor A, serial port 1 -> Motor B")
    confirm = input("Use the default assignment? (y/n, press Enter for y): ").strip().lower()

    if confirm == 'n':
        try:
            idx_a = int(input(f"  Enter the serial port index for Motor A (0~{len(ports)-1}): ").strip())
            idx_b = int(input(f"  Enter the serial port index for Motor B (0~{len(ports)-1}): ").strip())
            port_a, port_b = ports[idx_a], ports[idx_b]
        except (ValueError, IndexError):
            print("Invalid input; falling back to default assignment.")
            port_a, port_b = ports[0], ports[1]
    else:
        port_a, port_b = ports[0], ports[1]

    print(f"\nMotor A -> {port_a}")
    print(f"Motor B -> {port_b}")

    ser_a = open_serial(port_a)
    ser_b = open_serial(port_b)

    if not ser_a or not ser_b:
        print("Failed to open serial ports; exiting.")
        close_all(ser_a, ser_b)
        return

    try:
        wait_for_init(ser_a, "Motor A")
        wait_for_init(ser_b, "Motor B")

        print("\nDual-motor calibration complete; control link is open!")
        print("-" * 40)
        print("Please choose a test mode:")
        print("1: Manual mode  - Enter target angles for both motors one by one")
        print("2: Auto sweep   - Motors sweep oppositely between -1.57 and 1.57 rad")
        print("q: Quit")
        print("-" * 40)

        mode = input("Enter choice (1 / 2 / q): ").strip()

        if mode == '1':
            print("\nManual mode entered. Enter target angles (radians) for Motor A and Motor B each round.")
            print("Type 'q' to quit.\n")
            while True:
                val_a = input("Motor A target angle (radians, or q to quit): ").strip()
                if val_a.lower() == 'q':
                    break
                val_b = input("Motor B target angle (radians, or q to quit): ").strip()
                if val_b.lower() == 'q':
                    break
                try:
                    angle_a = float(val_a)
                    angle_b = float(val_b)
                    send_angle(ser_a, "Motor A", angle_a)
                    send_angle(ser_b, "Motor B", angle_b)
                except ValueError:
                    print("Please enter a valid number!")

        elif mode == '2':
            print("\nAuto dual-axis sweep mode. Motors A and B rotate oppositely. Press Ctrl+C to stop.")
            target_angles = [0.0, 1.57, 0.0, -1.57]
            idx = 0
            while True:
                angle = target_angles[idx % len(target_angles)]
                send_angle(ser_a, "Motor A",  angle)
                send_angle(ser_b, "Motor B", -angle)
                idx += 1
                time.sleep(2)

        elif mode.lower() == 'q':
            print("Exiting test.")

    except KeyboardInterrupt:
        print("\nInterrupt received; stopping program.")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        close_all(ser_a, ser_b)
        print("Program exited safely.")


if __name__ == "__main__":
    main()
