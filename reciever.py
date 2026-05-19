import serial
import time
import json
import struct
from common_transfer import (
    UART_PORT, BAUD_RATE,
    build_packet, parse_packet,
    MSG_DATA, MSG_ACK, MSG_NACK
)

START_BYTE = 0xAA

def read_packet_from_serial(ser: serial.Serial, timeout: float) -> bytes:
    """Read one complete packet from serial, same logic as sender side."""
    ser.timeout = timeout

    while True:
        byte = ser.read(1)
        if not byte:
            raise serial.SerialTimeoutException("Timeout waiting for start byte")
        if byte[0] == START_BYTE:
            break

    header = ser.read(3)
    if len(header) < 3:
        raise ValueError("Incomplete header")

    length = struct.unpack(">H", header[1:3])[0]
    remainder = ser.read(length + 2)
    if len(remainder) < length + 2:
        raise ValueError("Incomplete packet body")

    return bytes([START_BYTE]) + header + remainder

def handle_packet(ser: serial.Serial, raw: bytes):
    """Parse a packet and respond with ACK or NACK."""
    try:
        msg_type, payload = parse_packet(raw)

        if msg_type == MSG_DATA:
            data = json.loads(payload.decode("utf-8"))
            print(f"[Receiver] Data received from '{data.get('sat_id', '?')}':")
            print(f"           State     : {data.get('docking_state')}")

            # Send ACK
            ack = build_packet(MSG_ACK, b"")
            ser.write(ack)
            ser.flush()
            print("[Receiver] ACK sent.")

    except (ValueError, json.JSONDecodeError) as e:
        print(f"[Receiver] Bad packet: {e}. Sending NACK.")
        nack = build_packet(MSG_NACK, b"")
        ser.write(nack)
        ser.flush()

def main():
    print(f"[Receiver] Listening on {UART_PORT} at {BAUD_RATE} baud...")
    with serial.Serial(UART_PORT, BAUD_RATE, timeout=1) as ser:
        ser.reset_input_buffer()

        while True:
            try:
                raw = read_packet_from_serial(ser, timeout=5.0)
                handle_packet(ser, raw)
            except serial.SerialTimeoutException:
                # No data yet — normal while waiting for dock
                print("[Receiver] Waiting for sender...")
            except Exception as e:
                print(f"[Receiver] Unexpected error: {e}")
                time.sleep(0.2)

if __name__ == "__main__":
    main()