import serial
import time
import json
from common_transfer import (
    UART_PORT, BAUD_RATE,
    build_packet, parse_packet,
    MSG_DATA, MSG_ACK, MSG_NACK
)

MAX_RETRIES = 3
ACK_TIMEOUT = 2.0  # seconds

def send_data(ser: serial.Serial, payload_dict: dict) -> bool:
    """
    Serialize a dict to JSON, send it, and wait for ACK.
    Returns True on success, False after exhausting retries.
    """
    payload = json.dumps(payload_dict).encode("utf-8")
    packet = build_packet(MSG_DATA, payload)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[Sender] Attempt {attempt}: sending {len(payload)} bytes...")
        ser.write(packet)
        ser.flush()

        try:
            # Read response packet (fixed header size + dynamic payload)
            response_raw = read_packet_from_serial(ser, timeout=ACK_TIMEOUT)
            msg_type, _ = parse_packet(response_raw)

            if msg_type == MSG_ACK:
                print("[Sender] ACK received. Transfer successful.")
                return True
            elif msg_type == MSG_NACK:
                print("[Sender] NACK received. Retrying...")
        except (ValueError, serial.SerialTimeoutException) as e:
            print(f"[Sender] Error on attempt {attempt}: {e}")

        time.sleep(0.5)

    print("[Sender] All retries exhausted. Transfer failed.")
    return False

def read_packet_from_serial(ser: serial.Serial, timeout: float) -> bytes:
    """
    Read one complete packet from the serial buffer.
    Blocks until START_BYTE is found or timeout expires.
    """
    import struct
    ser.timeout = timeout
    START_BYTE = 0xAA

    # Seek to start byte
    while True:
        byte = ser.read(1)
        if not byte:
            raise serial.SerialTimeoutException("Timeout waiting for start byte")
        if byte[0] == START_BYTE:
            break

    # Read fixed header fields: msg_type (1B) + length (2B)
    header = ser.read(3)
    if len(header) < 3:
        raise ValueError("Incomplete header")

    msg_type = header[0]
    length = struct.unpack(">H", header[1:3])[0]

    # Read payload + checksum + end byte
    remainder = ser.read(length + 2)
    if len(remainder) < length + 2:
        raise ValueError("Incomplete packet body")

    return bytes([START_BYTE]) + header + remainder

def main():
    with serial.Serial(UART_PORT, BAUD_RATE, timeout=1) as ser:
        time.sleep(0.1)  # Let port settle
        ser.reset_input_buffer()

        # payload
        data = {
            "sat_id" : "sender",
            "docking_state": "connected"
        }

        success = send_data(ser, data)
        if not success:
            print("[Sender] Consider re-attempting after reconnect.")

if __name__ == "__main__":
    main()