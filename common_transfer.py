import struct

BAUD_RATE = 115200
UART_PORT = "/dev/serial0"  # Hardware UART on Pi 5

START_BYTE = 0xAA
END_BYTE   = 0x55

def compute_checksum(data: bytes) -> int:
    """Simple XOR checksum over payload bytes."""
    result = 0
    for b in data:
        result ^= b
    return result

def build_packet(msg_type: int, payload: bytes) -> bytes:
    """
    Packet structure:
      [START_BYTE][msg_type][payload_len (2B)][payload][checksum][END_BYTE]
    """
    length = len(payload)
    checksum = compute_checksum(payload)
    header = struct.pack(">BBH", START_BYTE, msg_type, length)
    footer = struct.pack(">BB", checksum, END_BYTE)
    return header + payload + footer

def parse_packet(raw: bytes):
    """
    Returns (msg_type, payload) or raises ValueError on bad packet.
    """
    if len(raw) < 6:
        raise ValueError("Packet too short")
    if raw[0] != START_BYTE or raw[-1] != END_BYTE:
        raise ValueError("Invalid start/end bytes")

    msg_type = raw[1]
    length = struct.unpack(">H", raw[2:4])[0]
    payload = raw[4 : 4 + length]
    checksum = raw[4 + length]

    if len(payload) != length:
        raise ValueError("Payload length mismatch")
    if compute_checksum(payload) != checksum:
        raise ValueError("Checksum mismatch")

    return msg_type, payload

# Message types
MSG_DATA = 0x01
MSG_ACK  = 0x02
MSG_NACK = 0x03

