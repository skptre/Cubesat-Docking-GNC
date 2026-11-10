
"""
benchmark_transfer.py — measure real data rate across the pogo-pin UART link.

Usage (two Pis, faces mated):
    Pi A (target):  python3 benchmark_transfer.py recv
    Pi B (chaser):  python3 benchmark_transfer.py send

Measures:
    - Goodput (payload bytes/sec)
    - Mean round-trip time
    - Packet delivery and failure counts
    - Retries and NACKs
    - Efficiency against the 8N1 physical ceiling

Results are incrementally saved to benchmark_results.json.
"""

import json
import os
import struct
import sys
import time
from statistics import mean

import serial

from common_transfer import (
    UART_PORT,
    BAUD_RATE,
    build_packet,
    parse_packet,
    MSG_ACK,
    MSG_NACK,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MSG_BENCH = 0x04

PAYLOAD_SIZES = [16, 64, 256, 1024, 4096]

# Start with a smaller workload to verify pogo-pin reliability.
PACKETS_PER_SIZE = 10

ACK_TIMEOUT = 1.0
MAX_RETRIES = 2

PROGRESS_INTERVAL = 5

RESULT_FILE = "benchmark_results.json"
TEMP_RESULT_FILE = "benchmark_results.tmp"

# 8N1 = 10 line bits per byte.
LINE_CEILING_BPS = BAUD_RATE / 10.0


# --------------------------------------------------
# JSON result saving
# --------------------------------------------------

def save_results(results):
    """
    Save benchmark results after each completed payload size.

    Write to a temporary file first to avoid leaving a
    partially written JSON file if the program is interrupted.
    """
    try:
        with open(TEMP_RESULT_FILE, "w") as f:
            json.dump(results, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(TEMP_RESULT_FILE, RESULT_FILE)

        print(f"[bench-send] Results checkpoint saved: {RESULT_FILE}")

    except OSError as e:
        print(f"[bench-send] WARNING: Could not save JSON: {e}")


# --------------------------------------------------
# UART packet reading
# --------------------------------------------------

def read_exactly(ser, count):
    """
    Read exactly count bytes.

    Returns None if the read times out before all bytes arrive.
    """
    data = bytearray()

    while len(data) < count:
        chunk = ser.read(count - len(data))

        if not chunk:
            return None

        data.extend(chunk)

    return bytes(data)


def read_packet_from_serial(ser, timeout):
    """
    Read one complete packet.

    Packet structure assumed by the existing protocol:
        0xAA | type | length (2 bytes) | payload | checksum (2 bytes)

    Returns the complete raw packet.
    """

    old_timeout = ser.timeout
    deadline = time.monotonic() + timeout

    try:
        while time.monotonic() < deadline:

            remaining = deadline - time.monotonic()
            ser.timeout = max(0, remaining)

            start = ser.read(1)

            if not start:
                raise serial.SerialTimeoutException(
                    "Timeout waiting for start byte"
                )

            if start[0] == 0xAA:
                break

        else:
            raise serial.SerialTimeoutException(
                "Timeout waiting for start byte"
            )

        remaining = deadline - time.monotonic()
        ser.timeout = max(0, remaining)

        header = read_exactly(ser, 3)

        if header is None:
            raise serial.SerialTimeoutException(
                "Incomplete packet header"
            )

        length = struct.unpack(">H", header[1:3])[0]

        remaining = deadline - time.monotonic()
        ser.timeout = max(0, remaining)

        rest = read_exactly(ser, length + 2)

        if rest is None:
            raise serial.SerialTimeoutException(
                "Incomplete packet body"
            )

        return start + header + rest

    finally:
        ser.timeout = old_timeout


# --------------------------------------------------
# Receiver
# --------------------------------------------------

def run_receiver():

    print("=" * 55)
    print("[bench-recv] UART DATA TRANSFER BENCHMARK")
    print("=" * 55)

    print(f"[bench-recv] Port: {UART_PORT}")
    print(f"[bench-recv] Baud: {BAUD_RATE}")
    print("[bench-recv] Waiting for sender...")
    print("[bench-recv] Press Ctrl+C to stop.\n")

    good = 0
    bad = 0

    try:
        with serial.Serial(
            UART_PORT,
            BAUD_RATE,
            timeout=1
        ) as ser:

            ser.reset_input_buffer()
            ser.reset_output_buffer()

            while True:

                try:
                    raw = read_packet_from_serial(
                        ser,
                        timeout=5.0
                    )

                except serial.SerialTimeoutException:
                    continue

                try:
                    msg_type, payload = parse_packet(raw)

                    if msg_type == MSG_BENCH:

                        # Acknowledge received benchmark packet.
                        ack = build_packet(MSG_ACK, b"")

                        ser.write(ack)
                        ser.flush()

                        good += 1

                        if good % 5 == 0:
                            print(
                                f"[bench-recv] "
                                f"Received: {good} packets "
                                f"| Corrupt: {bad}"
                            )

                except ValueError:

                    # Tell sender that the packet was corrupted.
                    nack = build_packet(MSG_NACK, b"")

                    ser.write(nack)
                    ser.flush()

                    bad += 1

                    print(
                        f"[bench-recv] "
                        f"Corrupt packet #{bad}"
                    )

    except KeyboardInterrupt:

        print("\n[bench-recv] Receiver stopped.")

    finally:

        print("\n========== RECEIVER SUMMARY ==========")
        print(f"Successfully received: {good}")
        print(f"Corrupted packets:     {bad}")
        print("======================================")


# --------------------------------------------------
# Send one packet
# --------------------------------------------------

def send_one(ser, payload):
    """
    Send one benchmark packet and wait for its ACK.

    Returns:
        ok, retries, nacks, rtt
    """

    packet = build_packet(MSG_BENCH, payload)

    retries = 0
    nacks = 0

    for attempt in range(MAX_RETRIES + 1):

        t0 = time.perf_counter()

        try:

            ser.write(packet)
            ser.flush()

            deadline = time.monotonic() + ACK_TIMEOUT

            while time.monotonic() < deadline:

                remaining = deadline - time.monotonic()

                try:
                    raw = read_packet_from_serial(
                        ser,
                        timeout=remaining
                    )

                    msg_type, _ = parse_packet(raw)

                    if msg_type == MSG_ACK:

                        rtt = time.perf_counter() - t0

                        return True, retries, nacks, rtt

                    elif msg_type == MSG_NACK:

                        nacks += 1
                        break

                    else:
                        # Ignore unrelated messages.
                        continue

                except serial.SerialTimeoutException:
                    break

                except ValueError:
                    # Invalid response. Continue waiting.
                    continue

        except serial.SerialException as e:

            print(f"[bench-send] UART error: {e}")
            break

        if attempt < MAX_RETRIES:
            retries += 1

    return False, retries, nacks, None


# --------------------------------------------------
# Sender
# --------------------------------------------------

def run_sender():

    results = []

    print("=" * 60)
    print("[bench-send] UART DATA TRANSFER BENCHMARK")
    print("=" * 60)

    print(f"[bench-send] Port: {UART_PORT}")
    print(f"[bench-send] Baud: {BAUD_RATE}")
    print(f"[bench-send] Packets per size: {PACKETS_PER_SIZE}")
    print(f"[bench-send] ACK timeout: {ACK_TIMEOUT}s")
    print(f"[bench-send] Maximum retries: {MAX_RETRIES}")
    print(f"[bench-send] Physical ceiling: {LINE_CEILING_BPS:.0f} B/s")
    print("=" * 60)

    try:

        with serial.Serial(
            UART_PORT,
            BAUD_RATE,
            timeout=1,
            write_timeout=2
        ) as ser:

            time.sleep(0.5)

            ser.reset_input_buffer()

            for size in PAYLOAD_SIZES:

                print(f"\n[bench-send] Testing payload size: {size} bytes")

                payload = os.urandom(size)

                ok_count = 0
                retries = 0
                nacks = 0
                fails = 0

                rtts = []

                t_start = time.perf_counter()

                try:

                    for i in range(PACKETS_PER_SIZE):

                        ok, r, n, rtt = send_one(
                            ser,
                            payload
                        )

                        retries += r
                        nacks += n

                        if ok:

                            ok_count += 1
                            rtts.append(rtt)

                        else:

                            fails += 1

                        # Progress reporting.
                        if (
                            (i + 1) % PROGRESS_INTERVAL == 0
                            or (i + 1) == PACKETS_PER_SIZE
                        ):

                            print(
                                f"[bench-send] "
                                f"{i + 1}/{PACKETS_PER_SIZE} packets "
                                f"| delivered={ok_count} "
                                f"| failed={fails} "
                                f"| retries={retries}"
                            )

                except KeyboardInterrupt:

                    print("\n[bench-send] Benchmark interrupted.")

                    break

                elapsed = time.perf_counter() - t_start

                # Goodput counts successfully delivered payload bytes.
                goodput = (
                    (ok_count * size) / elapsed
                    if elapsed > 0 else 0.0
                )

                efficiency = (
                    100.0 * goodput / LINE_CEILING_BPS
                    if LINE_CEILING_BPS > 0 else 0.0
                )

                mean_rtt = (
                    mean(rtts) * 1000
                    if rtts else None
                )

                result = {
                    "payload_bytes": size,
                    "packets_sent": PACKETS_PER_SIZE,
                    "delivered": ok_count,
                    "failed": fails,
                    "retries": retries,
                    "nacks": nacks,
                    "elapsed_s": round(elapsed, 3),
                    "goodput_Bps": round(goodput, 1),
                    "efficiency_pct": round(efficiency, 1),
                    "mean_rtt_ms": (
                        round(mean_rtt, 2)
                        if mean_rtt is not None else None
                    ),
                }

                results.append(result)

                # Save after every completed payload size.
                save_results(results)

                print(
                    f"\n[bench-send] {size:5d} B : "
                    f"{goodput:8.1f} B/s "
                    f"({efficiency:4.1f}% of ceiling) "
                    f"| RTT: {mean_rtt if mean_rtt is not None else 'N/A'} ms "
                    f"| retries: {retries} "
                    f"| fails: {fails}"
                )

    except KeyboardInterrupt:

        print("\n[bench-send] Benchmark interrupted.")

    except serial.SerialException as e:

        print(f"\n[bench-send] UART connection error: {e}")

    finally:

        if results:
            save_results(results)

        print("\n========== BENCHMARK SUMMARY ==========")

        if results:

            print(
                "\n payload |  goodput  | efficiency | "
                "mean RTT | retries | fails"
            )

            print(
                "---------+-----------+------------+"
                "----------+---------+------"
            )

            for r in results:

                print(
                    f"{r['payload_bytes']:7d}B | "
                    f"{r['goodput_Bps']:8.1f} | "
                    f"{r['efficiency_pct']:9.1f}% | "
                    f"{str(r['mean_rtt_ms']):>7}ms | "
                    f"{r['retries']:7d} | "
                    f"{r['failed']:5d}"
                )

        else:

            print("No completed payload sizes.")

        print(
            f"\nPhysical ceiling @ {BAUD_RATE} baud 8N1: "
            f"{LINE_CEILING_BPS:.0f} B/s"
        )

        print(f"Results file: {RESULT_FILE}")

        print("=======================================")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else ""

    if mode == "recv":

        run_receiver()

    elif mode == "send":

        run_sender()

    else:

        print("Usage:")
        print("  python3 benchmark_transfer.py recv")
        print("  python3 benchmark_transfer.py send")
