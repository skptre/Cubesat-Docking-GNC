"""
benchmark_transfer.py — measure real data rate across the pogo-pin UART link.

Usage (two Pis, faces mated):
    Pi A (target):  python3 benchmark_transfer.py recv
    Pi B (chaser):  python3 benchmark_transfer.py send

The sender sweeps packet sizes, firing PACKETS_PER_SIZE packets at each size,
and reports goodput (payload bytes/s), mean round-trip time, retries, and
failures — plus efficiency against the 8N1 physical ceiling.

Results are printed as a table and saved to benchmark_results.json so the
sweep can be plotted later (throughput vs packet size is the money figure).
"""

import json
import os
import struct
import sys
import time

import serial

from common_transfer import (
    UART_PORT, BAUD_RATE,
    build_packet, parse_packet,
    MSG_ACK, MSG_NACK,
)

MSG_BENCH = 0x04            # benchmark payload (raw bytes)

PAYLOAD_SIZES = [16, 64, 256, 1024, 4096]   # bytes
PACKETS_PER_SIZE = 50
ACK_TIMEOUT = 1.0 
MAX_RETRIES = 3

# 8N1: 10 line bits per data byte
LINE_CEILING_BPS = BAUD_RATE / 10.0         # bytes/sec, physical maximum


def read_packet_from_serial(ser, timeout):
    # Read one complete packet
    ser.timeout = timeout
    while True:
        b = ser.read(1)
        if not b:
            raise serial.SerialTimeoutException("timeout waiting for start byte")
        if b[0] == 0xAA:
            break
    header = ser.read(3)
    if len(header) < 3:
        raise ValueError("incomplete header")
    length = struct.unpack(">H", header[1:3])[0]
    rest = ser.read(length + 2)
    if len(rest) < length + 2:
        raise ValueError("incomplete body")
    return bytes([0xAA]) + header + rest


def run_receiver():
    print(f"[bench-recv] listening on {UART_PORT} @ {BAUD_RATE}; Ctrl-C to stop")
    good = bad = 0
    with serial.Serial(UART_PORT, BAUD_RATE, timeout=1) as ser:
        ser.reset_input_buffer()
        while True:
            try:
                raw = read_packet_from_serial(ser, timeout=5.0)
            except serial.SerialTimeoutException:
                continue
            try:
                msg_type, payload = parse_packet(raw)
                if msg_type == MSG_BENCH:
                    ser.write(build_packet(MSG_ACK, b""))
                    ser.flush()
                    good += 1
            except ValueError:
                ser.write(build_packet(MSG_NACK, b""))
                ser.flush()
                bad += 1
                print(f"[bench-recv] corrupt packet #{bad}")


def send_one(ser, payload):
    # Send one packet, wait for ACK. Returns (ok, retries, nacks, rtt).
    packet = build_packet(MSG_BENCH, payload)
    retries = nacks = 0
    for attempt in range(1 + MAX_RETRIES):
        t0 = time.perf_counter()
        ser.write(packet)
        ser.flush()
        try:
            raw = read_packet_from_serial(ser, timeout=ACK_TIMEOUT)
            msg_type, _ = parse_packet(raw)
            rtt = time.perf_counter() - t0
            if msg_type == MSG_ACK:
                return True, retries, nacks, rtt
            nacks += 1
        except (ValueError, serial.SerialTimeoutException):
            pass
        retries += 1
    return False, retries, nacks, None


def run_sender():
    results = []
    with serial.Serial(UART_PORT, BAUD_RATE, timeout=1) as ser:
        time.sleep(0.1)
        ser.reset_input_buffer()

        for size in PAYLOAD_SIZES:
            payload = os.urandom(size)
            ok_count = retries = nacks = fails = 0
            rtts = []

            t_start = time.perf_counter()
            for _ in range(PACKETS_PER_SIZE):
                ok, r, n, rtt = send_one(ser, payload)
                retries += r
                nacks += n
                if ok:
                    ok_count += 1
                    rtts.append(rtt)
                else:
                    fails += 1
            elapsed = time.perf_counter() - t_start

            goodput = (ok_count * size) / elapsed if elapsed > 0 else 0.0
            results.append({
                "payload_bytes": size,
                "packets_sent": PACKETS_PER_SIZE,
                "delivered": ok_count,
                "failed": fails,
                "retries": retries,
                "nacks": nacks,
                "elapsed_s": round(elapsed, 3),
                "goodput_Bps": round(goodput, 1),
                "efficiency_pct": round(100.0 * goodput / LINE_CEILING_BPS, 1),
                "mean_rtt_ms": round(1000 * sum(rtts) / len(rtts), 2) if rtts else None,
            })
            r = results[-1]
            print(f"[bench-send] {size:5d} B : "
                  f"{r['goodput_Bps']:8.1f} B/s "
                  f"({r['efficiency_pct']:4.1f}% of ceiling)  "
                  f"rtt {r['mean_rtt_ms']} ms  "
                  f"retries {retries}  fails {fails}")

    print("\n payload |  goodput  | efficiency |  mean RTT | retries | fails")
    print(" --------+-----------+------------+-----------+---------+------")
    for r in results:
        print(f" {r['payload_bytes']:6d}B | {r['goodput_Bps']:7.1f}Bs "
              f"| {r['efficiency_pct']:9.1f}% | {str(r['mean_rtt_ms']):>7}ms "
              f"| {r['retries']:7d} | {r['failed']:5d}")
    print(f"\n physical ceiling @ {BAUD_RATE} baud 8N1: "
          f"{LINE_CEILING_BPS:.0f} B/s")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(" results saved to benchmark_results.json")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "recv":
        run_receiver()
    elif mode == "send":
        run_sender()
    else:
        print("usage: python3 benchmark_transfer.py [send|recv]")