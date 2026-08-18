"""
provision_cluster.py — one-time upload of an SPI-flash image into the GD32's 8MB
SPI flash over the UART4 console (CH340). The firmware provisions TWO independent
regions, each printing its own "[prov:<label>] READY <n>" when its magic is
absent; this script sends whichever image you pass to whatever is asking:
  - cluster (5 experts):   --img cluster_image.bin   (already provisioned earlier)
  - LM size bank (s0p6/m1p35): --img lmbank_image.bin

Protocol: read "...READY <total>"; send the image in 4096-byte sectors, waiting
for a 'K' ack after each (abort on 'E').

Run (bank, the new region — reset the board first so it re-enters provisioning):
  python provision_cluster.py --port COM7 --img lmbank_image.bin
(If BOTH magics are absent, the MCU asks for cluster first then bank — run this
 once per image, resetting/relaunching as each "READY" appears.)
"""
import argparse
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pip install pyserial")

SEC = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--img", default="cluster_image.bin")
    args = ap.parse_args()

    data = open(args.img, "rb").read()
    if len(data) % SEC != 0:
        data += b"\xff" * (SEC - len(data) % SEC)
    nsec = len(data) // SEC
    print(f"image {len(data)} bytes = {nsec} sectors")

    sp = serial.Serial(args.port, args.baud, timeout=10)
    # Discard boot text retained by the Windows USB-serial driver from an
    # earlier session. Only a fresh, size-matching READY may start a transfer.
    sp.reset_input_buffer()
    time.sleep(0.3)
    # wait for READY
    t0 = time.time()
    while True:
        line = sp.readline().decode("utf-8", "replace").strip()
        if line:
            print("MCU:", line)
        if "READY" in line:
            try:
                ready_size = int(line.rsplit(None, 1)[-1])
            except ValueError:
                ready_size = -1
            if ready_size == len(data):
                break
            print(f"ignore READY size {ready_size}; expected {len(data)}")
        if time.time() - t0 > 300:
            sys.exit("no READY from MCU within 300s (reset the board so it re-enters provisioning)")

    sp.reset_input_buffer()
    t0 = time.time()
    # Continuous send: the firmware masks interrupts during each sector's receive
    # burst, so no pacing is needed (and pacing gaps could exceed the MCU per-byte
    # timeout now that the spin loop runs uninterrupted). One write per sector,
    # lock-stepped by the 'K' ack.
    for k in range(nsec):
        sp.write(data[k * SEC:(k + 1) * SEC])
        sp.flush()
        ack = sp.read(1)
        if ack != b"K":
            print(f"sector {k}: bad ack {ack!r} -- draining MCU output:")
            for _ in range(12):
                rest = sp.readline().decode("utf-8", "replace").strip()
                if rest:
                    print("  MCU:", rest)
            sys.exit(f"stopped at sector {k}")
        if (k + 1) % 32 == 0 or k == nsec - 1:
            print(f"  {k+1}/{nsec} sectors ({time.time()-t0:.0f}s)")
    print(f"done in {time.time()-t0:.0f}s; MCU tail:")
    for _ in range(4):
        line = sp.readline().decode("utf-8", "replace").strip()
        if line:
            print("MCU:", line)
    sp.close()


if __name__ == "__main__":
    main()
