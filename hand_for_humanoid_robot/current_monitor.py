#!/usr/bin/env python3
from dynamixel_sdk import *
import time

PORT = "/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_F42208A55157375037202020FF122F34-if00"
BAUD = 57600
ADDR_PRESENT_CURRENT = 126

port = PortHandler(PORT)
packet = PacketHandler(2.0)
port.openPort()
port.setBaudRate(BAUD)

print("Monitoring current (mA) — press Ctrl+C to stop")
print(f"{'M1':>8} {'M2':>8} {'M3':>8} {'M4':>8}")
print("-" * 40)

try:
    while True:
        vals = []
        for mid in range(1, 5):
            val, res, _ = packet.read2ByteTxRx(port, mid, ADDR_PRESENT_CURRENT)
            if res == 0:
                if val > 32767:
                    val -= 65536
                vals.append(f"{abs(val):>8}")
            else:
                vals.append(f"{'ERR':>8}")
        print("  ".join(vals), end='\r')
        time.sleep(0.05)
except KeyboardInterrupt:
    pass

port.closePort()
print("\nDone.")
