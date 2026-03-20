import time
from dynamixel_sdk import * # --- Configuration ---
# You MUST update these variables to match your setup

PROTOCOL_VERSION = 2.0      # XC430-W240-T uses Protocol 2.0
DXL_ID           = 1       # The ID of your motor
BAUDRATE         = 2000000   # Default is 57600. Update if you changed it (e.g., 1000000)

# Device Name:
# Windows: "COM1", "COM3", etc.
# Linux/Mac: "/dev/ttyUSB0", "/dev/tty.usbserial", etc.
DEVICENAME       = '/dev/ttyUSB0' 

# --- Setup Handlers ---
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

def main():
    # 1. Open Port
    if portHandler.openPort():
        print(f"Succeeded to open the port: {DEVICENAME}")
    else:
        print(f"Failed to open the port: {DEVICENAME}")
        print("Press any key to terminate...")
        input()
        quit()

    # 2. Set Baudrate
    if portHandler.setBaudRate(BAUDRATE):
        print(f"Succeeded to change the baudrate to {BAUDRATE}")
    else:
        print(f"Failed to change the baudrate")
        print("Press any key to terminate...")
        input()
        quit()

    # 3. Send Reboot Command
    print(f"\nAttempting to reboot Dynamixel ID: {DXL_ID}...")
    dxl_comm_result, dxl_error = packetHandler.reboot(portHandler, DXL_ID)

    # 4. Handle Errors / Results
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ERROR] Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"[ERROR] Motor returned an error: {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"[SUCCESS] Dynamixel ID {DXL_ID} has been successfully rebooted.")
        
        # Give the motor's internal MCU a moment to physically restart
        print("Waiting for reboot to complete...")
        time.sleep(1.5)
        print("Motor is ready.")

    # 5. Close Port
    portHandler.closePort()

if __name__ == '__main__':
    main()