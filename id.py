import socket
import struct
import time

# Motive Default Settings
SERVER_ADDR = "127.0.0.1"  # Change to Motive PC IP if different
COMMAND_PORT = 1510        # Default Motive Command Port
MCAST_GRP = '239.255.42.99'
DATA_PORT = 1511

def get_ids():
    # Create Command Socket
    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_sock.settimeout(2.0)
    
    # NatNet Command Packet for "Request Model Definitions"
    # Message ID 2 is "Request", Packet starts with Message ID (2 bytes) then Size (2 bytes)
    packet = struct.pack('<HH', 2, 0) + b"RequestModelDef\0"
    
    print(f"Connecting to Motive at {SERVER_ADDR}...")
    try:
        cmd_sock.sendto(packet, (SERVER_ADDR, COMMAND_PORT))
        print("Command sent: RequestModelDef")
        
        # Motive usually responds on the same port or via the data stream
        # Here we listen for a short burst of data to catch the response
        data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        data_sock.bind(('', DATA_PORT))
        
        mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
        data_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        data_sock.settimeout(5.0)

        print("Listening for Model Definitions (5s timeout)...")
        start_time = time.time()
        while time.time() - start_time < 5:
            data, addr = data_sock.recvfrom(65535)
            message_id = struct.unpack('<H', data[0:2])[0]
            
            # Message ID 5 is Model Definition
            if message_id == 5:
                print("\n--- Found Model Definitions ---")
                # The data contains number of datasets at offset 4
                dataset_count = struct.unpack('<I', data[4:8])[0]
                print(f"Total tracked assets: {dataset_count}")
                # Note: Manual parsing of names from raw bytes is complex,
                # but often the first integer after the name string is the ID.
                print("Check Motive 'Properties' pane for 'Streaming ID' to be 100% sure.")
                return
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cmd_sock.close()

if __name__ == "__main__":
    get_ids()