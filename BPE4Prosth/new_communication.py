import socket
import struct
import time
import sys

def get_input_values():
    """Prompts the user to enter a list of float values in format [x1, x2, x3, x4]"""
    try:
        user_input = input("Enter values [x1, x2, x3, x4] or press Enter to skip: ")
        if not user_input.strip():
            return None  # Return None if nothing is entered
        
        # Check if input has the proper format with brackets
        user_input = user_input.strip()
        if user_input.startswith('[') and user_input.endswith(']'):
            # Remove brackets and split by commas
            values_str = user_input[1:-1].split(',')
            # Convert to floats
            values = [float(val.strip()) for val in values_str]
            
            # We specifically need 4 values
            if len(values) == 4:
                return values
            else:
                print("Error: Please enter exactly 4 values in format [x1, x2, x3, x4]")
                return None
        else:
            print("Error: Please use the format [x1, x2, x3, x4]")
            return None
    except ValueError:
        print("Error: Please enter valid numeric values")
        return None

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    HOST = "192.168.0.201"  # Check your desktop IP address
    PORT = 57105
    server.bind((HOST, PORT))
    server.listen()
    print(f"Listening on {HOST}:{PORT}")
 
    try:
        while True:
            print("\nWaiting for connection...")
            conn, addr = server.accept()
            print(f"Connection established with {addr}")
     
            try:
                while True:
                    # Get values from terminal
                    values = get_input_values() # [50, 1.8, -50, 0.4] 

                    
                    # If no values entered, continue without sending
                    if values is None:
                        continue
                    
                    # Prepare and send data
                    try:
                        data = struct.pack('!4f', *values)  # Big-endian format for 4 floats
                        conn.sendall(data)
                        print(f"Sent: {values}")
                    except struct.error as se:
                        print(f"Formatting error: {se}")
                    except Exception as e:
                        print(f"Error while sending: {e}")
                        break
           
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"Client disconnected: {e}")
            
            finally:
                conn.close()
                print("Connection closed")
    
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    finally:
        print("Server closed")