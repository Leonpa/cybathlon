import subprocess

def pair_device(address):
    print(f"Pairing with {address}...")
    try:
        result = subprocess.run(['/usr/bin/bluetoothctl', 'pair', address], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        print(result.stderr)
        if "Pairing successful" in result.stdout or "org.bluez.Error.AlreadyExists" in result.stdout or "org.bluez.Error.AlreadyExists" in result.stderr:
            subprocess.run(['/usr/bin/bluetoothctl', 'trust', address], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            subprocess.run(['/usr/bin/bluetoothctl', 'connect', address], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"Successfully paired with {address}")
            return True
        else:
            print(f"Failed to pair with {address}")
            return False
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace with the actual Bluetooth address you want to pair with
SERVER_BLUETOOTH_ADDRESS = "B8:27:EB:D1:35:D4"

pair_device(SERVER_BLUETOOTH_ADDRESS)
