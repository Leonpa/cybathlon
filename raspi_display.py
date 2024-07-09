import tkinter as tk
import threading
import json
from bluedot.btcomm import BluetoothServer


def draw_map(data, canvas):
    # Clear previous drawings
    canvas.delete("all")
    # Draw new map
    for obj in data["map"]:
        x = obj["x"]
        y = obj["y"]
        shape = obj["shape"]
        text = obj["text"]
        color = "blue"

        if shape == "square":
            canvas.create_rectangle(x, y, x + 20, y + 20, fill=color)
        elif shape == "round":
            canvas.create_oval(x, y, x + 20, y + 20, fill=color)

        canvas.create_text(x + 25, y + 10, text=text, anchor=tk.W)


def close_application(event=None):
    root.destroy()


def scan_action():
    print("Scan button pressed")


def data_received(data):
    # Split the incoming data by newline character
    messages = data.split('\n')
    for message in messages:
        if message.strip():  # Ensure that the message is not empty
            try:
                parsed_data = json.loads(message)
                print("Received data:", parsed_data)
                x, y = parsed_data["center"]
                label = parsed_data["label"]
                obj = {"x": x, "y": y, "shape": "square", "text": label}
                received_data["map"].append(obj)
                draw_map(received_data, canvas)
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON message: {e}")


def server_thread():
    server = BluetoothServer(data_received, port=1)
    print("Bluetooth server started")


# Initialize Tkinter window
root = tk.Tk()
root.title("Map Display")

# Make the window fullscreen
root.attributes("-fullscreen", True)

# Bind the Escape key to close the application
root.bind("<Escape>", close_application)

# Create the main frame
frame = tk.Frame(root)
frame.pack(expand=True, fill='both')

# Create the canvas
canvas = tk.Canvas(frame)
canvas.pack(expand=True, fill='both')

# Add the close button in the upper right corner
close_button = tk.Button(root, text="X", command=close_application)
close_button.place(relx=1.0, rely=0.0, anchor='ne')

# Add the scan button at the bottom center
scan_button = tk.Button(root, text="Scan", command=scan_action)
scan_button.place(relx=0.5, rely=1.0, anchor='s')

# Received data
received_data = {
    "map": []
}

# Start the server thread
threading.Thread(target=server_thread, daemon=True).start()

# Start the Tkinter main loop
root.mainloop()
