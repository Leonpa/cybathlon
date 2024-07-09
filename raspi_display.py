import tkinter as tk
import threading
import json
from bluedot.btcomm import BluetoothServer
import time

# Define categories that should get a circle
CIRCLE_CATEGORIES = ["cat_2-hard", "cat_3-soft"]

# Theoretical dimensions
THEORETICAL_WIDTH = 1600
THEORETICAL_HEIGHT = 1200

# Actual display dimensions (adjust these values as per your 3.5-inch display resolution)
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# Scaling factors
SCALE_X = DISPLAY_WIDTH / THEORETICAL_WIDTH
SCALE_Y = DISPLAY_HEIGHT / THEORETICAL_HEIGHT

# Label mapping
LABEL_MAPPING = {
    "cat_2-hard": "Hard Cylinder",
    "cat_3-soft": "Soft Cylinder",
    "background-soft": "Soft Cube",
    "background-hard": "Hard Cube"
}


def draw_map(data, canvas):
    # Clear previous drawings
    canvas.delete("all")

    # Draw the theoretical bounding box
    canvas.create_rectangle(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT, outline="black", width=6)

    # Draw new map
    for obj in data["map"]:
        x = int(obj["x"] * SCALE_X)
        y = int(obj["y"] * SCALE_Y)
        shape = obj["shape"]
        text = obj["text"]
        color = obj["color"]

        if shape == "square":
            canvas.create_rectangle(x, y, x + 30 * SCALE_X, y + 30 * SCALE_Y, fill=color)
        elif shape == "round":
            canvas.create_oval(x, y, x + 30 * SCALE_X, y + 30 * SCALE_Y, fill=color)

        canvas.create_text(x + 25 * SCALE_X, y + 10 * SCALE_Y, text=text, anchor=tk.W)


def close_application(event=None):
    root.destroy()


def scan_action():
    print("Scan button pressed")


def data_received(data):
    global last_processed_time
    # Clear previous data
    received_data["map"].clear()
    # Split the incoming data by newline character
    messages = data.split('\n')
    for message in messages:
        if message.strip():  # Ensure that the message is not empty
            try:
                parsed_data = json.loads(message)
                print(f"Received data at {time.time()}: {parsed_data}")
                x, y = parsed_data["center"]
                label = parsed_data["label"]

                # Map label to display name
                display_name = LABEL_MAPPING.get(label, label)

                # Determine shape and color based on category
                shape = "round" if label in CIRCLE_CATEGORIES else "square"
                color = "red" if "-soft" in label else "green"

                obj = {"x": x, "y": y, "shape": shape, "text": display_name, "color": color}
                received_data["map"].append(obj)
                last_processed_time = time.time()
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON message: {e}")

    draw_map(received_data, canvas)


def server_thread():
    server = BluetoothServer(data_received, port=1)
    print("Bluetooth server started")
    while True:
        time.sleep(0.1)  # Adjust sleep time as necessary


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

# Create the canvas with the actual display dimensions
canvas = tk.Canvas(frame, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
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

last_processed_time = time.time()

# Start the server thread
threading.Thread(target=server_thread, daemon=True).start()

# Start the Tkinter main loop
root.mainloop()
