import tkinter as tk

def draw_map(data, canvas):
    # Clear previous drawings
    canvas.delete("all")
    # Draw new map
    # Replace with your actual drawing logic
    for obj in data["map"]:
        canvas.create_rectangle(obj["x"], obj["y"], obj["x"] + 10, obj["y"] + 10, fill="blue")

def close_application(event=None):
    root.destroy()

def scan_action():
    # Placeholder for the scan button action
    print("Scan button pressed")

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

# Simulated received data
received_data = {"map": [{"x": 50, "y": 50}, {"x": 100, "y": 100}]}
draw_map(received_data, canvas)

# Start the Tkinter main loop
root.mainloop()
