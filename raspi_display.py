import tkinter as tk

def draw_map(data, canvas):
    # Clear previous drawings
    canvas.delete("all")
    # Draw new map
    # Replace with your actual drawing logic
    for obj in data["map"]:
        canvas.create_rectangle(obj["x"], obj["y"], obj["x"] + 10, obj["y"] + 10, fill="blue")

# Initialize Tkinter window
root = tk.Tk()
root.title("Map Display")

# Make the window fullscreen
root.attributes("-fullscreen", True)
root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))  # Pressing Escape will exit fullscreen

# Create and pack the canvas to fill the entire window
canvas = tk.Canvas(root)
canvas.pack(expand=True, fill='both')

# Simulated received data
received_data = {"map": [{"x": 50, "y": 50}, {"x": 100, "y": 100}]}
draw_map(received_data, canvas)

# Start the Tkinter main loop
root.mainloop()
