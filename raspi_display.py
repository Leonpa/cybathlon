import tkinter as tk

root = tk.Tk()
root.title("Map Display")

canvas = tk.Canvas(root, width=480, height=320)
canvas.pack()

def draw_map(data):
    canvas.delete("all")
    for obj in data["map"]:
        canvas.create_rectangle(obj["x"], obj["y"], obj["x"] + 10, obj["y"] + 10, fill="blue")


received_data = {"map": [{"x": 50, "y": 50}, {"x": 100, "y": 100}]}
draw_map(received_data)

root.mainloop()
