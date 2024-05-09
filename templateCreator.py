import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageDraw

class PaintBox:
    def __init__(self, root):
        self.root = root
        self.root.title("Austrian painter")

        self.canvas = tk.Canvas(self.root, width=50, height=50, bg='white')
        self.canvas.pack()

        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.save_button = ttk.Button(self.toolbar, text="Save", command=self.save_image)
        self.save_button.pack(side=tk.LEFT)

        self.clear_button = ttk.Button(self.toolbar, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.close_button = ttk.Button(self.toolbar, text="Close", command=self.close_window)
        self.close_button.pack(side=tk.LEFT)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("RGB", (50, 50), "white")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x = event.x
        y = event.y
        self.canvas.create_rectangle(x, y, x + 2, y + 2, fill='black')  # Draw a small rectangle
        self.draw.rectangle([x, y, x + 2, y + 2], fill='black')  # Draw on the image

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            resized_image = self.image.resize((28, 28))
            resized_image.save(file_path)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (50, 50), "white")
        self.draw = ImageDraw.Draw(self.image)

    def close_window(self):
        self.root.destroy()

root = tk.Tk()
app = PaintBox(root)
root.mainloop()

