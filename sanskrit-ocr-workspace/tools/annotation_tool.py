import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk

class AnnotationTool:
    def __init__(self, master):
        self.master = master
        self.master.title("Sanskrit Character Annotation Tool")
        
        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.load_button = tk.Button(master, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.char_entry = tk.Entry(master)
        self.char_entry.pack()

        self.save_button = tk.Button(master, text="Save Annotation", command=self.save_annotation)
        self.save_button.pack()

        self.annotations = []

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select Character Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.current_image = Image.open(file_path)
            self.display_image()

    def display_image(self):
        self.current_image.thumbnail((400, 400))
        self.tk_image = ImageTk.PhotoImage(self.current_image)
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image

    def save_annotation(self):
        character = self.char_entry.get()
        if character and hasattr(self, 'current_image'):
            annotation = {
                'image_path': self.current_image.filename,
                'character': character
            }
            self.annotations.append(annotation)
            self.char_entry.delete(0, tk.END)
            messagebox.showinfo("Success", "Annotation saved!")
        else:
            messagebox.showwarning("Warning", "Please load an image and enter a character.")

    def save_annotations_to_file(self, output_file):
        with open(output_file, 'w') as f:
            for annotation in self.annotations:
                f.write(f"{annotation['image_path']},{annotation['character']}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationTool(root)
    root.mainloop()