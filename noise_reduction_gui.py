import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import time

def binarize(img_gray):
    hist, _ = np.histogram(img_gray.flatten(), bins=256, range=[0,256])
    total = img_gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, w_f, var_max, threshold = 0, 0, 0, 0, 0
    for i in range(256):
        w_b += hist[i]
        if w_b == 0: continue
        w_f = total - w_b
        if w_f == 0: break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f)**2
        if var_between > var_max:
            var_max = var_between
            threshold = i
    binary = np.where(img_gray > threshold, 255, 0).astype(np.uint8)
    return binary, threshold

def identify_noise(binary_img):
    padded = np.pad(binary_img, 1, mode='edge')
    noise_img = binary_img.copy()
    rows, cols = binary_img.shape
    for i in range(rows):
        for j in range(cols):
            center = padded[i+1, j+1]
            neighbors = padded[i:i+3, j:j+3].flatten()
            neighbors = np.delete(neighbors, 4)
            if all(center != neighbors):
                noise_img[i, j] = 128
    return noise_img

def median_filter(img, k=3):
    pad = k // 2
    padded = np.pad(img, pad, mode='edge')
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j] = np.median(padded[i:i+k, j:j+k])
    return output

def erosion(img, k=3):
    pad = k // 2
    padded = np.pad(img, pad, mode='constant', constant_values=255)
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j] = np.min(padded[i:i+k, j:j+k])
    return output

def dilation(img, k=3):
    pad = k // 2
    padded = np.pad(img, pad, mode='constant', constant_values=0)
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j] = np.max(padded[i:i+k, j:j+k])
    return output

def morphology_opening(img, k=3):
    return dilation(erosion(img, k), k)

def mse(img1, img2):
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

def psnr(img1, img2):
    err = mse(img1, img2)
    if err == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(err))

class NoiseReductionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Noise Reduction GUI")

        self.image_label = tk.Label(root, text="No image selected")
        self.image_label.pack()

        self.open_btn = tk.Button(root, text="Open Image (.jpg)", command=self.open_image)
        self.open_btn.pack()

        self.kernel_label = tk.Label(root, text="Select Window Size:")
        self.kernel_label.pack()

        self.kernel_combo = ttk.Combobox(root, values=[3, 5, 7])
        self.kernel_combo.current(0)
        self.kernel_combo.pack()

        self.process_btn = tk.Button(root, text="Process", command=self.process_image)
        self.process_btn.pack()

        self.result_text = tk.Text(root, height=12, width=60)
        self.result_text.pack()

        self.frame_results = tk.Frame(root)
        self.frame_results.pack()

        self.label_median = tk.Label(self.frame_results, text="Median Filtering Result")
        self.label_median.grid(row=0, column=0, padx=10, pady=10)

        self.label_morph = tk.Label(self.frame_results, text="Morphology Opening Result")
        self.label_morph.grid(row=0, column=1, padx=10, pady=10)

        self.image_array = None
        self.original_image = None
        self.ref_array = None

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("JPG Files", "*.jpg")])
        if path:
            self.original_image = Image.open(path)
            img_gray = self.original_image.convert('L')
            self.image_array = np.array(img_gray)
            img_thumbnail = img_gray.resize((200, 200))
            self.tk_img_input = ImageTk.PhotoImage(img_thumbnail)
            self.image_label.config(image=self.tk_img_input, text="")

            self.ref_array = self.image_array.copy()

    def tampilkan_hasil(self, array, label):
        img = Image.fromarray(array.astype('uint8'))
        img = img.resize((200, 200))
        tk_img = ImageTk.PhotoImage(img)
        label.config(image=tk_img)
        label.image = tk_img

    def process_image(self):
        if self.image_array is None:
            return

        k = int(self.kernel_combo.get())
        binary_img, threshold = binarize(self.image_array)

        noise_identified = identify_noise(binary_img)

        start1 = time.time()
        median_result = median_filter(binary_img, k)
        end1 = time.time()

        start2 = time.time()
        morph_result = morphology_opening(binary_img, k)
        end2 = time.time()

        mse_median = mse(self.ref_array, median_result)
        psnr_median = psnr(self.ref_array, median_result)

        mse_morph = mse(self.ref_array, morph_result)
        psnr_morph = psnr(self.ref_array, morph_result)

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END,
            f"[Median Filtering]\nMSE: {mse_median:.2f}, PSNR: {psnr_median:.2f} dB\nTime: {end1 - start1:.3f} s\n\n"
            f"[Morphology Opening]\nMSE: {mse_morph:.2f}, PSNR: {psnr_morph:.2f} dB\nTime: {end2 - start2:.3f} s")

        self.tampilkan_hasil(median_result, self.label_median)
        self.tampilkan_hasil(morph_result, self.label_morph)
        
        print("\n=== Evaluasi Metode Reduksi Noise ===")
        if mse_median < mse_morph and psnr_median > psnr_morph:
            print("Metode Median Filtering lebih efektif untuk mereduksi noise.")
        elif mse_morph < mse_median and psnr_morph > psnr_median:
            print("Metode Morphology Opening lebih efektif untuk mereduksi noise.")
        else:
            print("Hasil evaluasi metode tidak konsisten, silakan cek hasil secara visual dan nilai MSE/PSNR.")

        self.show_result_window(binary_img, noise_identified, median_result, morph_result, threshold)

    def show_result_window(self, binary_img, noise_img, median_result, morph_result, threshold):
        win = tk.Toplevel(self.root)
        win.title("All Result Images")

        images = [
            (self.original_image, "Original Image"),
            (self.original_image.convert('L'), "Grayscale Image"),
            (Image.fromarray(binary_img), f"Binary Image (Threshold={threshold})"),
            (Image.fromarray(noise_img), "Binary with Noise Identification"),
            (Image.fromarray(median_result), "Median Filtering Result"),
            (Image.fromarray(morph_result), "Morphology Opening Result"),
        ]

        self.result_labels = []
        self.result_pil_images = []
        self.result_tk_images = []

        for idx, (img, title) in enumerate(images):
            row = idx // 3
            col = idx % 3
            tk.Label(win, text=title).grid(row=row*2, column=col, padx=10, pady=5)

            lbl = tk.Label(win)
            lbl.grid(row=row*2 + 1, column=col, padx=10, pady=5)
            self.result_labels.append(lbl)
            self.result_pil_images.append(img)

        def resize_and_update(event=None):
            win.update_idletasks()
            max_width, max_height = 600, 600

            width = max((win.winfo_width() - 60) // 3, 50)
            height = max((win.winfo_height() - 120) // 2, 70)

            width = min(width, max_width)
            height = min(height, max_height)

            self.result_tk_images.clear()
            for i, pil_img in enumerate(self.result_pil_images):
                pil_w, pil_h = pil_img.size
                ratio = min(width / pil_w, height / pil_h)
                new_w = int(pil_w * ratio)
                new_h = int(pil_h * ratio)

                img_resized = pil_img.resize((new_w, new_h))
                tk_img = ImageTk.PhotoImage(img_resized)
                self.result_tk_images.append(tk_img)
                self.result_labels[i].config(image=tk_img)
                self.result_labels[i].image = tk_img

        win.bind("<Configure>", resize_and_update)
        win.update_idletasks()
        resize_and_update()

if __name__ == "__main__":
    root = tk.Tk()
    app = NoiseReductionGUI(root)
    root.mainloop()
