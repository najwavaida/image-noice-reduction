import os
import numpy as np
from PIL import Image
import pandas as pd
import time

def load_image(path):
    return np.array(Image.open(path).convert('L'))

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
    return np.where(img_gray > threshold, 255, 0).astype(np.uint8)

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
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def psnr(img1, img2):
    err = mse(img1, img2)
    if err == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(err))

# === Main Batch Evaluation ===
folder = "gambar_batch"
os.makedirs(folder, exist_ok=True)

# Simulasi data
for i in range(1, 6):
    base = np.tile(np.linspace(100, 200, 200), (200, 1)).astype(np.uint8)
    Image.fromarray(base).save(f"{folder}/referensi{i}.jpg")
    noisy = base.copy()
    coords = np.random.randint(0, 200, (100, 2))
    for x, y in coords:
        noisy[x, y] = 0 if i % 2 == 0 else 255
    Image.fromarray(noisy).save(f"{folder}/noisy{i}.jpg")

# Evaluasi batch
results = []
for i in range(1, 6):
    noisy_img = load_image(f"{folder}/noisy{i}.jpg")
    ref_img = load_image(f"{folder}/referensi{i}.jpg")
    bin_img = binarize(noisy_img)

    t1 = time.time()
    median_result = median_filter(bin_img, 3)
    t2 = time.time()
    morph_result = morphology_opening(bin_img, 3)
    t3 = time.time()

    results.append({
        "image": f"noisy{i}.jpg",
        "mse_median": mse(ref_img, median_result),
        "psnr_median": psnr(ref_img, median_result),
        "time_median": t2 - t1,
        "mse_morph": mse(ref_img, morph_result),
        "psnr_morph": psnr(ref_img, morph_result),
        "time_morph": t3 - t2
    })

df = pd.DataFrame(results)
df.to_excel("hasil_evaluasi_batch.xlsx", index=False)
print("Evaluasi selesai! Hasil disimpan ke hasil_evaluasi_batch.xlsx")
