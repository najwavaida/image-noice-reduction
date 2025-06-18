import numpy as np
from PIL import Image

def add_salt_and_pepper_noise(img_array, amount=0.02):
    noisy_img = img_array.copy()
    num_pixels = img_array.size
    num_noisy = int(amount * num_pixels)

    # Tambahkan salt (putih)
    for _ in range(num_noisy // 2):
        x = np.random.randint(0, img_array.shape[0])
        y = np.random.randint(0, img_array.shape[1])
        noisy_img[x, y] = 255

    # Tambahkan pepper (hitam)
    for _ in range(num_noisy // 2):
        x = np.random.randint(0, img_array.shape[0])
        y = np.random.randint(0, img_array.shape[1])
        noisy_img[x, y] = 0

    return noisy_img

# Contoh penggunaan
img = Image.open('gambar.jpg')
img_array = np.array(img)
noisy = add_salt_and_pepper_noise(img_array)
Image.fromarray(noisy.astype(np.uint8)).save('noisy.jpg')
