# image-noise-reduction
 
This program is a Python-based GUI application (using Tkinter) for reducing noise in images. Users can load a .jpg image, which is then converted to grayscale and binarized. The program detects noise and applies two noise reduction methods:

- Median Filter
- Morphological Opening

After processing, the results of both methods are displayed along with MSE (Mean Squared Error) and PSNR (Peak Signal-to-Noise Ratio) values to compare their effectiveness. Visual results are also shown so users can clearly see the differences before and after processing.

📁 Project Structure
- `noisy_code.py` — Add noise (e.g. salt & pepper) to an input image
- `noise_reduction_gui.py` — GUI to reduce noise using:
  - Median Filtering
  - Morphological Opening
- `batch_evaluation.py` — Evaluate and compare methods in batch
- `noisy.jpg` / `referensi.jpg` — Sample noisy and reference images
- `Jurnal.pdf` — Supporting report/document

✨ Features
- Load .jpg grayscale images
- Apply Otsu thresholding for binarization
- Detect isolated noise pixels
- Perform noise reduction with:
  - Median Filtering
  - Morphological Opening
- Compare results using MSE and PSNR
- Visualize all steps side-by-side

🛠️ Requirements
- Python 3.x
- numpy
- Pillow
- tkinter (usually pre-installed)

🚀 How to Use
1. Generate Noisy Image
'''bash
python noisy_code.py
'''
2. Open GUI for Noise Reduction
'''bash
python noise_reduction_gui.py
'''
3. Run Batch Evaluation
'''bash
python batch_evaluation.py
'''

📝 Note
- Make sure the image is in grayscale or .jpg format
- Results include image previews and MSE/PSNR comparison
