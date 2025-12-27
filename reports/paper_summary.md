# Ringkasan Paper: ImageNet Classification with Deep Convolutional Neural Networks

**Penulis:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
**Tahun:** 2012  
**Konferensi:** NIPS 2012 (Neural Information Processing Systems)

## 1. Motivasi Penelitian

### 1.1 Latar Belakang
- Sebelum AlexNet, sistem klasifikasi gambar skala besar masih menggunakan metode tradisional yang terbatas
- Dataset ImageNet dengan jutaan gambar dan ribuan kelas membutuhkan pendekatan baru
- Komputasi GPU yang semakin powerful memungkinkan training deep neural networks yang lebih kompleks

### 1.2 Tujuan
- Mengembangkan deep convolutional neural network yang dapat mengklasifikasikan 1.2 juta gambar ImageNet ke dalam 1000 kategori
- Mencapai error rate yang lebih rendah dibandingkan metode state-of-the-art saat itu
- Membuktikan bahwa deep learning dapat mengungguli metode tradisional dalam computer vision

## 2. Arsitektur AlexNet

### 2.1 Struktur Umum
AlexNet terdiri dari **8 layer** yang dapat dibagi menjadi:
- **5 Convolutional Layers** (Conv1-Conv5)
- **3 Fully Connected Layers** (FC6-FC7-FC8)

### 2.2 Detail Arsitektur

#### Layer Convolutional

**Conv1:**
- Input: 224×224×3 (RGB image)
- Filters: 96 kernels of size 11×11×3
- Stride: 4 pixels
- Output: 55×55×96
- Activation: ReLU
- Local Response Normalization (LRN)
- Max Pooling: 3×3, stride 2 → 27×27×96

**Conv2:**
- Input: 27×27×96
- Filters: 256 kernels of size 5×5×96
- Padding: 2
- Output: 27×27×256
- Activation: ReLU
- Local Response Normalization (LRN)
- Max Pooling: 3×3, stride 2 → 13×13×256

**Conv3:**
- Input: 13×13×256
- Filters: 384 kernels of size 3×3×256
- Padding: 1
- Output: 13×13×384
- Activation: ReLU
- No pooling

**Conv4:**
- Input: 13×13×384
- Filters: 384 kernels of size 3×3×384
- Padding: 1
- Output: 13×13×384
- Activation: ReLU
- No pooling

**Conv5:**
- Input: 13×13×384
- Filters: 256 kernels of size 3×3×384
- Padding: 1
- Output: 13×13×256
- Activation: ReLU
- Max Pooling: 3×3, stride 2 → 6×6×256

#### Layer Fully Connected

**FC6:**
- Input: 6×6×256 = 9216 neurons
- Output: 4096 neurons
- Activation: ReLU
- Dropout: 0.5

**FC7:**
- Input: 4096 neurons
- Output: 4096 neurons
- Activation: ReLU
- Dropout: 0.5

**FC8 (Output):**
- Input: 4096 neurons
- Output: 1000 neurons (1000 classes ImageNet)
- Softmax activation

### 2.3 Karakteristik Kunci
- **Total Parameters:** ~60 juta parameters
- **Input Size:** 224×224×3 RGB images
- **Output:** 1000 class probabilities

## 3. Teknik Inovatif

### 3.1 ReLU (Rectified Linear Unit) Activation
- **Inovasi:** Menggantikan tanh/sigmoid dengan ReLU: f(x) = max(0, x)
- **Keuntungan:**
  - Training lebih cepat (6x lebih cepat dibanding tanh)
  - Mengurangi vanishing gradient problem
  - Komputasi lebih efisien
- **Dampak:** Menjadi standar activation function untuk deep learning modern

### 3.2 Dropout Regularization
- **Teknik:** Randomly set 50% neurons ke 0 selama training (hanya di FC layers)
- **Tujuan:** Mencegah overfitting dengan mengurangi co-adaptation antar neurons
- **Hasil:** Mengurangi error rate secara signifikan

### 3.3 Data Augmentation
- **Horizontal Flipping:** Random horizontal flip
- **Color Jittering:** Mengubah intensitas RGB channels
- **PCA-based Color Augmentation:** Menambahkan noise berdasarkan principal components
- **Tujuan:** Meningkatkan generalisasi dan mengurangi overfitting

### 3.4 Local Response Normalization (LRN)
- Normalisasi lokal antar adjacent feature maps
- Meningkatkan generalisasi (meskipun kemudian digantikan oleh Batch Normalization)

### 3.5 Multi-GPU Training
- **Inovasi:** Split model training across 2 GPUs (GTX 580)
- **Metode:** 
  - GPUs berkomunikasi hanya pada specific layers (Conv2, Conv4, Conv5)
  - Parallel training dengan gradient synchronization
- **Hasil:** Training time berkurang drastis, memungkinkan training model yang lebih besar

### 3.6 Overlapping Pooling
- Max pooling dengan stride < kernel size menghasilkan overlapping
- Mengurangi top-1 error rate sebesar 0.4% dan top-5 error rate sebesar 0.3%

## 4. Hasil Eksperimen

### 4.1 ImageNet 2012 Competition
- **Top-5 Error Rate:** 15.3% (baseline: ~26%)
- **Top-1 Error Rate:** 37.5%
- **Peringkat:** Juara 1 (mengalahkan runner-up dengan margin 10.9%)

### 4.2 Analisis Komponen
- **ReLU vs Tanh:** ReLU mengurangi error rate dari 18.2% ke 15.3%
- **Multi-GPU:** Meningkatkan training speed tanpa mengurangi accuracy
- **Dropout:** Mengurangi overfitting secara signifikan
- **Data Augmentation:** Meningkatkan generalisasi model

### 4.3 Visualisasi
- Paper menunjukkan bahwa layer pertama belajar edge detectors dan color blobs
- Layer-layers berikutnya belajar texture dan object parts
- Layer terakhir menunjukkan high-level semantic features

## 5. Dampak Penelitian

### 5.1 Revolusi Deep Learning
- **Membuka Era Deep Learning:** AlexNet membuktikan bahwa deep CNNs dapat bekerja dengan baik pada dataset skala besar
- **GPU Computing:** Memopulerkan penggunaan GPU untuk deep learning training
- **Computer Vision:** Mengubah paradigma dari hand-crafted features ke learned features

### 5.2 Pengaruh Jangka Panjang
- **Arsitektur Modern:** Menjadi fondasi untuk VGG, ResNet, dan arsitektur modern lainnya
- **Transfer Learning:** Pre-trained AlexNet digunakan untuk berbagai tugas computer vision
- **Industri:** Memicu investasi besar-besaran dalam AI dan deep learning

### 5.3 Teknologi yang Diterapkan
- ReLU menjadi standar activation function
- Dropout menjadi teknik regularisasi yang umum
- Data augmentation menjadi best practice
- Multi-GPU training menjadi standar untuk model besar

## 6. Kesimpulan

AlexNet merupakan breakthrough yang mengubah landscape computer vision dan deep learning. Dengan kombinasi teknik inovatif (ReLU, dropout, data augmentation, multi-GPU training) dan arsitektur yang tepat, paper ini membuktikan bahwa deep convolutional neural networks dapat mengungguli metode tradisional dalam klasifikasi gambar skala besar. Dampaknya masih terasa hingga hari ini, dengan teknik-teknik yang diperkenalkan menjadi standar dalam deep learning modern.

## Referensi

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25.



