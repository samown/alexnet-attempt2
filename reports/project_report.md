# Laporan Proyek: AlexNet & iFood 2019 Challenge

## 1. Ringkasan Eksekutif

Proyek ini mengimplementasikan dan memodifikasi arsitektur AlexNet untuk klasifikasi fine-grained makanan pada dataset iFood 2019. Kami melakukan empat eksperimen untuk membandingkan performa baseline AlexNet dengan modifikasi menggunakan Batch Normalization dan LeakyReLU activation.

## 2. Ringkasan Paper AlexNet

*(Ringkasan lengkap dapat dilihat di [paper_summary.md](paper_summary.md))*

### 2.1 Motivasi
Paper "ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky et al., 2012) merupakan breakthrough dalam deep learning yang membuktikan bahwa deep convolutional neural networks dapat mengungguli metode tradisional dalam klasifikasi gambar skala besar.

### 2.2 Arsitektur
- 5 Convolutional layers
- 3 Fully Connected layers
- ReLU activation
- Dropout (0.5) untuk regularisasi
- Output: 1000 classes (ImageNet)

### 2.3 Teknik Inovatif
- **ReLU Activation**: Menggantikan tanh/sigmoid, training 6x lebih cepat
- **Dropout**: Mencegah overfitting dengan randomly disabling neurons
- **Data Augmentation**: Rotation, flipping, color jittering
- **Multi-GPU Training**: Parallel training across 2 GPUs

### 2.4 Hasil
- Top-5 Error Rate: 15.3% (baseline: ~26%)
- Juara 1 ImageNet 2012 Competition
- Membuka era deep learning modern

## 3. Metodologi

### 3.1 Dataset: iFood 2019
- **Jumlah kelas**: 251 fine-grained food categories
- **Training images**: ~120,000 images
- **Validation images**: ~12,000 images
- **Test images**: ~28,000 images
- **Tantangan**: Class imbalance, fine-grained classification, noisy labels

### 3.2 Data Preprocessing

#### 3.2.1 Data Augmentation
- **Training**: Random rotation (±15°), horizontal flip, color jitter, random crop (224x224)
- **Validation/Test**: Resize to 224x224, normalization only
- **Normalization**: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]

#### 3.2.2 Class Imbalance Handling
- Analisis distribusi kelas menunjukkan imbalance ratio tinggi
- Implementasi class weighting dalam loss function menggunakan metode 'balanced'
- Formula: `weight = n_samples / (n_classes * class_count)`

### 3.3 Model Architectures

#### 3.3.1 Baseline AlexNet
- Arsitektur original AlexNet
- 5 Conv layers + 3 FC layers
- ReLU activation
- Dropout 0.5 di FC layers
- Output: 251 classes

#### 3.3.2 Modified 1: Batch Normalization
- Menambahkan Batch Normalization setelah setiap conv dan FC layer
- Meningkatkan training stability dan convergence speed
- Mengurangi internal covariate shift

#### 3.3.3 Modified 2: LeakyReLU
- Mengganti ReLU dengan LeakyReLU (negative_slope=0.01)
- Mencegah dying ReLU problem
- Memungkinkan gradient flow untuk negative values

#### 3.3.4 Combined: BN + LeakyReLU
- Kombinasi Batch Normalization dan LeakyReLU
- Menggabungkan keuntungan kedua modifikasi

### 3.4 Training Configuration

#### 3.4.1 Hyperparameters
- **Optimizer**: SGD dengan momentum 0.9
- **Learning rate**: 0.001
- **Learning rate scheduler**: StepLR (step_size=30, gamma=0.1)
- **Batch size**: 64
- **Epochs**: 50
- **Weight decay**: 0.0005
- **Early stopping**: Patience=10, min_delta=0.001

#### 3.4.2 Experiment Tracking
- Menggunakan Weights & Biases (wandb) untuk logging metrics
- Track: train/val loss, train/val accuracy, learning rate
- Save checkpoints untuk setiap epoch
- Save best model berdasarkan validation accuracy

## 4. Hasil Eksperimen

### 4.1 Eksperimen A: Baseline AlexNet

**Konfigurasi:**
- Model: AlexNet baseline
- Activation: ReLU
- Regularization: Dropout 0.5

**Hasil:**
- Best Train Accuracy: [TO BE FILLED]
- Best Val Accuracy: [TO BE FILLED]
- Test Accuracy: [TO BE FILLED]
- Training Time: [TO BE FILLED]

**Observasi:**
- [Observations about training dynamics, convergence, etc.]

### 4.2 Eksperimen B: AlexNet Modified 1 (Batch Normalization)

**Konfigurasi:**
- Model: AlexNet dengan Batch Normalization
- Activation: ReLU
- Regularization: Dropout 0.5 + Batch Normalization

**Hasil:**
- Best Train Accuracy: [TO BE FILLED]
- Best Val Accuracy: [TO BE FILLED]
- Test Accuracy: [TO BE FILLED]
- Training Time: [TO BE FILLED]

**Observasi:**
- [Observations about impact of Batch Normalization]

### 4.3 Eksperimen C: AlexNet Modified 2 (LeakyReLU)

**Konfigurasi:**
- Model: AlexNet dengan LeakyReLU
- Activation: LeakyReLU (negative_slope=0.01)
- Regularization: Dropout 0.5

**Hasil:**
- Best Train Accuracy: [TO BE FILLED]
- Best Val Accuracy: [TO BE FILLED]
- Test Accuracy: [TO BE FILLED]
- Training Time: [TO BE FILLED]

**Observasi:**
- [Observations about impact of LeakyReLU]

### 4.4 Eksperimen D: AlexNet Combined (BN + LeakyReLU)

**Konfigurasi:**
- Model: AlexNet dengan Batch Normalization + LeakyReLU
- Activation: LeakyReLU (negative_slope=0.01)
- Regularization: Dropout 0.5 + Batch Normalization

**Hasil:**
- Best Train Accuracy: [TO BE FILLED]
- Best Val Accuracy: [TO BE FILLED]
- Test Accuracy: [TO BE FILLED]
- Training Time: [TO BE FILLED]

**Observasi:**
- [Observations about combined modifications]

## 5. Analisis Hasil

### 5.1 Perbandingan Performa

| Model | Train Acc (%) | Val Acc (%) | Test Acc (%) | Training Time |
|-------|----------------|-------------|--------------|---------------|
| Baseline | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| Modified 1 (BN) | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| Modified 2 (LeakyReLU) | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| Combined | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |

### 5.2 Analisis Training Curves

[Include plots of training/validation curves for all experiments]

**Key Observations:**
- [Analysis of convergence speed]
- [Analysis of overfitting/underfitting]
- [Comparison of training stability]

### 5.3 Confusion Matrix Analysis

[Include confusion matrices for all models]

**Key Observations:**
- [Analysis of class-wise performance]
- [Identification of confusing classes]
- [Patterns in misclassifications]

### 5.4 Impact of Modifications

**Batch Normalization:**
- [Impact on training speed]
- [Impact on final accuracy]
- [Impact on training stability]

**LeakyReLU:**
- [Impact on gradient flow]
- [Impact on final accuracy]
- [Comparison with ReLU]

**Combined Modifications:**
- [Synergistic effects]
- [Overall improvement]

## 6. Kesimpulan

### 6.1 Model Terbaik
[Identify best performing model based on test accuracy and other metrics]

### 6.2 Key Findings
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

### 6.3 Lessons Learned
- [Lesson 1]
- [Lesson 2]
- [Lesson 3]

### 6.4 Future Work
- [Suggestion 1]
- [Suggestion 2]
- [Suggestion 3]

## 7. Challenges & Solutions

### 7.1 Challenges Encountered
1. **Class Imbalance**: Dataset memiliki distribusi kelas yang tidak seimbang
   - **Solution**: Implementasi class weighting dalam loss function

2. **Training Time**: Training model membutuhkan waktu lama
   - **Solution**: Optimasi batch size dan penggunaan GPU

3. **Memory Constraints**: Model besar membutuhkan banyak memory
   - **Solution**: Gradient accumulation atau mengurangi batch size

### 7.2 Technical Challenges
- [Any other technical challenges and solutions]

## 8. Reproducibility

### 8.1 Environment Setup
Semua dependencies tercantum di `requirements.txt` dan `environment.yml`.

### 8.2 Random Seeds
Random seed diset ke 42 untuk reproducibility.

### 8.3 Configuration
Semua hyperparameters tercantum di `src/config/config.yaml`.

## 9. Referensi

1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25.

2. iFood 2019 Challenge: https://www.kaggle.com/c/ifood-2019-fgvc6

3. iFood 2019 Dataset: https://github.com/karansikka1/iFood_2019

## 10. Appendix

### 10.1 Code Structure
```
alexnet-and-ifood-2019-challenge-kelompok/
├── src/
│   ├── models/          # Model implementations
│   ├── data/            # Data loading and preprocessing
│   ├── training/        # Training and evaluation utilities
│   └── config/          # Configuration files
├── notebooks/           # Jupyter notebooks for experiments
├── scripts/             # Utility scripts
└── reports/             # Project reports
```

### 10.2 Hyperparameter Details
[Detailed hyperparameter settings for each experiment]

### 10.3 Additional Visualizations
[Any additional plots, graphs, or visualizations]

