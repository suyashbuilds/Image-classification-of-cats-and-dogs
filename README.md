# 🐶🐱 Cat vs Dog Image Classifier using CNN

## 📌 Project Overview
This project builds a **Convolutional Neural Network (CNN)** from scratch to classify images of **cats and dogs**. Using the Kaggle Dogs vs. Cats (filtered 25K) dataset, the model is trained on labeled image data and includes a **Gradio interface** for live predictions using uploaded images or webcam input.

---

## ⚙ Technologies Used
- Python  
- TensorFlow / Keras  
- OpenCV  
- Matplotlib  
- Gradio  
- Kaggle API  
- Google Colab

---

## 📁 Dataset
The dataset used is:  
**[Dogs vs. Cats - filtered 25K on Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats)**

- Total: ~25,000 images  
- Balanced classes of cat and dog images  
- Split into `/train` and `/test` folders  

---

## 🚀 How It Works

1. **Download Dataset**:  
   Uses Kaggle API to download and unzip dataset

2. **Data Loading**:  
   Images are loaded using `image_dataset_from_directory` with image resizing to `256x256`

3. **Preprocessing**:
   - Normalized pixel values to [0, 1]
   - Batched and shuffled datasets

4. **CNN Model Architecture**:
   - `Conv2D → BatchNorm → MaxPooling` (×3)
   - `Flatten → Dense(128) → Dropout`
   - `Dense(64) → Dropout → Output layer (sigmoid)`

5. **Training**:
   - Optimizer: `Adam`
   - Loss: `Binary Crossentropy`
   - Accuracy metric tracked over 10–25 epochs

6. **Evaluation**:
   - Accuracy and loss plotted using Matplotlib
   - Model performance visualized over epochs

7. **Prediction on Custom Images**:
   - OpenCV used to read and resize custom images
   - Output is a confidence score with a class prediction

8. **Gradio Interface (Optional)**:
   - Upload or webcam input
   - Displays prediction and confidence score

---

## 📊 Results

- Achieved ~85–90% accuracy with proper training
- Consistent results on both training and validation sets
- Can detect dogs and cats from custom images

---

## 🖼 Example Output

Prediction:  
**Dog (Prediction Score: 0.9876)**  
**Cat (Prediction Score: 0.0211)**  

---

## ✅ Requirements

Install required libraries using:

```bash
pip install tensorflow opencv-python matplotlib gradio numpy
```

---

## 📂 How to Run

1. Upload your `kaggle.json` to authenticate with Kaggle  
2. Run the notebook step by step in **Google Colab**  
3. Use the last section to test on custom images or launch the Gradio app

---

## 🧑‍💻 Developed By

**Suyash Srivastava**  
Student at United College of Engineering and Research, Naini, Prayagraj


