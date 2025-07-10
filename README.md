# 📊 Synthetic Stock Data Generator using Variational Autoencoder (VAE)

This project uses a **Variational Autoencoder (VAE)** to generate synthetic Apple stock data (1980–2020). It helps simulate realistic financial data when original datasets are limited or sensitive.

## 🧠 Project Summary

- **Model Used**: VAE with TensorFlow
- **Data**: AAPL stock (Open, High, Low, Close, Volume)
- **Techniques**:
  - Normalization via `StandardScaler`
  - Visual comparisons via PCA and t-SNE
- **Output**: `synthetic_AAPL.csv`

## 📂 Folder Structure

| Folder/File        | Description                              |
|--------------------|------------------------------------------|
| `src/train_vae.py` | Main VAE training & generation script    |
| `data/`            | Raw input stock data                     |
| `visualizations/`  | PCA and t-SNE plots                      |
| `models/`          | (Optional) Trained model weights         |
| `README.md`        | Project summary and instructions         |
| `requirements.txt` | Python packages needed                   |

## 📈 Visual Results

Example of PCA & t-SNE plot output:

![PCA](visualizations/pca_vs_synthetic.png)
	
![t-SNE](visualizations/tsne_vs_synthetic.png)

## 🛠 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

##  Author

- Meet Patel
