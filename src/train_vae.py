"""
COSC 4207 - Seminars in Computer Science
Title: Synthetic Data Generation using VAE
Team Members: Meet Patel, Agam Jammu, Naga Siva Sai Kumar Kudulla
Course: Seminars in Computer Science (COSC 4207)
Submission Date: April 14, 2025

Description:
This script builds and trains a Variational Autoencoder (VAE) using TensorFlow
to generate synthetic financial time-series data from the stock dataset.
The model is trained on normalized stock features, and visualizations (PCA & t-SNE)
are used to compare the distributions of real and synthetic data.

References:
- Pezoulas, I., et al. (2024). Synthetic data generation methods in healthcare: A review on open-source tools and methods.
- TensorFlow VAE Tutorial. URL: https://www.tensorflow.org/tutorials/generative/cvae
- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. URL: https://arxiv.org/abs/1312.6114
- YouTube: "Variational Autoencoders in TensorFlow 2.0" by Aladdin Persson (For learning purpose).
- Dataset: Oleh Onyshchak's "Stock Market Dataset" on Kaggle, Originally sourced from Yahoo Finance.
"""

#%% STEP 1: IMPORT LIBRARIES
# Standard numerical and ML tools for data processing and modeling 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras import layers, models

#%% STEP 2: LOAD AND PREPROCESS DATA
# Bring in stock data and select relevant features for modeling
file_path = "/home/meetpatel/Desktop/Reseach_Project/data/AAPL_stock.csv"   
data = pd.read_csv(file_path)

# Set random seeds for reproducibility
np.random.seed(2026)
tf.random.set_seed(2026)
 
# Select key financial indicators to model
features = ['Open', 'High', 'Low', 'Close', 'Volume']
data = data[features]

# Normalize features using StandardScaler (zero mean, unit variance)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

#%% STEP 3: BUILD VAE MODEL
# Define encoder and decoder to map between input and latent space
latent_dim = 5
input_dim = data_scaled.shape[1]


# --- Encoder: maps input to latent distribution (mean and log-variance) ---
inputs = layers.Input(shape=(input_dim,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(32, activation='relu')(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# Reparameterization trick: sample latent vector z from distribution
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# Create the encoder model
encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
print(" The Encoder model is created.")

#--- Decoder: maps latent space back to original input dimensions ---
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(32, activation='relu')(latent_inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(input_dim)(x)  #  Linear activation for regression

decoder = models.Model(latent_inputs, outputs, name='decoder')
print("Decoder model created.")

# STEP 4: VAE CLASS WRAPPER
# Custom training loop using Model subclass to calculate VAE loss
class VAE(models.Model):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def compile(self, optimizer):
        super(VAE, self).compile()
        self.optimizer = optimizer
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Reconstruction loss: mean squared error between input and output
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction), axis=1))
            
            # KL divergence: regularizes the latent space to follow standard normal
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {"loss": self.total_loss_tracker.result()}

# Instantiate and compile the VAE model
vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())
print("VAE model compiled successfully.")

#%% STEP 5: TRAIN VAE
# Train VAE on normalized stock data
print("Starting training...")
vae.fit(data_scaled, epochs=50, batch_size=64, verbose=1)
print("Training completed.")

#%% STEP 6: GENERATE SYNTHETIC DATA
#Sample random latent vectors, decode them, and inverse-transform back

n_samples = data_scaled.shape[0]
z_synthetic = tf.random.normal(shape=(n_samples, latent_dim))   # Random latent vectors
synthetic_scaled = decoder.predict(z_synthetic)                 # Generate synthetic data
synthetic_data = scaler.inverse_transform(synthetic_scaled)     # Inverse transform to original scale

# Save the synthetic data
synthetic_df = pd.DataFrame(synthetic_data, columns=features)
synthetic_df.to_csv("synthetic_AAPL.csv", index=False)
print("Synthetic data saved to synthetic_AAPL.csv")

#%% STEP 7: VISUALIZE OVERLAP (PCA & t-SNE)
# Compare distributions of real vs synthetic data visually
real_label = np.zeros(len(data_scaled))
synth_label = np.ones(len(synthetic_scaled))
combined_data = np.vstack([data_scaled, synthetic_scaled])
combined_labels = np.hstack([real_label, synth_label])

# Color legend for plotting
colors = ['blue' if label == 0 else 'red' for label in combined_labels]

# --- PCA Visualization ---
pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined_data)
plt.figure(figsize=(6, 5))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.6)
plt.title("PCA: Real vs Synthetic")
plt.xlabel("PC1")
plt.ylabel("PC2")
custom_labels = [plt.Line2D([0], [0], marker='o', color='w', label='Real', markerfacecolor='blue', markersize=7),
                 plt.Line2D([0], [0], marker='o', color='w', label='Synthetic', markerfacecolor='red', markersize=7)]
plt.legend(handles=custom_labels)
plt.grid(True)
plt.show()

# --- t-SNE Visualization ---
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_result = tsne.fit_transform(combined_data)
plt.figure(figsize=(6, 5))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, alpha=0.6)
plt.title("t-SNE: Real vs Synthetic")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend(handles=custom_labels)
plt.grid(True)
plt.show()


