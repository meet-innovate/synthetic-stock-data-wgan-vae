######################################################################
# COSC 4027                                                          #
# Synthetic Data Sets for ML                                         #
# Agam Jammu, Meet Patel & Naga Siva Sai Kumar Kudulla               #
#                                                                    #
# Based on:                                                          #                                             
#   Arjovsky, M., Chintala, S., & Bottou, L. (2017).                 #
#   "Wasserstein GAN", Courant Institute of Mathematical Sciences.   #
#   Facebook AI Research                                             #
#                                                                    #
# Dataset Source:                                                    #
#   Oleh Onyshchak's "Stock Market Dataset" on Kaggle,               #
#   originally sourced from Yahoo Finance.                           #
######################################################################

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

# Set up a directory to stash all plots.
PLOT_DIR = "plots/WGAN"
os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------
# LOAD & PREPROCESS DATA
# -------------------

# Bring in real AAPL stock data and do the usual housekeeping—
# sort by date and drop adjusted close (we don't need it here).
raw_data_path = "AAPL_stock.csv"
df = pd.read_csv(raw_data_path, parse_dates=["Date"])
df = df.sort_values(by="Date")
df = df.drop(columns=["Adj Close"])

# Normalize numerical features
scaler = MinMaxScaler()
df[["Open", "High", "Low", "Close", "Volume"]] = scaler.fit_transform(
    df[["Open", "High", "Low", "Close", "Volume"]]
)

# Pull out only the numeric features (we don't need the date column anymore).
data = df[["Open", "High", "Low", "Close", "Volume"]].values

# -------------------
# MODEL HYPERPARAMETERS
# -------------------
LATENT_DIM = 100          # Dimensionality of the generator's input noise
FEATURES = data.shape[1]  # Stock features: open, high, low, close, volume
BATCH_SIZE = 1024         # Training batch size
EPOCHS = 2000               
CRITIC_ITER = 5           # Train critic more often than the generator (per Arjovsky et al.). 5 critic iterations for every generator step
LAMBDA_GP = 15            # Gradient penalty weight to enforce Lipschitz constraint


# -------------------
# GENERATOR DEFINITION
# -------------------

# This generator maps a latent vector to stock-like features.
# used sigmoid at the end, tried linear but it gave negative stock prices which isn't ideal.
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(FEATURES, activation="sigmoid")  # for non-negative stock prices
    ])
    return model


# -------------------
# CRITIC DEFINITION
# -------------------
# The critic (a.k.a. the WGAN "discriminator") gives real-valued scores 
# instead of probabilities. No sigmoid here.
# Approximating the Kantorovich-Rubinstein dual, per Arjovsky et al.
def build_critic():
    model = tf.keras.Sequential([
        layers.Dense(512, activation="relu", input_shape=(FEATURES,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="linear")  # Output a single score
    ])
    return model

# ---------------------------------------------
# Wasserstein GAN with Gradient Penalty
# ---------------------------------------------
# This follows the WGAN-GP formulation from Arjovsky et al.
# Gradient penalty stabilizes training and ensures our critic
# is a 1-Lipschitz function (the central assumption of WGAN theory).
class WGAN(tf.keras.Model):
    def __init__(self, generator, critic):
        super(WGAN, self).__init__()
        self.generator = generator
        self.critic = critic
        self.gen_optimizer = Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)
        self.crit_optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)

    def compile(self):
        super(WGAN, self).compile()
    
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]

        # Train critic more times than generator to better approximate the EM distance.
        # This is key to WGANs' stability and supported by the paper's theoretical grounding.
        # This follows Arjovsky's advice to estimate the Wasserstein distance better
        for _ in range(CRITIC_ITER):
            noise = tf.random.normal([batch_size, LATENT_DIM])
            fake_data = self.generator(noise, training=True)
            
            with tf.GradientTape() as tape:
                real_score = self.critic(real_data, training=True)
                fake_score = self.critic(fake_data, training=True)
                gp = self.gradient_penalty(real_data, fake_data)
                critic_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + LAMBDA_GP * gp
            
            gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.crit_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        # Now update the generator to fool the critic (i.e., increase fake scores)
        noise = tf.random.normal([batch_size, LATENT_DIM])
        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            fake_score = self.critic(fake_data, training=True)
            gen_loss = -tf.reduce_mean(fake_score)

        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        return {"critic_loss": critic_loss, "generator_loss": gen_loss}

    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform([BATCH_SIZE, 1], 0., 1.)
        interpolated = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.critic(interpolated)
        grad = tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

generator = build_generator()
critic = build_critic()
wgan = WGAN(generator, critic)
wgan.compile()

# Convert real data to TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices(data.astype("float32")).shuffle(buffer_size=1024).batch(BATCH_SIZE, drop_remainder=True)

# Train WGAN
history = wgan.fit(dataset, epochs=EPOCHS, verbose=1)

# -------------------
# GENERATE SYNTHETIC STOCK DATA
# -------------------
def generate_synthetic_data(generator, num_samples):
    noise = tf.random.normal([num_samples, LATENT_DIM])
    synthetic_data = generator(noise, training=False).numpy()
    return synthetic_data

# Generate synthetic data with same number of samples as real data
num_samples = data.shape[0]
synthetic_stock_data = generate_synthetic_data(generator, num_samples)

# Save synthetic data
synthetic_df = pd.DataFrame(synthetic_stock_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
synthetic_df.to_csv("../data/WGAN_synthetic_AAPL_stock.csv", index=False)
print("Synthetic stock data saved as WGAN_synthetic_AAPL.csv")

# -------------------
# WASSERSTEIN DISTANCE
# -------------------
# For each feature, compute the actual Wasserstein-1 distance between real and synthetic distributions
# it's nice when it's close to 0
wass_distances = {}
for i, col in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
    wass_distances[col] = wasserstein_distance(data[:, i], synthetic_stock_data[:, i])

# Print Wasserstein distances
print("Wasserstein Distances:")
for feature, dist in wass_distances.items():
    print(f"{feature}: {dist:.4f}")

# Plot Distances
plt.figure(figsize=(8, 6))
plt.bar(wass_distances.keys(), wass_distances.values(), color='b', alpha=0.7)
plt.xlabel("Stock Feature")
plt.ylabel("Wasserstein Distance")
plt.title("Wasserstein Distance between Real and Synthetic Data")
plt.xticks(rotation=45)
plt.savefig(f"{PLOT_DIR}/wasserstein_distance.png")
plt.show()


# -------------------
# PCA & t-SNE
# -------------------

# Combine real and synthetic data
combined_data = np.vstack([data, synthetic_stock_data])
labels = np.array([0] * len(data) + [1] * len(synthetic_stock_data))  # 0: Real, 1: Synthetic

# PCA Visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined_data)

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[labels==0][:, 0], pca_result[labels==0][:, 1], alpha=0.6, label="Real Data", marker='o')
plt.scatter(pca_result[labels==1][:, 0], pca_result[labels==1][:, 1], alpha=0.6, label="Synthetic Data", marker='x')
plt.legend()
plt.title("PCA: Real vs Synthetic Stock Data")
plt.savefig(f"{PLOT_DIR}/pca_real_vs_synthetic.png")
plt.show()

# t-SNE Visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(combined_data)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[labels==0][:, 0], tsne_result[labels==0][:, 1], alpha=0.6, label="Real Data", marker='o')
plt.scatter(tsne_result[labels==1][:, 0], tsne_result[labels==1][:, 1], alpha=0.6, label="Synthetic Data", marker='x')
plt.legend()
plt.title("t-SNE: Real vs Synthetic Stock Data")
plt.savefig(f"{PLOT_DIR}/tsne_real_vs_synthetic.png")
plt.show()
