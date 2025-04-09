import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy import linalg
import tensorflow_addons as tfa


# 1. Load Data
def load_data(file_path):    
    df = pd.read_csv(file_path, sep=',', parse_dates=['timestamp'])
    return df


def prepare_multivariate_data(df, window_size=24, test_size=0.2):
    """Prepares 3D time series data with non-overlapping train-test split."""
    countries = df.columns[1:]  # Exclude timestamp
    
    # Split original data into train/test based on timesteps
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Fit scaler on training data only
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_df[countries])
    scaled_test = scaler.transform(test_df[countries])  # Use same scaler
    
    # Create sliding windows for train/test
    def create_sequences(data):
        return np.array([
            data[i:i+window_size] 
            for i in range(len(data) - window_size + 1)
        ], dtype='float32')
    
    X_train = create_sequences(scaled_train)
    X_test = create_sequences(scaled_test)
    
    return scaler, X_train, X_test

def inverse_transform_data(scaler, windowed_data, original_columns):
    """
    Converts 3D normalized data back to original scale and reconstructs the original time series.
    Args:
        scaler: Fitted MinMaxScaler instance
        windowed_data: 3D array of shape (samples, timesteps, features)
        original_columns: List of original column names (country names from input dataframe)
    Returns:
        DataFrame in original scale with the entire reconstructed time series.
    """
    samples, timesteps, features = windowed_data.shape
    
    # Reshape to 2D and inverse transform
    data_2d = windowed_data.reshape(-1, features)
    inverted_data = scaler.inverse_transform(data_2d)
    
    # Reshape back to 3D
    inverted_3d = inverted_data.reshape(samples, timesteps, features)
    
    # Reconstruct the original timesteps by taking the last occurrence of each timestep
    total_timesteps = samples + timesteps - 1
    reconstructed_data = np.zeros((total_timesteps, features))
    for i in range(samples):
        for j in range(timesteps):
            t = i + j
            reconstructed_data[t] = inverted_3d[i, j]
    
    # Create DataFrame with original columns
    df = pd.DataFrame(reconstructed_data, columns=original_columns)
    return df

from tensorflow_addons.layers import SpectralNormalization

def build_wgan_generator(latent_dim, window_size, num_features):
    noise = layers.Input(shape=(latent_dim,))
    x = layers.Dense(6*256, activation='relu')(noise)
    # the 128 noise vector is projected into a higher-dimensional space (768 units) to create an initial structure (3 timesteps, 256 channels) that can be upsampled.
    x = layers.Reshape((6, 256))(x)
    
    # Upsampling layers
    x = layers.Conv1DTranspose(128, 8, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1DTranspose(64, 8, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1DTranspose(32, 8, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    # Attention Mechanism (using self-attention)
    #attn_output = layers.MultiHeadAttention(
    #    num_heads=4,
    #    key_dim=64,
    #    attention_axes=(1,)  # Apply attention along time dimension
    #)(x, x, x)  # query, key, value (self-attention)
    #x = layers.Add()([x, attn_output])  # Residual connection
    x = layers.LayerNormalization()(x)
    
        
    outputs = layers.Conv1D(num_features, 8, padding='same', activation='tanh')(x)
    
    model = keras.Model(inputs=noise, outputs=outputs, name="generator")
    return model


def build_wgan_discriminator(window_size, num_features):
    inputs = layers.Input(shape=(window_size, num_features))
    x = layers.Conv1D(64, 8, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    #x = layers.MaxPooling1D(1)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(128, 8, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    #x = layers.MaxPooling1D(1)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(256, 8, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="discriminator")
    return model

# 4. WGAN-GP Model
class WGAN_GP(keras.Model):
    def __init__(self, critic, generator, latent_dim, n_critic=5, gp_weight=10.0):
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        #self.critic_train_counter = tf.Variable(0, trainable=False, dtype=tf.int32)  # Track critic steps
               
        # Explicit input specification
        self.build((None, latent_dim))
        
    def build(self, input_shape):
        # Initialize weights by building the generator
        self.generator.build(input_shape)
        super().build(input_shape)
                                              
    def call(self, inputs, training=None):
        return self.generator(inputs, training=training)
    
    def get_config(self):  # NEW: Serialization support
        return {
            "critic": keras.models.clone_model(self.critic),
            "generator": keras.models.clone_model(self.generator),
            "latent_dim": self.latent_dim,
            "gp_weight": self.gp_weight
        }

    @classmethod
    def from_config(cls, config):  # NEW: Deserialization support
        return cls(**config)

    def compile(self, c_optimizer, g_optimizer):
        super().compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_loss_metric = keras.metrics.Mean(name="c_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
    
    def gradient_penalty(self, real, fake):
        batch_size = tf.shape(real)[0]
        alpha = tf.random.uniform([batch_size, 1, 1], 0., 1.)
        interpolated = alpha * real + (1 - alpha) * fake
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.critic(interpolated)
        
        gradients = tape.gradient(pred, interpolated)
        
        # Feature-specific gradient penalty
        #gradients_per_feature = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        #gp = tf.reduce_mean((gradients_per_feature - 1.0) ** 2)
        
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2]))
        gp = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        
        return gp
   
    def train_step(self, real_data):
        
        for _ in range(self.n_critic):  # Train Critic More Often
            noise = tf.random.normal([tf.shape(real_data)[0], self.latent_dim])
            with tf.GradientTape() as tape:
                fake_data = self.generator(noise, training=True)
                real_pred = self.critic(real_data, training=True)
                fake_pred = self.critic(fake_data, training=True)

                c_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
                gp = self.gradient_penalty(real_data, fake_data)
                c_total_loss = c_loss + self.gp_weight * gp

            c_grads = tape.gradient(c_total_loss, self.critic.trainable_weights)
            self.c_optimizer.apply_gradients(zip(c_grads, self.critic.trainable_weights))
            self.c_loss_metric.update_state(c_total_loss)
            
    
        noise = tf.random.normal([tf.shape(real_data)[0], self.latent_dim])
        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            fake_pred = self.critic(fake_data, training=True)
            g_loss = -tf.reduce_mean(fake_pred)
            
        g_grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in [self.c_loss_metric, self.g_loss_metric]}


from scipy import linalg
import numpy as np
from scipy.stats import wasserstein_distance

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, real_data, latent_dim, window_size, num_features, every_n=10):
        super().__init__()
        self.real_data = real_data
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.num_features = num_features
        self.every_n = every_n
        self.best_fid = float('inf')
        self.fid_history = []
        self.temp_corr_history = []
        self.real_corr_history = []
        self.wasserstein_dist_history = []
        
        # Build feature extractor for fid calculation
        self.feature_extractor = self.build_time_series_feature_extractor()
        
    def build_time_series_feature_extractor(self):
        """CNN-based feature extractor for multivariate time-series data."""
        inputs = keras.Input(shape=(self.window_size, self.num_features))
        
        x = layers.Conv1D(64, kernel_size=3, strides=1, activation='relu', padding='same')(inputs)
        x = layers.Conv1D(128, kernel_size=3, strides=1, activation='relu', padding='same')(x)  # Second Conv1D layer
        x = layers.GlobalAveragePooling1D()(x)  # Pooling to get a fixed-size feature vector
        
        model = keras.Model(inputs, x, name="TimeSeriesFeatureExtractor")
        return model
    
    def calculate_fid(self, real, synthetic):
        """Frechet Inception Distance using critic's features"""
        real_features = self.feature_extractor.predict(real, verbose=0)
        syn_features = self.feature_extractor.predict(synthetic, verbose=0)
        
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_syn, sigma_syn = np.mean(syn_features, axis=0), np.cov(syn_features, rowvar=False)
        
        #diff = mu_real - mu_syn
        # Compute squared mean difference
        ssdiff = np.sum((mu_real - mu_syn)**2)
        
        # Compute sqrt of covariance product (handling singular matrix case)
        covmean = linalg.sqrtm(sigma_real @ sigma_syn, disp=False)[0].real
        #if np.iscomplexobj(covmean):
        #    covmean = covmean.real  # Ensure real-valued result
        
        fid = ssdiff + np.trace(sigma_real + sigma_syn - 2*covmean)
        return float(fid)
        #return diff.dot(diff) + np.trace(sigma_real + sigma_syn - 2*covmean)
    
    @staticmethod
    def temporal_correlation_score(data, lag=1):
        """Normalized autocorrelation calculation"""
        if tf.is_tensor(data):
            data = data.numpy()
            
        scores = []
        for sample in data:  # (timesteps, features)
            for feature in range(data.shape[2]):
                series = sample[:, feature]
                if np.var(series) < 1e-6:  # Skip constant series
                    continue
                acf = np.correlate(series - np.mean(series), 
                                 series - np.mean(series), 
                                 mode='full')
                norm_acf = acf[len(series)-1 + lag] / (len(series) * np.var(series))
                scores.append(norm_acf)
        return np.nanmean(scores) if scores else 0.0
    
    def compute_wasserstein_distance(self, real_data: np.ndarray, synthetic_data: np.ndarray):
        """Compute Wasserstein Distance between real and synthetic distributions."""
        if tf.is_tensor(real_data):
            real_data = real_data.numpy()
        if tf.is_tensor(synthetic_data):
            synthetic_data = synthetic_data.numpy() 
            
        # Reshape data to 2D: (samples * time steps, features)
        real_flat = real_data.reshape(-1, real_data.shape[2])  # Shape: (samples * time steps, features)
        synthetic_flat = synthetic_data.reshape(-1, synthetic_data.shape[2])  # Shape: (samples * time steps, features)

        # Compute Wasserstein distance per feature
        wd_per_feature = [wasserstein_distance(real_flat[:, i], synthetic_flat[:, i]) for i in range(real_data.shape[2])]

        # Compute mean Wasserstein distance across all features
        mean_wd = np.mean(wd_per_feature)

        return mean_wd
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n == 0:
            # Generate synthetic data
            noise = tf.random.normal([self.real_data.shape[0], self.latent_dim])
            synthetic = self.model.generator(noise, training=False)
            
            # Calculate metrics
            #real_features = self.feature_extractor.predict(self.real_data)
            #fake_features = self.feature_extractor.predict(synthetic)

            #fid = self.calculate_fid(real_features, fake_features)
            fid = self.calculate_fid(self.real_data, synthetic)
            
            real_corr = self.temporal_correlation_score(self.real_data)
            temp_corr = self.temporal_correlation_score(synthetic)
            
            wasserstein_dist = self.compute_wasserstein_distance(self.real_data, synthetic)
            
            # Save best model
            if fid < self.best_fid:
                self.best_fid = fid
                self.model.generator.save_weights("best_generator_weights.h5")
            
            # Store and print
            self.fid_history.append(fid)
            self.temp_corr_history.append(temp_corr)
            self.real_corr_history.append(real_corr)
            self.wasserstein_dist_history.append(wasserstein_dist)
            # Print metrics
            print(f"\nEpoch {epoch+1} Metrics:")
            print(f"FID: {fid:.2f} (Lower better)")
            print(f"TempCorr: {temp_corr:.2f} (Real: {real_corr:.2f})")
            print(f"wasserstein_dist: {wasserstein_dist:.2f} (Lower better)")


# Load data
df = load_data("price_subsampled_2percent.csv")


# Get original column names
original_columns = df.columns.tolist()  # ['timestamp', 'AT', 'BE', ...]
original_columns= original_columns[1:]

# Prepare data
preserve_scaler, X_train_wgan, X_test_wgan = prepare_multivariate_data(df)
num_features = X_train_wgan.shape[2]  

X_train_wgan.shape


# Build models
wgan_generator = build_wgan_generator(latent_dim=128, window_size=24, num_features=num_features)
wgan_discriminator = build_wgan_discriminator(window_size=24, num_features=num_features)
    
# Compile
wgan = WGAN_GP(critic=wgan_discriminator, generator=wgan_generator, latent_dim=128, n_critic = 2, gp_weight=10.0)
wgan.compile(
    c_optimizer=keras.optimizers.Adam(0.001, beta_1=0.0, beta_2=0.9),
    g_optimizer=keras.optimizers.Adam(0.001, beta_1=0.0, beta_2=0.9)
)

# Create dataset
train_ds = tf.data.Dataset.from_tensor_slices(X_train_wgan).batch(200)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/wgan-gp/checkpoint_{epoch}",
    save_weights_only=True,
    save_freq=100*32  # Save every 100 batches
)

lr_adjustment = keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.95 if epoch % 50 == 0 else lr
        )


# Train with monitoring
monitor = GANMonitor(X_test_wgan, latent_dim=128, window_size=24, num_features=num_features)


# Training execution
wgan_history = wgan.fit(
    train_ds,
    epochs=1000,  # Train longer
    callbacks=[checkpoint, monitor]
)



# After training
plt.figure(figsize=(12,6))

# FID Plot
plt.subplot(1,3,1)
plt.plot(monitor.fid_history, label='FID')
plt.xlabel('Epoch (x10)')
plt.ylabel('FID Distance')
plt.title('Frechet Inception Distance')
plt.grid(True)

# Temporal Correlation Plot
plt.subplot(1,3,2)
plt.plot(monitor.temp_corr_history, label='Synthetic', color='orange')
plt.plot(monitor.real_corr_history, color='green', linestyle='--', label='Real Data')
plt.xlabel('Epoch (x10)')
plt.ylabel('Correlation Score')
plt.title('Temporal Correlation')
plt.legend()
plt.grid(True)

# Temporal Correlation Plot
plt.subplot(1,3,3)
plt.plot(monitor.wasserstein_dist_history, label='Wassertein Distances')
plt.xlabel('Epoch (x10)')
plt.ylabel('mean wassertein distances')
plt.title('Wassertein Distances')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



# After training
plt.figure(figsize=(12,6))

# CWGAN-GP Loss Plot
plt.plot(wgan_history.history['c_loss'], color='orange', linestyle='--', label='Discriminator Loss')
plt.plot(wgan_history.history['g_loss'], color='blue', linestyle='--', label='Generator Loss')
plt.title('CWGAN-GP Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Save
wgan.save_weights("wgan_training/models/wgan_weights.weights.h5")

# Load
#wgan = WGAN_GP(...)  # Rebuild architecture first
#wgan.load_weights("wgan_weights.weights.h5")

# Save
wgan.save("wgan_training/models/wgan_gp_timeseries.keras")

# Load
#loaded_model = keras.models.load_model(
#    "wgan_gp_timeseries",
#    custom_objects={"WGAN_GP": WGAN_GP}
#)


# Generate synthetic data
wgan_noise = tf.random.normal([len(X_train_wgan), 128])
wgan_synthetic = wgan.generator.predict(wgan_noise)
print(wgan_synthetic.shape)  # Should be (num, 24, 21)


# Generate synthetic data
wgan_noise_test = tf.random.normal([len(X_test_wgan), 128])
wgan_synthetic_test = wgan.generator.predict(wgan_noise_test)
print(wgan_synthetic_test.shape)  # Should be (num, 24, 21)


import seaborn as sns

def plot_feature_tsne(
    real_data: np.ndarray,  # Shape: (samples, 24, 21)
    synthetic_data: np.ndarray,
    country_names: list,
    max_samples: int = 500
):
    num_features = real_data.shape[2]
    n_cols = 3
    n_rows = int(np.ceil(num_features / n_cols))
    
    plt.figure(figsize=(18, 5*n_rows))
    
    for feature_idx in range(num_features):
        ax = plt.subplot(n_rows, n_cols, feature_idx+1)
        
        # Extract country data
        # num of samples used in the t-SNE plot
        used_samples = min(real_data.shape[0], max_samples)
        real_feature = real_data[:used_samples, :, feature_idx]
        synthetic_feature = synthetic_data[:used_samples, :, feature_idx]
        
        # Combine and reduce
        combined = np.vstack([real_feature, synthetic_feature])
        tsne = TSNE(n_components=2, perplexity=40, random_state=42)
        tsne_results = tsne.fit_transform(combined)
        
        # Plot
        ax.scatter(tsne_results[:used_samples, 0], tsne_results[:used_samples, 1], 
                   c='red', alpha=0.5, label='Real',s=100)
        ax.scatter(tsne_results[used_samples:, 0], tsne_results[used_samples:, 1], 
                   c='blue', alpha=0.5, label='Synthetic',s=100)
        ax.set_title(f"t-SNE for {country_names[feature_idx]}")
        ax.legend()
    
    plt.tight_layout()
    plt.legend()
    plt.show()


plot_feature_tsne(X_train_wgan, wgan_synthetic, original_columns)


def avg_over_dim(data: np.ndarray, axis: int) -> np.ndarray:
    """
    Average over the feature dimension of the data.

    Args:
        data (np.ndarray): The data to average over.
        axis (int): Axis to average over.

    Returns:
        np.ndarray: The data averaged over the feature dimension.
    """
    return np.mean(data, axis=axis)


def plot_tsne(
    samples1: np.ndarray,
    samples1_name: str,
    samples2: np.ndarray,
    samples2_name: str,
    scenario_name: str,
    max_samples: int = 1000,
) -> None:
    """
    Visualize the t-SNE of two sets of samples and save to file.

    Args:
        samples1 (np.ndarray): The first set of samples to plot.
        samples1_name (str): The name for the first set of samples in the plot title.
        samples2 (np.ndarray): The second set of samples to plot.
        samples2_name (str): The name for the second set of samples in the
                             plot title.
        scenario_name (str): The scenario name for the given samples.
        max_samples (int): Maximum number of samples to use in the plot. Samples should
                           be limited because t-SNE is O(n^2).
    """
    if samples1.shape != samples2.shape:
        raise ValueError(
            "Given pairs of samples dont match in shapes. Cannot create t-SNE.\n"
            f"sample1 shape: {samples1.shape}; sample2 shape: {samples2.shape}"
        )

    samples1_2d = avg_over_dim(samples1, axis=2)
    samples2_2d = avg_over_dim(samples2, axis=2)

    # num of samples used in the t-SNE plot
    used_samples = min(samples1_2d.shape[0], max_samples)

    # Combine the original and generated samples
    combined_samples = np.vstack(
        [samples1_2d[:used_samples], samples2_2d[:used_samples]]
    )

    # Compute the t-SNE of the combined samples
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300, random_state=42)
    tsne_samples = tsne.fit_transform(combined_samples)

    # Create a DataFrame for the t-SNE samples
    tsne_df = pd.DataFrame(
        {
            "tsne_1": tsne_samples[:, 0],
            "tsne_2": tsne_samples[:, 1],
            "sample_type": [samples1_name] * used_samples
            + [samples2_name] * used_samples,
        }
    )

    # Plot the t-SNE samples
    plt.figure(figsize=(8, 8))
    for sample_type, color in zip([samples1_name, samples2_name], ["red", "blue"]):
        if sample_type is not None:
            indices = tsne_df["sample_type"] == sample_type
            plt.scatter(
                tsne_df.loc[indices, "tsne_1"],
                tsne_df.loc[indices, "tsne_2"],
                label=sample_type,
                color=color,
                alpha=0.5,
                s=100,
            )

    plt.title(f"t-SNE for {scenario_name}")
    plt.legend()
    plt.show()



plot_tsne(samples1=X_train_wgan,samples1_name="real_price", samples2=wgan_synthetic, samples2_name="synthetic_price", scenario_name="price")