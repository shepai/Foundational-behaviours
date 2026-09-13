import os
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras import layers, Model


# ============================================================
# Configuration
# ============================================================
IMAGE_SIZE = 64
CHANNELS = 3

LATENT_DIM = 32

BATCH_SIZE = 64
EPOCHS = 30

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# Load video dataset
# ============================================================

def load_videos(filename):
    """
    Load a NumPy dataset containing multiple video streams.

    Expected format:
        videos = [
            video_1,
            video_2,
            ...
        ]

    Each video should have shape:

        (frames, height, width, channels)

    Videos can have different heights, widths and numbers of frames.
    """
    videos = np.load(filename, allow_pickle=True)
    return videos


# ============================================================
# Resize videos
# ============================================================

def preprocess_videos(videos, image_size=IMAGE_SIZE):
    """
    Convert videos of different spatial sizes into a common
    dataset of frames.

    Returns:
        X : numpy array
            Shape = (total_frames, image_size, image_size, 3)
    """

    frames = []

    for video_number, video in enumerate(videos):

        video = np.asarray(video)

        print(
            f"Video {video_number}: "
            f"shape={video.shape}"
        )

        # ----------------------------------------------------
        # Handle grayscale videos
        # ----------------------------------------------------

        if video.ndim == 3:
            # (frames, height, width)
            video = np.expand_dims(video, axis=-1)

        # ----------------------------------------------------
        # Convert grayscale -> RGB
        # ----------------------------------------------------

        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)

        # ----------------------------------------------------
        # Resize every frame
        # ----------------------------------------------------

        for frame in video:

            frame = cv2.resize(
                frame,
                (image_size, image_size),
                interpolation=cv2.INTER_AREA
            )

            frames.append(frame)

    X = np.asarray(frames, dtype=np.float32)

    # --------------------------------------------------------
    # Normalize pixels
    # --------------------------------------------------------

    X /= 255.0

    print("Training data shape:", X.shape)

    return X


# ============================================================
# Build convolutional autoencoder
# ============================================================

def build_autoencoder(
    image_size=IMAGE_SIZE,
    latent_dim=LATENT_DIM
):

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    encoder_input = layers.Input(
        shape=(image_size, image_size, 3),
        name="image"
    )

    x = layers.Conv2D(
        32,
        3,
        strides=2,
        padding="same",
        activation="relu"
    )(encoder_input)

    x = layers.Conv2D(
        64,
        3,
        strides=2,
        padding="same",
        activation="relu"
    )(x)

    x = layers.Conv2D(
        128,
        3,
        strides=2,
        padding="same",
        activation="relu"
    )(x)

    x = layers.Conv2D(
        256,
        3,
        strides=2,
        padding="same",
        activation="relu"
    )(x)

    x = layers.Flatten()(x)

    latent = layers.Dense(
        latent_dim,
        name="latent"
    )(x)

    encoder = Model(
        encoder_input,
        latent,
        name="encoder"
    )

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    decoder_input = layers.Input(
        shape=(latent_dim,),
        name="latent_input"
    )

    x = layers.Dense(
        4 * 4 * 256,
        activation="relu"
    )(decoder_input)

    x = layers.Reshape((4, 4, 256))(x)

    x = layers.Conv2DTranspose(
        128,
        3,
        strides=2,
        padding="same",
        activation="relu"
    )(x)

    x = layers.Conv2DTranspose(
        64,
        3,
        strides=2,
        padding="same",
        activation="relu"
    )(x)

    x = layers.Conv2DTranspose(
        32,
        3,
        strides=2,
        padding="same",
        activation="relu"
    )(x)

    decoder_output = layers.Conv2DTranspose(
        3,
        3,
        strides=2,
        padding="same",
        activation="sigmoid",
        name="reconstruction"
    )(x)

    decoder = Model(
        decoder_input,
        decoder_output,
        name="decoder"
    )

    # --------------------------------------------------------
    # Full autoencoder
    # --------------------------------------------------------

    autoencoder_output = decoder(
        encoder(encoder_input)
    )

    autoencoder = Model(
        encoder_input,
        autoencoder_output,
        name="autoencoder"
    )

    return encoder, decoder, autoencoder


# ============================================================
# Main training
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    videos = load_videos("videos.npy")

    # --------------------------------------------------------
    # Convert all videos to frames of identical size
    # --------------------------------------------------------

    X = preprocess_videos(videos)

    # --------------------------------------------------------
    # Build models
    # --------------------------------------------------------

    encoder, decoder, autoencoder = build_autoencoder()

    autoencoder.summary()

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="mse"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history = autoencoder.fit(
        X,
        X,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        shuffle=True
    )

    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------

    autoencoder.save(
        os.path.join(
            MODEL_DIR,
            "autoencoder.keras"
        )
    )

    encoder.save(
        os.path.join(
            MODEL_DIR,
            "encoder.keras"
        )
    )

    decoder.save(
        os.path.join(
            MODEL_DIR,
            "decoder.keras"
        )
    )

    print("Models saved.")