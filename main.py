import glob
import os

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tqdm import tqdm

from IPython import display

BATCH_SIZE = 64
IMG_HEIGHT = 200
IMG_WIDTH = 200

DATA_DIR = "D:/Datasets/wikiart/"

files_enumerator = tqdm(glob.glob(os.path.join(DATA_DIR, "*/*")))
wrong_files = 0

for file in files_enumerator:
    files_enumerator.set_description("Scanning files")
    files_enumerator.set_postfix({"wrong": wrong_files})
    if file.endswith('.jpg'):
        with open(file, 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            wrong_files += 1
            os.remove(file)
    else:
        print(file)
        os.remove(file)

train_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  seed=674518236,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE,
  label_mode=None,
  shuffle=True)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(int(IMG_HEIGHT / 4) * int(IMG_WIDTH / 4) * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((int(IMG_HEIGHT / 4), int(IMG_WIDTH / 4), 256)))
    assert model.output_shape == (None, int(IMG_HEIGHT / 4), int(IMG_WIDTH / 4), 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, int(IMG_HEIGHT / 4), int(IMG_WIDTH / 4), 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, int(IMG_HEIGHT / 2), int(IMG_WIDTH / 2), 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, IMG_HEIGHT, IMG_WIDTH, 3)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[IMG_HEIGHT, IMG_WIDTH, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator = make_generator_model()

discriminator = make_discriminator_model()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
SAVE_MODEL_EVERY = 15
noise_dim = 100
num_examples_to_generate = 1

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss


def train(dataset, epochs):
    gen_loss = 0
    disc_loss = 0
    for epoch in range(epochs):
        dataset_enumerator = tqdm(dataset)
        for image_batch in dataset_enumerator:
            dataset_enumerator.set_description(f'Training Epoch ({epoch + 1:03d}/{epochs:03d})')
            dataset_enumerator.set_postfix({
                "gen-loss": f'{gen_loss:.6f}',
                "disc-loss": f'{disc_loss:.6f}'
            })
            gen_loss, disc_loss = train_step(image_batch)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Save the model every 15 epochs
        if (epoch + 1) % SAVE_MODEL_EVERY == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             seed)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(1, 1))
    plt.imshow(predictions[0])
    plt.axis('off')
    if not os.path.isdir('./images'):
        os.mkdir('./images')
    plt.savefig('./images/image_at_epoch_{:04d}.png'.format(epoch))


train(train_ds, EPOCHS)
