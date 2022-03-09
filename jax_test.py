import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp
import flaxmodels as fm

# Seed
key = jax.random.PRNGKey(0)

# Input noise
z = jax.random.normal(key, shape=(4, 512))

generator = fm.stylegan2.Generator(pretrained='metfaces')
params = generator.init(key, z)
images = generator.apply(params, z, train=False)

# Normalize images to be in range [0, 1]
images = (images - jnp.min(images)) / (jnp.max(images) - jnp.min(images))

# Save images
for i in range(images.shape[0]):
    Image.fromarray(np.uint8(images[i] * 255)).save(f'image_{i}.jpg')