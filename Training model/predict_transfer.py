import numpy as np
from PIL import Image

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

target_size = (229, 229)

def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  classes = preds.argmax(axis=-1)
  return preds[0], classes

model = load_model("data/cash/models/inceptionv3_large_data.model")

image_location = "data/cash/test1/2.jpg"

img = Image.open(image_location)

preds, classes = predict(model, img, target_size)

print('\n \n --------------- \n')

if(classes == 1.0):
	print("Image is of a Rs.10")
else:
	print("Image is of a Rs.20")

