# Load Libraries
import os
import numpy as np
import PIL
import keras

#Define prediction Function
def predict_only(model,image_path):
    # Load and resize the image using PIL.
    img = PIL.Image.open(image_path)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

    # Convert the PIL image to a numpy-array with the proper shape.
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    #Make Predictions
    pred = model.predict(img_array)
    
    if(pred[0][0] > pred[0][1]):
        print("Rs.10")
    else:
        print("Rs.20")

# Set Input Shape
input_shape = (224,224)

# Load Model from File
transfer_model = keras.models.load_model('finetune_3.model')

# Predict from a file
predict_only(transfer_model, image_path='data/train/twenty/tw_635.jpg')


