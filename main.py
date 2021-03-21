from tensorflow import keras
from PIL import Image
import streamlit as st
import numpy as np

hotdog_model = keras.models.load_model('hotdog.hdf5')
classes = ['Hotdog', 'Not Hotdog']

heading = st.title('Image classifier')
img = st.sidebar.file_uploader('📁 Upload Image', type=['png', 'jpeg', 'jpg'])
if img != None:
  st.image(img.getvalue())
  image = Image.open(img)
  image_array = np.array(image)
  image_array = np.resize(image_array, (1, 224, 224, 3))
  pred = hotdog_model.predict(image_array)
  
  if pred[0][0] >= 0.50:
    heading.title(classes[0])
  else:
    heading.title(classes[1])
