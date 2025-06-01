import tensorflow as tf
import numpy as np
import cv2

categories = ['spider', 'pencil', 'house', 'apple', 'airplane', 'bird', 'cat', 'fish', 'flower', 'black']
model = tf.keras.models.load_model('./assets/model/batter_nine.h5', compile=False)

def predict_image(image_path, model=model, categories=categories):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot open image file: {image_path}")
    img = cv2.resize(img, (224, 224))  
    image_array = np.asarray(img)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)[0]
    max_index = np.argmax(prediction)
    predicted_class = categories[max_index]
    confidence = prediction[max_index]

    return predicted_class, confidence

# predicted_class, confidence= predict_image('./photo/player_screenshot_82.png')
# print(f"Predicted class: {predicted_class} with confidence: {confidence:.2f}")