# DeepStream

**DeepStream** is a powerful, multi-functional app built with Streamlit, leveraging state-of-the-art deep learning models. This platform offers a suite of applications designed to solve a variety of tasks, ranging from image classification to text analysis.

## Features

### 1. Card Classification App
- **Purpose**: Identifies the type of card (Spades, Hearts, Diamonds, Clubs, or Joker) from an uploaded image.
- **Model**: EfficientNet (pretrained) with a custom classification layer.
- **Input**: Images of playing cards.
- **Output**: Predicted card type and visualization of probabilities.

### 2. Name Generator
- **Purpose**: Generates creative names based on a given starting letter.
- **Model**: Custom TensorFlow model trained on character-level sequences.
- **Input**: A single letter as the starting character.
- **Output**: Multiple generated names.

### 3. Blood Donation Prediction
- **Purpose**: Predicts whether a person is likely to donate blood.
- **Models**: Logistic Regression and a Neural Network.
- **Input**: Features like recency, frequency, time since first donation, and monetary value.
- **Output**: Predicted likelihood of donating blood.

### 4. Suicide Tweet Classifier
- **Purpose**: Detects suicidal intent in a given tweet.
- **Models**: XGBoost and Deep Learning with ensemble predictions.
- **Input**: A tweet or short text.
- **Output**: Classification as "Suicidal" or "Not Suicidal."

### 5. Weed Detection
- **Purpose**: Detects weeds in plant images and highlights the affected areas.
- **Model**: Keras-based Inception model for image segmentation.
- **Input**: Images of plants.
- **Output**: Heatmap overlay indicating weed-prone areas.

---

## How to Run the App

1. Clone the repository and navigate to the project directory.

2. Install the required dependencies by using the provided requirements file.

3. Run the Streamlit app.

4. Open the provided local URL in your browser.

---

## Project Structure

DeepStream/ │ ├── app.py # Main Streamlit app ├── models/ # Directory for deep learning models ├── requirements.txt # Python dependencies └── README.md # Project documentation


---

## Dependencies

Ensure you have the following libraries installed:

- torch
- tensorflow
- streamlit
- numpy
- pillow
- matplotlib
- joblib
- nltk
- opencv-python
- timm

For a full list, see the `requirements.txt` file.

---

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue to discuss potential improvements.

---

## License

This project is licensed under the MIT License. See the LICENSE file for more details.


