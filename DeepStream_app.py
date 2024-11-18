import streamlit as st

# Import necessary libraries for each app
import torch
import torchvision.transforms as transforms
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import timm
import tensorflow as tf
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import re
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import cv2

# Download necessary NLTK resources for the text classification app
nltk.download('punkt')
nltk.download('stopwords')

# App selection via sidebar
st.sidebar.title("Select an App")
app_options = ["Card Classification App", 
               "Name Generator", 
               "Blood Donation Prediction", 
               "Suicide Tweet Classifier", 
               "Weed Detection"]
selected_app = st.sidebar.selectbox("Choose the app you want to run:", app_options)

# 1. Card Classification App
def card_classification_app():
    st.title("Card Classification App")
    
    class CardClassifier(nn.Module):
        def __init__(self, num_classes=53):
            super(CardClassifier, self).__init__()
            self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
            self.features = nn.Sequential(*list(self.base_model.children())[:-1])
            enet_out_size = 1280
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(enet_out_size, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    # Load model using state dict
    def load_model_state_dict(model_path, device):
        model = CardClassifier()  # Initialize the model architecture
        model.load_state_dict(torch.load(model_path, map_location=device))  # Load the weights
        model.eval()  # Set to evaluation mode
        return model

    # Preprocess the image
    def preprocess_image(image, transform):
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        return image, image_tensor

    # Predict class probabilities
    def predict(model, image_tensor, device):
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities.cpu().numpy().flatten()

    # Visualize the predictions (original card name)
    def visualize_predictions(original_image, probabilities, class_names):
        fig, axarr = plt.subplots(1, 2, figsize=(14, 7))
        axarr[0].imshow(original_image)
        axarr[0].axis("off")
        axarr[1].barh(class_names, probabilities, color='skyblue')
        axarr[1].set_xlabel("Probability")
        axarr[1].set_title("Class Predictions")
        axarr[1].set_xlim(0, 1)
        st.pyplot(fig)

    # Map card class names to card types (Spade, Heart, Diamond, Club, Joker)
    def get_card_type(full_card_name):
        if "Spades" in full_card_name:
            return "Spade"
        elif "Hearts" in full_card_name:
            return "Heart"
        elif "Diamonds" in full_card_name:
            return "Diamond"
        elif "Clubs" in full_card_name:
            return "Club"
        else:
            return "Joker"

    # Main logic for Card Classification App
    model_path = "DeepStream_Models/card_classification_model_state_dict.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_state_dict(model_path, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # For EfficientNet
    ])

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        original_image, image_tensor = preprocess_image(image, transform)
        probabilities = predict(model, image_tensor, device)

        # Original class names (full card names)
        class_names = ["Ace of Spades", "Two of Spades", "Three of Spades", "Four of Spades", "Five of Spades",
        "Six of Spades", "Seven of Spades", "Eight of Spades", "Nine of Spades", "Ten of Spades",
        "Jack of Spades", "Queen of Spades", "King of Spades",
        "Ace of Hearts", "Two of Hearts", "Three of Hearts", "Four of Hearts", "Five of Hearts",
        "Six of Hearts", "Seven of Hearts", "Eight of Hearts", "Nine of Hearts", "Ten of Hearts",
        "Jack of Hearts", "Queen of Hearts", "King of Hearts",
        "Ace of Diamonds", "Two of Diamonds", "Three of Diamonds", "Four of Diamonds", "Five of Diamonds",
        "Six of Diamonds", "Seven of Diamonds", "Eight of Diamonds", "Nine of Diamonds", "Ten of Diamonds",
        "Jack of Diamonds", "Queen of Diamonds", "King of Diamonds",
        "Ace of Clubs", "Two of Clubs", "Three of Clubs", "Four of Clubs", "Five of Clubs",
        "Six of Clubs", "Seven of Clubs", "Eight of Clubs", "Nine of Clubs", "Ten of Clubs",
        "Jack of Clubs", "Queen of Clubs", "King of Clubs",
        "Joker"]

        # Get the index of the highest probability class (most likely class)
        predicted_index = probabilities.argmax()

        # Get the full card name and card type from the prediction
        predicted_card_name = class_names[predicted_index]
        predicted_card_type = get_card_type(predicted_card_name)

        # Show the predicted card type (Spade, Heart, Diamond, Club, or Joker)
        st.write(f"Predicted Card Type: {predicted_card_type}")

# 2. Name Generator App
def name_generator_app():
    st.title("Name Generator")

    def character_acc(y_true, y_pred):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), tf.float32))

    model_path = "C:DeepStream_Models\\dino_model.keras"
    with tf.keras.utils.custom_object_scope({'character_acc': character_acc}):
        dino_model = tf.keras.models.load_model(model_path)

    char_to_idx = {
        'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
        'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19,
        't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, 'END': 0
    }

    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    def sample(predictions, temperature=1.0):
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions + 1e-7) / temperature
        exp_preds = np.exp(predictions)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(predictions), p=preds)

    def generate_name(model, char_to_idx, idx_to_char, first_letter, max_length=20, temperature=1.0):
        input_seq = np.zeros((1, 1, len(char_to_idx)))

        if first_letter in char_to_idx:
            input_seq[0, 0, char_to_idx[first_letter]] = 1
        else:
            raise ValueError(f"First letter '{first_letter}' not found in char_to_idx mapping.")

        generated_name = [first_letter]
        for _ in range(max_length - 1):
            predictions = model.predict(input_seq, verbose=0)[0, -1, :]
            next_idx = sample(predictions, temperature)
            next_char = idx_to_char[next_idx]

            if next_char == 'END':
                break

            generated_name.append(next_char)
            new_input_seq = np.zeros((1, 1, len(char_to_idx)))
            new_input_seq[0, 0, next_idx] = 1
            input_seq = np.concatenate([input_seq, new_input_seq], axis=1)

        return ''.join(generated_name)

    first_letter = st.text_input("First Letter", value='A').lower()

    if st.button("Generate Names"):
        if first_letter and len(first_letter) == 1 and first_letter in char_to_idx:
            generated_names = [generate_name(dino_model, char_to_idx, idx_to_char, first_letter, temperature=1.0) for _ in range(5)]
            st.write("Generated Names:")
            for name in generated_names:
                st.write(name)
        else:
            st.warning("Please enter a single valid letter from 'a' to 'z'.")

# 3. Blood Donation Prediction App
def blood_donation_prediction_app():
    st.title("Blood Donation Prediction App")

    logreg_model = joblib.load("C:DeepStream_Models\\Tabular_data_logistic_regression_model.pkl")
    nn_model = load_model("C:DeepStream_Models\\Tabular_data_neural_network_model.h5")

    recency = st.number_input("Recency (months since last donation)", min_value=0, max_value=100, value=2)
    frequency = st.number_input("Frequency (total number of donations)", min_value=0, max_value=100, value=1)
    time = st.number_input("Time (months since first donation)", min_value=0, max_value=100, value=1)
    monetary_log = st.number_input("Logarithmic Monetary Value (log of total blood donated)", min_value=0.0, value=10.0)

    input_data = np.array([[recency, frequency, time, monetary_log]])

    logreg_pred = logreg_model.predict_proba(input_data)[:, 1]
    nn_pred = nn_model.predict(input_data).flatten()

    combined_pred = 0.9 * logreg_pred + 0.1 * nn_pred
    final_pred = (combined_pred >= 0.5).astype(int)

    if st.button("Predict"):
        st.write(f"Prediction (1 = will donate, 0 = will not donate): {final_pred[0]}")
        st.write(f"Logistic Regression Probability: {logreg_pred[0]:.4f}")
        st.write(f"Neural Network Probability: {nn_pred[0]:.4f}")
        st.write(f"Combined Prediction Probability: {combined_pred[0]:.4f}")

# 4. Suicide Tweet Classifier App
def suicide_tweet_classifier_app():
    st.title("Suicide Tweet Classifier")

    # Load models and preprocessors
    xgb_model = joblib.load("C:DeepStream_Models\\Text_Classification_xgb_model.pkl")
    dl_model = load_model("C:DeepStream_Models\\Text_classification_model_dl.h5")
    vectorizer = joblib.load("C:DeepStream_Models\\Text_Classificationvectorizer.pkl")
    tokenizer = joblib.load("C:DeepStream_Models\\Text_Classification_tokenizer.pkl")

    # Helper functions for cleaning and preprocessing
    def cleaner(text):
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'(.)\1+', r'\1', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        return text

    def preprocessor(text):
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        text = word_tokenize(text.lower())
        text = [stemmer.stem(word) for word in text if word not in stop_words]
        text = ' '.join(text)
        return text

    # Hardcoded suicidal phrases/words
    suicidal_phrases = [
        "shoot myself", "death", "kill myself", "end myself",
        "suicide", "i want to die", "i can't go on", "no reason to live", "end it all"
    ]

    def contains_suicidal_phrases(text):
        # Check if any of the hardcoded suicidal phrases are present in the input text
        text_lower = text.lower()
        for phrase in suicidal_phrases:
            if phrase in text_lower:
                return True
        return False

    # Streamlit input
    text = st.text_area("Enter the tweet:")

    if st.button("Classify"):
        # Check for hardcoded suicidal phrases
        if contains_suicidal_phrases(text):
            st.write("Prediction: Suicidal")
        else:
            # Clean and preprocess text
            text_cleaned = preprocessor(cleaner(text))
            text_seq = tokenizer.texts_to_sequences([text_cleaned])
            text_seq = pad_sequences(text_seq, maxlen=101)

            # Model-based predictions
            xgb_prediction = xgb_model.predict(vectorizer.transform([text_cleaned]))
            dl_prediction = (dl_model.predict(text_seq) > 0.5).astype("int32")
            final_prediction = 0.9 * xgb_prediction + 0.1 * dl_prediction.flatten()

            categories = ['Not Suicidal', 'Suicidal']
            st.write(f"Prediction: {categories[int(final_prediction[0])]}")


# 5. Weed Detection App
def weed_detection_app():
    st.title("Weed Detection with Keras Inception Model")

    @st.cache_resource
    def load_model():
        model = tf.keras.models.load_model("C:DeepStream_Models\\Image_Segmentation_plant_disease_model_inception.h5")
        return model

    def preprocess_image(uploaded_image, target_size=(139, 139)):
        img = Image.open(uploaded_image).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, np.array(img)

    def generate_heatmap(model, img_array, predicted_class):
        last_conv_layer = model.get_layer('mixed10')
        heatmap_model = tf.keras.models.Model(model.inputs, [last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = heatmap_model(img_array)
            loss = predictions[:, predicted_class]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = np.dot(conv_outputs, pooled_grads[..., np.newaxis])
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap

    def overlay_heatmap(heatmap, original_image):
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        if original_image.dtype != np.uint8:
            original_image = np.uint8(original_image)
        superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        return superimposed_img

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        img_array, original_image = preprocess_image(uploaded_image, target_size=(139, 139))
        model = load_model()
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        heatmap = generate_heatmap(model, img_array, predicted_class)
        overlay = overlay_heatmap(heatmap, original_image)

        st.image(original_image, caption="Original Image", use_column_width=True)
        st.image(heatmap, caption="Activation Heatmap", use_column_width=True, clamp=True)
        st.image(overlay, caption="Overlay of Heatmap on Original Image", use_column_width=True)

# Main
if selected_app == "Card Classification App":
    card_classification_app()
elif selected_app == "Name Generator":
    name_generator_app()
elif selected_app == "Blood Donation Prediction":
    blood_donation_prediction_app()
elif selected_app == "Suicide Tweet Classifier":
    suicide_tweet_classifier_app()
elif selected_app == "Weed Detection":
    weed_detection_app()
