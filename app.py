import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import gdown

# Check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Google Drive file ID and URL
file_id = '1lSun6s685Ysmytz3ij0y46aAmGR6Uwia'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'ResNet152v2_tmodel.pt'

@st.cache_data(show_spinner=False)
def download_model():
    gdown.download(url, output, quiet=False)
    return output

# Download the model if not already downloaded
model_path = download_model()

# Load the model
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# Class names for the predictions
class_names = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image):
    image = Image.open(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    return image

def predict(image):
    img = preprocess_image(image)
    with torch.no_grad():
        output = model(img)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_idx = torch.max(probabilities, 0)
    return top_prob.item(), top_idx.item()

# Streamlit app
def main():
    st.title("Cotton Plant Disease Classification")
    st.text("Upload an image for classification")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Classify the image
        if st.button('Classify'):
            with st.spinner('Classifying...'):
                probability, index = predict(uploaded_file)
                # Display the top prediction
                st.write(f"Prediction :  {class_names[index].capitalize()} ({probability * 100:.2f}%)")

if __name__ == '__main__':
    main()
