import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import gdown
import base64


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("Designer - 2024-06-27T043428.404.png")

page_bg_img = f"""
<style>

h1#cloud-based-plant-disease-detection-system-cbpdds {{
    background: #f5ffc4a3;
    font-family: "Source Sans Pro", sans-serif;
    font-weight: 700;
    color: rgb(14, 17, 23);
    padding: 1.25rem 20px 2rem;
    margin: 0px;
    line-height: 1.2;
    border-radius: 20px;
    text-align: center;
}}

.st-emotion-cache-uzeiqp p {{
    background: #f5ffc4a3;
    font-family: "Source Sans Pro", sans-serif;
    font-weight: 700;
    color: rgb(14, 17, 23);
    padding: 1.25rem 20px 2rem;
    margin: 0px;
    line-height: 1.2;
    border-radius: 20px;
    text-align: center;
}}


.st-emotion-cache-7ym5gk {{
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 2.5rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    user-select: none;
    background-color: rgb(255, 255, 255);
    border: 1px solid rgb(100 171 9 / 82%);
}}    

/* Hover state */
.st-emotion-cache-7ym5gk:hover {{
    background-color: rgb(173, 255, 47); /* Yellow-Green background */
    border: 1px solid rgb(173, 255, 47); /* Yellow-Green border */
    color : black
}}

/* Active state */
.st-emotion-cache-7ym5gk:active {{
    background-color: rgb(154, 205, 50); /* Darker Yellow-Green background */
    border: 1px solid rgb(154, 205, 50); /* Darker Yellow-Green border */
    color : black
}}

/* visited state */
.st-emotion-cache-7ym5gk:visited {{
    background-color: rgb(52, 210, 235); /* Darker Yellow-Green background */
    border: 1px solid rgb(154, 205, 50); /* Darker Yellow-Green border */
    color : black
}}

/* focus:not(:active) state */
.st-emotion-cache-7ym5gk:focus:not(:active) {{
    border-color: rgb(154, 205, 50);
    color: rgb(14, 17, 23);
}}

.st-emotion-cache-183lzff.exotz4b0 {{
font-family: 'Poppins', monospace;
font-size: 14px;
overflow-x: auto;
color: white;
font-weight: bold;
}}

.st-emotion-cache-y4bq5x.ewgb6651 {{
    padding: 4px;
    background: rgb(14 17 23 / 50%);
    border-radius: 0.5rem;
}}

.st-emotion-cache-y4bq5x {{
    display: flex;
    visibility: visible;
    vertical-align: middle;
    flex-direction: row;
    -webkit-box-align: center;
    align-items: center;
    margin-top: 0.5rem;
}}



label.st-emotion-cache-ue6h4q.e1y5xkzn3 {{
    border-radius: 0.5rem 0.5rem 0 0;
    margin-bottom: 0px;
    background: rgb(154 205 50 / 85%);
    flex-direction: column;
    color: rgb(255, 255, 255);
}}

section.st-emotion-cache-1gulkj5.e1b2p2ww15 {{
    border-radius: 0 0 0.5rem 0.5rem;
    background: rgb(255 255 255 / 88%);
}}

.st-emotion-cache-fis6aj {{
    background: rgb(240 242 246 / 77%);
    border-radius: 0.5rem;
    padding: 10px;
    margin-top: 0.5rem;
    margin-bottom : 0;
}}

.st-emotion-cache-1v0mbdj.e115fcil1 {{
    color: aliceblue;
    padding: 0.5rem;
    background: #f0f8ffe3;
    border-radius: 0.5rem;
}}

[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}


[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

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
    probability = top_prob
    index = top_idx
    if probability >= 0.99:
        if index in [0,1]:
                return  f'<div style="background-color: rgb(244,67,54,0.8); font-weight: bold;color: black; padding: 20px;border-radius : 0.5rem;text-align : center">Prediction  :  {class_names[index].capitalize()} ({probability * 100:.2f}%)</div>'
        else:
                return f'<div style="background-color: rgba(173,255,47, 0.8); font-weight: bold;color:black; padding: 20px;border-radius : 0.5rem;text-align : center">Prediction  :  {class_names[index].capitalize()} ({probability * 100:.2f}%)</div>'

    else:
        return  f'<div style="background-color: rgba(255, 0, 0, 0.8); font-weight: bold;color: black; padding: 20px;border-radius : 0.5rem;text-align : center">Error : Invalid image - Please upload a valid cotton plant or leaf image </div>'

# Streamlit app
def main():
    # Add background image
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("Cloud Based Plant Disease Detection System (CBPDDS)")
    st.text("Upload a cotton plant or leaf image")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Classify the image
        if st.button('Classify'):
            with st.spinner('Classifying...'):
                predicition = predict(uploaded_file)
                # Display the prediction
                st.markdown(predicition,unsafe_allow_html=True)
            

                    
if __name__ == '__main__':
    main()



