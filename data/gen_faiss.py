import numpy as np
import pandas as pd
import os
import faiss
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

def preprocess_and_encode_images(df, processor, model):
    image_features_list = []
    image_paths = []

    for idx, row in df.iterrows():
        img_path = f'data/images/{row["breed"]}/{row["id"]}.jpg'
        try:
            # Getting images
            image = Image.open(img_path).convert("RGB")
            # Processing images
            inputs = processor(images=image, return_tensors="pt")
            # Getting image vectors
            image_features = model.get_image_features(**inputs).detach().cpu().numpy()
            # collecting all images
            image_features_list.append(image_features)
            # collecting paths
            image_paths.append(img_path)

            print(f"Processed image {img_path}")
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    return np.vstack(image_features_list), image_paths


def load_or_create_faiss_index(image_features_array, index_path='data/faiss.index'):
    if os.path.exists(index_path):
        # Load existing index if it exists
        print(f'Loading existing FAISS index from: {os.path.abspath(index_path)}')
        index = faiss.read_index(index_path)
    else:
        # Create new index if it doesn't exist
        print(f'Creating new FAISS index at: {os.path.abspath(index_path)}')
        d = image_features_array.shape[1]
        index = faiss.IndexFlatL2(d)  # creating index
        index.add(image_features_array)  # adding image vectors
        faiss.write_index(index, index_path)  # saving index

    return index

# Load the CLIP model and processor from Hugging Face
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load the training data
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'train_cat_breeds.csv'))
df_train = pd.read_csv(data_path)

# Preprocess images and encode them
image_features_array, image_paths = preprocess_and_encode_images(df_train, processor, model)

# Create and save FAISS index
faiss_index = load_or_create_faiss_index(image_features_array)
