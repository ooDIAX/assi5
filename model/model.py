import os
import torch
import faiss
import pandas as pd
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'faiss.index'))
index = faiss.read_index(index_path)


data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'train_cat_breeds.csv'))
df_train = pd.read_csv(data_path)
image_paths = []
for img_id, breed in zip(df_train['id'].values, df_train['breed'].values):
    img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'images', breed, f'{str(img_id)}.jpg'))
    image_paths.append(img_path)

device = torch.device("cpu")
model.to(device)


def query(text_input, model = model, processor = processor, index = index, image_paths = image_paths, device = device, k=5):
    model.eval()
    with torch.no_grad():
        # processing texts
        text_inputs = processor(text=[text_input], return_tensors="pt").input_ids.to(device)
        # vectorizing text
        text_features = model.get_text_features(text_inputs).cpu().numpy()
        # searching over the database
        D, I = index.search(text_features, k)

    top_k_indices = I[0]
    top_k_distances = D[0]

    outputs = []
    for i in range(k):
        img_path = image_paths[top_k_indices[i]]
        title = df_train.iloc[top_k_indices[i]]['breed']
        outputs.append((img_path, title))
    return outputs
