import os
import torch
# import faiss
import pandas as pd
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from data.gen_faiss import get_index

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


index_path = os.path.join('path', 'to', 'directory', 'faiss.index')
faiss_index = get_index()


data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'train_cat_breeds.csv'))
df_train = pd.read_csv(data_path)
image_paths = []
for img_id, breed in zip(df_train['id'].values, df_train['breed'].values):
    img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'images', breed, f'{str(img_id)}.jpg'))
    image_paths.append(img_path)

device = "cpu"
model.to(device)

index = 1
device = 1

def query(text_input, model = model, processor = processor, index = index, image_paths = image_paths, device = device, k=5):
    # return [(image_paths[0], 'Abyssinian')]   
    model.eval()
    with torch.no_grad():
        # processing texts
        text_inputs = processor(text=[text_input], return_tensors="pt")
        # vectorizing text
        text_features = model.get_text_features(text_inputs).detach().cpu().numpy()
        # searching over the database
        D, I = index.search(text_features, k)

    top_k_indices = I[0]
    top_k_distances = D[0]
    # plotting
    # fig, axes = plt.subplots(1, k, figsize=(20, 5))

    outputs = []
    for i in range(k):
        img_path = image_paths[top_k_indices[i]]
        title = df_train.iloc[top_k_indices[i]]['breed']
        image = Image.open(img_path)
        outputs.append((image, title))
        # axes[i].imshow(image)
        # axes[i].set_title(f'{title}\n{top_k_distances[i]:.4f}')
        # axes[i].axis('off')
