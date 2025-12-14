import os
import shutil
import requests

# Download SAM model (Base Model)

def download_model():
    url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    response = requests.get(url)
    with open('checkpoints/sam_vit_b_01ec64.pth', 'wb') as f:
        f.write(response.content)

if __name__ == '__main__':
    download_model()