from PIL import Image
import os

imgPaths = os.listdir("data/membrane/train/label")
for imgpath in imgPaths:
    print(os.path.join("data/membrane/train/label", imgpath))
    img = Image.open(os.path.join("data/membrane/train/label", imgpath)).convert('LA')
    img.save(os.path.join("data/membrane/train/label", imgpath))