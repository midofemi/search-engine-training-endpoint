from src.components.data_preprocessing import DataPreprocessing
from src.entity.config_entity import ImageFolderConfig, EmbeddingsConfig
from src.utils.database_handler import MongoDBClient
from torch.utils.data import Dataset, DataLoader
from src.components.model import NeuralNet
from typing import List, Dict
from torchvision import transforms
from collections import namedtuple
from PIL import Image
from torch import nn
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import torch
import os
from pathlib import Path

ImageRecord = namedtuple("ImageRecord", ["img", "label", "s3_link"])


class ImageFolder(Dataset):
    """
    So here we stored every image in a path. This is where we create our custom dataset. I did a Dev notebook on how it done. It looks like this:
    image_records: [ImageRecord(img=WindowsPath('C:/Users/midof/OneDrive/Desktop/INeuron/Data_Science_Project/Industry_Ready_Proj/CNN/
    Search_Engine/search-engine-training-endpoint/notebooks/caltech-101/accordion/image_0001.jpg'), label='accordion'), AND SO ON]. It will be
    on a list. So each index would have a path, images and URL
    """
    def __init__(self, label_map: Dict):
        self.config = ImageFolderConfig()
        self.config.LABEL_MAP = label_map
        self.transform = self.transformations()
        self.image_records: List[ImageRecord] = []
        self.record = ImageRecord

        file_list = os.listdir(self.config.ROOT_DIR)

        for class_path in file_list: #Read all the images
            path = os.path.join(self.config.ROOT_DIR, f"{class_path}")
            images = os.listdir(path)
            for image in tqdm(images):
                image_path = Path(f"""{self.config.ROOT_DIR}/{class_path}/{image}""")
                self.image_records.append(self.record(img=image_path,
                                                      label=self.config.LABEL_MAP[class_path],
                                                      s3_link=self.config.S3_LINK.format(self.config.BUCKET, class_path,
                                                                                         image)))

    def transformations(self):
        """
        Transformation function to transform our data
        """
        TRANSFORM_IMG = transforms.Compose(
            [transforms.Resize(self.config.IMAGE_SIZE),
             transforms.CenterCrop(self.config.IMAGE_SIZE),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]
        )

        return TRANSFORM_IMG

    def __len__(self):
        return len(self.image_records) #This gives us the length of the list we created above of our custom dataset

    def __getitem__(self, idx):
        """
        Remember our __init__ function above.? We are now interating through that LIST of our custom dataset we have created. We want
        the images, targets and the URL link we generated above
        """
        record = self.image_records[idx]
        images, targets, links = record.img, record.label, record.s3_link
        images = Image.open(images)

        if len(images.getbands()) < 3:
            images = images.convert('RGB')
        images = np.array(self.transform(images))
        targets = torch.from_numpy(np.array(targets))
        images = torch.from_numpy(images)

        return images, targets, links


class EmbeddingGenerator:
    def __init__(self, model, device):
        self.config = EmbeddingsConfig()
        self.mongo = MongoDBClient()
        self.model = model
        self.device = device
        self.embedding_model = self.load_model()
        self.embedding_model.eval()

    def load_model(self):
        """
        Load the model we have trained. This would be the RESNET + CV. After we load it, we then took the final out (now we left just with the)
        flatten layer in a vector form
        """
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.config.MODEL_STORE_PATH, map_location=self.device)) #FIne tune folder path
        return nn.Sequential(*list(model.children())[:-1]) #This is where we remove the neural network so we can have the embeddings of our flatten layer

    def run_step(self, batch_size, image, label, s3_link):
        records = dict()

        images = self.embedding_model(image.to(self.device))
        images = images.detach().cpu().numpy()

        records['images'] = images.tolist()
        records['label'] = label.tolist()
        records['s3_link'] = s3_link

        df = pd.DataFrame(records)
        records = list(json.loads(df.T.to_json()).values())
        self.mongo.insert_bulk_record(records) #Store the embeddings in MongoDB

        return {"Response": f"Completed Embeddings Generation for {batch_size}."}


if __name__ == "__main__":
    dp = DataPreprocessing()
    loaders = dp.run_step()

    data = ImageFolder(label_map=loaders["valid_data_loader"][1].class_to_idx)
    dataloader = DataLoader(dataset=data, batch_size=64, shuffle=True)
    embeds = EmbeddingGenerator(model=NeuralNet(), device="cpu")

    for batch, values in tqdm(enumerate(dataloader)):
        img, target, link = values
        print(embeds.run_step(batch, img, target, link))
