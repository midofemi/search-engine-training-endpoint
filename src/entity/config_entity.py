from from_root import from_root
import os


class DatabaseConfig:
    def __init__(self):
        self.USERNAME: str = os.environ["DATABASE_USERNAME"]
        self.PASSWORD: str = os.environ["DATABASE_PASSWORD"]
        #self.URL: str = "mongodb+srv://<username>:<password>@cluster0.msojpmv.mongodb.net"
        self.URL = "mongodb+srv://{os.getenv('ATLAS_CLUSTER_USERNAME')}:{os.getenv('ATLAS_CLUSTER_PASSWORD')}@cluster0.msojpmv.mongodb.net"
        self.DBNAME: str = "ReverseImageSearchEngine"
        self.COLLECTION: str = "Embeddings"

    def get_database_config(self):
        return self.__dict__

class DataIngestionConfig:
    """
    This is just our data ingestion config. Avnish did similar format with the Linear regression and Sensor project.
    We are just specifying where we can finds things and then use them in our data ingestion component. You can also see these
    configurations in our flowchart: Search_Engine\search-engine-training-endpoint\flowcharts\001_data_ingestion.png
    """
    def __init__(self):
        #self.PREFIX = "images"
        #self.RAW = os.path.join("data", "raw")
        #self.SPLIT = os.path.join("data", "splitted")
        self.PREFIX: str = "images/"
        self.RAW: str = "data/raw/"
        self.SPLIT: str = "data/splitted"
        self.BUCKET: str = "data-collection-s3bucket"
        self.SEED: int = 1337 #This is just set to select our images randomly. It a random seed
        self.RATIO: tuple = (0.8, 0.1, 0.1)

    def get_data_ingestion_config(self):
        return self.__dict__


class DataPreprocessingConfig:
    """
    Same idea above for data ingestion. This is our config for our data preprocessing component
    """
    def __init__(self):
        self.BATCH_SIZE = 32 #Here batch size is just the number of images. In this case we will be tranforming or processing 32 images at a time
        self.IMAGE_SIZE = 256 #This is the size of the image. 256 by 256. Since some images are of different format. We want to standardize it
        self.TRAIN_DATA_PATH = os.path.join(from_root(), "data", "splitted", "train") #Our training images/data will be in this folder
        self.TEST_DATA_PATH = os.path.join(from_root(), "data", "splitted", "test") #Our testing images/data will be in this folder
        self.VALID_DATA_PATH = os.path.join(from_root(), "data", "splitted", "valid") #Our validation images/data will be in this folder

    def get_data_preprocessing_config(self):
        return self.__dict__


class ModelConfig:
    def __init__(self):
        self.LABEL = 101 #Number of labels
        self.STORE_PATH = os.path.join(from_root(), "model", "benchmark") #Where our RESNET model will be stored
        self.REPOSITORY = 'pytorch/vision:v0.10.0' #The repo for our Resnet model
        self.BASEMODEL = 'resnet18' #We are using RESNET
        self.PRETRAINED = True #Here resnet will be pretrained

    def get_model_config(self):
        return self.__dict__


class TrainerConfig:
    def __init__(self):
        self.MODEL_STORE_PATH = os.path.join(from_root(), "model", "finetuned", "model.pth") #Here our ResNet + CV design model will be stored
                                                   #This would be the path where our model will be stored for prediction
        self.EPOCHS = 2
        self.Evaluation = True

    def get_trainer_config(self):
        return self.__dict__


class ImageFolderConfig:
    def __init__(self):
        self.ROOT_DIR = os.path.join(from_root(), "data", "raw", "images")
        self.IMAGE_SIZE = 256
        self.LABEL_MAP = {}
        self.BUCKET: str = "data-collection-s3bucket"
        self.S3_LINK = "https://{0}.s3.us-east-2.amazonaws.com/images/{1}/{2}"

    def get_image_folder_config(self):
        return self.__dict__


class EmbeddingsConfig:
    def __init__(self):
        self.MODEL_STORE_PATH = os.path.join(from_root(), "model", "finetuned", "model.pth")

    def get_embeddings_config(self):
        return self.__dict__


class AnnoyConfig:
    def __init__(self):
        self.EMBEDDING_STORE_PATH = os.path.join(from_root(), "data", "embeddings", "embeddings.ann")

    def get_annoy_config(self):
        return self.__dict__


class s3Config:
    def __init__(self):
        self.ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
        self.SECRET_KEY = os.getenv("AWS_SECRET_KEY")
        self.REGION_NAME = "us-east-2"
        self.BUCKET_NAME = "data-collection-s3bucket"
        self.KEY = "model"
        self.ZIP_NAME = "artifacts.tar.gz"
        self.ZIP_PATHS = [(os.path.join(from_root(), "data", "embeddings", "embeddings.json"), "embeddings.json"),
                          (os.path.join(from_root(), "data", "embeddings", "embeddings.ann"), "embeddings.ann"),
                          (os.path.join(from_root(), "model", "finetuned", "model.pth"), "model.pth")]

    def get_s3_config(self):
        return self.__dict__
