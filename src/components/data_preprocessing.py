from src.entity.config_entity import DataPreprocessingConfig
#####################################################################################
#Here we are not using tensorflow for transformation and all. We are using Pytorch
#What you can try is instead of using pytorch, why not use tensor flow. You have wokred on a tensorflow project before via CV.
#It not that hard to implement here I believe
from torchvision.datasets import ImageFolder #So what this does is organise my data in a yaml kinda way so it can be easily access in pytorch
                                             #kinda way. Please use chatGPT and it will give you an example which will make complete sense
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
#####################################################################################


class DataPreprocessing:
    """
    Here data preprocessing is just data transformation. Because we are dealing with alot of images. It possible that some images has
    different sizes. So this is what our transformation try to do. Thus we have images of the same size before training our model.
    Our model need it to be this way (You should know this) as in any ML or AI project.
    """
    def __init__(self):
        self.config = DataPreprocessingConfig()

    def transformations(self):
        try:
            """
            Transformation Method Provides TRANSFORM_IMG object. Its pytorch's transformation class to apply on images.
            :return: TRANSFORM_IMG. Here is where our transformation happens. One thing I was think is that, your past CV project
            can be implemented here. Meaning, you could do the transformation differently than the way it been one here (MAYBE????)
            """
            TRANSFORM_IMG = transforms.Compose(
                [transforms.Resize(self.config.IMAGE_SIZE),
                 transforms.CenterCrop(256),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])]
            )#Here we are resizing the image based on our config_entity = 256 by 256 and then Normalize the image

            return TRANSFORM_IMG #Returned our tranform image
        except Exception as e:
            raise e

    def create_loaders(self, TRANSFORM_IMG):
        """
        The create_loaders method takes Transformations and create dataloaders (please ask chatgpt if you dont remember what data loader does).
        This is what makes tensorflwo and pytorch different in training our model. Pytorch does it this way.
        :param TRANSFORM_IMG:
        :return: Dict of train, test, valid Loaders
        """
        try:
            print("Generating DataLoaders : ")
            result = {}
            for _ in tqdm(range(1)):
                #Take the path of our training, test and validation data and implement ImageFolder library on it
                train_data = ImageFolder(root=self.config.TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
                test_data = ImageFolder(root=self.config.TEST_DATA_PATH, transform=TRANSFORM_IMG)
                valid_data = ImageFolder(root=self.config.TEST_DATA_PATH, transform=TRANSFORM_IMG)

                #Now that we have implemented an imagefolder for it. We can now apply our data loader.
                train_data_loader = DataLoader(train_data, batch_size=self.config.BATCH_SIZE,
                                               shuffle=True, num_workers=1)
                test_data_loader = DataLoader(test_data, batch_size=self.config.BATCH_SIZE,
                                              shuffle=False, num_workers=1)
                valid_data_loader = DataLoader(valid_data, batch_size=self.config.BATCH_SIZE,
                                               shuffle=False, num_workers=1)

                result = {
                    "train_data_loader": (train_data_loader, train_data),
                    "test_data_loader": (test_data_loader, test_data),
                    "valid_data_loader": (valid_data_loader, valid_data)
                }
            return result
        except Exception as e:
            raise e

    def run_step(self):
        try:
            """
            This methods calls all the private methods.
            :return: Response of Process
            """
            TRANSFORM_IMG = self.transformations()
            result = self.create_loaders(TRANSFORM_IMG)
            return result
        except Exception as e:
            raise e


if __name__ == "__main__":
    # Data Ingestion Can be replaced like this
    # https://aws.amazon.com/blogs/machine-learning/announcing-the-amazon-s3-plugin-for-pytorch/
    dp = DataPreprocessing()
    loaders = dp.run_step()
    for i in loaders["train_data_loader"][0]:
        break
