from src.entity.config_entity import DataIngestionConfig
from src.utils.storage_handler import S3Connector
from from_root import from_root
import splitfolders
import os
"""
It would be a good idea to look at the flow chart when going through these component for your pipeline. Maybe it will help you visualize what
is going on. Please see: Search_Engine\search-engine-training-endpoint\flowcharts\001_data_ingestion.png if curious
"""

class DataIngestion:
    """
    Here we have a data ingestion class
    """
    def __init__(self):
        self.config = DataIngestionConfig() #Here we are getting the data ingestion config from our config_entity so we can acccess those
                                #configuration in our code: Search_Engine\search-engine-training-endpoint\src\entity\config_entity.py.
                                #Please see our config_entity as config.yaml like in PPV which you created. 
                              #Actually, all this could have been done using a yaml file as well. just an alternative way of doing things

    def download_dir(self):
        """
        Here we are just connecting to our S3 bucket (Please see flow chart. It is exactly as we wrote the code so you can understand)
        params:
        - prefix: pattern to match in s3
        - local: local path to folder in which to place files
        - bucket: s3 bucket with target contents
        - client: initialized s3 client object

        """
        try:
            print("\n====================== Fetching Data ==============================\n")
            data_path = os.path.join(from_root(), self.config.RAW, self.config.PREFIX)
            os.system(f"aws s3 sync s3://data-collection-s3bucket/images/ {data_path} --no-progress")
            print("\n====================== Fetching Completed ==========================\n")

        except Exception as e:
            raise e

    def split_data(self):
        """
        This Method is Responsible for splitting. DUHHHHH. With any modeling. This is expected so don't overthink it LOL.
        Here we are just spliting the data based on our data ingestion config: Search_Engine\search-engine-training-endpoint\src\entity
        \config_entity.py
        :return:
        """
        try:
            splitfolders.ratio(
                input=os.path.join(self.config.RAW, self.config.PREFIX),
                output=self.config.SPLIT,
                seed=self.config.SEED,
                ratio=self.config.RATIO,
                group_prefix=None, move=False
            )
        except Exception as e:
            raise e

    def run_step(self):
        """
        Now run the steps above
        """
        self.download_dir()
        self.split_data()
        return {"Response": "Completed Data Ingestion"}


if __name__ == "__main__":
    paths = ["data", r"data\raw", r"data\splitted", r"data\embeddings",
             "model", r"model\benchmark", r"model\finetuned"]

    for folder in paths:
        path = os.path.join(from_root(), folder)
        print(path)
        if not os.path.exists(path):
            os.mkdir(folder)
    print("Passed this Point")
    dc = DataIngestion()
    print(dc.run_step())
