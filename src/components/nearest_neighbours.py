from src.utils.database_handler import MongoDBClient
from src.entity.config_entity import AnnoyConfig
from annoy import AnnoyIndex
from typing_extensions import Literal
from tqdm import tqdm
import json


class CustomAnnoy(AnnoyIndex): #Here AnnoyIndex = 256 because that the embedding space we are using. REMEMBER, model training
    """
    Please see note if you don't understand what annoy is about. I'll assume you know so I would go straight into the code and what is happening.

    """
    def __init__(self, f: int, metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"]): #Here we are just listing the metric we
                                                #we will be using
        super().__init__(f, metric)
        self.label = []

    # noinspection PyMethodOverriding
    def add_item(self, i: int, vector, label: str) -> None:
        """
        Here we are just passing our embeddings and index to Annoy
        """
        super().add_item(i, vector)
        self.label.append(label)

    def get_nns_by_vector(
            self, vector, n: int, search_k: int = ..., include_distances: Literal[False] = ...):
        """
        Here once we give annoy some vectors and the number of labels we want that are similar to our input. It will give us a labels
        based on the similarity of our input enbeddings. So we give in vectors and output gives us labels.
        FYI: This algorithm was modify a bit. In reality, ANNOY gives index of our image but we modify some part where instead we want
        the image as an output and not the indexes
        """
        indexes = super().get_nns_by_vector(vector, n, search_k, include_distances)
        labels = [self.label[link] for link in indexes] #Because our images are saved in a list. We can get them via indexes. So 0 will give us
                                  #the first URL of that list based on the search we are doing. These URL are S3 links (REMEMBER)
        return labels

    def load(self, fn: str, prefault: bool = ...):
        super().load(fn)
        path = fn.replace(".ann", ".json")
        self.label = json.load(open(path, "r"))

    def save(self, fn: str, prefault: bool = ...):
        super().save(fn)
        path = fn.replace(".ann", ".json")
        json.dump(self.label, open(path, "w"))


class Annoy(object):
    def __init__(self):
        self.config = AnnoyConfig()
        self.mongo = MongoDBClient()
        self.result = self.mongo.get_collection_documents()["Info"]

    def build_annoy_format(self):
        Ann = CustomAnnoy(256, 'euclidean')
        print("Creating Ann for predictions : ")
        for i, record in tqdm(enumerate(self.result), total=8677):
            Ann.add_item(i, record["images"], record["s3_link"])

        Ann.build(100)
        Ann.save(self.config.EMBEDDING_STORE_PATH)
        return True

    def run_step(self):
        self.build_annoy_format()


if __name__ == "__main__":
    ann = Annoy()
    ann.run_step()
