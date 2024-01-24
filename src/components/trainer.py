from src.components.data_preprocessing import DataPreprocessing
from src.entity.config_entity import TrainerConfig
from torch import nn
import torch
import numpy as np
from src.components.model import NeuralNet
from typing import Dict
from tqdm import tqdm


class Trainer:
    def __init__(self, loaders: Dict, device: str, net):
        self.config = TrainerConfig()
        #Get the data loaders we created in data_preprocessing.py
        self.trainLoader = loaders["train_data_loader"][0]
        self.testLoader = loaders["test_data_loader"][0]
        self.validLoader = loaders["valid_data_loader"][0]
        self.device = device # Whether our training will happen on GPU or CPU
        self.criterion = nn.CrossEntropyLoss()
        self.model = net.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4) #Adam Optimizzer
        self.evaluation = self.config.Evaluation

    def train_model(self):
        """
        This is where training is done
        """
        print("Start training...\n")
        for epoch in range(self.config.EPOCHS):
            print(f'Epoch Number : {epoch}')
            running_loss = 0.0
            running_correct = 0
            for data in tqdm(self.trainLoader):
                data, target = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, target) #Calculate the loss
                #Calculate the accuracy
                running_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                running_correct += (preds == target).sum().item()

                loss.backward() #Backward Propagation
                self.optimizer.step() #Optimize the weight

            loss = running_loss / len(self.trainLoader.dataset)
            accuracy = 100. * running_correct / len(self.trainLoader.dataset)

            val_loss, val_accuracy = self.evaluate()

            print(f"Train Acc : {accuracy:.2f}, Train Loss : {loss:.4f}, "
                  f"Validation Acc : {val_accuracy:.2f}, Validation Loss : {val_loss:.4f}")

        print("Training complete!...\n")

    def evaluate(self, validate=False):
        """
        After the completion of each training epoch, measure the model's performance
        on our validation dataset.
        """
        self.model.eval()
        val_accuracy = []
        val_loss = []

        loader = self.testLoader if not validate else self.validLoader

        with torch.no_grad(): #This mean no gradient would be use during evaluation
            for batch in tqdm(loader):
                img = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                logits = self.model(img)
                loss = self.criterion(logits, labels)
                val_loss.append(loss.item())
                preds = torch.argmax(logits, dim=1).flatten()

                accuracy = (preds == labels).cpu().numpy().mean() * 100
                val_accuracy.append(accuracy)

        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy

    def save_model_in_pth(self):
        """
        Self Explanatory. We are saving the model for prediction
        """
        model_store_path = self.config.MODEL_STORE_PATH
        print(f"Saving Model at {model_store_path}")
        torch.save(self.model.state_dict(), model_store_path)


if __name__ == "__main__":
    dp = DataPreprocessing()
    loaders = dp.run_step()
    trainer = Trainer(loaders, "cpu", net=NeuralNet())
    trainer.train_model()
    trainer.evaluate(validate=True)
    trainer.save_model_in_pth()
