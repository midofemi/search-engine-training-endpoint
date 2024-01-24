from src.entity.config_entity import ModelConfig
from torch import nn
import torch
#So before we get into it. There were 2 archetecture used for training our model (RESNET and then standard CV archetecture - which you are 
#familiar with). The reason was because resnet ended up giving us 1000 vectors as embeddings as it final layer and we can't just add our NN on 
#that large of a vector. This is why we added/design 3 more layers ontop of RESNET which you can see below.
#If you look at the flow chart, it also explain it visually for you as well. It convert let say for egs 1000 vector to 256 which is reasonable.
#This is great as well because it will reduce the size of our model doing the dimensionalty. These are things you have to know and think
#thru when doing big project like this
#This is no different as the CNN class you have taken. Please see: C:\Users\midof\OneDrive\Desktop\INeuron\CNN\CNN Practical Intuition.ipynb
#to refresh your memory if you have forgotten.
#Please don't be confuse here. 512 are just the INPUT kernels we are using in our input image and 32 are the OUTPUT kernels we want.
#So use 512 kernel to find patterns in our image but I only want 32 of the most important patterns.
#Another assignment would be, script the archteceture the same way you did on ..\INeuron\CNN\CNN Practical Intuition.ipynb
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ModelConfig()
        self.base_model = self.get_model() #Let say 1000x1000 dimension which is too much so let reduce that by designing a CV
        self.conv1 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #This is where we started with the archetecture
                                                                                    #The size of the image doesn't change
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #Size doesn't change
        self.conv3 = nn.Conv2d(16, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #Size doens't change
        self.flatten = nn.Flatten() #BoOOOM. Flatten for our NN. We can now apply our NN.
        self.final = nn.Linear(4 * 8 * 8, self.config.LABEL) #Remember the discussion we had on AWS-MongoDB-Setup specifically STEP 4 – MONGODB SETUP. 
                                                             #This is where we implement it with self.config.LABEL. Also, 4*8*8 just mean
                                                          #our starting Neural network will have 256 Neurons with an output layer of 101
        #We can now say if we split the Linear layer, we would have 256 vectors (embeddings) on one side.???? YESSSSSSSSSSSSSSSS

    def get_model(self):
        """
        This is our RESNET archetecture which we will be using with our CV archetecture (The one we designed to reduce dimensionality). 
        So Resnet is already pretrained so "model" just get that model for us.
        """
        torch.hub.set_dir(self.config.STORE_PATH)
        model = torch.hub.load(
            self.config.REPOSITORY,
            self.config.BASEMODEL,
            pretrained=self.config.PRETRAINED
        )
        #For the nn.Sequential. You may be curious what is going on here. Thanks to ChatGpt. This just mean, we don't want the classification
        #Layer nor the layer that came before that (average pooling layer) for our Resnet Model. Please give us the layer that is still
        #in NxN dimension So we can continue building/designing our CV which will give us a reasonable dimension (256x256) which we can use 
        #for our NN. I think this made alot of sense to me. Thanks to ChatGPt
        return nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        """
        This made complete sense once you understand what is going on above.
        """
        x = self.base_model(x) #Pretrained Resnet Model which will be passed to our CV
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.final(x)
        return x


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = NeuralNet()
    net.to(device)
