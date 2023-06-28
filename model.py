import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


# 生成器网络
class Generator(nn.Module):
    def __init__(self, input_length:int):
        super().__init__()
        self.input_length = input_length
        self.model1 = nn.Sequential(
            nn.Linear(in_features=input_length, out_features=1024*2),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024*2, out_features=1024*4),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024*4, out_features=1024*8),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024*8, out_features=1024*16),\
            nn.LeakyReLU()
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=2),
        )
    
    def forward(self, x):
        img = self.model1(x)
        img = img.view(-1, 16, 32, 32) # batch_size = -1(32), C = 32, H = 32, W = 32
        img = self.model2(img)
        return img

# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 输入为 N * C * W(28) * H(28)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(p=0.25),
            nn.Flatten(),
            nn.Linear(in_features=36864, out_features=128),
            # nn.Linear(in_features=9216, out_features=128),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=20) # 0-9表示真实图片，10-19表示伪造图片
        )
    
    # 正向传导
    def forward(self, x):
        logits = self.model(x)
        return logits

# 制作一个生成器输入
def make_input(value:int, input_length:int) -> tuple[torch.Tensor, int]:
    gen_input:torch.Tensor = torch.normal(value, 5, (input_length,))
    return (gen_input, value)

# 制作一批生成器输入
def make_input_batch(values:list[int], input_length:int) -> tuple[torch.Tensor, torch.Tensor]:
    input_batch = []
    label_batch = []

    for value in values:
        gen_input, gen_label = make_input(value, input_length)
        input_batch.append(gen_input)
        label_batch.append(gen_label)

    return (torch.stack(input_batch), torch.Tensor(label_batch).long())

class GAN(object):
    def __init__(self, 
                 device:str,
                 dataloader:DataLoader=None, 
                 generator:Generator=None,
                 discriminator:Discriminator=None,
                 generator_optimizer:Optimizer=None,
                 discriminator_optimizer:Optimizer=None,
                 loss_fn:nn.Module=None,
                 ) -> None:
        super().__init__()
        self.__device:str = device
        self.__dataloader:DataLoader = dataloader
        self.__generator:Generator = generator.to(self.__device)
        self.__discriminator:Discriminator = discriminator.to(self.__device)
        self.__generator_optimizer:Optimizer = generator_optimizer
        self.__discriminator_optimizer:Optimizer = discriminator_optimizer
        self.__loss_fn:nn.Module = loss_fn

    def train(self) -> None:
        for batch, (input_img, input_label) in enumerate(self.__dataloader):
            input_img, input_label = input_img.to(self.__device), input_label.to(self.__device)

            # 训练判别器
            self.__discriminator.train()
            self.__generator.eval()
            self.__discriminator_optimizer.zero_grad()

            # 用输入图片训练判别器
            prediction:torch.Tensor = self.__discriminator(input_img)
            input_loss:torch.Tensor = self.__loss_fn(prediction, input_label)
            input_loss.backward()
            # print(f"__discriminator input loss:{input_loss}")

            # 生成图片，随机生成会导致部分数值训练得多，部分数值训练得少，input_label 中每个数值的数量是一样的，因此跟随它进行生成
            gen_input, gen_label = make_input_batch(input_label, self.__generator.input_length) # 
            gen_input, gen_label = gen_input.to(self.__device), gen_label.to(self.__device)
            gen_img:torch.Tensor = self.__generator(gen_input)

            # 用生成的图片训练判别器
            prediction:torch.Tensor = self.__discriminator(gen_img)
            gen_loss:torch.Tensor = self.__loss_fn(prediction, gen_label + 10)
            gen_loss.backward()
            # print(f"__discriminator gen loss:{gen_loss}")

            # 更新判别器权重
            self.__discriminator_optimizer.step()

            # 训练生成器
            self.__generator.train()
            self.__discriminator.eval()
            self.__generator_optimizer.zero_grad()

            # 生成图片
            gen_input, gen_label = make_input_batch(input_label, self.__generator.input_length)
            gen_input, gen_label = gen_input.to(self.__device), gen_label.to(self.__device)
            gen_img:torch.Tensor = self.__generator(gen_input)

            # 用生成的图片训练生成器
            gen_loss:torch.Tensor = self.__loss_fn(self.__discriminator(gen_img), gen_label)
            gen_loss.backward()
            # print(f"__generator gen loss:{gen_loss}")

            # 更新生成器权重
            self.__generator_optimizer.step()

    def generate(self, values:list[int]) -> tuple[torch.Tensor, float, float]:
        with torch.no_grad():
            self.__generator.eval()
            self.__discriminator.eval()

            # 生成图片
            gen_input, gen_label = make_input_batch(values, self.__generator.input_length)
            gen_input, gen_label = gen_input.to(self.__device), gen_label.to(self.__device)
            gen_imgs:torch.Tensor = self.__generator(gen_input)

            # 计算 loss
            prediction:torch.Tensor = self.__discriminator(gen_imgs)
            loss_G:torch.Tensor = self.__loss_fn(prediction, gen_label)
            loss_D:torch.Tensor = self.__loss_fn(prediction, gen_label + 10)

            # 归一化
            gen_imgs:torch.Tensor = (gen_imgs + 1) / 2
            return gen_imgs, loss_G.item(), loss_D.item()

    def state_dict(self) -> dict[str, any]:
        return {
            "generator": self.__generator.state_dict(),
            "discriminator": self.__discriminator.state_dict(),
            "generator_optimizer": self.__generator_optimizer.state_dict(),
            "discriminator_optimizer": self.__discriminator_optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict:dict[str, any]) -> None:
        if "generator" in state_dict:
            self.__generator.load_state_dict(state_dict["generator"])

        if "discriminator" in state_dict:
            self.__discriminator.load_state_dict(state_dict["discriminator"])

        if "generator_optimizer" in state_dict:
            self.__generator_optimizer.load_state_dict(state_dict["generator_optimizer"])

        if "discriminator_optimizer" in state_dict:
            self.__discriminator_optimizer.load_state_dict(state_dict["discriminator_optimizer"])


class Archive(object):
    def __init__(self, file:str) -> None:
        super().__init__()
        self.__file:str = file

    def load(self) -> None:
        try:
            self.__model = torch.load(self.__file, map_location="cpu")
        except FileNotFoundError:
            self.__model = {}

    def save(self) -> None:
        torch.save(self.__model, self.__file)

    def generator_loss(self) -> list[float]:
        try:
            return self.__model["generator_loss"]
        except:
            return []

    def discriminator_loss(self) -> list[float]:
        try:
            return self.__model["discriminator_loss"]
        except:
            return []
    
    def gan_state_dict(self) -> dict[str, any]:
        try:
            return self.__model["gan_state_dict"]
        except:
            return {}
    
    def set_generator_loss(self, generator_loss:list[float]) -> None:
        self.__model["generator_loss"] = generator_loss

    def set_discriminator_loss(self, discriminator_loss:list[float]) ->None:
        self.__model["discriminator_loss"] = discriminator_loss

    def set_gan_state_dict(self, gan_state_dict:dict[str, any]) -> None:
        self.__model["gan_state_dict"] = gan_state_dict
