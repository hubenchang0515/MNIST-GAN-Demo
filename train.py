#! /usr/bin/env python3

import torch
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.optim import Adadelta
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os
from model import Generator, Discriminator, GAN, Archive

def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST GAN Demo")
    parser.add_argument("-r", "--rate", help="learing rate", type=float, default=1e-3)
    parser.add_argument("-b", "--batch", help=f"batch size", type=int, default=64)
    parser.add_argument("-e", "--epoch", help=f"epoches", type=int, default=1000)
    parser.add_argument("-l", "--length", help=f"generator input length", type=int, default=1024)
    parser.add_argument("-i", "--interval", help=f"interval of preview", type=int, default=50)
    parser.add_argument("-m", "--model", help="model file", type=str, default="model.pt")
    args = parser.parse_args()

    device:str = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = datasets.MNIST(root="data", train=True, transform=ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=args.batch)

    generator = Generator(args.length)
    discriminator = Discriminator()

    optimizer_G = Adadelta(generator.parameters(), lr=args.rate)
    optimizer_D = Adadelta(discriminator.parameters(), lr=args.rate)

    loss_fn = CrossEntropyLoss()

    gan = GAN(device, dataloader, generator, discriminator, optimizer_G, optimizer_D, loss_fn)

    archive = Archive(args.model)
    archive.load()
    generator_loss:list[float] = archive.generator_loss()
    discriminator_loss:list[float] = archive.discriminator_loss()
    gan.load_state_dict(archive.gan_state_dict())

    for epoch in range(args.epoch):
        start:datetime = datetime.now()
        gan.train()
        imgs, loss_G, loss_D = gan.generate([i for i in range(10)])
        end:datetime = datetime.now()
        print(f"epoch:{epoch} loss_G:{loss_G} loss_D:{loss_D} time cost:{end - start}")

        plt.figure("epoch loss")
        generator_loss.append(loss_G)
        discriminator_loss.append(loss_D)
        x:list[input] = [i for i in range(len(generator_loss))]
        plt.plot(x, generator_loss, "s-", color="blue", label="loss of generator")
        plt.plot(x, discriminator_loss, "o-", color="green", label="loss of discriminator")
        plt.legend(loc=0)
        plt.savefig(f"./loss.png")
        plt.close()
        
        if len(generator_loss) % args.interval != 0:
            continue

        archive.set_generator_loss(generator_loss)
        archive.set_discriminator_loss(discriminator_loss)
        archive.set_gan_state_dict(gan.state_dict())
        archive.save()

        if not os.path.exists("./preview"):
            os.mkdir("./preview")

        plt.figure("preview")
        fig, axs = plt.subplots(1, 10, figsize=(10, 1))
        for i, img in enumerate(imgs.cpu()):
            axs[i].imshow(img.numpy()[0], cmap="gray")
            axs[i].axis("off")
        plt.savefig(f"./preview/epoch-{len(generator_loss):>04d}.png")
        plt.close()

        

        
    
if __name__ == "__main__":
    main()