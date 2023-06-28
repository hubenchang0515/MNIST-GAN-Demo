#! /usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import argparse
from model import Generator, Archive, make_input_batch

def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST GAN Demo")
    parser.add_argument("-m", "--model", help="model file", type=str, default="model.pt")
    parser.add_argument("-o", "--output", help="output file", type=str, default="generate.png")
    args = parser.parse_args()

    device:str = "cuda" if torch.cuda.is_available() else "cpu"

    generator = Generator(1024).to(device)

    archive = Archive(args.model)
    archive.load()
    generator.load_state_dict(archive.gan_state_dict()["generator"])

    with torch.no_grad():
        gen_input, _ = make_input_batch([0,1,2,3,4,5,6,7,8,9], generator.input_length)
        gen_input = gen_input.to(device)
        imgs = generator(gen_input)

    plt.figure("preview")
    fig, axs = plt.subplots(1, len(gen_input), figsize=(len(gen_input), 1))
    for i, img in enumerate(imgs.cpu()):
        axs[i].imshow(img.numpy()[0], cmap="gray")
        axs[i].axis("off")
    plt.savefig(args.output)
    plt.close()
        
    
if __name__ == "__main__":
    main()