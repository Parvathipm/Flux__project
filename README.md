
# Project Title

This project demonstrates a simple Convolutional Neural Network (CNN) model built using Flux.jl in Julia. The goal is to load a preprocessed image tensor, pass it through a CNN model, and generate a forward pass output.




## Project Structure
The project is divided into three parts as mentioned below:

- **Image Generation**: Uses the Stable Diffusion model to generate realistic images from text prompts.  
- **Image Preprocessing**: Converts generated images to tensor format, resizes them, and normalizes them for input into a CNN model.  
- **CNN Model**: Builds a Convolutional Neural Network (CNN) using Flux.jl in Julia to classify images.

## Image Generation
The first step is to generate images using Stable Diffusion v1.5.

📜 Code for Image Generation

The code below uses the diffusers library to generate realistic images based on the text prompt provided.

    from diffusers import StableDiffusionPipeline

    import torch

Load Stable Diffusion model

    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/   stable-diffusion-v1-5")

Set to GPU if available, otherwise CPU

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)

Text prompt for generating images

    prompt = "a serene sunset over a futuristic city"

Generate images


    for i in range(1, 2):
        image = pipeline(prompt).images[0]
        image.save(f"generated_image_{i}.png")
        print(f"Image {i} saved as 'generated_image_{i}.png'")

## Image Preprocessing

The second step is to preprocess the generated images to convert them into tensors and normalize them.

📜 Code for Image Preprocessing
The code below uses torchvision to resize the image and convert it into tensor format.

    import torch
    from torchvision import transforms
    from PIL import Image

Define image preprocessing steps

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
    ])

Preprocess each image

    for i in range(1, 2):
    image = Image.open(f"generated_image_{i}.png")
    tensor_image = transform(image)

Save the tensor as a .pt file

    torch.save(tensor_image, f"preprocessed_image_{i}.pt")
    print(f"Image {i} preprocessed and saved as 'preprocessed_image_{i}.pt'")

##  Building CNN Model in Julia

Code for Building CNN Model

The code below builds and applies a CNN model in Julia.
using Flux
using JLD2
using MLUtils   ✅ Import MLUtils for flatten function

Load the image tensor from HDF5 file

    println("Loading the image tensor...")
    data = jldopen("preprocessed_image_1.h5", "r") do file
    read(file, "image_tensor")
    end

Reshape the tensor to fit the CNN model

    input_data = reshape(data, (28, 28, 3, 1))

Define the CNN Model

    println("Creating the CNN model...")
    model = Chain(
    Conv((3, 3), 3 => 32, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 32 => 64, relu),
    MaxPool((2, 2)),
    MLUtils.flatten,  # ✅ This is the correct way to flatten now
    Dense(1600, 128, relu),
    Dense(128, 2)
)

Perform a forward pass (prediction)

    println("Performing the forward pass...")
    output = model(input_data)

Get the prediction

    class_index = argmax(output)
    if class_index == 1
        println("Prediction: 🌅 Serene Sunset")
    else
        println("Prediction: 🏙️ Futuristic City")
    end

## Output

The CNN model will classify the image as either:

🌅 Serene Sunset

🏙️ Futuristic City
## Environment Setup

To run this project, make sure you have the following packages installed.

Python Libraries

    pip install torch torchvision diffusers pillow

Julia Libraries

    using Pkg
    Pkg.add("Flux")
    Pkg.add("JLD2")
    Pkg.add("MLUtils")


## Challenges Encountered

Here are some of the key challenges encountered while working on this project:

Image Generation Delay:
 Generating high-quality images using Stable Diffusion took considerable time on low-end hardware.

Tensor Reshape Issue:
 While feeding data into CNN, reshaping the tensor was slightly complex due to image size constraints.

Flux.jl Compatibility:
 Initially faced issues while using flatten() function in Flux.jl, later resolved it by using MLUtils.flatten.
## Assumptions Made

The generated images are assumed to have a size of 224x224 pixels for processing.

The CNN model has a simple architecture with two convolutional layers and two dense layers.

The output classes were binary (Sunset or Futuristic City).
GPU is recommended for faster image generation.
