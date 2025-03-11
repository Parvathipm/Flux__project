# 🧠 Flux CNN Model for Image Classification  
This project demonstrates a simple **Convolutional Neural Network (CNN)** model built using **Flux.jl** in **Julia**.  
The goal is to **load a preprocessed image tensor**, pass it through a **CNN model**, and generate a **forward pass output**.  

---

## 📜 Project Structure  
The project contains the following files:  

- **model.jl** → The main Julia script containing the CNN model and forward pass.  
- **preprocessed_image_1.bson** → The image tensor saved in **BSON format**.  
- **README.md** → Project description and setup instructions.  

---

## ✅ Requirements  
To run this project, you need to install the following Julia packages:  

| Package   | Purpose                                        |
|-----------|------------------------------------------------|
| **Flux.jl** | For building and running the CNN model.       |
| **BSON.jl** | For loading and saving the image tensor.      |
| **MLUtils.jl** | For flattening the CNN model output.       |

---

## 💻 Installation  
Follow these steps to set up your environment:  

### **Step 1: Install Julia Packages**  
Open your **Julia REPL (terminal)** and run the following commands:  

```julia
using Pkg
Pkg.add("Flux")
Pkg.add("BSON")
Pkg.add("MLUtils")
---
### **hj**
