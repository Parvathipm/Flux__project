using Flux
using JLD2
using MLUtils   # ✅ Import MLUtils for flatten function

# Step 1: Load the image tensor from HDF5 file
println("Loading the image tensor...")
data = jldopen("preprocessed_image_1.h5", "r") do file
    read(file, "image_tensor")
end

# Step 2: Reshape the tensor to fit the CNN model
input_data = reshape(data, (28, 28, 3, 1))

# Step 3: Define the CNN Model
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

# Step 4: Perform a forward pass (prediction)
println("Performing the forward pass...")
output = model(input_data)

# Step 5: Get the prediction
class_index = argmax(output)
if class_index == 1
    println("Prediction: 🌅 Serene Sunset")
else
    println("Prediction: 🏙️ Futuristic City")
end
