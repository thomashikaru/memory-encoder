import torch
from encoder import MemoryEncoder
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Model Initialization
    model = MemoryEncoder()

    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8)

    epochs = 20
    outputs = []
    losses = []
    for epoch in range(epochs):
        for (image, _) in loader:

            # Reshaping the image to (-1, 784)
            image = image.reshape(-1, 28 * 28)

            # Output of Autoencoder
            reconstructed = model(image)

            # Calculating the loss function
            loss = loss_function(reconstructed, image)

            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            losses.append(loss)
            outputs.append((epochs, image, reconstructed))

    # Defining the Plot Style
    plt.style.use("fivethirtyeight")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    # Plotting the last 100 values
    plt.plot(losses[-100:])
