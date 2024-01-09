import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.decomposition import PCA

# Load the MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()

# Number of samples per digit
samples_per_digit = 500

# Initialize arrays to store selected data
selected_x_train = []

# Randomly select 500 samples per digit
for digit in range(10):
    digit_indices = np.where(y_train == digit)[0]
    selected_indices = np.random.choice(digit_indices, samples_per_digit, replace=False)
    selected_x_train.extend(x_train[selected_indices])

# Convert selected data to NumPy array and flatten to 1D
selected_x_train = np.array(selected_x_train).reshape(-1, 28 * 28)

# Display 50 images from your own dataset
amount = 50
lines = 5
columns = 10

fig = plt.figure()
for i in range(amount):
    ax = fig.add_subplot(lines, columns, 1 + i)
    # Ensure that the image data is in the correct range (0-255)
    image = selected_x_train[i].astype(np.uint8).reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.sca(ax)
    ax.set_xticks([])
    ax.set_yticks([])
plt.savefig("Q4_1.png")
plt.show()

#----------------------------------------------------------------------------------------------------------
# 4.2
# Normalize the data
mean = np.mean(selected_x_train, axis=0)
std = np.std(selected_x_train, axis=0)
normalized_x_train = (selected_x_train - mean) / (std + 1e-8)  # Adding a small constant to avoid division by zero

# Compute the covariance matrix
cov_matrix = np.cov(normalized_x_train.T)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenpairs by descending eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Printing the top eigenvalues
top_eigenvalues = eigenvalues[:10]
print("Top 10 Eigenvalues:")
print(top_eigenvalues)

#----------------------------------------------------------------------------------------------------------
# 4.3
# Define the desired reduced dimensions
reduced_dimensions = [500, 300, 100, 50]

for k in reduced_dimensions:
    # Perform PCA with the specified dimensionality
    pca = PCA(n_components=k)
    reduced_data = pca.fit_transform(normalized_x_train)

    # Reconstruct the data
    reconstructed_data = pca.inverse_transform(reduced_data)

    # Display 10 decoding results for each digit
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for digit in range(10):
        digit_indices = np.where(y_train == digit)[0]

        # Ensure there are at least 10 samples for this digit
        if len(digit_indices) >= 10:
            for i in range(10):
                ax = axes[digit, i]
                original_image = selected_x_train[digit_indices[i]].reshape(28, 28)
                reconstructed_image = reconstructed_data[digit_indices[i]].reshape(28, 28)
                ax.imshow(reconstructed_image, cmap='binary')
                ax.set_xticks([])
                ax.set_yticks([])
    plt.suptitle(f"Decoding Results (k={k})", fontsize=16)
    plt.savefig(f"Q4_3_(k={k})", fontsize=16)
    plt.show()
