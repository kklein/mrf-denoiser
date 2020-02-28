import cv2
import numpy as np
import matplotlib.pyplot as plt


class Configuration:
    def __init__(self, potential_f1, potential_f2, n_epochs=10, alpha_1=1, alpha_2=.5,
                 file_path='imgs/mondrian.png', noise_parameter=.2):
        # Potential function defined between old and new graph.
        self.potential_f1 = potential_f1
        # Potential function defined within new graph.
        self.potential_f2 = potential_f2
        self.n_epochs = n_epochs
        # Hyperparameter of first potential function.
        self.alpha_1 = alpha_1
        # Hyperparameter of second potential function.
        self.alpha_2 = alpha_2
        self.file_path = file_path
        # 'Amount' of noise to be added to true image.
        self.noise_parameter = noise_parameter


def get_binarized_img(img):
    return (img > 128).astype(int)


def get_graph_from_img(img):
    result = img.copy()
    result[result == 0] = -1
    return result


def get_img_from_graph(graph):
    result = graph.copy()
    result[result == -1] = 0
    return result


def get_neighborhood_of_pixel(i, j, width, height):
    neighborhood = []
    if i - 1 >= 0:
        neighborhood.append((i - 1, j))
    if i + 1 < height:
        neighborhood.append((i + 1, j))
    if j - 1 >= 0:
        neighborhood.append((i, j - 1))
    if j + 1 < width:
        neighborhood.append((i, j + 1))
    return neighborhood


def get_neighborhood_sum(graph, i, j, width, height):
    neighborhood_sum = 0
    neighborhood = get_neighborhood_of_pixel(i, j, width, height)
    for (k, l) in neighborhood:
        neighborhood_sum += graph[k, l]
    return neighborhood_sum


def get_energy(cfg, new_value, value_1, value_2):
    potential_1 = cfg.potential_f1(new_value, value_1)
    potential_2 = cfg.potential_f2(new_value, value_2)
    return cfg.alpha_1 * potential_1 + cfg.alpha_2 * potential_2


def show_img(img, title):
    plt.imshow(img)
    plt.set_cmap('gray')
    plt.title(title)
    plt.axis("off")
    plt.show()


def main(cfg):
    img = cv2.imread(cfg.file_path, cv2.IMREAD_GRAYSCALE)
    true_img = get_binarized_img(img)

    (height, width) = true_img.shape
    noisy_img = np.copy(true_img)
    for i in range(height):
        for j in range(width):
            if np.random.binomial(1, cfg.noise_parameter):  # Flip value
                noisy_img[i, j] = 1 - true_img[i, j]

    # The potential functions are defined to work on binary values {-1, 1} instead of {0, 1}.
    old_graph = get_graph_from_img(noisy_img)
    new_graph = old_graph.copy()

    for epoch_index in range(cfg.n_epochs):
        for _ in range(height * height):
            i = np.random.choice(height)
            j = np.random.choice(width)

            neighborhood_sum = get_neighborhood_sum(new_graph, i, j, width, height)
            value_1 = -1
            energy_1 = get_energy(cfg, value_1, old_graph[i, j], neighborhood_sum)
            value_2 = 1
            energy_2 = get_energy(cfg, value_2, old_graph[i, j], neighborhood_sum)
            # If both energies are equal, don't update.
            if energy_1 > energy_2:
                new_graph[i, j] = value_1
            elif energy_2 > energy_1:
                new_graph[i, j] = value_2
        print(f'Finished epoch {epoch_index}.')
        print((old_graph != new_graph).sum())

    denoised_img = get_img_from_graph(new_graph)
    show_img(true_img, 'True Image')
    show_img(noisy_img, 'Noisy Image')
    show_img(denoised_img, 'Denoised Image')


if __name__ == "__main__":
    # Use same potential function with different values for both potentials.
    def potential_f(new_value, value):
        return new_value * value

    configuration = Configuration(potential_f, potential_f)
    main(configuration)
