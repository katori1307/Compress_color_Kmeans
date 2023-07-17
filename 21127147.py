import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
import PIL as pil
import time

def assign_label(img_1d, centroids):
    distances = np.linalg.norm((img_1d[:, None] - centroids), axis=2)
    return np.argmin(distances, axis=1)

def initialize_centroids(k_cluster, img_1d, init_centroids):
    if init_centroids == 'random':
        return np.random.choice(256, size=(k_cluster, img_1d.shape[1]), replace=False)
    elif init_centroids == 'in_pixels':
        random_indices = np.random.choice(img_1d.shape[0], size=k_cluster, replace=False)
        return img_1d[random_indices]
    else:
        raise ValueError("init_centroids must be 'random' or 'in_pixels'")

def update_centroids(img_1d, labels, k_cluster):
    new_centroids = np.zeros((k_cluster, img_1d.shape[1]))
    for i in range(k_cluster):
        cluster_points = np.where(labels == i)[0]
        # centroid = np.mean(img_1d[cluster_points], axis=0)
        # kiểm tra kích thước của cluster_points:
        if cluster_points.size > 0:
            centroid = np.mean(img_1d[cluster_points], axis=0)
        else:
            centroid = np.zeros(img_1d.shape[1])
        new_centroids[i] = centroid
    return new_centroids

def convergence_check(old_centroids, new_centroids):
    if old_centroids.shape != new_centroids.shape:
        return False
    else:
        return np.allclose(old_centroids, new_centroids)


def kmeans(img_1d, k_cluster, max_iter, init_centroids):
    centroids = initialize_centroids(k_cluster, img_1d, init_centroids)
    labels = np.zeros((img_1d.shape[0], img_1d.shape[1]))

    while max_iter:
        labels = assign_label(img_1d=img_1d, centroids=centroids)
        old_centroids = centroids
        centroids = update_centroids(img_1d, labels, k_cluster)
        if convergence_check(old_centroids, centroids):
            break
        max_iter -= 1

    modified_img = np.zeros((img_1d.shape[0], img_1d.shape[1]))
    for i in range(k_cluster):
        modified_img[labels == i] = centroids[i]
    return centroids, labels, modified_img



if __name__ == '__main__':
    # pic_name = input('Enter pictures name (including the suffix (.jpg)): ')
    pic_name = 'loinho.jpg'
    img = mpimage.imread(pic_name)
    # print('Before:\n', img)
    img_1d = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    k_cluster = int(input('Enter k_clusters: '))
    # print('img_1g:\n', img_1d)
    # print()
    max_iter = 100
    init_centroids = 'random'

    # centroids = initialize_centroids(k_cluster, img_1d, init_centroids)
    # print('centroids:\n', centroids)
    # # labels = print(assign_label(img_1d, centroids))
    # distances = np.linalg.norm((img_1d[:, None] - centroids), axis=2)
    # # print(distances)
    # print()
    # labels = np.argmin(distances, axis=1)
    # print(labels)
    # print(img_1d[:, None])
    
    start_time = time.time()
    centroids, labels, new_img = kmeans(img_1d, k_cluster, max_iter, init_centroids)
    end_time = time.time()
    new_img = new_img.reshape((img.shape[0], img.shape[1], img.shape[2]))
    print('Centroids: ', centroids)
    print('Labels: ', labels)
    print('Execution time: ', end_time - start_time, 's')
    plt.imshow(new_img.astype('uint8'))
    pic_name = pic_name.replace('.jpg', '')
    # plt.savefig(pic_name + '_' + str(k_cluster) + '_clusters' + '.png')
    # plt.show()
    