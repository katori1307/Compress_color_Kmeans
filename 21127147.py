import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
import PIL as pil
import time

def assign_label(img_1d, centroids):
    # tính chuẩn (norm), khoảng cách giữa từng điểm ảnh trong img_1d tới centroids[i]
    distances = np.linalg.norm((img_1d[:, None] - centroids), axis=2)
    # với mỗi điểm ảnh, trả về centroids mà có khoảng cách tới điểm ảnh đó là nhỏ nhất
    return np.argmin(distances, axis=1)

def initialize_centroids(k_cluster, img_1d, init_centroids):
    if init_centroids == 'random':
        # với 'random': chọn ngẫu nhiên giá trị centroid là số nguyên trong khoảng [0, 255].
        return np.random.choice(256, size=(k_cluster, img_1d.shape[1]), replace=False)
    elif init_centroids == 'in_pixels':
        # với 'in_pixels': chọn ngẫu nhiên giá trị centroid là giá trị màu của các điểm ảnh trong img_1d.
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
            # gán centroid mới là trung bình của các điểm ảnh
            centroid = np.mean(img_1d[cluster_points], axis=0)
        else:
            # nếu kích thước cluster_points <= 0 thì centroid mới sẽ được khởi tạo là các giá trị 0
            centroid = np.zeros(img_1d.shape[1])
        new_centroids[i] = centroid
    return new_centroids

def convergence_check(old_centroids, new_centroids):
    if old_centroids.shape != new_centroids.shape:
        return False
    else:
        # nếu bộ centroids mới không có sự khác biệt đáng kể so với centroids cũ thì trả về True
        # nếu không thì trả về False
        return np.allclose(old_centroids, new_centroids)


def kmeans(img_1d, k_cluster, max_iter, init_centroids):
    # khởi tạo bộ centroids và labels.
    centroids = initialize_centroids(k_cluster, img_1d, init_centroids)
    labels = np.zeros((img_1d.shape[0], img_1d.shape[1]))
    # sử dụng vòng lặp với max_iter lần lặp
    while max_iter:
        labels = assign_label(img_1d=img_1d, centroids=centroids)
        # lưu bộ centroids hiện tại ra old_centroids để có thể tính toán sự khác biệt.
        old_centroids = centroids
        # cập nhật bộ centroids mới.
        centroids = update_centroids(img_1d, labels, k_cluster)
        # so sánh với bộ centroids cũ.
        if convergence_check(old_centroids, centroids):
            break
        max_iter -= 1
    # tạo 1 ma trận mới để lưu giá trị của từng điểm ảnh là bộ centroids đã được xử lý
    modified_img = np.zeros((img_1d.shape[0], img_1d.shape[1]))
    for i in range(k_cluster):
        modified_img[labels == i] = centroids[i]
    # trả về centroids, labels và modified_img
    return centroids, labels, modified_img



if __name__ == '__main__':  
    # nhập tên ảnh. VD: <ten>.jpg
    pic_name = input('Enter pictures name (including the suffix (.jpg)): ')
    # pic_name = 'loinho.jpg'
    img = mpimage.imread(pic_name)
    # thay đổi số chiều của img thì mảng 1 chiều với các phần tử là các điểm màu mang giá trị R, G, B
    img_1d = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    # nhập số cluster
    k_cluster = int(input('Enter k_clusters: '))
    max_iter = 100
    init_centroids = 'random'
    start_time = time.time()
    centroids, labels, new_img = kmeans(img_1d, k_cluster, max_iter, init_centroids)
    end_time = time.time()
    new_img = new_img.reshape((img.shape[0], img.shape[1], img.shape[2]))
    print('Centroids: ', centroids)
    print('Labels: ', labels)
    print('Execution time: ', end_time - start_time, 's')
    plt.imshow(new_img.astype('uint8'))
    pic_name = pic_name.replace('.jpg', '')
    plt.savefig(pic_name + '_' + str(k_cluster) + '_clusters' + '.png')
    