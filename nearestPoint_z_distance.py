import open3d as o3d
import numpy as np


def mean_z_difference(path1, path2):


    # 读取点云
    source = o3d.io.read_point_cloud(path1)
    target = o3d.io.read_point_cloud(path2)

    # 构建目标点云的KD树
    target_tree = o3d.geometry.KDTreeFlann(target)

    # 提取源点云和目标点云的点数据
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    # 存储每个源点到目标点云的最近点
    correspondences = []

    for source_point in source_points:
        # 搜索最近邻
        [_, idx, _] = target_tree.search_knn_vector_3d(source_point, 1)
        nearest_point = target_points[idx[0]]
        correspondences.append((source_point, nearest_point))

    # # 打印每个源点及其对应的最近点
    # for src, tgt in correspondences:
    #     print(f"Source point: {src}, Nearest point in target: {tgt}")

    # 存储每对点的Z轴差值
    z_differences = []

    for src, tgt in correspondences:
        # 提取每对点的Z轴坐标
        z_src = src[2]  # 假设 src 是一个三维点 (x, y, z)，那么 src[2] 是 Z 坐标
        z_tgt = tgt[2]  # tgt 同样是三维点 (x, y, z)，那么 tgt[2] 是 Z 坐标

        # 计算 Z 轴差值
        z_difference = abs(z_tgt - z_src)

        # 将差值添加到列表中
        z_differences.append(z_difference)

    # 计算 Z 轴差值的平均值
    average_z_difference = np.mean(z_differences)
    std_z_difference = np.std(z_differences)
    max_z_difference = np.max(z_differences)
    min_z_difference = np.min(z_differences)

    print(f"Average Z-axis difference: {average_z_difference}")
    print(f"Standard deviation of Z-axis difference: {std_z_difference}")
    print(f"Max Z-axis difference: {max_z_difference}")
    print(f"Min Z-axis difference: {min_z_difference}")


path1 = "071_072/071_box1top.ply"
path2 = "071_072/072_box1top_reg.ply"

mean_z_difference(path1, path2)