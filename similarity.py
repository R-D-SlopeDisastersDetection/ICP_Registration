import open3d as o3d
import numpy as np
import math


def hausdorff_distance(pcd1, pcd2):
    """
    Hausdorff 距离衡量两个点集之间的最大最小距离，是评估点云相似度的一种严格标准。
    :param path1: 点云1路径
    :param path2:  点云2路径
    :return: hausdorff distance
    """

    # 计算 Hausdorff 距离
    distances = pcd1.compute_point_cloud_distance(pcd2)
    hausdorff = np.max(distances)
    print(f"Hausdorff distance: {hausdorff_distance}")
    return hausdorff


def point2point_mean_and_std_deviation(pcd1, pcd2, console_output=False, output_zero=True):
    """
    返回两个点云的平均值差和方差
    :param pcd1:
    :param pcd2:
    :param console_output:
    :param output_zero: 是否将nan替换为0值
    :return: mean_distance: 平均值距离
    :return: std_distance: 标准差距离
    """
    # 计算点到点距离
    distances = pcd1.compute_point_cloud_distance(pcd2)
    mean_distance = 0 if math.isnan(np.mean(distances)) and output_zero else np.mean(distances)
    std_distance = 0 if math.isnan(np.std(distances)) and output_zero else np.std(distances)

    if console_output:
        print(f"Mean distance: {mean_distance}")
        print(f"Standard deviation of distances: {std_distance}")
    return mean_distance, std_distance


def compute_repeated_part(pcd1, pcd2):
    """
    使用 ICP（Iterative Closest Point）算法对齐两个点云，然后计算重叠的点数或比例。
    :param pcd2:
    :param pcd1:
    :return: overlap_ratio: 重叠率
    """

    # 使用 ICP 进行点云配准
    threshold = 0.02
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print(f"Transformation matrix:\n{reg_p2p.transformation}")

    # 计算重叠部分
    pcd1.transform(reg_p2p.transformation)
    distances = pcd1.compute_point_cloud_distance(pcd2)
    overlap_ratio = np.sum(np.asarray(distances) < threshold) / len(distances)
    print(f"Overlap ratio: {overlap_ratio}")
    return overlap_ratio


def voxel_similarity(pcd1, pcd2):
    """
    将点云转换为体素网格，然后计算体素网格的重叠率。
    :param path1: 点云1路径
    :param path2:  点云2路径
    :return: voxel_similarity: 体素相似度
    """

    # 设置体素大小
    voxel_size = 0.02

    # 将点云转换为体素网格
    voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd1, voxel_size)
    voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd2, voxel_size)

    # 计算体素网格相似度
    intersection = voxel_grid1 + voxel_grid2
    union = voxel_grid1 | voxel_grid2
    voxel_similarity = len(intersection.voxels) / len(union.voxels)
    print(f"Voxel similarity: {voxel_similarity}")
    return voxel_similarity


def compute_z_axis(pcd_a, pcd_b):
    distances, indices = pcd_a.compute_point_cloud_distance(pcd_b)
    closest_points_a = pcd_a.points[np.asarray(indices)]
    closest_points_b = pcd_b.points
    print(closest_points_b[1])




