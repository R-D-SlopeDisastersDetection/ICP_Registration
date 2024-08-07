import open3d as o3d
import copy
import numpy as np
import similarity as sim


def draw_registration_result(source, target, transformation):
    """
    :param source:
    :param target:
    :param transformation:
    :return:
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, size, down_sample=False):
    """
    :param down_sample: bool, use the voxel down sample or not
    :param pcd: open3d.geometry.PointCloud, origin CloudPoint Object
    :param voxel_size: the size of voxel in meter, 0.05 as usual
    :return: pcd: pointcloud after processed
    :return: pcd_fpfh:The FPFH feature extract result of original pcd
    """
    if down_sample:
        # 体素线下采样
        print(":: Downsample with a voxel size %.3f." % size)
        pcd = pcd.voxel_down_sample(size)

    # 法线计算
    radius_normal = size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # FPFH特征提取
    radius_feature = size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh


def prepare_dataset(path1, path2, size, voxel_sample=False):
    """
    :param path1: PointCloud1's Path
    :param path2: PointCloud2's Path
    :param voxel_sample:
    :param size:
    :return:
    """
    print(":: Load two point clouds and disturb initial pose.")

    source = o3d.io.read_point_cloud(path1)
    target = o3d.io.read_point_cloud(path2)
    # What is the mean of Trans Matrix?
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    # Feature Extraction
    source_down, source_fpfh = preprocess_point_cloud(source, size, voxel_sample)
    target_down, target_fpfh = preprocess_point_cloud(target, size, voxel_sample)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source, target, source_fpfh,
                                target_fpfh, voxel_size):
    """
    :param source:
    :param target:
    :param source_fpfh:
    :param target_fpfh:
    :param voxel_size:
    :return:
    """
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def execute_icp_registration_by_time(source, target, result_ransac, count, threshold = 0.05):
    trans_init = result_ransac.transformation
    '''
    fitness，用于测量重叠面积（内点对应数/目标点数）。 值越高越好。
    inlier_rmse，它测量所有内点对应的 RMSE。越低越好。
    '''
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)

    while count > 0:
        print("Apply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        # draw_registration_result(source, target, reg_p2p.transformation)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        count = count-1

    draw_registration_result(source, target, reg_p2p.transformation)
    return source, target, reg_p2p


def execute_icp_registration_by_rmse(source, target, result_ransac, threshold=0.05 , rmse_threshold=0.05):
    trans_init = result_ransac.transformation

    '''
    fitness，用于测量重叠面积（内点对应数/目标点数）。 值越高越好。
    inlier_rmse，它测量所有内点对应的 RMSE。越低越好。
    '''
    print("Initial alignment")
    reg_p2p = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(reg_p2p)

    while reg_p2p.inlier_rmse > rmse_threshold:
        print("Apply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        # draw_registration_result(source, target, reg_p2p.transformation)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        count = count-1

    draw_registration_result(source, target, reg_p2p.transformation)
    return source, target, reg_p2p


def evaluation_matrix(pcd1, pcd2, reg_p2p):
    hausdorff_distance = sim.hausdorff_distance(pcd1, pcd2)
    mean, std = sim.point2point_mean_and_std_deviation(target)
    print("The Hausdorff Distance is ", hausdorff_distance)
    print("The mean is ", mean, ", the std is ", std)
    print("The inliner RMSE is ", reg_p2p.inlier_rmse, ", the fitness is ", reg_p2p.fitness)
    draw_registration_result(source, target, reg_p2p.transformation)


path1 = "dataset_reg/scnu_066_20m_2ms_box_faceonly.pcd"
path2 = "dataset_reg/scnu_079_20m_4ms_box_face_only.pcd"
o3d.visualization.draw_geometries([o3d.io.read_point_cloud(path1), o3d.io.read_point_cloud(path2)])
size = 0.01  # means 5cm for this dataset
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
    path1, path2, size, False)
result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            size)
print(result_ransac)
# draw_registration_result(source_down, target_down, result_ransac.transformation)

mean, std = sim.point2point_mean_and_std_deviation(source, target)
print("RANSAC Finish!\nThe mean is ", mean, ", the std is ", std)

source, target, reg_result = execute_icp_registration_by_time(source, target, result_ransac, 10)
evaluation_matrix(source, target, reg_result)



