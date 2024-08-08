import open3d as o3d
import copy
import numpy as np
import similarity as sim


def draw_registration_result(source, target, transformation):
    """
    Use to show two cloud point, this function will draw the blue and yellow to cloud point
    , and move and rotate the source cloud point according to the transformation matrix
    :param source: open3d.geometry.PointCloud, source cloud point
    :param target: open3d.geometry.PointCloud, target cloud point
    :param transformation:`4 x 4` float64 numpy array: The estimated transformation matrix
    :return: 
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def fpfh_feature_extract(pcd, size):
    """
    Use Fast Point Feature Histogram(FPFH) Feature Extraction to get the feature of Cloud Point
    :param pcd: open3d.geometry.PointCloud, origin CloudPoint Object
    :param size: int, the resolution of process, the resolution of Estimate Normals, FPFH is 2*size, 5*size
    :return: pcd_fpfh: open3d.geometry.PointCloud, The FPFH feature extract result of original pcd
    """

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
    return pcd_fpfh


def preprocess_dataset(source, target, size):
    """
    Initial the Cloud Point Data and extract the feature of the Cloud Point
    :param source: open3d.geometry.PointCloud, PointCloud1
    :param target: open3d.geometry.PointCloud, PointCloud2
    :param size: int ,the resolution of process, it will be used in Estimate Normals and FPFH
    In details, the resolution of Estimate Normals, FPFH is 2*size, 5*size respectively
    :return: open3d.geometry.PointCloud, return the source and target and their FPFH feature extract result
    """
    print(":: Load two point clouds and disturb initial pose.")

    # What is the mean of Trans Matrix?
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    # Feature Extraction
    source_fpfh = fpfh_feature_extract(source, size)
    target_fpfh = fpfh_feature_extract(target, size)
    return source, target, source_fpfh, target_fpfh


def execute_global_registration(source, target, source_fpfh,
                                target_fpfh, threshold):
    """
    Global Registration with RANSAC
    :param source: open3d.geometry.PointCloud, the Cloud Point that need to be moved and rotated to registrate the
                target Cloud Point
    :param target: open3d.geometry.PointCloud,, the Cloud Point that stay still and the target that source Cloud Point
                need to registrate
    :param source_fpfh: open3d.geometry.PointCloud, source Cloud Point Feature
    :param target_fpfh: open3d.geometry.PointCloud, target Cloud Point Feature
    :param threshold: float, distance_threshold in RANSAC
    :return: open3d.pipelines.registration.RegistrationResult, result of RANSAC
    """
    distance_threshold = threshold * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
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


def apply_icp_registration(source, target, threshold, trans_init):
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
    return reg_p2p


def execute_icp_registration(source, target, result_ransac, iter_method, iter_threshold, dis_threshold=0.05):
    """
    Accurate Registration with ICP
    :param source: the Cloud Point that need to be moved and rotated to registrate the target Cloud Point
    :param target: the Cloud Point that stay still and the target that source Cloud Point need to registrate
    :param result_ransac: open3d.pipelines.registration.RegistrationResult, the Global Registration Result, included the
        fitness, rmse and transformation
    :param iter_method: string, use "rmse", "fitness" or "max_iteration" to modify the iteration method
    :param iter_threshold: the threshold of iteration
    :param dis_threshold: float, distance threshold, radius of K-NN in ICP
    :return:
        open3d.geometry.PointCloud
        open3d.geometry.PointCloud
        open3d.pipelines.registration.RegistrationResult
    """
    trans_init = result_ransac.transformation
    '''
    fitness，用于测量重叠面积（内点对应数/目标点数）。 值越高越好。
    inlier_rmse，它测量所有内点对应的 RMSE。越低越好。
    '''
    print("Initial alignment")
    reg_p2p = o3d.pipelines.registration.evaluate_registration(
        source, target, dis_threshold, trans_init)
    print(reg_p2p)

    print("Apply point-to-point ICP")
    if iter_method == "rmse":
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, dis_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_rmse=iter_threshold))
    elif iter_method == "fitness":
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, dis_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=iter_threshold))
    elif iter_method == "max_iteration":
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, dis_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iter_threshold))
    else:
        pass
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)

    draw_registration_result(source, target, reg_p2p.transformation)
    return source, target, reg_p2p


def execute_icp_registration_by_rmse(source, target, result_ransac, threshold=0.05, rmse_threshold=0.05):
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
        count = count - 1

    draw_registration_result(source, target, reg_p2p.transformation)
    return source, target, reg_p2p


def evaluation_matrix(pcd1, pcd2, reg_p2p):
    """
    Compare the difference of pcd1 and pcd2, includes Hausdorff Distance , Mean Distance, Standard Deviation
    , Fitness, RMSE and Correspondence Set
    :param pcd1:open3d.geometry.PointCloud
    :param pcd2:open3d.geometry.PointCloud
    :param reg_p2p: result of reg, return from execute_icp_registration
    :return:
    """
    hausdorff_distance = sim.hausdorff_distance(pcd1, pcd2)
    mean, std = sim.point2point_mean_and_std_deviation(pcd1, pcd2)
    print("The Hausdorff Distance is ", hausdorff_distance)
    print("The mean is ", mean, ", the std is ", std)
    print("The inliner RMSE is ", reg_p2p.inlier_rmse, ", the fitness is ", reg_p2p.fitness)
    draw_registration_result(pcd1, pcd2, np.identity(4))
    # o3d.visualization.draw_geometries([pcd1, pcd2])


def cloudpoint_registration(path1, path2, size=0.05):
    """
    :param path1: PointCloud1's path
    :param path2: PointCloud2's path
    :param size: data process iteration size
    :return:
    """
    o3d.visualization.draw_geometries([o3d.io.read_point_cloud(path1), o3d.io.read_point_cloud(path2)])
    pcd1 = o3d.io.read_point_cloud(path1)
    pcd2 = o3d.io.read_point_cloud(path2)
    source, target, source_fpfh, target_fpfh = preprocess_dataset(
        pcd1, pcd2, size)
    result_ransac = execute_global_registration(source, target,
                                                source_fpfh, target_fpfh,
                                                size)
    print(result_ransac)
    # draw_registration_result(source_down, target_down, result_ransac.transformation)

    mean, std = sim.point2point_mean_and_std_deviation(source, target)
    print("RANSAC Finish!\nThe mean is ", mean, ", the std is ", std)

    source, target, reg_result = execute_icp_registration(source, target, result_ransac, "max_iteration", 10000, 10)
    reg_source = source.transform(reg_result.transformation)

    evaluation_matrix(reg_source, target, reg_result)
    return reg_source, target, reg_result


path1 = "dataset_reg/scnu_066_20m_2ms_box_faceonly.pcd"
path2 = "dataset_reg/scnu_079_20m_4ms_box_face_only.pcd"
cloudpoint_registration(path1, path2, size=0.02)
