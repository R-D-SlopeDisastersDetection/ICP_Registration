import open3d as o3d
import numpy as np

# 加载点云
source = o3d.io.read_point_cloud("dataset_reg/scnu_079_20m_4ms_box_face_only.ply")
target = o3d.io.read_point_cloud("dataset_reg/scnu_066_20m_2ms_box_faceonly.ply")

# 计算法线
source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# # 特征提取
radius_feature = 0.1
source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    source,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    target,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

# 设定配准参数
distance_threshold = 0.2
ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source, target, source_fpfh, target_fpfh, True, distance_threshold,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
    o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

# 精配准
distance_threshold = 0.02
icp_result = o3d.pipelines.registration.registration_icp(
    source, target, distance_threshold, ransac_result.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())

print("ICP result: \n", icp_result)
print("ICP Transformation Matrix: \n", icp_result.transformation)

print("RANSAC result: \n", ransac_result)
print("RANSAC Transformation Matrix: \n", ransac_result.transformation)

# 应用 RANSAC 变换
source.transform(ransac_result.transformation)

# 应用 ICP 变换
source.transform(icp_result.transformation)

# 可视化配准结果
source.paint_uniform_color([1, 0, 0])  # 红色表示源点云
target.paint_uniform_color([0, 1, 0])  # 绿色表示目标点云
o3d.visualization.draw_geometries([source, target])
