# import open3d as o3d
# import numpy as np
#
#
# print("->正在加载点云... ")
# pcd = o3d.io.read_point_cloud("bunny.pcd")
# print(pcd)

# # 将点云设置为灰色
# pcd.paint_uniform_color([0.5, 0.5, 0.5])
#
# # 建立KDTree
# pcd_tree = o3d.geometry.KDTreeFlann(pcd)
#
# # 将第1500个点设置为紫色
# pcd.colors[1500] = [0.5, 0, 0.5]
#
# # 使用K近邻，将第1500个点最近的5000个点设置为蓝色
# print("使用K近邻，将第1500个点最近的5000个点设置为蓝色")
# k = 5000    # 设置K的大小
# [num_k, idx_k, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], k)    # 返回邻域点的个数和索引
# np.asarray(pcd.colors)[idx_k[1:], :] = [0, 0, 1]  # 跳过最近邻点（查询点本身）进行赋色
# print("k邻域内的点数为：", num_k)
#
# # 使用半径R近邻，将第1500个点半径（0.02）范围内的点设置为红色
# print("使用半径R近邻，将第1500个点半径（0.02）范围内的点设置为红色")
# radius = 0.02   # 设置半径大小
# [num_radius, idx_radius, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], radius)   # 返回邻域点的个数和索引
# np.asarray(pcd.colors)[idx_radius[1:], :] = [1, 0, 0]  # 跳过最近邻点（查询点本身）进行赋色
# print("半径r邻域内的点数为：", num_radius)
#
# # 使用混合邻域，将半径R邻域内不超过max_num个点设置为绿色
# print("使用混合邻域，将第1500个点半径R邻域内不超过max_num个点设置为绿色")
# max_nn = 200   # 半径R邻域内最大点数
# [num_hybrid, idx_hybrid, _] = pcd_tree.search_hybrid_vector_3d(pcd.points[1500], radius, max_nn)
# np.asarray(pcd.colors)[idx_hybrid[1:], :] = [0, 1, 0]  # 跳过最近邻点（查询点本身）进行赋色
# print("混合邻域内的点数为：", num_hybrid)
#
# print("->正在可视化点云...")
# o3d.visualization.draw_geometries([pcd])






import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("test.pcd")
print(pcd)

print("->正在DBSCAN聚类...")
eps = 0.5           # 同一聚类中最大点间距
min_points = 50     # 有效聚类的最小点数
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps, min_points, print_progress=True))
max_label = labels.max()    # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])

