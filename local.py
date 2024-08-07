import copy
import curved_rebuild
import similarity as sim
import open3d as o3d
import numpy as np

#
# pcd = o3d.io.read_point_cloud("terra_pcd\\cloudb14209456efa07c_Block_6.pcd")
# pcd.estimate_normals()
# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
# o3d.io.write_triangle_mesh("mesh\\poisson_mesh2.ply", mesh)

# 读取mesh文件
# mesh = o3d.io.read_point_cloud("mesh\\Ground_points.pcd")

# translation = [0.0, 0.0, 100.0]

# pcd.translate(translation)
#
# vis = o3d.visualization.Visualizer()
# vis.create_window()
#
# vis.add_geometry(mesh)
# vis.run()
#
# vis.destroy_window()

# 进行深拷贝
# mesh_t = copy.deepcopy(mesh)
#
#
# 可视化mesh
# o3d.visualization.draw_geometries([mesh],
#                                     window_name = '可视化多个mesh',
#                                     mesh_show_wireframe = True,
#                                     mesh_show_back_face = True
# )


# curved_rebuild.poisson_mesh_rebuild("mesh\\Ground_points.pcd")
# ply = o3d.io.read_point_cloud("E:\\datasets\\dji_terra_data\\PCGSPRO_1718517410\\14718155972\\240717_Shanzhan_k533_90_LiDAR\lidars\\terra_point_ply\\cloud_merged.ply")
# chair = o3d.io.read_point_cloud("dataset_reg/scnu_078_20m_3ms_box_single.pcd")
#
# chair.paint_uniform_color([1, 0.706, 0])

# o3d.visualization.draw_geometries([chair])

# try:
#     # 使用内置示例点云数据
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(
#         [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
#     )
#
#     # 对点云进行整体着色（例如：红色）
#     pcd.paint_uniform_color([1, 0, 0])
#
#     # 可视化点云
#     o3d.visualization.draw_geometries([pcd],
#                                       zoom=0.7,
#                                       front=[0.5439, -0.2333, -0.8060],
#                                       lookat=[2.4615, 2.1331, 1.338],
#                                       up=[-0.1781, -0.9708, 0.1608])
# except Exception as e:
#     print(f"An error occurred: {e}")

ply = o3d.io.read_point_cloud("dataset_reg/scnu_066_20m_2ms_box_faceonly.ply")
# ply1 = o3d.io.read_point_cloud("dataset_reg/scnu_066_20m_2ms_box_faceonly.ply")
# translation = np.array([1.0, 2.0, 3.0])
# ply1.translate(translation)
# ply1.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([ply])




