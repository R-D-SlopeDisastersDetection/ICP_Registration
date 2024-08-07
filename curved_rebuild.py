import open3d as o3d
import numpy as np


# ---------------------- 定义点云体素化函数 ----------------------
def get_mesh(_relative_path):
    mesh = o3d.io.read_triangle_mesh(_relative_path)
    mesh.compute_vertex_normals()
    return mesh


def alpha_shape_rebuild(path):
    # Alpha Shape
    # --------------------------- 加载点云 ---------------------------
    print("->正在加载点云... ")
    pcd = o3d.io.read_point_cloud(path)
    print("原始点云：", pcd)
    # ==============================================================

    # ------------------------- Alpha shapes -----------------------
    alpha = 0.01
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    # ==============================================================


def ball_pivoting_rebuild(path):
    # ------------------------- Ball pivoting --------------------------
    print("->Ball pivoting...")
    N = 20000  # 将点划分为N个体素
    pcd = get_mesh(path).sample_points_poisson_disk(N)
    o3d.visualization.draw_geometries([pcd])

    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    # o3d.visualization.draw_geometries([pcd, rec_mesh])
    # 保存结果为文件
    o3d.io.write_triangle_mesh("mesh/bpa_mesh.ply", rec_mesh)
    # ==============================================================


def poisson_mesh_rebuild(path):
    # 读取点云数据
    pcd = o3d.io.read_point_cloud(path)

    # 估计点云的法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    # 检查是否成功估计法线
    if not pcd.has_normals():
        raise ValueError("点云数据没有法线信息，法线估计失败")
    else:
        print("法线创建成功")

    # 执行泊松曲面重建
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)

    # 裁剪低密度的三角形，去除可能的噪声
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    poisson_mesh.remove_vertices_by_mask(vertices_to_remove)

    # 保存结果为文件
    o3d.io.write_triangle_mesh("mesh/poisson_mesh3.ply", poisson_mesh)

    # 可视化结果
    # o3d.visualization.draw_geometries([poisson_mesh], mesh_show_back_face=True)
