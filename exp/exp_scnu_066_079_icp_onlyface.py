import open3d as o3d
import similarity as sim


ply_079 = o3d.io.read_point_cloud("dataset_reg/scnu_079_20m_4ms_box_face_only.ply")
ply_066 = o3d.io.read_point_cloud("dataset_reg/scnu_066_20m_2ms_box_faceonly.ply")
ply_066_icp = o3d.io.read_point_cloud("dataset_reg/scnu_066_20m_2ms_box_single_onlyface_icp.ply")
ply_079_icp = o3d.io.read_point_cloud("dataset_reg/scnu_079_20m_4ms_box_onlyface_icp.ply")

hausdorff = sim.hausdorff_distance(ply_066, ply_079)
hausdorff_icp = sim.hausdorff_distance(ply_066_icp, ply_079_icp)
print("Hausdorff Distance in SCNU_066 and SCNU_079 before ICP Reg is ", hausdorff, ", after ICP Reg is ", hausdorff_icp)

p2p = sim.point2point_mean_and_std_deviation(ply_066, ply_079)
p2p_icp = sim.point2point_mean_and_std_deviation(ply_066_icp, ply_079_icp)
print("P2P Std and Mean in SCNU_066 and SCNU_079 before ICP Reg is ", p2p, ", after ICP Reg is ", p2p_icp)

# voxel_sim = sim.voxel_similarity(ply_066, ply_079)
# voxel_sim_icp = sim.voxel_similarity(ply_066_icp, ply_079_icp)
# print("Voxel Similarity in SCNU_066 and SCNU_079 before ICP Reg is ", p2p, ", after ICP Reg is ", p2p_icp)