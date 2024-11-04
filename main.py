import math

from BlockAnalyze.BlockAnalyze import *
from BlockAnalyze.BlockAnanlyzeByChange import BlockAnalyzeByChange

# las1 = laspy.read("dataset_reg/K533/118-k533-origin-90-ground.las")
# las2 = laspy.read("dataset_reg/K533/119-k533-origin-90-ground.las")
# las3 = laspy.read("dataset_reg/K533/135-k533-origin-90-ground.las")
#
# las1.write("dataset_reg/K533/118-k533-origin-90-ground.ply")
# las2.write("dataset_reg/K533/119-k533-origin-90-ground.ply")
# las3.write("dataset_reg/K533/135-k533-origin-90-ground.ply")


pcd1 = o3d.io.read_point_cloud("dataset_reg/K533/CSF_Test/118_Disaster1.ply")
pcd2 = o3d.io.read_point_cloud("dataset_reg/K533/CSF_Test/119_Disaster1.ply")
pcd3 = o3d.io.read_point_cloud("dataset_reg/K533/CSF_Test/135_Disaster1.ply")
# pcd4 = o3d.io.read_point_cloud("dataset_reg/scnu/125-scnu-40-40-0cm - Cloud.ply")
# pcd5 = o3d.io.read_point_cloud("dataset_reg/scnu/129-scnu-40-40-10cm - Cloud.ply")
# o3d.visualization.draw_geometries([pcd1])
# o3d.visualization.draw_geometries([pcd2])
# o3d.visualization.draw_geometries([pcd3])

xyz = np.array(pcd1.points)
colors = np.array(pcd1.colors)

x_min = xyz[:, 0].min()
x_max = xyz[:, 0].max()
y_min = xyz[:, 1].min()
y_max = xyz[:, 1].max()

x_blocks_num = math.ceil((x_max - x_min) / 0.5)
y_blocks_num = math.ceil((y_max - y_min) / 0.5)

print("Grid Cutting With X_Blocks ", x_blocks_num, ", Y_Blocks ", y_blocks_num)

block_analyze = BlockAnalyzeByChange(pcd1, pcd2, pcd3, x_blocks_num, y_blocks_num, "118_Disaster_2", "135")
block_analyze.block_analyze(0.1)
block_analyze.visualize_color_pcd(block_analyze.color_pcd, True)
block_analyze.statistics_analyze(0, 1)
block_analyze.visualize_color_pcd(block_analyze.exception_pcd, True)
block_analyze.save_result_as_csv()
block_analyze.save_color_blocks()

# block_analyze = BlockAnalyzeByTerra(pcd1, pcd3, 60, 100)
# block_analyze.block_analyze()
# block_analyze.visualize_color_pcd(True)
# block_analyze.save_result_as_csv()
# block_analyze.save_color_blocks(True, 0.2)


