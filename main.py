from BatchRun import BatchRun
import open3d as o3d
from BlockAnalyze import *
from PointCloudGridCutting.CloudPointGridCutting import CloudPointGridCutting
import csv

pcd1 = o3d.io.read_point_cloud("dataset_reg/scnu/089_0cm - Cloud.ply")
pcd2 = o3d.io.read_point_cloud("dataset_reg/scnu/092_5cm - Cloud.ply")
pcd3 = o3d.io.read_point_cloud("dataset_reg/scnu/096_10cm - Cloud.ply")

# arr = np.random.rand(3, 4, 2)
# print(arr)
# print(arr[:,:,1])
# print(arr[:,:,1].tolist())
# o3d.visualization.draw_geometries([pcd1.translate([0,0,1])])
# pcd3.translate([0.1, 0, 0])
# o3d.io.write_point_cloud("grid_analyse_result/result_visualize/color_down_pcd_10.ply", pcd3)

# o3d.visualization.draw_geometries([pcd2])

block_analyze = BlockAnalyzeByTerra(pcd1, pcd3, 60, 100, "125-scnu-40-40-0cm", "130-scnu-40-40-10cm")
block_analyze.block_analyze()
block_analyze.visualize_color_pcd(block_analyze.color_pcd, True)
block_analyze.statistics_analyze(0, 2)
block_analyze.visualize_color_pcd(block_analyze.exception_pcd, True)
# block_analyze.save_result_as_csv()
# block_analyze.save_color_blocks(True, 0.2)

# block_analyze = BlockAnalyzeByTerra(pcd1, pcd3, 60, 100)
# block_analyze.block_analyze()
# block_analyze.visualize_color_pcd(True)
# block_analyze.save_result_as_csv()
# block_analyze.save_color_blocks(True, 0.2)



