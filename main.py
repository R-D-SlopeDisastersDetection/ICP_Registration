from BatchRun import BatchRun
import open3d as o3d
from BlockAnalyze import BlockAnalyze
from PointCloudGridCutting.CloudPointGridCutting import CloudPointGridCutting


pcd1 = o3d.io.read_point_cloud("dataset_reg/simtest5m/simtest5/simtest5m_csf/Ag.ply")
pcd2 = o3d.io.read_point_cloud("dataset_reg/simtest5m/simtest5/simtest5m_csf/C30g.ply")
# pcd3 = o3d.io.read_point_cloud("grid_analyse_result/result_visualize/color_down_pcd.ply")
pcd4 = o3d.io.read_point_cloud("grid_analyse_result/result_visualize/color_down_pcd_10.ply")

o3d.visualization.draw_geometries([pcd4, pcd2])
# pcd3.translate([0.1, 0, 0])
# o3d.io.write_point_cloud("grid_analyse_result/result_visualize/color_down_pcd_10.ply", pcd3)

# o3d.visualization.draw_geometries([pcd2])

# block_analyze = BlockAnalyze(pcd1, pcd2, 10, 10)
# block_analyze.block_analyze(0.5)
# block_analyze.save_result_as_txt()
# block_analyze.save_color_blocks(True, 0.2)

# cloud_point_grid_cutting = CloudPointGridCutting(10, 10, pcd2, 'PointCloudGridCutting/output')
# blocks = cloud_point_grid_cutting.grid_cutting()
# cloud_point_grid_cutting.output_files()



