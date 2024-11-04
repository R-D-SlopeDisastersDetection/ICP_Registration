from BlockAnalyze.BlockAnalyze import BlockAnalyze
from PointCloudGridCutting.CloudPointGridCutting import CloudPointGridCutting as Grid
import csv
import os
import sys
from datetime import datetime
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import similarity as sim


class BlockAnalyzeByTerra(BlockAnalyze):
    """
    网格化分析类，用于对网格化后的区块进行分析
    TODO: 改进方向：使用历史数据横向对比进行着色，而非直接使用均值着色
    """

    def __init__(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, block_x: int, block_y: int,
                 source_name: str = "Default Source", target_name: str = "Default Target"):
        super().__init__(source, target, block_x, block_y, source_name, target_name)
        self.color_pcd = o3d.geometry.PointCloud()

    def block_analyze(self, color_threshold: float = 0.1, drop_threshold: float = 0.05, decimal_places: int = 5):
        """
        分块分析：获取以二维数组存储的点云分块，原始网格和目标网格的对应块进行分析，若块非空，则将分析结果放入analyse_result矩阵中。
        该函数会将源点云划分为block_x行block_y列的二维数组，数组元素为点云。可视作为标准二维坐标轴顺时针旋转90°的所得到的网格
        :param color_threshold:
        :param decimal_places: 保留几位小数
        :param drop_threshold: 用于规定点数低于所有区块平均点数的多少的块会被视为空块丢弃，不做处理
        :return:
        """
        source_grid = Grid(self.block_y, self.block_x, self.source)
        target_grid = Grid(self.block_y, self.block_x, self.target)
        source_blocks = source_grid.grid_cutting()[0]
        target_blocks = target_grid.grid_cutting()[0]
        empty_array = self.block_empty(source_blocks, drop_threshold)

        for i in range(self.block_x):
            for j in range(self.block_y):
                if empty_array[i][j] == 1:
                    mean, std = sim.point2point_mean_and_std_deviation(source_blocks[i][j], target_blocks[i][j])
                    res = o3d.pipelines.registration.evaluate_registration(source_blocks[i][j], target_blocks[i][j],
                                                                           0.05, np.identity(4))
                    self.analyse_result[i, j, 0], self.analyse_result[i, j, 1], self.analyse_result[i, j, 2], \
                        self.analyse_result[i, j, 3] = round(mean, decimal_places), round(std, decimal_places), \
                        round(res.inlier_rmse, decimal_places), round(res.fitness, decimal_places)
                    self.update_min_matrix(mean, std, res.inlier_rmse, res.fitness)
        self.draw_color(target_blocks, empty_array, color_threshold)

    def save_result_as_csv(self):
        file = open(self.filepath, "w+")
        writer = csv.writer(file)
        abstract = [['Grid Analyse Result\n'],
                    ['Source Cloud Points: ' + self.source_name + '\n',
                     'Target Cloud Points: ' + self.target_name + '\n']]
        performance_matrix = ["Mean", "Std Deviation", "RMSE", "Fitness"]
        writer.writerows(abstract)
        for i in range(0, 4):
            writer.writerows([[], [performance_matrix[i]]])
            writer.writerows(self.analyse_result[:, :, i].tolist())
        file.close()
