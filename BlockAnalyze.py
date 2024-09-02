import open3d as o3d
import numpy as np
from PointCloudGridCutting.CloudPointGridCutting import CloudPointGridCutting as Grid
import similarity as sim
from datetime import datetime
import os
import sys


class BlockAnalyze:
    """
    网格化分析类，用于对网格化后的区块进行分析
    TODO: 改进方向：使用历史数据横向对比进行着色，而非直接使用均值着色
    """
    def __init__(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, block_x: int, block_y: int,
                 source_name: str = "Default Source", target_name: str = "Default Target"):
        self.source = source
        self.target = target
        self.source_name = source_name
        self.target_name = target_name
        self.block_x = block_x
        self.block_y = block_y
        self.analyse_result = np.zeros((self.block_x, self.block_y, 4))
        self.color_pcd = o3d.geometry.PointCloud()
        '''
        一个4*1的矩阵，用于存储所有块的最小评估值。其中第一个为最小均值，第二个为最小标准差，第三个为最小rmse，第四个为最小重叠率
        '''
        self.evaluate_min = np.full((1, 4), sys.float_info.max)

        date = datetime.now().strftime("%Y-%m-%d")
        count = 1
        filename = date + " grid_analyse_result_" + str(count) + ".txt"
        dir_path = "grid_analyse_result/"
        filepath = dir_path + filename
        while os.path.exists(filepath):
            count = count + 1
            filename = date + " grid_analyse_result_" + str(count) + ".txt"
            filepath = dir_path + filename
        self.filepath = filepath

    def block_analyze(self, drop_threshold: float = 0.05, decimal_places: int = 5):
        """
        分块分析：获取以二维数组存储的点云分块，原始网格和目标网格的对应块进行分析，若块非空，则将分析结果放入analyse_result矩阵中
        :param decimal_places: 保留几位小数
        :param drop_threshold: 用于规定点数低于所有区块平均点数的多少的块会被视为空块丢弃，不做处理
        :return:
        """
        source_grid = Grid(self.block_x, self.block_y, self.source)
        target_grid = Grid(self.block_x, self.block_y, self.target)
        source_blocks = source_grid.grid_cutting()
        target_blocks = target_grid.grid_cutting()
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
        self.draw_color(source_blocks, empty_array)

    def block_empty(self, source_blocks: np.ndarray, threshold):
        """
        测量哪些块为空块，空块有两种情况：1则是最常见的，直接点云数量为0，这种则直接认为是空块；2是在切分过程中切分的边边角角，比如说只
        切到了边缘的一小块，这种情况也可被认为是空块。现在采用最简单直接的方法，即当块的点数少于所有块的均值的threshold时，认为该块时边缘
        块，可被抛弃，一般为5%

        :TODO：
          - 但此法和地面点分离算法共用时，可能会地面点较少植被点偏多的块在滤除了植被后，可能会因为点太少被误判为空块。后续可通过分析疑似
          - 空块内部的点分布规律进一步优化空块判断

        返回一个block_x*block_y的np数组，其中0表示为空块，1表示非空块
        :param source_blocks:
        :param threshold
        :return:
        """
        empty_array = np.zeros((self.block_x, self.block_y))
        avg = 0
        count = 0
        for j in range(self.block_y):
            for i in range(self.block_x):
                if len(source_blocks[i][j].points) != 0:
                    count += 1
                avg += len(source_blocks[i][j].points)
        avg = avg / count  # 均值

        for j in range(self.block_y):
            for i in range(self.block_x):
                if len(source_blocks[i][j].points) < avg * threshold:
                    empty_array[i][j] = 0
                else:
                    empty_array[i][j] = 1
        return empty_array

    def draw_color(self, source_block: np.ndarray, empty_array: np.ndarray, standard: int = 0):
        """
        将source_block中的点云复制为新点云，并且根据评估值的大小对点云进行着色：其中变化越小的越偏向绿色，变化越大的越偏向红色。
        需要注意的是，我们将读取所有分块中评估值最小的作为起始点，而非将0作为起始点。因为系统的误差普遍在3-5cm内，因此直接将0作为起始点没有
        评估意义，但此法仍有问题：如果一块点云中所有的点均出现了大范围偏移，那么使用评估最小值作为起始点反而导致总体偏绿。
        :param empty_array:
        :param source_block:
        :param standard: 用于规定使用哪种评估标准来进行着色，0为均值，1为标准差，2为RMSE，3为重叠率
        :return:
        """
        color_block = source_block.copy()
        color_rgb = [0.0, 0.0, 0.0]
        for j in range(self.block_y):
            for i in range(self.block_x):
                if empty_array[i][j] == 1:
                    # o3d着色的值为0到1，0.1 m为最大值，0.1 米以上的统一着色为(1,0,0)
                    color_rgb[0] = min(
                        (self.analyse_result[i][j][standard] - self.evaluate_min[0, standard]) * (1 / 0.1), 1)
                    color_rgb[1] = max(
                        1 - ((self.analyse_result[i][j][standard] - self.evaluate_min[0, standard]) * (1 / 0.1)), 0)
                    color_block[i][j].paint_uniform_color(color_rgb)
                    self.color_pcd += color_block[i][j]
        o3d.visualization.draw_geometries([self.color_pcd])

    def update_min_matrix(self, mean, std, rmse, fitness):
        """
        用于更新最小评估矩阵，最小评估矩阵用于记录所有块中的最小评估值，比如所有块中的最小均值，在着色的时候最小值将会用作为起始点
        :return:
        """
        if self.evaluate_min[0][0] > mean:
            self.evaluate_min[0][0] = mean
        if self.evaluate_min[0][1] > std:
            self.evaluate_min[0][1] = std
        if self.evaluate_min[0][2] > rmse:
            self.evaluate_min[0][2] = rmse
        if self.evaluate_min[0][3] > fitness:
            self.evaluate_min[0][3] = fitness

    def save_result_as_txt(self):
        file = open(self.filepath, "w+")
        abstract = ["Grid Analyse Result\n",
                    "Source Cloud Points: " + self.source_name + "\n",
                    "Target Cloud Points: " + self.target_name + "\n",
                    "----------------------------\n"]
        item_name = ["Mean", "Std Deviation", "RMSE", "Fitness"]
        file.writelines(abstract)
        for k in range(4):
            detail = "--------" + item_name[k] + "----------\n"
            for i in range(self.block_x):
                for j in range(self.block_y):
                    detail = detail + str(self.analyse_result[i, j, k]) + "  "
                detail += "\n"
            detail += "\n\n\n"
            file.write(detail)
        file.close()

    def save_color_blocks(self, voxel_sample=False, voxel_size=0.05):
        """
        :TODO 当color_block没有点的时候抛出异常
        :param voxel_size:
        :param voxel_sample: 是否使用体素下采样
        :return:
        """
        if voxel_sample:
            color_down_pcd = self.color_pcd.voxel_down_sample(voxel_size)
            o3d.io.write_point_cloud("grid_analyse_result/result_visualize/color_down_pcd.ply", color_down_pcd)
        else:
            o3d.io.write_point_cloud("grid_analyse_result/result_visualize/color_pcd.ply", self.color_pcd)
