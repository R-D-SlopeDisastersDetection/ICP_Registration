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


class BlockAnalyzeByChange(BlockAnalyze):
    """
    基于点云变化的网格化分析工具
    使用该方法需要在决定监测某边的时候，至少使用同样飞行参数进行两次扫描。一次是source，一次是baseline
    基础点云source，摸底式的扫描，以记录边坡原始的几何信息
    基准点云baseline，将会和source点云进行对比，得到边坡原始的基础点对点均值等基础信息evaluate_base。

    往后日常寻飞将会形成target点云，target点云将会和source进行对比，得到边坡当前的点对点均值信息evaluate_now，将evaluate_now和和
    evaluate_base做对比，找出当前边坡和原来的异同，并且将其标记出来
    """
    def __init__(self, source: o3d.geometry.PointCloud, baseline: o3d.geometry.PointCloud,
                 target: o3d.geometry.PointCloud, block_x: int, block_y: int,
                 source_name: str = "Default Source", target_name: str = "Default Target"):
        super().__init__(source, target, block_x, block_y, source_name, target_name)
        self.baseline = baseline
        self.baseline_result = np.zeros((block_x, block_y, 4))
        self.target_result = np.zeros((block_x, block_y, 4))

    def block_analyze(self, color_threshold: float = 0.1, decimal_places: int = 5):
        """
        分块分析：获取以二维数组存储的点云分块，原始网格和目标网格的对应块进行分析，若块非空，则将分析结果放入analyse_result矩阵中。
        该函数会将源点云划分为block_x行block_y列的二维数组，数组元素为点云。可视作为标准二维坐标轴顺时针旋转90°的所得到的网格
        :param color_threshold:
        :param decimal_places: 用于规定结果的小数点位数，默认为5
        :return:
        """
        # 获取网格化后的点云数组，网格化后将会返回一个[block_x, block_y]的二维numpy数组，数组元素为点云
        source_grid = Grid(self.block_y, self.block_x, self.source)
        target_grid = Grid(self.block_y, self.block_x, self.target)
        baseline_grid = Grid(self.block_y, self.block_x, self.baseline)
        source_blocks = source_grid.grid_cutting()[0]
        target_blocks = target_grid.grid_cutting()[0]
        baseline_block = baseline_grid.grid_cutting()[0]

        """
        遍历整个点云数组，将基础点云source和基准点云作对比得到mean_base和std_base；将基础点云source和当前巡飞得到的当前点云做对比得到
        mean_target和std_target。对比mean_base和mean_target之间的差、std_base和std_target之间的差，并且存入结果数组中
        """
        for i in range(self.block_x):
            for j in range(self.block_y):
                if target_blocks[i][j].points != 0:
                    mean_base, std_base = sim.point2point_mean_and_std_deviation(source_blocks[i][j],
                                                                                 baseline_block[i][j])
                    mean_target, std_target = sim.point2point_mean_and_std_deviation(source_blocks[i][j],
                                                                                     target_blocks[i][j])

                    self.baseline_result[i][j][0], self.baseline_result[i][j][1] = \
                        round(mean_base, decimal_places), round(std_base, decimal_places)

                    self.target_result[i][j][0], self.target_result[i][j][1] = \
                        round(mean_target, decimal_places), round(std_target, decimal_places)

                    # if self.target_result[i][j][0] == Nan
                    self.analyse_result[i][j][0], self.analyse_result[i][j][1] = \
                        round(mean_target - mean_base, decimal_places), round(std_target - std_base, decimal_places)

        # 调用着色函数，根据结果着色
        self.draw_color(target_blocks, np.ones((self.block_x, self.block_y)), color_threshold)

    def save_result_as_csv(self):
        """
        将结果保存为csv文件，其中包含如下信息：
        source和baseline的评估数组
        source和target的评估数组
        两个评估数组的差值
        基于z-scores的异常块监测结果，包括块的行号、列号及其数值
        :return:
        """
        file = open(self.filepath, "w")
        writer = csv.writer(file)
        abstract = [['Grid Analyse Result\n'],
                    ['Source Cloud Points: ' + self.source_name + '\n',
                     'Target Cloud Points: ' + self.target_name + '\n']]
        writer.writerows(abstract)
        # performance_matrix = ["Mean", "Std Deviation", "RMSE", "Fitness"]
        # item = ["baseline", "target", "change"]
        writer.writerows([[], [], ["Mean: Change Between BaseLine And Target"]])
        writer.writerows(self.analyse_result[:, :, 0].tolist())
        writer.writerows([[], [], ["Mean: Baseline"]])
        writer.writerows(self.baseline_result[:, :, 0].tolist())
        writer.writerows([[], [], ["Mean: Target"]])
        writer.writerows(self.target_result[:, :, 0].tolist())
        if len(self.exception_block_arr) != 0:
            writer.writerows([[], [], ["Exception Block Detail INFO"], ["Row", "Column", "Mean Change"]])
            for i in range(len(self.exception_block_arr)):
                writer.writerow([self.exception_block_arr[i][0], self.exception_block_arr[i][1],
                                 self.exception_block_arr[i][2]])
        file.close()

    def draw_color(self, target_block: np.ndarray, empty_array: np.ndarray, threshold: float = 0.1, standard: int = 0):
        """
        将target_block中的点云复制为新点云，并且根据评估值的大小对点云进行着色：其中变化越小的越偏向绿色，变化越大的越偏向红色。
        需要注意的是，我们将读取所有分块中评估值最小的作为起始点，而非将0作为起始点。因为系统的误差普遍在3-5cm内，因此直接将0作为起始点没有
        评估意义，但此法仍有问题：如果一块点云中所有的点均出现了大范围偏移，那么使用评估最小值作为起始点反而导致总体偏绿。
        :param threshold: 阈值，以最小均值距离min为起点，[min, mean+threshold]区间由绿变红，超过mean+threshold一律为红，默认为0.1m
        :param target_block: 目标点云，基于目标点云着色
        :param empty_array: 空块数组，用于指定哪些块为空
        :param standard: 用于规定使用哪种评估标准来进行着色，0为均值，1为标准差，2为RMSE，3为重叠率，目前只支持0
        :return:
        """
        color_block = target_block.copy()
        color_rgb = [0.0, 0.0, 0.0]
        for j in range(self.block_y):
            for i in range(self.block_x):
                if empty_array[i][j] == 1:
                    # o3d着色的值为0到1，0.1 m为最大值，0.1 米以上的统一着色为(1,0,0)
                    color_rgb[0] = min(abs(self.analyse_result[i][j][standard]) * (1 / threshold), 1)
                    color_rgb[1] = max(1 - (abs(self.analyse_result[i][j][standard]) * (1 / threshold)), 0)
                    color_block[i][j].paint_uniform_color(color_rgb)
                    self.color_pcd += color_block[i][j]
