import ICP_Reg
import open3d as o3d
import copy
import numpy as np
from datetime import datetime
import os
import similarity as sim


class BatchRun:
    def __init__(self, source, target, times, threshold):
        self.times = times
        self.source = source
        self.target = target
        self.reg = ICP_Reg.Registration(self.source, self.target, threshold)

        date = datetime.now().strftime("%Y-%m-%d")
        count = 1
        filename = date + "batch_run_result_" + str(count) + ".txt"
        dir = "result/"
        filepath = dir + filename
        while os.path.exists(filepath):
            count = count + 1
            filename = date + " batch_run_result_" + str(count) + ".txt"
            filepath = dir + filename
        self.filepath = filepath

    def registration_without_global_reg(self, icp_method: str, iter_threshold: float, dis_threshold: float):
        """

        :param icp_method:
        :param iter_threshold:
        :param dis_threshold:
        :return:
        """
        self.reg.reg_result = o3d.pipelines.registration.evaluate_registration(self.source, self.target, 0.05,
                                                                               np.identity(4))
        self.reg.preprocess_dataset()
        self.reg.execute_global_registration()
        self.reg.execute_icp_registration(icp_method, iter_threshold, dis_threshold)
        self.reg.reg_source = copy.deepcopy(self.source)
        self.reg.reg_source.transform(self.reg.reg_result.transformation)
        o3d.visualization.draw_geometries([self.reg.source, self.target])
        o3d.visualization.draw_geometries([self.source, self.target])

    def start_batch_run(self, icp_method: str = "max_iteration", iter_threshold: float = 10000, dis_threshold: float = 0.05):
        """

        :param icp_method:
        :param iter_threshold:
        :param dis_threshold:
        :return:
        """
        file = open(self.filepath, "w+")
        decimal_places = 6  # 结果保留几位小数
        basic_info = ["Basic Info:\n",
                      "============================\n",
                      "Batch Run of Registration\n",
                      "Start Time:" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n",
                      "Iteration Times:" + str(self.times) + "\n",
                      "_____ICP Reg Parameters______\n",
                      "ICP Method:" + icp_method + "\n",
                      "Iteration Threshold:" + str(iter_threshold) + "\n",
                      "Distance Threshold:" + str(dis_threshold) + "\n",
                      "==========Each Time Detail Data========\n"]
        file.writelines(basic_info)
        avg_mean, avg_std, avg_rmse, avg_fitness, index = 0.0, 0.0, 0.0, 0.0, 0

        for i in range(1, self.times + 1):
            self.registration_without_global_reg(icp_method, iter_threshold, dis_threshold)
            mean, std = sim.point2point_mean_and_std_deviation(self.reg.reg_source, self.target)
            fitness = self.reg.reg_result.fitness
            rmse = self.reg.reg_result.inlier_rmse
            epoch_info = [str(index) + "  [",
                          "Mean:" + str(round(mean, decimal_places)) + ",",
                          "Standard Derivation:" + str(round(std, decimal_places)) + ",",
                          "RMSE:" + str(round(rmse, decimal_places)) + ",",
                          "Fitness:" + str(round(fitness, decimal_places)) + "]\n"]
            file.writelines(epoch_info)
            avg_mean, avg_std, avg_rmse, avg_fitness, index = \
                avg_mean + mean, avg_std + std, avg_rmse + rmse, avg_fitness + fitness, index + 1

        avg_mean, avg_std, avg_rmse, avg_fitness = \
            avg_mean / self.times, avg_std / self.times, avg_rmse / self.times, avg_fitness / self.times
        summary = ["==================================\n",
                   "Batch Run Finish! Printing Result\n",
                   "Mean:" + str(round(avg_mean, decimal_places)) + "\n",
                   "Standard Derivation:" + str(round(avg_std, decimal_places)) + "\n",
                   "RMSE:" + str(round(avg_rmse, decimal_places)) + "\n",
                   "Fitness:" + str(round(avg_fitness, decimal_places)) + "]\n"]
        file.writelines(summary)
        file.close()
