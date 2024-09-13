import open3d as o3d
import copy
import numpy as np
import similarity as sim


def fpfh_feature_extract(pcd, size):
    """
    Use Fast Point Feature Histogram(FPFH) Feature Extraction to get the feature of Cloud Point
    :param pcd: open3d.geometry.PointCloud, origin CloudPoint Object
    :param size: int, the resolution of process, the resolution of Estimate Normals, FPFH is 2*size, 5*size
    :return: pcd_fpfh: open3d.geometry.PointCloud, The FPFH feature extract output of original pcd
    """
    # 法线计算
    radius_normal = size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # FPFH特征提取
    radius_feature = size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh


class Registration:
    def __init__(self, source, target, threshold, console_output=False):
        self.source = source
        self.target = target
        self.source_fpfh = None
        self.target_fpfh = None
        self.reg_source = None
        self.reg_result = None
        self.threshold = threshold
        self.raw_trans = None
        self.console_output = console_output

    def draw_registration_result(self, transformation):
        """
        Used to show two cloud points, this function will draw the blue and yellow to cloud point
        , and move and rotate the source cloud point according to the transformation matrix
        :param transformation:`4 x 4` float64 numpy array: The estimated transformation matrix
        :return:
        """
        source_temp = copy.deepcopy(self.source)
        target_temp = copy.deepcopy(self.target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def preprocess_dataset(self):
        """
        Initial the Cloud Point Data and extract the feature of the Cloud Point
        :return:
        """
        print(":: Load two point clouds and disturb initial pose.")

        # Feature Extraction
        self.source_fpfh = fpfh_feature_extract(self.source, self.threshold)
        self.target_fpfh = fpfh_feature_extract(self.target, self.threshold)

    def execute_global_registration(self):
        """
        Global Registration with RANSAC
        """
        distance_threshold = self.threshold * 5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        self.reg_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            self.source, self.target, self.source_fpfh, self.target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.99))
        self.raw_trans = copy.deepcopy(self.reg_result.transformation)

    def execute_icp_registration(self, iter_method, iter_threshold, dis_threshold=0.05):
        """
        Accurate Registration with ICP
        :param iter_method: string, use "rmse", "fitness" or "max_iteration" to modify the iteration method
        :param iter_threshold: the threshold of iteration
        :param dis_threshold: float, distance threshold, radius of K-NN in ICP
        :return:
        """
        trans_init = self.reg_result.transformation
        '''
        fitness，用于测量重叠面积（内点对应数/目标点数）。 值越高越好。
        inlier_rmse，它测量所有内点对应的 RMSE。越低越好。
        '''
        print("Initial alignment")
        reg_p2p = o3d.pipelines.registration.evaluate_registration(
            self.source, self.target, dis_threshold, trans_init)
        print(reg_p2p)

        print("Apply point-to-point ICP")
        if iter_method == "rmse":
            self.reg_result = o3d.pipelines.registration.registration_icp(
                self.source, self.target, dis_threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_rmse=iter_threshold))
        elif iter_method == "fitness":
            self.reg_result = o3d.pipelines.registration.registration_icp(
                self.source, self.target, dis_threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=iter_threshold))
        elif iter_method == "max_iteration":
            self.reg_result = o3d.pipelines.registration.registration_icp(
                self.source, self.target, dis_threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iter_threshold))
        else:
            pass
        print(self.reg_result)
        print("Transformation is:")
        print(self.reg_result.transformation)

    def evaluation_matrix(self, reg_p2p):
        """
        Compare the difference of pcd1 and pcd2, includes Hausdorff Distance , Mean Distance, Standard Deviation
        , Fitness, RMSE and Correspondence Set
        :return:
        """
        hausdorff_distance = sim.hausdorff_distance(self.reg_source, self.target)
        mean, std = sim.point2point_mean_and_std_deviation(self.reg_source, self.target)
        print("The Hausdorff Distance is ", hausdorff_distance)
        print("The mean is ", mean, ", the std is ", std)
        print("The inliner RMSE is ", reg_p2p.inlier_rmse, ", the fitness is ", reg_p2p.fitness)
        self.draw_registration_result(reg_p2p.transformation)

    def cloudpoint_registration(self, icp_method="max_iteration", iter_threshold=10000, dis_threshold=0.05):
        """
        :param icp_method:
        :param iter_threshold:
        :param dis_threshold:
        :return:
        """
        o3d.visualization.draw_geometries([self.source, self.target])
        self.preprocess_dataset()
        self.execute_global_registration()
        print(self.reg_result)

        mean, std = sim.point2point_mean_and_std_deviation(self.source, self.target)
        print("RANSAC Finish!\nThe mean is ", mean, ", the std is ", std)
        self.draw_registration_result(np.identity(4))

        self.execute_icp_registration(icp_method, iter_threshold, dis_threshold)
        self.reg_source = copy.deepcopy(self.source)
        self.reg_source.transform(self.reg_result.transformation)
        self.evaluation_matrix(self.reg_result)


# if __name__ == "__main__":
#     pcd1 = o3d.io.read_point_cloud("dataset_reg/simtest/format/5331.pcd")
#     pcd2 = o3d.io.read_point_cloud("dataset_reg/simtest/format/5332.pcd")
#     reg = Registration(pcd1, pcd2, 0.05)
#     reg.cloudpoint_registration()
#     output = reg.reg_source
#     print("内部结果")
#     o3d.visualization.draw_geometries([output, pcd2])
#     o3d.visualization.draw_geometries([pcd1.transform(reg.raw_trans), pcd2])
