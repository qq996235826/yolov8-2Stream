import copy

import numpy as np
import open3d as o3d


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: 使用大小为为{}的体素下采样点云.".format(voxel_size))
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: 使用搜索半径为{}估计法线".format(radius_normal))
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: 使用搜索半径为{}计算FPFH特征".format(radius_feature))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    print(":: 加载点云并转换点云的位姿.")
    source = o3d.io.read_point_cloud("point_cloud/nabati-2.ply")
    target = o3d.io.read_point_cloud("point_cloud/target_point_cloud.ply")
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: 对下采样的点云进行RANSAC配准.")
    print("   下采样体素的大小为： %.3f," % voxel_size)
    print("   使用宽松的距离阈值： %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
         ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


if __name__ == '__main__':
    # 粗配准
    # 相当于使用0.5cm的体素对点云进行均值操作
    voxel_size = 0.005  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)

    source = source.transform(result_ransac.transformation)
    o3d.io.write_point_cloud("point_cloud/source.ply", source)

    # 精配准？
    # print("1. Load two point clouds and show initial pose")
    # o3d.io.write_point_cloud("point_cloud/target.ply", target)
    # source.estimate_normals()
    # target.estimate_normals()
    # # draw initial alignment
    # current_transformation = np.identity(4)
    # draw_registration_result_original_color(source, target, current_transformation)
    # # 点到面的ICP
    # current_transformation = np.identity(4)
    # print("2. 在原始点云上应用点到平面ICP配准来精准对齐，距离阈值0.02。")
    #
    # result_icp = o3d.pipelines.registration.registration_icp(source, target, 0.02, current_transformation,
    #                                                          o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # print(result_icp)
    # draw_registration_result_original_color(source, target, result_icp.transformation)

    # 彩色点云配准
    # 在以下论文中实现
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [0.004, 0.002, 0.001]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("3. 彩色点云配准")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. 下采样的点云的体素大小： %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. 法向量估计.")
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. 应用彩色点云配准")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)
    # 可视化
    draw_registration_result_original_color(source, target,
                                            result_icp.transformation)




