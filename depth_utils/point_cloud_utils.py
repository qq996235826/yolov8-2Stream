import copy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from catch.config import CAMERA_INTRINSIC
from scipy.optimize import minimize
import cv2
from scipy.linalg import orthogonal_procrustes

colorBar = [[95, 134, 112], [0, 1, 0], [184, 0, 0], [81, 45, 109],
            [0, 193, 212], [245, 230, 202], [251, 147, 0],
            [161, 196, 90], [161, 196, 90], [241, 197, 80], [247, 98, 98], [33, 101, 131], [101, 192, 186],
            [207, 253, 248], [132, 185, 239], [251, 228, 201], [255, 93, 93], [149, 46, 75]]
"""
    将txt转化成为ply点云文件
"""


def txt2ply(txtpath, plypath):
    ## 数据读取
    np.set_printoptions(suppress=True)  # 取消默认的科学计数法
    points = np.loadtxt(txtpath, dtype=float,
                        delimiter=' ', usecols=(0, 1, 2), unpack=False)
    colors = np.loadtxt(txtpath, dtype=int,
                        delimiter=' ', usecols=(3, 4, 5), unpack=False)
    ## open3d数据转换
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    o3d.visualization.draw_geometries([pcd])
    ## 保存成ply数据格式
    o3d.io.write_point_cloud(plypath, pcd)


"""
使用颜色和深度图获得ply点云
"""
def depth2ply(colorpath, depthpath, plypath, camera_intrinsic):
    depth = cv2.imread(depthpath, cv2.IMREAD_ANYDEPTH)
    rgb = cv2.imread(colorpath)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
    # 不要忽略深度为0的点
    depth[depth == 0] = 1500
    # color_raw = o3d.io.read_image(colorpath)
    # depth_raw = o3d.io.read_image(depthpath)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
                                                                    convert_rgb_to_intensity=False)

    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = camera_intrinsic
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.io.write_point_cloud(plypath, pcd)
    return pcd


# 定义最小化的目标函数
def transformation_error(parameters, source, target):
    # parameters 包含平移、旋转和缩放参数
    translation = parameters[:3]
    rotation_matrix = parameters[3:12].reshape((3, 3))
    scale = parameters[12]

    # 对原始点进行变换
    transformed_source = scale * np.dot(source, rotation_matrix.T) + translation

    # 计算变换后的点与目标点的误差
    error = np.sum((transformed_source - target) ** 2)

    return error


def find_affine_transform(source_points, target_points):
    """
    寻找包含平移、旋转和缩放的非刚体变换的转换矩阵。

    参数:
    - A: 原始坐标 (numpy array)
    - B: 目标坐标 (numpy array)

    返回:
    - translation: 平移向量
    - rotation: 旋转矩阵
    - scale: 缩放因子
    """

    # 初始参数猜测
    initial_params = np.zeros(13)

    # 最小化误差函数，得到最优参数
    result = minimize(transformation_error, initial_params, args=(source_points, target_points), method='nelder-mead')

    # 获取最优参数
    optimized_params = result.x

    # 提取平移、旋转和缩放参数
    translation_optimized = optimized_params[:3]
    rotation_matrix_optimized = optimized_params[3:12].reshape((3, 3))
    scale_optimized = optimized_params[12]

    # 打印结果
    print("最优平移参数:", translation_optimized)
    print("最优旋转矩阵:")
    print(rotation_matrix_optimized)
    print("最优缩放参数:", scale_optimized)

    return translation_optimized, rotation_matrix_optimized, scale_optimized


"""
使用orb来获得两个图片之间的对应点
"""
def orb_keypoint(image1, image2, draw):
    # 使用SIFT算法检测关键点和计算描述符
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # 使用BFMatcher（暴力匹配器）进行特征匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 过滤匹配点，使用比值测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good_matches.append(m)

    # 设置距离阈值，过滤不准确的匹配点
    distance_threshold = 300
    filtered_matches = [m for m in good_matches if m.distance < distance_threshold]

    # 提取对应点的坐标
    points1 = [keypoints1[m.queryIdx].pt for m in filtered_matches]
    points2 = [keypoints2[m.trainIdx].pt for m in filtered_matches]

    # 将坐标转换为整数类型
    points1 = list(map(lambda x: (int(x[0]), int(x[1])), points1))
    points2 = list(map(lambda x: (int(x[0]), int(x[1])), points2))

    # 输出对应点的坐标
    print("Points in Image 1:", points1)
    print("Points in Image 2:", points2)
    if draw:
        # 绘制匹配结果
        matching_result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                          matchesMask=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # 显示匹配结果
        cv2.imshow('Matching Result', matching_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return points1, points2


def draw_keypoint(rgbimg, depthimg, keypoints):
    rgb_image = cv2.imread(rgbimg)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
    depth_image = cv2.imread(depthimg, cv2.IMREAD_ANYDEPTH)

    # 相机内参矩阵
    camera_matrix = np.array([[609.919868760916, 0, 327.571937094492],
                              [0, 607.733032143467, 241.738191162382],
                              [0, 0, 1]])

    # 获取图像的高度和宽度
    height, width = depth_image.shape

    # 生成点云
    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            # 获取深度值
            depth = depth_image[v, u]

            # 跳过深度值为0的点
            if depth == 0:
                continue

            # 反投影得到相机坐标系下的三维坐标
            camera_coord = np.dot(np.linalg.inv(camera_matrix), np.array([u, v, 1]) * depth)

            # 添加到点云
            points.append(camera_coord)
            colors.append(rgb_image[v, u])

    # 创建Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors) / 255.0)  # 颜色需要进行归一化
    # Flip it, otherwise the pointcloud will be upside down
    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # 可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加点云到可视化窗口
    vis.add_geometry(point_cloud)
    o3d.io.write_point_cloud('base.ply', point_cloud)

    # 设置要高亮显示的XY坐标列表
    highlight_coordinates = keypoints  # 根据实际情况设置

    # 高亮显示指定XY坐标下的点
    highlighted_points = []
    highlighted_colors = []

    for coord in highlight_coordinates:
        # 在点云中找到最接近指定XY坐标的点
        index = np.argmin(np.linalg.norm(np.array(points)[:, :2] - np.array(coord), axis=1))

        # 添加到高亮的点云列表
        highlighted_points.append(points[index])
        highlighted_colors.append([0, 1, 0])  # 使用绿色显示

    # 创建高亮的点云
    highlighted_point_cloud = o3d.geometry.PointCloud()
    highlighted_point_cloud.points = o3d.utility.Vector3dVector(np.array(highlighted_points))
    highlighted_point_cloud.colors = o3d.utility.Vector3dVector(np.array(highlighted_colors))
    # Flip it, otherwise the pointcloud will be upside down
    highlighted_point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # 添加高亮的点云到可视化窗口
    vis.add_geometry(highlighted_point_cloud)
    o3d.io.write_point_cloud('keypoint.ply', highlighted_point_cloud)

    # 设置渲染选项
    render_options = vis.get_render_option()
    render_options.point_size = 10.0  # 设置高亮的点的大小

    # 运行可视化窗口
    vis.run()

    # 关闭可视化窗口
    vis.destroy_window()


def get_3d_coordinates(u, v, depth, camera_matrix):
    # 使用相机内参进行反投影
    fx, fy, cx, cy = camera_matrix[0][0], camera_matrix[1][1], camera_matrix[0][2], camera_matrix[1][2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    return x, y, z


def depth_to_point_cloud(depth_image, rgb_image, camera_matrix):
    points = []
    colors = []

    height, width = depth_image.shape

    for v in range(height):
        for u in range(width):
            # 获取深度值
            depth = depth_image[v, u]

            # 跳过深度值为0的点
            if depth == 0:
                continue

            # 反投影得到相机坐标系下的三维坐标
            camera_coord = np.dot(np.linalg.inv(camera_matrix), np.array([u, v, 1]) * depth)

            # 获取RGB颜色
            rgb_color = rgb_image[v, u] / 255.0  # 归一化颜色到 [0, 1]

            # 添加到点云
            points.append(camera_coord)
            colors.append(rgb_color)

    return points, colors


def get_point_from_pcd(points, pcd, plypath):
    point_array = np.asarray(pcd.points)
    point_array = point_array.reshape((480, 640, 3))
    keypoint = []
    color = []
    index = 0
    for point in points:
        keypoint.append(point_array[point[1]][point[0]])
        color.append(np.asarray(colorBar[index]) / 255)
        index += 1
    # 将对应点写成PLY
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(keypoint))
    point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(color))
    # 保存为PLY文件
    o3d.io.write_point_cloud(plypath, point_cloud)
    return keypoint


# 通过一组三位点来计算旋转和平移
def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.matmul(np.transpose(AA), BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = np.matmul(Vt.T, U.T)

    t = -np.matmul(R, centroid_A) + centroid_B
    # err = B - np.matmul(A,R.T) - t.reshape([1, 3])
    return R, t


if __name__ == '__main__':
    # 分别读取RGB和深度图像
    depht_1 = cv2.imread('depth-2.png', cv2.IMREAD_ANYDEPTH)
    depht_2 = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)

    color_1 = cv2.imread('color-2.png', cv2.IMREAD_ANYDEPTH)
    color_1 = cv2.cvtColor(color_1, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
    color_2 = cv2.imread('color.png', cv2.IMREAD_ANYDEPTH)
    color_2 = cv2.cvtColor(color_2, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序

    # sift匹配，返回的是两张图片中的对应点（二维）
    points1, points2 = orb_keypoint(color_1, color_2, True)

    # 根据RGB和DEPTH生成PLY点云
    pcd_1 = depth2ply('color-2.png', 'depth-2.png', 'point_cloud/point_cloud_1.ply', CAMERA_INTRINSIC)
    pcd_2 = depth2ply('color.png', 'depth.png', 'point_cloud/point_cloud_2.ply', CAMERA_INTRINSIC)

    # 根据二维坐标获得关键点以及其ply文件
    pointset1 = np.asarray(get_point_from_pcd(points1, pcd_1, "point_cloud/keypoint_1.ply"))
    pointset2 = np.asarray(get_point_from_pcd(points2, pcd_2, "point_cloud/keypoint_2.ply"))

    # 把点转化为open3d格式
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(pointset1)
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(pointset2)

    # 计算旋转和平移
    R, t = rigid_transform_3D(pointset1, pointset2)
    print(R, t)
    homogeneous_matrix = np.identity(4)
    homogeneous_matrix[:3, :3] = R
    homogeneous_matrix[:3, 3] = t

    # 获得源点云
    source_cloud = o3d.io.read_point_cloud('point_cloud/point_cloud_1.ply')
    # 将源点云应用变换
    source_cloud.transform(homogeneous_matrix)
    # 变换后的点云保存为PLY文件
    o3d.io.write_point_cloud('point_cloud/point_cloud_transfrom.ply', source_cloud)
