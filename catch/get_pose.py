import os

import numpy
import numpy as np
import cv2 as cv
import open3d as o3d
from catch import config
from catch import yolo_item
from catch.config import CAMERA_INTRINSIC
from catch.config import CAMERA2BASE
from scipy.optimize import minimize

colorBar = [[95, 134, 112], [0, 1, 0], [184, 0, 0], [81, 45, 109],
            [0, 193, 212], [245, 230, 202], [251, 147, 0],
            [161, 196, 90], [161, 196, 90], [241, 197, 80], [247, 98, 98], [33, 101, 131], [101, 192, 186],
            [207, 253, 248], [132, 185, 239], [251, 228, 201], [255, 93, 93], [149, 46, 75]]


def apply_mask(items):
    # 删除之前的结果
    for file in os.scandir(config.MASKS_COLOR_PATH):
        os.remove(file.path)
    # 读取color图
    color = cv.imread(config.COLOR_PATH, -1)
    for index in range(len(items)):
        mask = np.uint8(items[index].mask)
        mask = mask.reshape((480, 640))
        mask = mask * 255
        masked = cv.bitwise_and(color, color, mask=mask)
        image_path = config.MASKS_COLOR_PATH + str(index) + "-" + items[index].name + "-" + str(
            items[index].confidence) + ".png"
        items[index].mask_image_path = image_path
        cv.imwrite(image_path, masked)


def depth2ply(colorpath, depthpath, plypath, camera_intrinsic):
    """
    使用颜色和深度图获得ply点云
    """
    depth = cv.imread(depthpath, cv.IMREAD_ANYDEPTH)
    rgb = cv.imread(colorpath)
    rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)  # cv2默认为bgr顺序
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
    # 翻转它，否则点云会倒置
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.io.write_point_cloud(plypath, pcd)
    return pcd


def get_point_from_pcd(points, pcd, plypath):
    point_array = np.asarray(pcd.points)
    point_array = point_array.reshape((480, 640, 3))
    keypoint = []
    color = []
    index = 0
    for point in points:
        keypoint.append(point_array[point[1]][point[0]])
        # color.append(np.asarray(colorBar[index]) / 255)
        index += 1
    # 将对应点写成PLY
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(keypoint))
    # point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(color))
    if plypath is not None:
        # 保存为PLY文件
        o3d.io.write_point_cloud(plypath, point_cloud)
    return keypoint


def get_catch_point_and_vector(pcd, matrix):
    """
    :param pcd: 模型点云
    :param matrix: 转换矩阵
    :return: 抓取点和向量
    """
    # 初始抓取点与法向量,默认抓取点在中心，抓取方向垂直向下
    catch_point = np.asarray([320, 240])
    vector = np.asarray([0, 0, 1])

    # 获得转换后的抓取点和法向量
    catch_point = np.asarray(get_point_from_pcd([catch_point], pcd, None))
    # 将三维点转换为齐次坐标
    catch_point = np.concatenate([catch_point[0], [1]])
    # 转换
    catch_point = np.dot(matrix, catch_point)
    # 将齐次坐标转换回三维坐标
    catch_point = catch_point[:3]

    # 转换法向量
    vector = np.concatenate([vector, [0]])
    # 进行变换
    vector = np.dot(matrix, vector)
    # 取变换后的齐次坐标的前三个分量，即为变换后的向量
    vector = vector[:3]

    return catch_point, vector


# 通过一组三位点来计算旋转和平移
def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    # 计算点云的质心
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # 居中点云
    AA = A - centroid_A
    BB = B - centroid_B

    # 计算矩阵 H
    H = np.dot(AA.T, BB)
    # 进行奇异值分解
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # 处理反射情况
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = np.matmul(Vt.T, U.T)

    # 计算平移矩阵 t
    t = -np.dot(R, centroid_A) + centroid_B

    # err = B - np.matmul(A,R.T) - t.reshape([1, 3])

    # 定义目标函数，即欧氏距离的平方
    def objective_function(scale):
        transformed_A = scale * np.dot(R, A.T).T + t
        return np.sum((B - transformed_A) ** 2)

    # 使用最小二乘法估计缩放因子
    result = minimize(objective_function, x0=1.0, bounds=[(0.1, 10.0)])
    scale = result.x[0]
    print('缩放:', scale)
    print('旋转:', R)
    print('平移:', t)
    return scale * R, t


def sift_keypoint(image1, image2, draw=False):
    """
    使用orb来获得两个图片之间的对应点
    """
    # 使用SIFT算法检测关键点和计算描述符
    sift = cv.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # 使用BFMatcher（暴力匹配器）进行特征匹配
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 过滤匹配点，使用比值测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good_matches.append(m)

    # 设置距离阈值，过滤不准确的匹配点
    distance_threshold = 200
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
        matching_result = cv.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                         matchesMask=None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # 显示匹配结果
        cv.imshow('Matching Result', matching_result)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return points1, points2


def get_key_points(color_real, class_name):
    """
    :param color_real: 实拍待抓取图像
    :param class_name: 类名
    :return: 返回匹配点数量最多的平面的匹配点和模板名
    """
    color_model_1 = cv.imread('catch_result/photo/model/' + class_name[:-2] + '/' + class_name + '-1-color.png')
    color_model_2 = cv.imread('catch_result/photo/model/' + class_name[:-2] + '/' + class_name + '-2-color.png')

    # 获得二维匹配点
    points_model_1, points_real_1 = sift_keypoint(color_model_1, color_real, True)
    points_model_2, points_real_2 = sift_keypoint(color_model_2, color_real, True)

    if len(points_real_1) < 3 and len(points_real_2) < 3:
        raise ValueError('无法获得至少3个匹配点')

    else:
        if len(points_real_1) > len(points_real_2):
            return points_model_1, points_real_1, class_name + '-1'
        else:
            return points_model_2, points_real_2, class_name + '-2'


def get_catch_pose_by_sift(results):
    items = yolo_item.get_items(results)
    if len(items) == 0:
        config.data['have_item'] = False
        print('抓取完毕，摄像头区域内已无目标')
        return None

    # 将图像分割成各个mask，存放在config.MASKS_COLOR_PATH
    apply_mask(items)

    # 输出识别结果
    yolo_item.print_items(items)
    sorted_by_confidence = sorted(items, key=lambda x: x.confidence)
    # 获得置信度最高的物体
    best_item = sorted_by_confidence[-1]

    # 读取实拍RGB和深度图，RGB图像已经经过mask处理
    color_real = cv.imread(best_item.mask_image_path)
    # depht_real = cv.imread('depth.png', cv.IMREAD_ANYDEPTH)

    # 获得二维匹配点
    points_model, points_real, model_name = get_key_points(color_real, best_item.name)

    # 根据RGB和DEPTH生成PLY点云
    pcd_model = depth2ply('catch_result/photo/model/' + model_name[:-4] + '/' + model_name + '-color.png',
                          'catch_result/photo/model/' + model_name[:-4] + '/' + model_name + '-depth.png',
                          'catch_result/point_cloud/point_cloud_model.ply', CAMERA_INTRINSIC)
    pcd_real = depth2ply('catch_result/photo/color.png', 'catch_result/photo/depth.png',
                         'catch_result/point_cloud/point_cloud_real.ply', CAMERA_INTRINSIC)

    # 根据二维坐标获得关键点以及其ply文件
    pointset_model = np.asarray(
        get_point_from_pcd(points_model, pcd_model, "catch_result/point_cloud/keypoint_model.ply"))
    pointset_real = np.asarray(get_point_from_pcd(points_real, pcd_real, "catch_result/point_cloud/keypoint_real.ply"))

    # 计算旋转和平移
    R, t = rigid_transform_3D(pointset_model, pointset_real)

    # 构造转换矩阵，4*4齐次
    homogeneous_matrix = np.identity(4)
    homogeneous_matrix[:3, :3] = R
    homogeneous_matrix[:3, 3] = t

    # 获得源点云
    source_cloud = o3d.io.read_point_cloud('catch_result/point_cloud/point_cloud_model.ply')
    # 将源点云应用变换
    source_cloud.transform(homogeneous_matrix)
    # 变换后的点云保存为PLY文件
    o3d.io.write_point_cloud('catch_result/point_cloud/point_cloud_transform.ply', source_cloud)

    # 获得抓取点和抓取方向
    catch_point, vector = get_catch_point_and_vector(pcd_model, homogeneous_matrix)
    print('抓取点:', catch_point)
    print('抓取向量:', vector)
    # 将相机坐标系下的抓取点和抓取方向转到机械臂坐标系下
    # 齐次
    catch_point = numpy.append(catch_point, 1)
    vector = numpy.append(vector, 1)
    # 乘上转换矩阵
    catch_point = np.dot(catch_point, CAMERA2BASE)
    vector = np.dot(vector, CAMERA2BASE)
    print('抓取点:', catch_point)
    print('抓取向量:', vector)
    best_item.catch_pose = np.append(catch_point, vector)
    return best_item
