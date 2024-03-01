import numpy as np
import open3d as o3d
from catch.config import CAMERA_INTRINSIC
import cv2
from scipy.optimize import minimize

colorBar = [[95, 134, 112], [0, 1, 0], [184, 0, 0], [81, 45, 109],
            [0, 193, 212], [245, 230, 202], [251, 147, 0],
            [161, 196, 90], [161, 196, 90], [241, 197, 80], [247, 98, 98], [33, 101, 131], [101, 192, 186],
            [207, 253, 248], [132, 185, 239], [251, 228, 201], [255, 93, 93], [149, 46, 75]]


def txt2ply(txtpath, plypath):
    """
        将txt转化成为ply点云文件
    """
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


def depth2ply(colorpath, depthpath, plypath, camera_intrinsic):
    """
    使用颜色和深度图获得ply点云
    """
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


def orb_keypoint(image1, image2, draw=False):
    """
    使用orb来获得两个图片之间的对应点
    """
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
            camera_coord = np.dot(np.linalg.inv(CAMERA_INTRINSIC), np.array([u, v, 1]) * depth)

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
    if plypath is not None:
        # 保存为PLY文件
        o3d.io.write_point_cloud(plypath, point_cloud)
    return keypoint


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


def get_catch_pose(classname, color_path, depth_path):
    # 读取模板第一个面RGB和深度图
    color_model_1 = cv2.imread('model/' + classname + '-1-color.png', cv2.IMREAD_ANYDEPTH)
    color_model_1 = cv2.cvtColor(color_model_1, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
    depht_model_1 = cv2.imread('model/' + classname + '-1-depth.png', cv2.IMREAD_ANYDEPTH)

    # 读取模板第二个面RGB和深度图
    color_model_2 = cv2.imread('model/' + classname + '-2-color.png', cv2.IMREAD_ANYDEPTH)
    color_model_2 = cv2.cvtColor(color_model_2, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
    depht_model_2 = cv2.imread('model/' + classname + '-2-depth.png', cv2.IMREAD_ANYDEPTH)

    # 读取实拍RGB和深度图，RGB图像已经经过mask处理
    color_real = cv2.imread(color_path, cv2.IMREAD_ANYDEPTH)
    color_real = cv2.cvtColor(color_real, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
    depht_real = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    # sift匹配，返回的是两张图片中的对应点（二维）
    points_model_1, points_real_1 = orb_keypoint(color_model_1, color_real, True)
    points_model_2, points_real_2 = orb_keypoint(color_model_2, color_real, True)


if __name__ == '__main__':
    # 读取模板RGB和深度图
    color_model = cv2.imread('color-2.png', cv2.IMREAD_ANYDEPTH)
    color_model = cv2.cvtColor(color_model, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
    depht_model = cv2.imread('depth-2.png', cv2.IMREAD_ANYDEPTH)

    # 读取实拍RGB和深度图，RGB图像已经经过mask处理
    color_real = cv2.imread('color.png', cv2.IMREAD_ANYDEPTH)
    color_real = cv2.cvtColor(color_real, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
    depht_real = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)

    # sift匹配，返回的是两张图片中的对应点（二维）
    points_model, points_real = orb_keypoint(color_model, color_real, True)

    # 根据RGB和DEPTH生成PLY点云
    pcd_model = depth2ply('color-2.png', 'depth-2.png', 'point_cloud/point_cloud_model.ply', CAMERA_INTRINSIC)
    pcd_real = depth2ply('color.png', 'depth.png', 'point_cloud/point_cloud_real.ply', CAMERA_INTRINSIC)

    # 根据二维坐标获得关键点以及其ply文件
    pointset_model = np.asarray(get_point_from_pcd(points_model, pcd_model, "point_cloud/keypoint_model.ply"))
    pointset_real = np.asarray(get_point_from_pcd(points_real, pcd_real, "point_cloud/keypoint_real.ply"))

    # 计算旋转和平移
    R, t = rigid_transform_3D(pointset_model, pointset_real)

    # 构造转换矩阵，4*4齐次
    homogeneous_matrix = np.identity(4)
    homogeneous_matrix[:3, :3] = R
    homogeneous_matrix[:3, 3] = t

    # 获得源点云
    source_cloud = o3d.io.read_point_cloud('point_cloud/point_cloud_face_model.ply')
    # 将源点云应用变换
    source_cloud.transform(homogeneous_matrix)
    # 变换后的点云保存为PLY文件
    o3d.io.write_point_cloud('point_cloud/point_cloud_transform.ply', source_cloud)

    # 获得抓取点和抓取方向
    catch_point, vector = get_catch_point_and_vector(pcd_model, homogeneous_matrix)
    print('抓取点:', catch_point)
    print('抓取向量:', vector)

    # 获得抓取位置
    # 获得源点云
    source_cloud = o3d.io.read_point_cloud('point_cloud/catch.ply')
    # 将源点云应用变换
    source_cloud.transform(homogeneous_matrix)
    # 变换后的点云保存为PLY文件
    o3d.io.write_point_cloud('point_cloud/catch_transform.ply', source_cloud)
