# -*- coding: utf-8 -*-
"""
@Time ： 2022/10/25 14:28
@Auth ： 邓浩然
@File ：calculate_pose.py
@IDE ：PyCharm
@Description：负责位姿估计的主要文件
"""
import os
import numpy as np
import cv2 as cv
import math
import matrix_util
import open3d as o3d
import config
import pyransac3d as pyrsc
import yolo_item


# 获得法向量
def get_normal_vector(point_cloud):
    x = []
    y = []
    z = []

    xy = np.ones((len(point_cloud), 3))
    d = np.zeros((len(point_cloud), 1))

    for i in range(len(point_cloud)):
        xy[i, 0] = point_cloud[i][0]
        xy[i, 1] = point_cloud[i][1]
        d[i, 0] = point_cloud[i][2]

        x.append(point_cloud[i][0])
        y.append(point_cloud[i][1])
        z.append(point_cloud[i][2])

    A_T = xy.T
    A1 = np.dot(A_T, xy)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2, A_T)
    X = np.dot(A3, d)
    print('平面拟合结果为：z = %.3f * x + %.3f * y + %.3f' % (X[0, 0], X[1, 0], X[2, 0]))

    return [X[0, 0], X[1, 0], -1]


# 计算抓取的面积
def get_area(xypoints):
    x = math.fabs(float(xypoints[2]) - float(xypoints[0]))
    y = math.fabs(float(xypoints[3]) - float(xypoints[1]))
    return x * y


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3)


# 返回权重
def get_weights(item, area_weight):
    area = item.area
    depth = item.catch_pose[2]
    return area / area_weight + depth


def point_weight(item):
    return item.weight


# 获得抓点在机械臂坐标系下的坐标,返回一个列表,存放x y z,在返回一个深度
def pixel_to_world2(depth_path, x, y, camera2base, camera_matrix):
    # 读取深度图并获得深度矩阵
    depth = cv.imread(depth_path, -1)
    depth = np.asarray(depth).T
    z = depth

    # 获得在深度图上的深度,由于像素坐标必定是整数,所以要先变成整数
    x1 = int(x)
    y1 = int(y)
    # 得到深度数据
    d = z[x1, y1]

    # 数学公式
    a2 = np.around(x * d, decimals=3)
    b2 = np.around(y * d, decimals=3)

    pixel = np.zeros((3, 1))
    pixel[0, 0] = a2
    pixel[1, 0] = b2
    pixel[2, 0] = d

    # 相机内参矩阵
    camera_matrix = np.mat(camera_matrix)

    camera_matrix = np.linalg.inv(camera_matrix)
    camera = np.matmul(camera_matrix, pixel)

    temp = 1
    aa = np.vstack((camera, temp))

    # 坐标系变换矩阵
    camera2base = np.matrix(camera2base)

    base = np.dot(camera2base, aa)

    # [x,y,z]
    return base[0:3].getA().tolist(), d


# 获得在机械臂坐标系下的全部点云
@cuda.jit
def get_base_point_cloud_by_cuda(point_cloud, camera2base):
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if tx <= 640 and ty <= 480:
        for i in range(3):
            point = point_cloud[ty][tx]
            pixel = [[point[0]][point[1]][point[2]]]
            normalized_point = np.vstack((pixel, 1))
            # 坐标系变换矩阵
            camera2base = np.matrix(camera2base)
            base = np.dot(camera2base, normalized_point)
            base = np.asarray(base)
            base = base[:3, 0].flatten()
            base = np.append(base, point[3:6])
            point_cloud[ty][tx] = base


def get_base_cloud_by_numba(point_cloud, camera2base):
    """获得拟合平面在机械臂坐标系下的点云,待优化,上面的版本想使用GPU加速"""
    base_point_cloud = []
    for point in point_cloud:
        #
        pixel = np.zeros((3, 1))
        pixel[0, 0] = point[0]
        pixel[1, 0] = point[1]
        pixel[2, 0] = point[2]
        normalized_point = np.vstack((pixel, 1))
        # 坐标系变换矩阵
        camera2base = np.matrix(camera2base)
        base = np.dot(camera2base, normalized_point)
        base = np.asarray(base)
        base = base[:3, 0].flatten()
        base = np.append(base, point[3:6])
        base_point_cloud.append(base)
    return base_point_cloud


def get_target_point_cloud(point_cloud, camera2base, center_point_pixel, point_area=40, ):
    base_point_cloud = []

    # 如果是取中心点附近的点云的话
    if center_point_pixel is not None:
        # 把base_cloud变成(480*640)的大小
        point_cloud = np.reshape(point_cloud, (480, 640, -1))
        # point_cloud = get_point_cloud()

        point_area = int(point_area / 2)

        point_cloud = point_cloud[center_point_pixel[1] - point_area:center_point_pixel[1] + point_area,
                      center_point_pixel[0] - point_area:center_point_pixel[0] + point_area]

        for yindex in range(len(point_cloud)):
            for xindex in range(len(point_cloud[0])):
                x = point_cloud[yindex][xindex][0]
                y = point_cloud[yindex][xindex][1]
                z = point_cloud[yindex][xindex][2]

                #
                pixel = np.zeros((3, 1))
                pixel[0, 0] = x
                pixel[1, 0] = y
                pixel[2, 0] = z

                temp = 1
                normalized_point = np.vstack((pixel, temp))
                # 坐标系变换矩阵
                camera2base = np.matrix(camera2base)
                base = np.dot(camera2base, normalized_point)
                base = np.asarray(base)
                base = base[:3, 0].flatten()
                base = np.append(base, point_cloud[yindex][xindex][3:6])
                base_point_cloud.append(base)
    return base_point_cloud


# 获得中心点坐标
def get_center_point(point_list):
    x = (float(point_list[2]) + float(point_list[0])) / 2
    y = (float(point_list[3]) + float(point_list[1])) / 2
    return x, y


# 格式化输出YOLO识别结果的方法
def print_items_info(items):
    for item in items:
        print('物体类别: {: ^15} 置信度: {}   坐标: ({},{}) ({},{})  序号：{}'.
              format(item[4], item[5], item[0], item[1], item[2], item[3], item[6]))


# 格式化输出catch_points的方法
def print_catch_points(catch_points):
    for point_info in catch_points:
        print('机械臂坐标系坐标: x:{} y:{} z:{} 面积: {} 序号:{}  权重:{} 类别:{}'.
              format(point_info[0], point_info[1], point_info[2], point_info[3], point_info[4], point_info[5],
                     point_info[6]))


# 格式化输出catch_points的方法
def print_point(point):
    print('机械臂坐标系坐标: x:{} y:{} z:{} 面积: {} 序号:{}  权重:{} 类别:{}'.
          format(point[0], point[1], point[2], point[3], point[4], point[5],
                 point[6]))


# 写入点云的方法,第一个参数是文件名,第二个是点云,第三个是点云内每个点有多少的数据
def write_cloud(filename, point_cloud, size):
    # 每个点包含了x y z R G B6个信息
    if size == 6:
        with open(filename, 'w') as f:
            for p in point_cloud:
                f.write(
                    str(np.float(p[0])) + ' ' + str(np.float(p[1])) + ' ' + str(np.float(p[2])) + ' '
                    + str(np.float(p[3])) + ' ' + str(np.float(p[4])) + ' ' + str(
                        np.float(p[5])) + '\n')
    # 每个点只有xyz3个信息
    if size == 3:
        with open(filename, 'w') as f:
            for p in point_cloud:
                f.write(
                    str(np.float(p[0])) + ' ' + str(np.float(p[1])) + ' ' + str(np.float(p[2])) + '\n')


# 读取YOLO识别出来的结果,包含坐标,类别名和置信度
def get_yolo_result(result_path):
    file = open(result_path)
    # 读取npy_of_mask矩阵,npy_of_mask[0]对应第一个物体的分割矩阵
    masks = np.load(config.MASKS_PATH)
    # 用于存放所有识别结果的二维数组,每一行的前四个是坐标,第五个是面的分类,第六个是置信度
    items = []
    index = 0

    for line in file.readlines():
        line = line.split()
        # 把位置文件中的每一行变成列表
        text = [i for i in line]
        text.append(index)
        text.append(masks[index])
        index += 1
        # 忽略置信度小于0.6的物体
        if float(text[5]) > config.CONFIDENCE:
            items.append(text)
    return items


# 获得指定平面的深度图
def get_item_depth(depth_img, item_area):
    depth = cv.imread(depth_img)
    area = cv.imread(item_area)
    matrix = np.multiply(depth, area)
    cv.imwrite("photo/img_item_depth.png", matrix)
    img = cv.imread("photo/img_item_depth.png")
    return img


def plane_fit(target_point_cloud, flag=False):
    """flag表示是否显示拟合的示意图,使用ransac来拟合平面,返回的是np.array (1, 4) 是Ax+By+Cy+D的系数 """
    # 初始化pointCloud
    pcd_load = o3d.geometry.PointCloud()
    # 转为numpy格式
    target_point_cloud = np.asarray(target_point_cloud)
    # 读取前三位,为xyz
    pcd_load.points = o3d.utility.Vector3dVector(target_point_cloud[:, 0:3])
    # 读取RGB
    pcd_load.colors = o3d.utility.Vector3dVector(target_point_cloud[:, 2:5])

    # 去掉离群点
    cl, ind = pcd_load.remove_statistical_outlier(nb_neighbors=80, std_ratio=3)  # 返回pointcloud和索引
    # cl, ind = pcd_load.remove_radius_outlier(nb_points=4, radius=1.0)  # 返回pointcloud和索引
    display_inlier_outlier(pcd_load, ind)

    # np.savetxt('target_point_cloud.txt',np.asarray(cl.points))
    plane1 = pyrsc.Plane()
    # 拟合平面,与被认为是内点的平面的阈值距离,第一个是数组,为AX+BY+CZ+D的四个系数,第二个四内点的集合
    best_eq, best_inliers = plane1.fit(np.asarray(cl.points))
    # best_eq, best_inliers = plane1.fit(np.asarray(pcd_load.points))
    plane = cl.select_by_index(best_inliers).paint_uniform_color([1, 0, 0])
    obb = cl.get_oriented_bounding_box()
    obb2 = cl.get_axis_aligned_bounding_box()
    obb.color = [0, 0, 1]
    obb2.color = [0, 1, 0]
    if True:
        o3d.visualization.draw_geometries([plane, obb, obb2], width=900, height=600, window_name="平面拟合")
    print("ransac拟合结果:%.3f * x + %.3f * y + %.3f z + %.3f " % (best_eq[0], best_eq[1], best_eq[2], best_eq[3]))
    return best_eq[0:3]


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    # 选中的点为灰色，未选中点为红色
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # 可视化
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])



# 分割出每个面，把分割出来的彩色图像放入指定位置
def split_pic(items, color_path=config.COLOR_PATH, depth_path=config.DEPTH_PATH,
              save_path=config.MASKS_COLOR_PATH):
    # 删除之前的结果
    for file in os.scandir(save_path):
        os.remove(file.path)
    # 读取color图
    color = cv.imread(color_path, -1)
    # 读取depth图
    depth = cv.imread(depth_path, -1)
    # 遍历每个mask，并进行覆盖
    for index in range(len(items)):
        mask = items[index].mask
        # 构造三通道mask
        mask_3 = np.zeros_like(color)
        mask_3[:, :, 0] = mask
        mask_3[:, :, 1] = mask
        mask_3[:, :, 2] = mask

        # if not np.max(masks):
        #     continue

        # 覆盖
        sub = color * mask_3
        depth_sub = depth * mask
        # 保存图片

        cv.imwrite(save_path + str(index) + "-" + items[index].name + "-" + str(items[index].confidence) + ".png", sub)
        cv.imwrite(save_path + str(index) + "-" + items[index].name + "-" + str(items[index].confidence) + "_depth" +
                   ".png", depth_sub)
        cv.imwrite(config.final_path + str(index) + "-" + items[index].name + str(items[index].confidence) + ".png",
                   sub)


# 获得腐蚀后的mask
def get_narrow_area(mask, index):
    # 构造腐蚀核，准备腐蚀边缘
    erode_decrease = cv.getStructuringElement(cv.MORPH_CROSS, (15, 15))
    # 得到腐蚀后的mask
    mask = cv.erode(mask[0], erode_decrease, iterations=1)
    # 读取颜色图
    img = cv.imread(config.COLOR_PATH, -1)
    # 读取深度图
    # depth_img = cv.imread(config.PREDICT_DEPTH_PATH, -1)
    np.asarray(img)
    mask_3 = np.zeros_like(img)
    mask_3[:, :, 0] = mask
    mask_3[:, :, 1] = mask
    mask_3[:, :, 2] = mask
    # 覆盖操作
    img = mask_3 * img
    # depth_img = depth_img * mask.astype(np.uint16)
    # depth_img = cv.bitwise_and(depth_img, mask.astype(np.unit8))
    # 保存图片
    cv.imwrite(config.MASKS_COLOR_PATH + str(index) + "_erode.png", img)
    cv.imwrite(config.final_path + str(index) + "-mask_erode.png", img)
    # cv.imwrite(config.MASKS_COLOR_PATH + str(index) + "_depth.png", depth_img.astype(np.uint16))
    return mask


# 从深度图获得相机坐标系下的全部点云
def get_point_cloud(depth_path, color_path, camera_intrinsic):
    # 读取RGB和深度图
    color_raw = o3d.io.read_image(color_path)
    depth_raw = o3d.io.read_image(depth_path)
    # 把RGB图和深度图合成为RGBD图
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,
                                                                    convert_rgb_to_intensity=False)
    # 返回480*640的深度点云
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = camera_intrinsic
    # PCD就是点云数据
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # 获得点数据
    p = pcd.points
    c = pcd.colors
    # 转化成numpy矩阵
    pc = np.asarray(p)
    cc = np.asarray(c)
    cc = cc * 255
    pc = np.append(pc, cc, axis=1)
    # 保存至txt文件
    np.savetxt(config.PREDICT_DEPTH_POINT_CLOUD_PATH, pc)
    return pc


# 计算放置物体的回转角角度
def cal_rotate_back_angle(original_mask):
    original_mask = original_mask.astype(np.uint8)
    kernel = np.ones((40, 40), np.uint8)

    original_mask = cv.morphologyEx(original_mask[0], cv.MORPH_CLOSE, kernel)

    edges = cv.Canny(original_mask, 0, 1)
    cv.imwrite("./photo/canny.png", edges)
    cv.imwrite(config.final_path + "canny.png", edges)
    x, y = np.where(edges == 255)
    point_list = []

    for i in range(len(x)):
        point = [x[i], y[i]]
        point_list.append(point)

    point_list = np.array(point_list)
    # 获得最小内切矩形
    out_rect = cv.minAreaRect(point_list)
    if out_rect[-2][0] < out_rect[-2][1]:
        return out_rect[-1] - 90
    # 最后一个参数是跟照片边框的角度
    return out_rect[-1]


def rotate_with_direct(angle, a, b, c):
    # 绕任意轴旋转任意角度公式
    # 不是我写的，我也不会，参考https://blog.csdn.net/FreeSouthS/article/details/112576370
    # 顺便帮我检查一下有没有敲错
    angle = angle
    c = float(c)
    a = np.array([(b ** 2 + c ** 2) * math.cos(angle) + a ** 2, a * b * (1 - math.cos(angle)) - c * math.sin(angle),
                  a * c * (1 - math.cos(angle)) + b * math.sin(angle),
                  a * b * (1 - math.cos(angle)) + c * math.sin(angle), b ** 2 + (1 - b ** 2) * math.cos(angle),
                  b * c * (1 - math.cos(angle)) - a * math.sin(angle),
                  a * c * (1 - math.cos(angle)) - b * math.sin(angle),
                  b * c * (1 - math.cos(angle)) + a * math.sin(angle), c ** 2 + (1 - c ** 2) * math.cos(angle)])
    a = a.reshape((3, 3))
    return a


# 计算抓取点和方向
def get_catch_point(results):
    items = yolo_item.get_items(results)
    if len(items) == 0:
        config.data['have_item'] = False
        print('抓取完毕，摄像头区域内已无目标')
        return None

    if config.MODEL == 1:
        # 从深度估计网络获得相机坐标系下的点云
        camera_point_cloud = get_point_cloud(config.PREDICT_DEPTH_PATH, config.COLOR_PATH, config.CAMERA_INTRINSIC)
    else:
        # 使用双目相机的数据
        camera_point_cloud = np.load(config.NPY_PATH)

    # masks_npy是一堆480*640的矩阵，每个矩阵对应一个物体的分割矩阵
    # 将图像分割成各个mask，存放在config.MASKS_COLOR_PATH
    split_pic(items, config.COLOR_PATH, config.DEPTH_PATH, config.MASKS_COLOR_PATH)

    # 输出识别结果
    yolo_item.print_items(items)

    # 记录每个抓点对应items里面的第几行
    index = 0
    # 开始处理所有待抓取的物体,找到第一个要抓的物体
    for item in items:
        # 获得中心点,也是抓取目标点
        x = item.center_point[0]
        y = item.center_point[1]
        # 得到中心点在机器臂坐标系下的位置
        base_point, d = pixel_to_world2(config.DEPTH_PATH, x, y, config.CAMERA2BASE, config.CAMERA_INTRINSIC)
        # 如果找到了有深度的中心点
        if d != 0:
            # 这一步之后,base_point已经变成了一个包含三个浮点数的list,分别是机械臂坐标系下的xyz
            base_point = [i[0] for i in base_point]
        else:
            print("无法获得中心点,尝试在其周围30*30的范围寻找替代点")
            for step in range(30):
                base_point, d = pixel_to_world2(config.DEPTH_PATH, x - step, y, config.CAMERA2BASE,
                                                config.CAMERA_INTRINSIC)
                if d != 0:
                    base_point = [i[0] for i in base_point]
                    break
                base_point, d = pixel_to_world2(config.DEPTH_PATH, x + step, y, config.CAMERA2BASE,
                                                config.CAMERA_INTRINSIC)
                if d != 0:
                    base_point = [i[0] for i in base_point]
                    break
                base_point, d = pixel_to_world2(config.DEPTH_PATH, x, y - step, config.CAMERA2BASE,
                                                config.CAMERA_INTRINSIC)
                if d != 0:
                    base_point = [i[0] for i in base_point]
                    break
                base_point, d = pixel_to_world2(config.DEPTH_PATH, x, y + step, config.CAMERA2BASE,
                                                config.CAMERA_INTRINSIC)
                if d != 0:
                    base_point = [i[0] for i in base_point]
                    break
                if step == 29:
                    print("在30*30的范围内都无法找到抓点,请检查抓取对象是否有反光,深度图是否出现大面积黑洞")
        # 为item添加抓取点数据
        item.catch_pose = np.asarray(base_point)
        # 添加权重数据
        item.weight = get_weights(item, config.AREA_WEIGHT)
        # 添加腐蚀处理后的mask
        item.narrow_mask = get_narrow_area(item.mask, item.num)
        index = index + 1

    # 根据权重进行排序
    # catch_points=[x,y,z,面积,序号,权重,类别]
    items.sort(key=point_weight)
    print('***********************************************')
    yolo_item.print_items_catch(items)
    # 获得权重最大的面,[x,y,z,面积,index,权重,类别,mask]
    item = items[-1]
    # 获得盒状物体短边和照片的夹角
    back_angle = cal_rotate_back_angle(item.mask)
    config.real_back_angle = back_angle
    item.angle = back_angle
    print("角度:", back_angle)
    print('***********************************************')
    print("目标点：")
    item.print_catch_info()
    catch_img = cv.imread("photo/color.png")
    catch_img = cv.circle(catch_img, (int(item.center_point[0]), int(item.center_point[1])), 10, (0, 0, 0), -1)
    # 保存抓取点图
    cv.imwrite(config.CATCH_IMG_PATH, catch_img)

    # 用来拟合平面的点云
    target_point_cloud_in_camera = []

    # 把base_cloud变成(480*640)的大小
    camera_point_cloud = np.reshape(camera_point_cloud, (480, 640, -1))

    # 获得腐蚀后的mask点云
    point_location = np.where(item.narrow_mask != 0)
    for x in range(len(point_location[0])):
        point_of_cloud = camera_point_cloud[point_location[0][x]][point_location[1][x]]
        # 先获得相机坐标系下的拟合平面点云
        target_point_cloud_in_camera.append(point_of_cloud)
    # 把拟合平面的点云写入txt
    if config.WRITER_TARGET_CLOUD:
        np.savetxt(config.TARGET_POINT_CLOUD_PATH, target_point_cloud_in_camera, fmt='%.5f %.5f %.5f %d %d %d',
                   delimiter=' ')
    # 获得拟合平面在机械臂坐标系下的点云,待优化
    target_point_cloud = get_base_cloud_by_numba(target_point_cloud_in_camera, config.CAMERA2BASE)
    # 保存目标点云
    item.target_point_cloud = target_point_cloud
    # else:
    #     # 获得机械臂坐标系下目标点周围一定范围的点云
    #     target_point_cloud = get_base_point_cloud(camera_point_cloud, config.CAMERA2BASE, center_point_pixel,
    #                                               config.POINT_AREA)

    # 最小二乘拟合
    # normal_vector = get_normal_vector(target_point_cloud)
    # ransac平面拟合
    normal_vector = plane_fit(target_point_cloud)
    # print("原法向量:")
    # print(normal_vector)
    normal_vector = np.asarray(normal_vector)
    if normal_vector[-1] > 0:
        normal_vector = -normal_vector
    # print("法向量坐标:")
    # print(normal_vector)
    # 把法向量写入item中
    item.normal_vector = normal_vector
    config.normal_vector1 = normal_vector[0]
    config.normal_vector2 = normal_vector[1]
    config.normal_vector3 = normal_vector[2]
    # 如果法向量与xy平面平行,这设置为一个预设的值
    origin_v = [0, 0, 0]
    if normal_vector[0] == 0 and normal_vector[1] == 0:
        v = [2.009, -2.434, 0.068]
    else:
        v = rotation_matrix_from_vectors([0, 0, 1], normal_vector)
        origin_v = v
        # v = np.dot(v, angle_mat)
        v = matrix_util.rm2rpy(v)
        v[2] = v[2]
        v = matrix_util.rpy2rv(v)

        origin_v = matrix_util.rm2rpy(origin_v)
        origin_v = matrix_util.rpy2rv(origin_v)

    # print("计算前的rxryrz")
    # print(origin_v)
    # print("旋转向量rxryrz")
    # print(v)
    # print("rpy角")
    # print(util.rv2rpy(v[0], v[1], v[2]) / 6.28 * 360)

    x = normal_vector[0]
    y = normal_vector[1]
    z = normal_vector[2]

    # alpha = math.atan(math.fabs(y / x))
    # theta = math.asin(math.fabs(z) / math.sqrt(x ** 2 + y ** 2 + z ** 2))
    beta = math.acos(math.fabs(z) / math.sqrt(x ** 2 + y ** 2 + z ** 2))
    # print(beta)
    # print(beta / math.pi * 180)

    if math.fabs(beta) < math.pi * 2 * 7 / 360:
        v = [math.pi, 0.0, 0.0]

    flag = [1, 1, 1]
    if x > 0:
        flag[0] = -1
    if y > 0:
        flag[1] = -1
    if z > 0:
        flag[2] = 1

    item.catch_pose = np.append(item.catch_pose, v)
    print("抓取点:", item.catch_pose[0], " ", item.catch_pose[1], " ", item.catch_pose[2])

    f = open(config.CATCH_POSE_PATH, "w")
    f.write(str(item.catch_pose[0]) + " " + str(item.catch_pose[1]) + " " + str(item.catch_pose[2]) + " " + str(
        item.catch_pose[3]) + " "
            + str(item.catch_pose[4]) + " " + str(item.catch_pose[5]) + " " + item.name)

    return item
