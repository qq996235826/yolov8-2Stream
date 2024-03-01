from ultralytics import YOLO
import cv2 as cv
from catch import calculate_pose, get_pose
from catch import config


if __name__ == '__main__':
    model = YOLO("weight/best-FF.pt")
    results = model(['depth_utils/color.png', 'depth_utils/viridis.png'])
    # 保存结果图
    res_plotted = results[0].plot()
    # 保存至photo文件夹下
    cv.imwrite("catch_result/photo/result.png", res_plotted)
    # 保存至历史
    cv.imwrite(config.final_path + "result.png", res_plotted)
    # 对照片进行处理并获得位姿信息
    item = get_pose.get_catch_pose_by_orb(results)