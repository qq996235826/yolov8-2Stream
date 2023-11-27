from ultralytics import YOLO
import cv2 as cv

if __name__ == '__main__':
    model = YOLO('D:/project/Python/twoStreamCatch/ultralytics/cfg/models/v8/yolov8m-seg-2stream.yaml')  # build a new model from YAML
    """
    更多详细配置在   ultralytics/yolo/cfg/default.yaml
    """
    model.train(data="D:/project/datasets/yolov8-10000/data.yaml", epochs=200, imgsz=640, device=0, batch=4, amp=False)

    # 推理
    # model = YOLO('best.pt')  # load a custom model
    #
    # # Predict with the model
    # results = model(['berlin_color.png','depth_viridis.png'])  # predict on an image
    # # 保存结果图
    # res_plotted = results[0].plot()
    # # 保存至photo文件夹下
    # cv.imwrite("result.png", res_plotted)