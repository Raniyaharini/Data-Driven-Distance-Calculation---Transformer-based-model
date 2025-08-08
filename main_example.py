import os
import torch
from yolo_nas import YOLO_nas
from yolo_v6 import YOLOv6

SHAREPOINT_PATH = r"E:\\Thesis\\all_images\\Images"
#PATH_TO_TRAIN = r"E:\\Thesis\\custom_dataset"
#Best_CKPT=r"C:\\Users\\pmdominator\\Desktop\\git\\research-yolo-framework\\checkpoints\\Wallbox_frames\\ckpt_best.pth"
# if torch.cuda.is_available():
#     print("CUDA (GPU) is available.")
# else:
#     print("CUDA (GPU) is not available.")
CUDA_LAUNCH_BLOCKING=1
if __name__ == "__main__":
    yolo = YOLOv6(SHAREPOINT_PATH, task="Thesis_roof")
    #yolo.dataset_params["data_dir"] = PATH_TO_TRAIN
    print(yolo)

    #yolo.train_val_split(image_folder=r"E:\\Thesis\\train_images_new", label_folder=r"E:\\Thesis\\train_labels")

    #yolo.extract_new_sample(200, tag='TEST1', out_path=r"E:\\Thesis\\TEST1")
    
    #yolo.train(warm_start = "C:\\Users\\pmdominator\\Desktop\\git\\research-yolo-framework\\checkpoints\\ckpt_latest.pth")

    result = yolo.infer("E:\\Thesis\\TEST1", weights = 'C:\\Users\\pmdominator\\Desktop\\git\\research-yolo-framework\\checkpoints\\Thesis_roof\\ckpt_best.pth')

    #result.show()

    #result.save(output_folder="E:\\Thesis\\pred_yolonas")

    #yolo.deploy_onnx(weights = "C:\\Users\\pmdominator\\Desktop\\git\\research-yolo-framework\\checkpoints\\Thesis_roof\\ckpt_best.pth")
