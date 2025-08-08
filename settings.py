from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)

# Choose the model you prefer
model_size = "yolo_nas_l"
# yolo_nas_s
# yolo_nas_m
# yolo_nas_l

dataset_params = {
    # Insert full path of the mother folder with the train and test pictures
    "data_dir": r"E:\Thesis\custom_dataset",
    # I suggest to not touch these
    "train_images_dir": "train/images",
    "train_labels_dir": "train/labels",
    "val_images_dir": "val/images",
    "val_labels_dir": "val/labels",
    # Put the correct number of labels and in the correct order
    "classes": ["hooks"],
}

dataloader_params = {"batch_size": 8, "num_workers": 2}

train_params = {
    # ENABLING SILENT MODE
    "silent_mode": False,
    "launch_tensorboard": True,
    "average_best_models": True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": 10,
    "mixed_precision": False,
    "save_checkpoint_interval": 10,  # Set 'n' to an appropriate value
    "save_best_only": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        # NOTE: num_classes needs to be defined here
        num_classes=len(dataset_params["classes"]),
        reg_max=16,
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            # NOTE: num_classes needs to be defined here
            num_cls=len(dataset_params["classes"]),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7,
            ),
        )
    ],
    "metric_to_watch": "mAP@0.50",
}
