import os
import shutil
import sys
import warnings

import pandas as pd

# FIXME: Workaround to show logging: https://github.com/Deci-AI/super-gradients/issues/1021
stdout = sys.stdout  # noqa: E402
from super_gradients.training import Trainer, models  # noqa: E402

sys.stdout = stdout

from super_gradients.training.dataloaders.dataloaders import (  # noqa: E402
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
from super_gradients.training.utils.checkpoint_utils import (  # noqa: E402
    load_checkpoint_to_model,
)
from tqdm import tqdm  # noqa: E402

from yolo_base import YoloBase  # noqa: E402

# FIXME: Workaround to show logging: https://github.com/Deci-AI/super-gradients/issues/1021
warnings.filterwarnings("default")


class YOLO_nas(YoloBase):
    """
    Manager of pictures to feed specifically YOLOnas and the output of YOLOnas. Performing checks of duplicates and more.

    """

    def __init__(self, source_data: str, task: str):
        super().__init__(source_data, task)

        self.CHECKPOINT_DIR = "checkpoints"

        if os.path.exists(os.path.join(self.CHECKPOINT_DIR, self.task)):
            warnings.warn(
                "You will overwrite best weights, save them before running the train or you will lose them!",
                stacklevel=2,
            )

        self.trainer = Trainer(
            experiment_name=self.task, ckpt_root_dir=self.CHECKPOINT_DIR
        )

        from settings import dataloader_params, dataset_params, model_size, train_params

        self.model_size = model_size
        self.dataset_params = dataset_params
        self.dataloader_params = dataloader_params
        self.train_params = train_params

    def create_folders_tree(self):
        """Creating a set of multiple folders based on train, res and validation"""

        dir_list = [
            "\\custom_dataset\\",
            "\\custom_dataset\\train",
            "\\custom_dataset\\train\\images",
            "\\custom_dataset\\train\\labels",
            "\\custom_dataset\\val",
            "\\custom_dataset\\val\\images",
            "\\custom_dataset\\val\\labels",
        ]

        for folder in dir_list:
            try:
                os.makedirs(f"{self.upper_path}{folder}")
            except OSError:
                # The directory already existed, nothing to do
                pass

    def train_val_split(self, image_folder, label_folder, ratio_val=0.2):
        """Specific function to split the data accordingly to YOLO version requirements"""

        if os.path.exists(f"sample_history/sample_history_{self.task}.csv"):
            history = pd.read_csv(f"sample_history/sample_history_{self.task}.csv")
            history_train_df = history.loc[history["sample_type"] == "TRAIN_VAL"]
            history_train = history_train_df["filename"].unique().tolist()
        else:
            raise FileExistsError(
                f"sample_history/sample_history_{self.task}.csv is not existing"
            )

        train_images = os.listdir(os.path.join(self.upper_path, image_folder))

        assert len(train_images) == len(
            history_train
        ), f"Images in {image_folder} are not the same number as in history."

        assert (
            len(list(set(train_images) & set(history_train)))
            == len(train_images)
            == len(history_train)
        ), f"There are unknown pictures in {image_folder}"

        self.create_folders_tree()

        for file in os.listdir(os.path.join(self.upper_path, label_folder)):
            with open(os.path.join(self.upper_path, label_folder, file)) as f:
                f_read = f.readline()
                label = f_read.split()[0]
                history_train_df.loc[
                    history_train_df["filename"].str.strip(".jpg")
                    == file.strip(".txt"),
                    "label",
                ] = int(
                    label
                )  # filename without jpg

        val_df = history_train_df.sample(frac=ratio_val, random_state=42)

        train_df = history_train_df.drop(val_df.index)
        for image in tqdm(train_df.filename.str.strip(".jpg").to_list(), desc="Train"):
            shutil.copy2(
                os.path.join(self.upper_path, image_folder, f"{image}.jpg"),
                os.path.join(self.upper_path, "custom_dataset", "train", "images"),
            )
            try:
                shutil.copy2(
                    os.path.join(self.upper_path, label_folder, f"{image}.txt"),
                    os.path.join(self.upper_path, "custom_dataset", "train", "labels"),
                )
            except FileNotFoundError:
                pass

        for image in tqdm(
            val_df.filename.str.strip(".jpg").to_list(), desc="Validation"
        ):
            shutil.copy2(
                os.path.join(self.upper_path, image_folder, f"{image}.jpg"),
                os.path.join(self.upper_path, "custom_dataset", "val", "images"),
            )
            try:
                shutil.copy2(
                    os.path.join(self.upper_path, label_folder, f"{image}.txt"),
                    os.path.join(self.upper_path, "custom_dataset", "val", "labels"),
                )
            except FileNotFoundError:
                pass

    def train(self, **kwargs):
        """Specific function to run the train script of YOLO"""

        train_data = coco_detection_yolo_format_train(
            dataset_params={
                "data_dir": self.dataset_params["data_dir"],
                "images_dir": self.dataset_params["train_images_dir"],
                "labels_dir": self.dataset_params["train_labels_dir"],
                "classes": self.dataset_params["classes"],
            },
            dataloader_params=self.dataloader_params,
        )

        val_data = coco_detection_yolo_format_val(
            dataset_params={
                "data_dir": self.dataset_params["data_dir"],
                "images_dir": self.dataset_params["val_images_dir"],
                "labels_dir": self.dataset_params["val_labels_dir"],
                "classes": self.dataset_params["classes"],
            },
            dataloader_params=self.dataloader_params,
        )

        model = models.get(
            self.model_size,
            num_classes=len(self.dataset_params["classes"]),
            pretrained_weights="coco",
        )

        if "warm_start" in kwargs:
            load_checkpoint_to_model(
                net=model,
                ckpt_local_path=kwargs["warm_start"],
            )
            print("Warm start loaded")

        self.trainer.train(
            model=model,
            training_params=self.train_params,
            train_loader=train_data,
            valid_loader=val_data,
        )

    def infer(self, path_image, weights="", conf=0.5):
        """Specific function to run the infer script of YOLO"""

        if weights == "":
            weights = f"{self.CHECKPOINT_DIR}/{self.task}/ckpt_best.pth"
        best_model = models.get(
            self.model_size,
            num_classes=len(self.dataset_params["classes"]),
            checkpoint_path=weights,
        )

        prediction = best_model.predict(path_image, conf=conf)

        return prediction

    def deploy_onnx(self, weights=""):
        """Deploy in onnx format"""

        if weights == "":
            weights = f"{self.CHECKPOINT_DIR}/{self.task}/ckpt_best.pth"

        model_chose = models.get(
            self.model_size,
            num_classes=len(self.dataset_params["classes"]),
            checkpoint_path=weights,
        )

        model_chose.export(f"{self.task}_{self.model_size}.onnx")
