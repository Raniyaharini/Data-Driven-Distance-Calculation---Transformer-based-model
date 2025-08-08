import os
import shutil
from tqdm import tqdm
import pandas as pd
import yaml

from yolo_base import YoloBase


class YOLOv6(YoloBase):
    """
    Manager of pictures to feed specifically YOLOv6 and the output of YOLOv6. Performing checks of duplicates and more.

    """

    def create_folders_tree(self):
        """Creating a set of multiple folders based on train and validation"""

        dir_list = [
            "\\custom_dataset\\",
            "\\custom_dataset\\images",
            "\\custom_dataset\\images\\train",
            "\\custom_dataset\\images\\val",
            "\\custom_dataset\\images\\test",
            "\\custom_dataset\\labels\\train",
            "\\custom_dataset\\labels\\val",
            "\\custom_dataset\\labels\\test",
        ]

        for folder in dir_list:
            try:
                os.makedirs(f"{self.upper_path}{folder}")
            except OSError:
                # The directory already existed, nothing to do
                pass

    def train_val_split(self, image_folder: str, label_folder: str, ratio_val: float):
        """Splits a dataset of images and their corresponding labels into two subsets for training and validation
           Creates the appropriate folder structure for each subset

           Note : Resultant images in the folder contain images with .jpg

        Parameters
        ----------
        image_folder : string
            A string representing the path to the directory containing the training images
        label_folder : dataframe
            A dataframe containing the path to the directory containing the labels for the training images
        ratio_val : float
            A float representing the ratio of the dataset to be used for validation

        Returns
        -------

        """

        # Coherency checks
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

        for image in tqdm(
            train_df.filename.str.strip(".jpg").to_list(), desc="Train"
        ):  # filename without .jpg
            shutil.copy2(
                os.path.join(self.upper_path, image_folder, f"{image}.jpg"),
                f"{self.upper_path}\\custom_dataset\\images\\train",
            )  # filename with .jpg
            try:
                shutil.copy2(
                    os.path.join(self.upper_path, label_folder, f"{image}.txt"),
                    f"{self.upper_path}\\custom_dataset\\labels\\train",
                )
            except FileNotFoundError:
                pass

        for image in tqdm(
            val_df.filename.str.strip(".jpg").to_list(), desc="Validation"
        ):  # filename without .jpg
            shutil.copy2(
                os.path.join(self.upper_path, image_folder, f"{image}.jpg"),
                f"{self.upper_path}\\custom_dataset\\images\\val",
            )  # filename with .jpg
            try:
                shutil.copy2(
                    os.path.join(self.upper_path, label_folder, f"{image}.txt"),
                    f"{self.upper_path}\\custom_dataset\\labels\\val",
                )
            except FileNotFoundError:
                pass

    def infer(
        self,
        weights: str,
        yaml_file: str,
        image_path: str,
        method: str = "best",
        output_dir: str = "",
        save_txt: bool = False,
        not_save_img: bool = False,
        not_run: bool = False,
    ) -> pd.DataFrame:
        """Runs the prediction process and stores the result in a text file. This infer method() returns the dataframe containing predicted labels and confidence

         Note : In this function predictions hold filenames without jpg
         The return dataframe pred_df contains filename along with jpg

        Parameters
         ----------
         weights : str
             The path to the weights file stored as a dictionary in the YAML file generated by YOLO.
         yaml_file : str
             The path of the YAML file that stores all the image names for prediction.
         image_path : str
             The path of the directory containing the images to be predicted.
         method : str, optional
             The method for extracting predictions. Default is 'best'.
         output_dir : str, optional
             The path of the directory where the predicted results should be saved. Default is an empty string.
         save_txt : bool, optional
             A flag indicating whether to save the predicted results in a text file. Default is False.
         not_save_img : bool, optional
             A flag indicating whether to not save the predicted images. Default is False.
         not_run : bool, optional
             A flag indicating whether to not run the prediction process. Default is False.

         Returns
         -------
         pd.DataFrame
             A DataFrame containing predicted labels and confidence.
        """

        from tools.infer import run

        self.subroutine_check(image_path)

        with open(yaml_file) as file:
            read_yaml = yaml.safe_load(file)

        label_dict = {}
        for i, label in enumerate(read_yaml["names"]):
            label_dict[i] = label

        if not not_run:
            run(
                weights=weights,
                source=os.path.join(self.upper_path, image_path),
                yaml=yaml_file,
                save_dir=os.path.join(self.upper_path, output_dir),
                device=0,
                view_img=False,
                save_txt=save_txt,
                not_save_img=not_save_img,
            )

        if save_txt:
            predictions = {}
            all_images = os.listdir(os.path.join(self.upper_path, image_path))

            for image in all_images:
                predictions[image.strip(".jpg")] = [
                    "NO PREDICTION",
                    None,
                ]  # Filename without .jpg

            ## THIS MUST BE CHANGE!
            folder_name = image_path.split("\\")[-1]
            label_path = os.path.join(self.upper_path, output_dir, folder_name)

            for file_pred in os.listdir(label_path):
                assert (
                    file_pred.strip(".txt") in predictions.keys()
                ), f"The following prediction refer to a not existing image: {file_pred}"

                with open(os.path.join(label_path, file_pred)) as f:
                    all_pred_raw = f.readlines()

                if method == "best":
                    pred, conf = self.output_parser_best(all_pred_raw, pred=True)
                    predictions[file_pred.strip(".txt")] = [label_dict[pred], conf]

                elif method == "all":
                    pred_list = self.output_parser_all(all_pred_raw)
                    predictions[file_pred.strip(".txt")] = pred_list

            if method == "best":
                final_col = ["prediction", "confidence"]
            elif method == "all":
                len_list = [len(i) for i in predictions.values()]
                longest = max(len_list) / 2
                res = [[f"pred_{i}"] + [f"conf_{i}"] for i in range(longest)]
                final_col = [item for sublist in res for item in sublist]

            pred_df = pd.DataFrame.from_dict(
                predictions, orient="index", columns=final_col
            )
            pred_df = pred_df.reset_index().rename(columns={"index": "filename"})
            pred_df["filename"] = pred_df["filename"].str[:] + ".jpg"

            return pred_df  # filename with jpg

        else:
            return pd.DataFrame({})

    def output_parser_best(self, file: list, pred: bool = True):
        # need to ask marco about the return type...I was thinking it was tuple (An integer representing the label and a float representing the confidence)
        """Parsing input of type:
        [
        [4 0.691667 0.329948 0.15 0.135937 0.957169],
        [3 0.691623 0.135937 0.12 0.691667 0.946731],
        [4 0.691667 0.329948 0.15 0.135937 0.977199]
        ]
        First number is the label, the consequent numbers are bounding box coords, last one is confidence
        Looping through the lists in the list and giving as output only the prediction with the highest confidence score.

        Parameters
        ----------
        file : list
            The list containing lists with the row information
        pred : bool, optional
            A flag used to check if there is or not the confidence

        Returns
        -------
        int
            The label
        float
            The confidence
        """

        prediction = ""
        best_conf = 0

        if pred:
            for pred_str in file:
                pred_num = pred_str.split()

                if float(pred_num[-1]) > best_conf:
                    best_conf = float(pred_num[-1])
                    prediction = int(pred_num[0])

            return prediction, best_conf

        else:
            pred_str = file[0]  # One line only
            pred_num = pred_str.split()
            return int(pred_num[0]), None

    def output_parser_all(self, file: list) -> list:
        """Parsing input of type and returning a list:
        [
        [label1, confidence1, label2, confidence2, ...]
        ]

        Parameters
        ----------
        file : list
            The list containing the row information.

        Returns
        -------
        list
            The list containing the labels and confidences.
        """
        pred_list = []
        for i, pred_str in enumerate(file):
            pred_num = pred_str.split()
            pred_list.append(pred_num[0])
            pred_list.append(pred_num[-1])

        return pred_list

    def train(
        self,
        path_to_data: str,
        config: str,
        batch_size: int = 8,
        epochs: int = 100,
        device: int = 0,
        **kwargs,
    ):
        """Description

        NOT WORKING!!


        Parameters
        ----------
        yaml : type1
            descr
        config_file : type2, optional
            descr
        weights_file:
            descr

        Returns
        -------
        int
            descr
        float, optional
            descr
        """

        from tools.train import main as main_train
        from tools.train import get_args_parser

        parser_for_yolo = get_args_parser()

        params = {
            "--data-path": path_to_data,
            "--conf-file": config,
            "--batch-size": str(batch_size),
            "--epochs": str(epochs),
            "--device": str(device),
        }

        param_list = [item for key in params for item in (key, params[key])]

        for opt in kwargs:
            param_list += [f"--{opt}", kwargs[opt]]

        args_for_train = parser_for_yolo.parse_args(param_list)

        main_train(args_for_train)

    def copy_files_with_tag(self, csv_file):
        """Copies files with a specific tag from a CSV file to a new folder named after the tag


         Parameters
        ----------
        csv_file : str
            Path to the CSV file that contains the tags and filenames

        Returns
        -------

        """

        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file, header=None, skiprows=1, names=["filename", "tag"])

        # Loop over each unique tag in the DataFrame
        for tag in df["tag"].unique():
            # Skip files with tag named TRAIN_VAL
            if tag != "TRAIN_VAL":
                # Create a new folder with the name of the tag
                tag_folder = os.path.join(self.upper_path, tag)
                os.makedirs(tag_folder, exist_ok=True)

                # Get the filenames associated with the current tag
                filenames = df.loc[df["tag"] == tag, "filename"]

                # Copy each file to the tag folder
                for filename in filenames:
                    source_file = os.path.join(self.source_data, filename)
                    dest_file = os.path.join(tag_folder, filename)
                    shutil.copyfile(source_file, dest_file)
