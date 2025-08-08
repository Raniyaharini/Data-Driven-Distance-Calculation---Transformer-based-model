import abc
import os
import warnings
import pandas as pd
import shutil
from tqdm import tqdm


class YoloBase(metaclass=abc.ABCMeta):
    """
    Base class to manage pictures and YOLO framework.
    Child classes needs to have three methods defined:
    1. train(), needed to train the model
    2. infer(), needed to infer on pictures starting from weights
    3. train_val_split(), method that create the structure of folders needed by YOLO


    Attributes
    ----------
    data_map : Pandas DataFrame
        Dataframe with all the pictures as:
                                        filename
        0              191660_BoxFrontFoto_1.jpg
        1      20210928bktEAB_BoxFrontFoto_1.jpg
        2      20210928bkUEAR_BoxFrontFoto_1.jpg
        3      20210930gd0EAB_BoxFrontFoto_1.jpg
        4      20211001ip9EAB_BoxFrontFoto_1.jpg
                    ......

    source_data : str
        Path to the images

    upper_path : str
        Upper path with respect to source_data attribute

    task : str
        The name of the task that we are performing

    """

    def __init__(self, source_data: str, task: str):
        """Reading all jpg files inside the folder and store them in an internal pandas dataframe.

        Parameters
        ----------
        source_data : str
            path to the images
        task : str
            The name of the task that we are performing. This parameter is important
            to retrieve the correct history of sample.

        Returns
        -------

        """

        self.source_data = source_data

        # Change to a library in the future to avoid problems between linux and windows
        # self.upper_path = '\\'.join(self.source_data.split("\\")[:-1])
        self.upper_path = os.path.split(self.source_data)[0]

        self.task = task

        if os.path.isdir(self.source_data):
            all_data = []

            for filename in os.listdir(self.source_data):
                if filename.endswith(".jpg"):
                    all_data.append(
                        filename
                    )  # files with jpg . Eg. 191660_BoxFrontFoto_1.jpg

                else:
                    warnings.warn(
                        "A file is not a picture in the data folder.", stacklevel=2
                    )

            self.data_map = pd.DataFrame({"filename": all_data})

            print(f"#### Initializing: {self.data_map.shape[0]} pictures found.")

        else:
            raise ValueError(f"{self.source_data} not existing.")

        if os.path.isdir("sample_history"):
            pass
        else:
            os.mkdir("sample_history")

    def __repr__(self):
        """Returns a string representation of the YoloBase object.

        Returns
        -------
        str
            String representation of the YoloBase object.
        """

        if os.path.exists(f"sample_history/sample_history_{self.task}.csv"):
            sample_history = pd.read_csv(
                f"sample_history/sample_history_{self.task}.csv"
            )
            return f"Total of {self.data_map.shape[0]} pictures are stored. There are {self.data_map.isna().sum().sum()} null values. \
            \nFound history file with a total of {sample_history.sample_type.unique().shape[0]} samples extracted. \
            \nTag found: \n{sample_history.groupby('sample_type', as_index=False)['filename'].count().rename(columns={'filename':'#'})}."

        else:
            return f"Total of {self.data_map.shape[0]} pictures are stored. There are {self.data_map.isna().sum().sum()} null values. \
            \nNo history has been found for this task."

    def extract_new_sample(
        self,
        size: int,
        tag: str,
        out_path: str,
        seed: int = 42,
        save_history: bool = True,
    ):
        """Extracts a new sample from the pool of images, saving the sample in a folder and keeping the history in a CSV file of the
        extracted pictures.

        Parameters
        ----------
        size : int
            Number of pictures to extract.
        tag : str
            Unique name to give to the extracted sample. The "TRAIN_VAL" sample is a special sample that is used for consistency checks.
        out_path : str
            Folder where to put the extracted pictures.
        seed : int, optional
            Random seed for sampling, passed to the random_state argument in the `.sample` method of Pandas. Default is 42.
        save_history : bool, optional
            Determines whether to look for the file "sample_history_{self.task}.csv", append the new sample to the file, and save it again.
            The default behavior is recommended. However, if you want to check the sample first, you can set it to False, check the pictures, and then set it back to True
            to extract the same sample again. Default is True.

        Returns
        -------
        None
        """

        if os.path.exists(f"sample_history/sample_history_{self.task}.csv"):
            sample_history = pd.read_csv(
                f"sample_history/sample_history_{self.task}.csv"
            )
            print("#### Sampling: History of samples found.")
            print(
                f"#### Sampling: There are {sample_history.sample_type.unique().shape[0]} samples extracted previously."
            )

            if tag in sample_history["sample_type"].values:
                raise ValueError(f"The tag {tag} has already been used!")

            data = self.data_map.loc[
                self.data_map["filename"].isin(sample_history["filename"])
                == False  # noqa: E712
            ]
            sample_filt = data.sample(size, random_state=seed)
            sample_filt["sample_type"] = tag

            self.copy_images(sample_filt, out_path)

            new_sample = pd.concat([sample_history, sample_filt])

            if save_history:
                self.subroutine_check(out_path, tag)
                new_sample.to_csv(
                    f"sample_history/sample_history_{self.task}.csv", index=False
                )
                print("#### Sampling: Recorded in history.")

        else:
            sample = self.data_map.sample(size, random_state=seed)
            sample["sample_type"] = tag

            self.copy_images(sample, out_path)

            if save_history:
                sample.to_csv(
                    f"sample_history/sample_history_{self.task}.csv", index=False
                )
                self.subroutine_check(out_path, tag)
                print("#### Sampling: Recorded in history.")

    def extend_sample(
        self, size: int, tag: str, out_path: str, seed: int = 42, train_data_check=True
    ):
        if os.path.exists(f"sample_history/sample_history_{self.task}.csv"):
            sample_history = pd.read_csv(
                f"sample_history/sample_history_{self.task}.csv"
            )

            if tag == "TRAIN_VAL":
                if train_data_check:
                    raise ValueError(
                        "You are extending the training set! if you are really sure of what you are doing suppress this check using train_data_check=False"
                    )
                warnings.warn(
                    "Attention, suppressing error for train dataset extension",
                    stacklevel=2,
                )

            if tag not in sample_history["sample_type"].values:
                raise ValueError(f"The tag {tag} is not in the history!")

            data = self.data_map.loc[
                self.data_map["filename"].isin(sample_history["filename"])
                == False  # noqa: E712
            ]
            sample_filt = data.sample(size, random_state=seed)
            sample_filt["sample_type"] = tag

            self.copy_images(sample_filt, out_path)

            new_sample = pd.concat([sample_history, sample_filt])

            self.subroutine_check(out_path, tag)
            new_sample.to_csv(
                f"sample_history/sample_history_{self.task}.csv", index=False
            )
            print("#### Sampling: Extension recorded in history.")

        else:
            raise FileExistsError(
                "Sample history not existing. You may want to create you first sample? Use extract_new_sample method."
            )

    def copy_images(self, df: pd.DataFrame, to_where: str):
        """Copying a set of specified images from the source to a new folder.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas DataFrame containing filenames of the pictures to move, stored in a column named "filename".
        to_where : str
            Folder where to store the copied pictures.

        Returns
        -------
        None
        """

        if os.path.isdir(os.path.join(self.upper_path, to_where)):
            if len(os.listdir(os.path.join(self.upper_path, to_where))) > 0:
                warnings.warn(
                    "Attention, there are already files inside the folder.",
                    stacklevel=2,
                )
                pass

            pass
            # raise ValueError(f'{to_where} is already existing, please provide a new name.')

        else:
            os.mkdir(os.path.join(self.upper_path, to_where))

        for filename in tqdm(df["filename"], desc="Copying images..."):
            shutil.copy(
                os.path.join(self.source_data, filename),
                os.path.join(self.upper_path, to_where),
            )

    def subroutine_check(self, path: str, tag: str):
        """Routine of consistency checks. Checks if pictures in the path (or one picture) exist in the source folder.
        Also checks that the pictures are not in the training data.

        Parameters
        ----------
        path : str
            Folder containing the pictures or filename of a picture.

        Returns
        -------
        None
        """

        if os.path.isdir(os.path.join(self.upper_path, path)):
            images = os.listdir(os.path.join(self.upper_path, path))
        elif os.path.exists(os.path.join(self.upper_path, path)):
            images = [path]

        assert len(set(images) & set(self.data_map.filename.to_list())) == len(
            images
        ), "Some pictures are not in the mother folder"

        if os.path.exists(f"sample_history/sample_history_{self.task}.csv"):
            history = pd.read_csv(f"sample_history/sample_history_{self.task}.csv")
            history_train_df = history.loc[history["sample_type"] == "TRAIN_VAL"]
            history_train = history_train_df["filename"].unique().tolist()
        else:
            raise FileExistsError(
                f"sample_history/sample_history_{self.task}.csv is not existing"
            )

        if len(set(images) & set(history_train)) != 0:
            if tag == "TRAIN_VAL":
                print("#" * 40)
                warnings.warn(
                    f"There are some training pictures in {path}! #: {len(list(set(images) & set(history_train)))}",
                    stacklevel=2,
                )
                print("#" * 40)
            else:
                raise RuntimeError(
                    f"There are some training pictures in {path}! #: {len(list(set(images) & set(history_train)))}"
                )
        else:
            pass

    def wrap_sample(self, tag: str) -> pd.DataFrame:
        """Performs sampling based on a CSV file.

        Parameters
        ----------
        tag : str
            Unique name to give to the extracted sample.
        file_link : str, optional
            A string representing a link to a file.   ### The purpose of file_link is not mentioned inside the function.

        Returns
        -------
        Pandas Dataframe
            Extracted sample
        """

        if os.path.exists(f"sample_history/sample_history_{self.task}.csv"):
            history = pd.read_csv(f"sample_history/sample_history_{self.task}.csv")
            sample = history.loc[history["sample_type"] == tag]
            return sample

        else:
            raise FileExistsError(
                f"sample_history/sample_history_{self.task}.csv is not existing"
            )

    @staticmethod
    def path_differences(path1: str, path2: str):
        """Finds the common files between two directories and prints the number of common files.

        Parameters
        ----------
        path1 : str
            A string representing the path to the first directory.
        path2 : str
            A string representing the path to the second directory.

        Returns
        -------
        None
        """

        all_images = []
        for image in os.listdir(path=path1):
            all_images.append(image)

        print(f"{len(all_images)} found in the first folder")

        # Loop on the second folder
        other_images = []
        for image in os.listdir(path=path2):
            other_images.append(image)

        print(f"{len(other_images)} found in the second folder")

        intersection = list(set(all_images) & set(other_images))

        if len(intersection) > 0:
            print(f"There are {len(intersection)} common files.")
        else:
            print("There are no common files.")

    @abc.abstractmethod
    def train(self):
        """Specific function to run the train script of YOLO"""

    @abc.abstractmethod
    def infer(self):
        """Specific function to run the infer script of YOLO"""

    @abc.abstractmethod
    def train_val_split(self):
        """Specific function to split the data accordingly to YOLO version requirements"""
