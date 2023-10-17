"""
ml_facepoints base module.

This is the principal module of the ml_facepoints project.
here you put your main classes and objects.

Be creative! do whatever you want!

If you want to replace this with a Flask application run:

    $ make init

and then choose `flask` as template.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config

from torch.utils.data import DataLoader, Dataset

TRAINING_SET_PATH = (
    r"/media/yanuar/Media/Dataset/Kaggle/facial-keypoints-detection/training.csv"
)

CATEGORY_NAMES = [
    "left_eye_center_x",
    "left_eye_center_y",
    "right_eye_center_x",
    "right_eye_center_y",
    "left_eye_inner_corner_x",
    "left_eye_inner_corner_y",
    "left_eye_outer_corner_x",
    "left_eye_outer_corner_y",
    "right_eye_inner_corner_x",
    "right_eye_inner_corner_y",
    "right_eye_outer_corner_x",
    "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x",
    "left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x",
    "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x",
    "right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x",
    "right_eyebrow_outer_end_y",
    "nose_tip_x",
    "nose_tip_y",
    "mouth_left_corner_x",
    "mouth_left_corner_y",
    "mouth_right_corner_x",
    "mouth_right_corner_y",
    "mouth_center_top_lip_x",
    "mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x",
    "mouth_center_bottom_lip_y",
]


class FacialDataset(Dataset):
    def __init__(self, data, train=True, transform=None):
        super().__init__()
        self.category = CATEGORY_NAMES
        self.transform = transform
        self.train = train
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.train:
            image = np.array(self.data.iloc[index, 30].split()).astype(np.float32)
            labels = np.array(self.data.iloc[index, :30].tolist())
            labels[np.isnan(labels)] = -1
        else:
            image = np.array(self.data.iloc[index, 1].split()).astype(np.float32)
            labels = np.zeros(30)

        ignore_indices = labels == -1
        labels = labels.reshape(15, 2)

        if self.transform:
            image = np.repeat(image.reshape(96, 96, 1), 3, 2).astype(np.uint8)
            augmentations = self.transform(image=image, keypoints=labels)
            image = augmentations["image"]
            labels = augmentations["keypoints"]

        labels = np.array(labels).reshape(-1)
        labels[ignore_indices] = -1

        return image, labels.astype(np.float32)


if __name__ == "__main__":
    data = pd.read_csv(TRAINING_SET_PATH, nrows=1000)

    ds = FacialDataset(
        data=data,
        train=True,
        transform=config.train_tx,
    )

    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    for idx, (x, y) in enumerate(loader):
        plt.imshow(x[0][0].detach().cpu().numpy(), cmap="gray")
        plt.plot(
            y[0][0::2].detach().cpu().numpy(),
            y[0][1::2].detach().cpu().numpy(),
            "go",
        )
        plt.show()
