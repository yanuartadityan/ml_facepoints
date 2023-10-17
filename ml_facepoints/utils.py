import torch
import numpy as np
import config
import pandas as pd

from tqdm import tqdm

LOOKUP_TABLE_PATH = (
    r"/media/yanuar/Media/Dataset/Kaggle/facial-keypoints-detection/IdLookupTable.csv"
)


def get_submission(loader, dataset, model_15, model_4):
    model_15.eval()
    model_4.eval()
    id_lookup = pd.read_csv(LOOKUP_TABLE_PATH)
    predictions = []
    image_id = 1

    for image, label in tqdm(loader):
        image = image.to(config.DEVICE)
        preds_15 = torch.clip(model_15(image).squeeze(0), 0.0, 96.0)
        preds_4 = torch.clip(model_4(image).squeeze(0), 0.0, 96.0)
        feat_names = id_lookup.loc[id_lookup["ImageId"] == image_id]["FeatureName"]

        for f in feat_names:
            feat_index = dataset.category.index(f)
            if feat_names.shape[0] < 10:
                predictions.append(preds_4[feat_index].item())
            else:
                predictions.append(preds_15[feat_index].item())

        image_id += 1

    df = pd.DataFrame(
        {
            "RowId": np.arange(1, len(predictions) + 1),
            "Location": predictions,
        }
    )
    df.to_csv("submission.csv", index=False)
    model_15.train()
    model_4.train()


def get_rmse(loader, model, loss_fn, device):
    model.eval()
    num_examples = 0
    losses = []
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = loss_fn(scores[targets != -1], targets[targets != -1])
        num_examples += scores[targets != -1].shape[0]
        losses.append(loss.item())

    model.train()
    print(f"Loss on val: {(sum(losses)/num_examples)**0.5}")


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
