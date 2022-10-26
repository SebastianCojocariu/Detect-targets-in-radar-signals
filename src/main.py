import pandas as pd
import numpy as np
import os
import cv2
import json

from sklearn.model_selection import train_test_split
from trainer import Trainer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from collections import Counter

# read the images in the same order specified by df["id"]
def read(corpus_path, df):
    file_paths = [os.path.join(corpus_path, name) for name in df["id"]]

    data = []
    for file_path in file_paths:
        image = cv2.imread(file_path)
        data.append(image)
    data = np.asarray(data, dtype=np.float32)

    # collect the labels as well (if exist)
    labels = None
    if "label" in df:
        labels = np.asarray(df["label"], dtype=np.uint8)
        print("#### Histogram: {}".format(Counter(labels)))

    return data, labels

# plot images per group (to better undestand the data)
def plot_per_group(data, labels, target_label):
    import matplotlib.pyplot as plt

    i, curr_idx = 0, 0
    plt.figure(figsize=(10, 10))

    while (i < len(data) and curr_idx < 9):
        image = (data[i] * 255).astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = labels[i]

        if label == target_label:
            plt.subplot(3, 3, curr_idx + 1)
            plt.imshow(image)
            plt.title(label)
            plt.axis("off")
            curr_idx += 1

        i += 1


if __name__ == "__main__":
    with open('config_file.json') as json_file:
        config_file = json.load(json_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config_file["device"])
    PROJECT_PATH = os.path.join(config_file["training_path"], "{}/".format(config_file["device"]))

    if os.path.exists(PROJECT_PATH):
        raise Exception("{} already exists. Try another path!".format(PROJECT_PATH))
    os.makedirs(PROJECT_PATH, exist_ok=True)

    config_file["checkpoint_args"]["filepath"] = os.path.join(PROJECT_PATH, config_file["checkpoint_args"]["filepath"])

    with open(os.path.join(PROJECT_PATH, "config_file.json"), 'w+') as f:
        json.dump(config_file, f, indent=4)

    df_train = pd.read_csv("corpus/train.csv")
    df_test = pd.read_csv("corpus/test.csv")

    all_train_data, all_train_labels = read(corpus_path="corpus/train/", df=df_train) # <--- collect all the images from the training dataset

    df_train = df_train.sample(frac=1).reset_index(drop=True) # <--- Shuffle the training data. Because of the augmentation step, we cannot shuffle during the `fit` call.
    df_train, df_validation = train_test_split(df_train, test_size=0.2) # <--- split into train + validation

    train_data, train_labels = read(corpus_path="corpus/train/", df=df_train) # <--- load the corresponding train images
    validation_data, validation_labels = read(corpus_path="corpus/train/", df=df_validation) # <--- load the corresponding validation images
    test_data, _ = read(corpus_path="corpus/test/", df=df_test) # <--- load the corresponding test images

    print("##### [Train]: Initial: {} -> {}".format(len(df_train), train_data.shape))
    print("##### [Validation]: Initial: {} -> {}".format(len(df_validation), validation_data.shape))
    print("##### [Test]: Initial: {} -> {}".format(len(df_test), test_data.shape))

    # for label in range(1, 6):
    #    plot_per_group(data=train_data, labels=train_labels, target_label=label)

    # instantiate a trainer object
    trainer = Trainer(all_train_data=all_train_data,
                      all_train_labels=all_train_labels,
                      train_data=train_data,
                      train_labels=train_labels,
                      validation_data=validation_data,
                      validation_labels=validation_labels,
                      loss_name=config_file["loss_name"],
                      model_type=config_file["model_type"],
                      learning_rate_decay=config_file["learning_rate_decay"],
                      enable_batchnorm=config_file["enable_batchnorm"],
                      batch_size=config_file["batch_size"],
                      epochs=config_file["epochs"],
                      early_stopping_args=config_file["early_stopping_args"],
                      checkpoint_args=config_file["checkpoint_args"],
                      task_type=config_file["task_type"],
                      learning_rate=config_file["learning_rate"],
                      weight_decay=config_file["weight_decay"],
                      num_classes=5)


    # Cross Validation case
    if config_file["type"] == "cross_validation":
        mae_list, mse_list, accuracy_list = trainer.cross_validation(epochs=75)
        history_cross_validation = {}
        history_cross_validation["mae_list"] = str(mae_list)
        history_cross_validation["mse_list"] = str(mse_list)
        history_cross_validation["accuracy_list"] = str(accuracy_list)
        with open(os.path.join(PROJECT_PATH, "history_cross_validation.json"), 'w+') as f:
            json.dump(history_cross_validation, f, indent=4)
    # Training case
    elif config_file["type"] == "train":
        history = trainer.train()
        for key in history:
            history[key] = str(history[key])

        train_predicted_labels, train_predictions = trainer.get_predictions(train_data)
        validation_predicted_labels, validation_predictions = trainer.get_predictions(validation_data)
        test_predicted_labels, test_predictions = trainer.get_predictions(test_data)

        # Accuracy score for train + validation
        accuracy_train = accuracy_score(train_labels, train_predicted_labels)
        accuracy_validation = accuracy_score(validation_labels, validation_predicted_labels)

        print("#### [Train] Accuracy for the best model: {}".format(accuracy_train))
        print("#### [Validation] Accuracy for the best model: {}".format(accuracy_validation))

        history["train_accuracy"] = str(accuracy_train)
        history["validation_accuracy"] = str(accuracy_validation)

        # For regression, add MAE and MSE too
        if config_file["task_type"] == "regression":
            mae_train = mean_absolute_error(train_labels, train_predictions)
            mae_validation = mean_absolute_error(validation_labels, validation_predictions)

            mse_train = mean_squared_error(train_labels, train_predictions)
            mse_validation = mean_squared_error(validation_labels, validation_predictions)

            history["train_mae"] = str(mae_train)
            history["validation_mae"] = str(mae_validation)

            history["train_mse"] = str(mse_train)
            history["validation_mse"] = str(mse_validation)

            print("#### [Train] Mae for the best model: {}".format(mae_train))
            print("#### [Validation] Mae for the best model: {}".format(mae_validation))

            print("#### [Train] Mse for the best model: {}".format(mse_train))
            print("#### [Validation] Mse for the best model: {}".format(mse_validation))

        df = pd.DataFrame.from_dict({"id": df_test["id"], "label": list(test_predicted_labels)})
        df.to_csv(os.path.join(PROJECT_PATH, "final_predictions.csv"), index=False)

        df_train["prediction"] = list(train_predictions) # <--- add predictions
        df_validation["prediction"] = list(validation_predictions) # <--- add predictions
        df_test["prediction"] = list(test_predictions) # <--- add predictions

        # merge back train + validation
        merged_train_df = pd.concat([df_train, df_validation], ignore_index=True) # <--- merge train + validation predictions
        merged_train_df = merged_train_df.sort_values(by=["id"]) # <--- sort by id

        merged_train_df.to_csv(os.path.join(PROJECT_PATH, "train_validation_predictions.csv"), index=False)
        df_test.to_csv(os.path.join(PROJECT_PATH, "test_predictions.csv"), index=False)

        with open(os.path.join(PROJECT_PATH, "history.json"), 'w+') as f:
            json.dump(history, f, indent=4)

        print(df)