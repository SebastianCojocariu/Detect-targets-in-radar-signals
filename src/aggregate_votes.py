import pandas as pd
import numpy as np
import os
import json
from collections import Counter

PATHS = []
SAVE_PATH = "all_preds.csv"
TRAINING_NUMBERS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
THRESHOLD_VAL_ACCURACY = 0.75

# Combine the predictions from various trainings
if __name__ == "__main__":
    names, votes, predictions = None, [], []
    selected_training_numbers = []
    print("#### Initial training numbers: {}\n".format(TRAINING_NUMBERS))

    for PATH in PATHS:
        for train_no in TRAINING_NUMBERS:
            train_no = str(train_no)

            # if no such path exists, continue.
            if not os.path.exists(os.path.join(PATH, "{}".format(train_no))):
                print("#### Training {} does not exist!".format(os.path.join(PATH, train_no)))
                continue

            # if history is not available, it means that the training is not finished yet, so continue.
            if not os.path.exists(os.path.join(PATH, "{}/history.json".format(train_no))):
                print("#### Training {} is not finished!".format(os.path.join(PATH, train_no)))
                continue

            with open(os.path.join(PATH, "{}/history.json".format(train_no))) as json_file:
                history = json.load(json_file)

            # if the validation accuracy is less than the THRESHOLD, we should discard it.
            if "validation_accuracy" not in history or float(history["validation_accuracy"]) < THRESHOLD_VAL_ACCURACY:
                print("#### Discared {}".format(os.path.join(PATH, train_no)))
                continue
            print("### Training {} -> val_accuracy: {}".format(os.path.join(PATH, train_no), history["validation_accuracy"]))

            selected_training_numbers.append(train_no)

            df = pd.read_csv(os.path.join(PATH, "{}/final_predictions.csv".format(train_no)))
            if not names:
                names = list(df["id"])
            else:
                assert names == list(df["id"]) # <--- assert that the id's are the same accros various trainings

            votes.append(list(df["label"]))

    if len(votes) == 0:
        raise Exception("#### All the trainings were discarded!!!")
    print(len(votes))

    votes = np.asarray(votes)
    majority_votes = []
    for i in range(votes.shape[1]):
        c = Counter(list(votes[:, i]))
        value, count = c.most_common()[0]
        majority_votes.append(value)

    final_df = pd.DataFrame.from_dict({"id": names, "label": majority_votes})
    file_name = SAVE_PATH
    final_df.to_csv(file_name, index=False)

    print(final_df)

