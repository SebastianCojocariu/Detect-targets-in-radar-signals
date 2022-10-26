import pandas as pd
import numpy as np
import os
import json
import ast

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

PATHS = []
SAVE_PATH = "xgboost_all_preds.csv"
TRAINING_NUMBERS = [0, 1, 2, 3, 4, 5, 6, 7]
THRESHOLD_VAL_ACCURACY = 0.745

# Combine the predictions from various trainings and train an XGBoost Meta-Learner
if __name__ == "__main__":
    selected_training_numbers = []
    print("#### Initial training numbers: {}\n".format(TRAINING_NUMBERS))

    X_train, X_test = [], []
    Y_train = []

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
                print("#### Discared training no: {}".format(os.path.join(PATH, train_no)))
                continue
            print("### Training {} -> val_accuracy: {}".format(os.path.join(PATH, train_no), history["validation_accuracy"]))

            selected_training_numbers.append(train_no)

            df_train_val = pd.read_csv(os.path.join(PATH, "{}/train_validation_predictions.csv".format(train_no)))

            df_train_val["prediction"] = df_train_val["prediction"].apply(lambda x: x if isinstance(x, float) else np.argmax(np.fromstring(x[1:-1], sep=' ').astype(float)))
            train_data = list(df_train_val["prediction"])
            X_train.append(train_data)

            if len(Y_train) == 0:
                Y_train = list(df_train_val["label"])
            else:
                assert Y_train == list(df_train_val["label"]) # <--- assert that the id's are the same accros various trainings

            df_test = pd.read_csv(os.path.join(PATH, "{}/test_predictions.csv".format(train_no)))
            df_test["prediction"] = df_test["prediction"].apply(lambda x: x if isinstance(x, float) else np.argmax(np.fromstring(x[1:-1], sep=' ').astype(float)))
            test_data = list(df_test["prediction"])
            X_test.append(test_data)

    print("\n#### Selected only the trainings: {}".format(selected_training_numbers))
    print(len(X_train))

    X_train = np.transpose(np.asarray(X_train, dtype=np.float32))
    X_test = np.transpose(np.asarray(X_test, dtype=np.float32))
    Y_train = np.transpose(np.asarray(Y_train, dtype=np.uint8))
    Y_train = Y_train - 1 # <--- because the classes starts from 1, instead of 0
    print(X_train)
    print(X_test)
    print(Y_train)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.20) # <--- split the training data into train + validation

    print("#### Train shape: {} {} ####".format(X_train.shape, Y_train.shape))
    print("#### Validation shape: {} {} ####".format(X_val.shape, Y_val.shape))
    print("#### Test shape: {} ####".format(X_test.shape))

    parameters = {
        'learning_rate': 0.1,
        'silent': True,
        'objective': 'multi:softprob',
        'num_class': 5,
        'max_depth': 5,
        'n_estimators': 100
        }
    num_round = 300  # the number of training iterations

    xgb_model = xgb.XGBClassifier(**parameters)

    xgb_model.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_val, Y_val)],
                  early_stopping_rounds=20,
                  verbose=True)

    accuracy_train = accuracy_score(Y_train, xgb_model.predict(X_train))
    accuracy_val = accuracy_score(Y_val, xgb_model.predict(X_val))

    print("#### [Train] Accuracy: {}".format(accuracy_train * 100.0))
    print("#### [Validation] Accuracy: {}".format(accuracy_val * 100.0))

    classes = xgb_model.predict(X_test) + 1 # <--- add 1 because the classes start from 0
    print(classes)

    final_df = pd.DataFrame.from_dict({"id": df_test["id"], "label": classes})

    file_name = SAVE_PATH
    final_df.to_csv(file_name, index=False)
    print(final_df)
    