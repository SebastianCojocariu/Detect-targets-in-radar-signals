import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import random
from tensorflow.keras import backend as K

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from data_augmenter import DataAugmenter
from accuracy_regression import accuracy_regression
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Input, MaxPool2D, Conv2D, BatchNormalization, AveragePooling2D, \
    Dropout, DepthwiseConv2D, Activation, Add, GaussianNoise, GlobalAveragePooling2D, Concatenate, Multiply

# prepare the labels in the suitable format (classification or regression)
def prepare_labels(labels, preparation_type):
    if preparation_type == "regression":
        labels = np.asarray(labels, dtype=np.float32)

    elif preparation_type == "classification":
        labels = np.asarray(labels, dtype=np.uint8)
        labels = labels - 1 # <--- because the labels starts from 1
        labels = np.eye(5)[labels]

    return labels

# custom loss function (margin loss)
def margin_loss(y_true, y_pred):
    L = K.maximum(0., K.square(y_true - y_pred) - 0.25)
    return K.mean(L)


class Trainer():
    def __init__(self, all_train_data, all_train_labels, train_data, train_labels, validation_data, validation_labels, loss_name, model_type, learning_rate_decay, enable_batchnorm, batch_size, epochs, early_stopping_args, checkpoint_args,
                 learning_rate, weight_decay, task_type, num_classes):
        self.all_train_data = all_train_data
        self.all_train_labels = all_train_labels
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.loss_name = loss_name
        self.model_type = model_type

        self.learning_rate_decay = learning_rate_decay
        self.enable_batchnorm = enable_batchnorm
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_args = early_stopping_args
        self.checkpoint_args = checkpoint_args
        self.num_classes = num_classes
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.initial_shape = train_data[0].shape

        #self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
        #self.optimizer = tf.keras.optimizers.Adam()
        self.optimizer = tfa.optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay)

        if self.task_type == "regression":
            self.early_stopping_args["monitor"] = "val_accuracy_regression"
            self.early_stopping_args["mode"] = "max"
            self.checkpoint_args["monitor"] = "val_accuracy_regression"
            self.checkpoint_args["mode"] = "max"
        elif self.task_type == "classification":
            self.early_stopping_args["monitor"] = "val_accuracy"
            self.early_stopping_args["mode"] = "max"
            self.checkpoint_args["monitor"] = "val_accuracy"
            self.checkpoint_args["mode"] = "max"
        else:
            raise Exception("#### Unknown task_type: {}".format(self.task_type))

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(**self.early_stopping_args)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(**self.checkpoint_args)
        learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr if epoch < 1 else min(0.1,
                 max(0.0001, lr * self.learning_rate_decay)))

        self.callbacks_list = [checkpoint_callback, early_stopping_callback, learning_rate_callback]


    def conv_block(self, input, filters, kernel_size, strides, activation="relu", enable_batchnorm=True):
        res = input

        res = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation=None, padding="same")(res)
        if activation is not None and activation != "":
            res = Activation(activation=activation)(res)

        if enable_batchnorm:
            res = BatchNormalization()(res)

        return res

    # layer to reweight channels/features importances
    def squeeze_and_excitation(self, input, shrink_ratio):
        no_channels = input.shape[-1]

        res = GlobalAveragePooling2D()(input)
        res = Dense(units=no_channels // shrink_ratio, activation="relu")(res)
        res = Dense(units=no_channels, activation="sigmoid")(res)
        res = Multiply()([input, res])

        return res

    # resnet block
    def resnet_block(self, input, kernel_size, shrink_ratio, strides, enable_batchnorm, out_filters=None):
        res, identity = input, input

        # shrink the number of intermediate channels.
        intermediate_channels = input.shape[-1] // shrink_ratio
        if out_filters is None:
            out_filters = input.shape[-1]

        res = self.conv_block(input=res, filters=intermediate_channels, kernel_size=(1, 1), strides=1, activation="relu", enable_batchnorm=enable_batchnorm)
        res = self.conv_block(input=res, filters=intermediate_channels, kernel_size=kernel_size, strides=strides, activation="relu", enable_batchnorm=enable_batchnorm)
        res = self.conv_block(input=res, filters=out_filters, kernel_size=(1, 1), strides=1, activation="relu", enable_batchnorm=enable_batchnorm)

        res = self.squeeze_and_excitation(input=res, shrink_ratio=4) # <--- squeeze and excitation layer

        if out_filters != identity.shape[-1] or strides > 1:
            identity = self.conv_block(input=identity, filters=out_filters, kernel_size=(1, 1), strides=strides, activation=None, enable_batchnorm=enable_batchnorm)

        res = res + identity
        return res


    # inception block
    def inception_block(self, input, channels_1x1, channels_3x3_reduce, channels_3x3, channels_5x5_reduce, channels_5x5, channels_projection, strides=1, enable_batchnorm=False):
        res = input

        conv1x1 = self.conv_block(input=res, filters=channels_1x1, kernel_size=(1, 1), strides=strides, activation="relu", enable_batchnorm=enable_batchnorm)

        conv3x3 = self.conv_block(input=res, filters=channels_3x3_reduce, kernel_size=(1, 1), strides=1,
                                  activation="relu", enable_batchnorm=enable_batchnorm)
        conv3x3 = self.conv_block(input=conv3x3, filters=channels_3x3, kernel_size=(3, 3), strides=strides,
                                  activation="relu", enable_batchnorm=enable_batchnorm)

        conv5x5 = self.conv_block(input=res, filters=channels_5x5_reduce, kernel_size=(1, 1), strides=1,
                                  activation="relu", enable_batchnorm=enable_batchnorm)
        conv5x5 = self.conv_block(input=conv5x5, filters=channels_5x5, kernel_size=(5, 5), strides=strides,
                                  activation="relu", enable_batchnorm=enable_batchnorm)

        projection = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(res)
        projection = self.conv_block(input=projection, filters=channels_projection, kernel_size=(1, 1), strides=strides,
                                  activation="relu", enable_batchnorm=enable_batchnorm)

        return Concatenate()([conv1x1, conv3x3, conv5x5, projection]) # <--- check axis value


    def transition_layer_dense(self, input, enable_batchnorm):
        res = input

        # reduce number of channels
        res = self.conv_block(input=res, filters=res.shape[-1] // 2, kernel_size=(1, 1), strides=1, activation="relu", enable_batchnorm=enable_batchnorm)
        # reduce dimension
        res = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(res)
        return res


    def dense_block(self, input, no_modules, out_channels_per_module, intermediate_channels_per_module, enable_batchnorm):
        res = input

        for _ in range(no_modules):
            intermediate_res = self.conv_block(input=res, filters=intermediate_channels_per_module, kernel_size=(1, 1),
                                               strides=1, activation="relu", enable_batchnorm=enable_batchnorm)
            intermediate_res = self.conv_block(input=intermediate_res, filters=out_channels_per_module, kernel_size=(3, 3),
                                               strides=1, activation="relu", enable_batchnorm=enable_batchnorm)
            intermediate_res = self.squeeze_and_excitation(input=intermediate_res, shrink_ratio=4) # <--- squeeze and excitation layer
            res = Concatenate()([res, intermediate_res])

        return res


    def mobilenet_v2(self, input, filters, kernel_size, strides, expansion_rate, enable_batchnorm):
        res, identity = input, input

        res = self.conv_block(input=res, filters=input.shape[-1] * expansion_rate, kernel_size=(1, 1), strides=1, activation="relu", enable_batchnorm=enable_batchnorm)

        res = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding="same")(res)
        res = Activation("relu")(res)

        res = self.conv_block(input=res, filters=filters, kernel_size=(1, 1), strides=1, activation="relu", enable_batchnorm=enable_batchnorm)

        res = self.squeeze_and_excitation(input=res, shrink_ratio=4)  # <--- squeeze and excitation layer

        if res.shape[-1] != identity.shape[-1] or strides > 1:
            identity = self.conv_block(input=identity, filters=res.shape[-1], kernel_size=(1, 1), strides=strides, activation=None, enable_batchnorm=enable_batchnorm)

        res = res + identity
        return res


    def create_model(self):
        inputs = Input(shape=self.initial_shape, name="input")
        res = inputs

        res = GaussianNoise(stddev=0.01)(res)

        if self.model_type == "linear":
            res1 = AveragePooling2D((3, 3))(res)
            res1 = Flatten()(res1)
            res2 = MaxPool2D((3, 3))(res)
            res2 = Flatten()(res2)
            res = Concatenate()([res1, res2])

        elif self.model_type == "vgg":
            res = self.conv_block(input=res, filters=64, kernel_size=(3, 3), strides=1, activation="relu", enable_batchnorm=self.enable_batchnorm)
            res = self.conv_block(input=res, filters=64, kernel_size=(3, 3), strides=1, activation="relu", enable_batchnorm=self.enable_batchnorm)
            res = AveragePooling2D((2, 2))(res)

            res = self.conv_block(input=res, filters=96, kernel_size=(3, 3), strides=1, activation="relu", enable_batchnorm=self.enable_batchnorm)
            res = self.conv_block(input=res, filters=96, kernel_size=(3, 3), strides=1, activation="relu", enable_batchnorm=self.enable_batchnorm)
            res = AveragePooling2D((2, 2))(res)

            res = self.conv_block(input=res, filters=128, kernel_size=(3, 3), strides=1, activation="relu", enable_batchnorm=self.enable_batchnorm)
            res = self.conv_block(input=res, filters=128, kernel_size=(3, 3), strides=1, activation="relu", enable_batchnorm=self.enable_batchnorm)
            res = AveragePooling2D((2, 2))(res)

            res = self.conv_block(input=res, filters=256, kernel_size=(3, 3), strides=1, activation="relu", enable_batchnorm=self.enable_batchnorm)
            res = self.conv_block(input=res, filters=256, kernel_size=(3, 3), strides=1, activation="relu", enable_batchnorm=self.enable_batchnorm)
            res = AveragePooling2D((2, 2))(res)

            res = GlobalAveragePooling2D()(res)

        elif self.model_type == "resnet":
            res = self.conv_block(input=res, filters=64, kernel_size=(3, 3), strides=1, activation="relu", enable_batchnorm=self.enable_batchnorm)

            #res = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(res)
            res = self.resnet_block(input=res, kernel_size=(3, 7), shrink_ratio=2, strides=2, out_filters=128, enable_batchnorm=self.enable_batchnorm)
            res = self.resnet_block(input=res, kernel_size=(3, 7), shrink_ratio=2, strides=1, out_filters=128, enable_batchnorm=self.enable_batchnorm)
            res = self.resnet_block(input=res, kernel_size=(3, 7), shrink_ratio=2, strides=1, out_filters=128, enable_batchnorm=self.enable_batchnorm)

            #res = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(res)
            res = self.resnet_block(input=res, kernel_size=(3, 5), shrink_ratio=3, strides=2, out_filters=196, enable_batchnorm=self.enable_batchnorm)
            res = self.resnet_block(input=res, kernel_size=(3, 5), shrink_ratio=3, strides=1, out_filters=196, enable_batchnorm=self.enable_batchnorm)
            res = self.resnet_block(input=res, kernel_size=(3, 5), shrink_ratio=3, strides=1, out_filters=196, enable_batchnorm=self.enable_batchnorm)

            #res = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(res)
            res = self.resnet_block(input=res, kernel_size=(3, 3), shrink_ratio=4, strides=2, out_filters=256, enable_batchnorm=self.enable_batchnorm)
            res = self.resnet_block(input=res, kernel_size=(3, 3), shrink_ratio=4, strides=1, out_filters=256, enable_batchnorm=self.enable_batchnorm)
            res = self.resnet_block(input=res, kernel_size=(3, 3), shrink_ratio=4, strides=1, out_filters=256, enable_batchnorm=self.enable_batchnorm)

            #res = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(res)
            res = self.resnet_block(input=res, kernel_size=(3, 3), shrink_ratio=4, strides=2, out_filters=512, enable_batchnorm=self.enable_batchnorm)
            res = self.resnet_block(input=res, kernel_size=(3, 3), shrink_ratio=4, strides=1, out_filters=512, enable_batchnorm=self.enable_batchnorm)
            res = self.resnet_block(input=res, kernel_size=(3, 3), shrink_ratio=4, strides=1, out_filters=512, enable_batchnorm=self.enable_batchnorm)

            res = GlobalAveragePooling2D()(res)

        elif self.model_type == "densenet":
            res = self.conv_block(input=res, filters=64, kernel_size=(3, 3), strides=1, activation="relu", enable_batchnorm=self.enable_batchnorm)

            res = self.dense_block(input=res, no_modules=6, out_channels_per_module=32, intermediate_channels_per_module=96, enable_batchnorm=self.enable_batchnorm)
            res = self.transition_layer_dense(input=res, enable_batchnorm=self.enable_batchnorm)

            res = self.dense_block(input=res, no_modules=8, out_channels_per_module=32, intermediate_channels_per_module=96, enable_batchnorm=self.enable_batchnorm)
            res = self.transition_layer_dense(input=res, enable_batchnorm=self.enable_batchnorm)

            res = self.dense_block(input=res, no_modules=12, out_channels_per_module=32, intermediate_channels_per_module=96, enable_batchnorm=self.enable_batchnorm)
            res = self.transition_layer_dense(input=res, enable_batchnorm=self.enable_batchnorm)

            res = self.dense_block(input=res, no_modules=16, out_channels_per_module=32, intermediate_channels_per_module=96, enable_batchnorm=self.enable_batchnorm)
            res = self.transition_layer_dense(input=res, enable_batchnorm=self.enable_batchnorm)

            res = GlobalAveragePooling2D()(res)

        elif self.model_type == "inception":
            res = self.conv_block(input=res, filters=64, kernel_size=(3, 3), strides=1, activation="relu", enable_batchnorm=self.enable_batchnorm)

            res = self.inception_block(input=res, channels_1x1=32, channels_3x3_reduce=16, channels_3x3=32, channels_5x5_reduce=16, channels_5x5=32, channels_projection=32, strides=2, enable_batchnorm=self.enable_batchnorm)
            res = self.inception_block(input=res, channels_1x1=48, channels_3x3_reduce=24, channels_3x3=48, channels_5x5_reduce=24, channels_5x5=48, channels_projection=48, strides=1, enable_batchnorm=self.enable_batchnorm)
            res = self.inception_block(input=res, channels_1x1=64, channels_3x3_reduce=32, channels_3x3=64, channels_5x5_reduce=32, channels_5x5=64, channels_projection=64, strides=2, enable_batchnorm=self.enable_batchnorm)
            res = self.inception_block(input=res, channels_1x1=96, channels_3x3_reduce=48, channels_3x3=96, channels_5x5_reduce=48, channels_5x5=96, channels_projection=96, strides=2, enable_batchnorm=self.enable_batchnorm)

            res = GlobalAveragePooling2D()(res)

        res = Dense(1024, activation="relu", use_bias=True)(res)
        res = Dropout(0.5)(res)
        #res = Dense(256, activation="relu", use_bias=True)(res)
        #res = Dropout(0.5)(res)

        print("#### Type: {} ####\n".format(self.task_type))
        if self.task_type == "classification":
            predictions = Dense(self.num_classes, activation="softmax", use_bias=True)(res)
            model = Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer=self.optimizer,
                          loss=self.loss_name,
                          metrics=["accuracy"])
        elif self.task_type == "regression":
            predictions = Dense(1, activation=None, use_bias=True)(res)
            model = Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer=self.optimizer,
                          loss=self.loss_name,
                          metrics=[accuracy_regression, "mae", "mse"])
        else:
            raise Exception("Unknown task_type: {}".format(self.task_type))

        self.model = model
        self.model.summary()


    def train(self):
        self.create_model()

        train_labels = prepare_labels(labels=self.train_labels, preparation_type=self.task_type) # <--- prepare the labels according to the task
        validation_labels = prepare_labels(labels=self.validation_labels, preparation_type=self.task_type) # <--- prepare the labels according to the task

        train_augmenter = DataAugmenter(images=self.train_data, labels=train_labels, transform_type="train") # <--- augment the images from the training set
        validation_augmenter = DataAugmenter(images=self.validation_data, labels=validation_labels, transform_type="predict") # <--- prepare the images from the validation set without augmenting.

        history = self.model.fit(x=train_augmenter.augmented_images,
                                 y=train_augmenter.augmented_labels,
                                 validation_data=(validation_augmenter.augmented_images, validation_augmenter.augmented_labels),
                                 callbacks=self.callbacks_list,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 epochs=self.epochs)

        if self.task_type == "regression":
            self.model = tf.keras.models.load_model(filepath=self.checkpoint_args["filepath"],
                                                    custom_objects={"accuracy_regression": accuracy_regression})
        else:
            self.model = tf.keras.models.load_model(filepath=self.checkpoint_args["filepath"])

        return history.history


    def cross_validation(self, epochs=1):
        # shuffle the training data (the same across multiple trainings)
        shuffled_indices_list = list(range(len(self.all_train_data)))
        random.Random(42).shuffle(shuffled_indices_list)

        all_train_data = self.all_train_data[shuffled_indices_list]
        all_train_labels = self.all_train_labels[shuffled_indices_list]

        # Start a 5-Fold Cross Validation. 
        k_fold = KFold(n_splits=5)
        mae_list, mse_list, accuracy_list = [], [], []
        for train_indices, test_indices in k_fold.split(all_train_data):
            X_train, X_test = all_train_data[train_indices], all_train_data[test_indices]
            y_train, y_test = all_train_labels[train_indices], all_train_labels[test_indices]

            y_train = prepare_labels(labels=y_train, preparation_type=self.task_type)  # <--- prepare the labels according to the task
            y_test = prepare_labels(labels=y_test, preparation_type=self.task_type)  # <--- prepare the labels according to the task

            train_augmenter = DataAugmenter(images=X_train, labels=y_train, transform_type="train")  # <--- augment the images from the training set
            test_augmenter = DataAugmenter(images=X_test, labels=y_test, transform_type="predict")  # <--- prepare the images from the validation set without augmentation.

            if self.model_type == "random":
                predictions = [random.randint(1, 5) for _ in range(len(y_test))]
            else:
                self.create_model()

                history = self.model.fit(x=train_augmenter.augmented_images,
                                         y=train_augmenter.augmented_labels,
                                         batch_size=self.batch_size,
                                         shuffle=True,
                                         epochs=epochs)
                predictions = self.model.predict(test_augmenter.augmented_images, batch_size=self.batch_size)
                predictions = np.reshape(predictions, (predictions.shape[0],))

            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)

            classes = np.around(predictions)
            # clip the predictions to [1,2,3,4,5]
            for i in range(len(classes)):
                if classes[i] <= 1:
                    classes[i] = 1
                elif classes[i] >= self.num_classes:
                    classes[i] = self.num_classes

            accuracy = accuracy_score(y_test, classes)

            print("#### MAE: {}, MSE: {}, Accuracy: {}".format(mae, mse, accuracy))

            mae_list.append(mae)
            mse_list.append(mse)
            accuracy_list.append(accuracy)


        print("#### MAE per folds: {}".format(mae_list))
        print("#### MSE per folds: {}".format(mse_list))
        print("#### Accuracy per folds: {}".format(accuracy_list))

        return mae_list, mse_list, accuracy_list


    def get_predictions(self, x):
        predict_augmenter = DataAugmenter(x, labels=[-1] * len(x), transform_type="predict") # <--- prepare the images, but dont augment. (using predict instead of train)
        predictions = self.model.predict(predict_augmenter.augmented_images, batch_size=self.batch_size)

        if self.task_type == "classification":
            classes = np.argmax(predictions, axis=1) + 1 # because the labels starts with 1.
        else:
            predictions = np.reshape(predictions, (predictions.shape[0],))
            classes = np.around(predictions)
            # clip the predictions to [1,2,3,4,5]
            for i in range(len(classes)):
                if classes[i] <= 1:
                    classes[i] = 1
                elif classes[i] >= self.num_classes:
                    classes[i] = self.num_classes

        classes = classes.astype(int)
        return classes, predictions
