import os
from keras.src.models import model
from keras.src.ops import numpy
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.saving.saved_model.load import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools



batch_size = 32
image_size = (224, 224)
classes = ['Empty parking lots', 'Parking lots full']

train_batches = ImageDataGenerator().flow_from_directory(
    directory=r"C:\Users\User\Downloads\project-Bina\python\data\train",
    classes=classes,
    class_mode='categorical',
    target_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

valid_batches = ImageDataGenerator().flow_from_directory(
    directory=r"C:\Users\User\Downloads\project-Bina\python\data\valid",
    classes=classes,
    class_mode='categorical',
    target_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

test_batches = ImageDataGenerator().flow_from_directory(
    directory=r"C:\Users\User\Downloads\project-Bina\python\data\test",
    classes=classes,
    class_mode='categorical',
    target_size=image_size,
    batch_size=batch_size,
    shuffle=False
)


def build_vgg16():
    vgg16_base = VGG16(input_shape=(224, 224, 3), include_top=True, weights='imagenet')

    vgg16_base = Model(inputs=vgg16_base.input, outputs=vgg16_base.get_layer('block5_pool').output)

    for layer in vgg16_base.layers:
        layer.trainable = False


    x = GlobalAveragePooling2D()(vgg16_base.output)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(len(classes), activation='softmax')(x)

    # instantiate new model
    myModel = Model(inputs=vgg16_base.input, outputs=predictions)

    return myModel


def compile_model():
    model = build_vgg16()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    return model


def fit_model():
    #es = EarlyStopping(monitor='val_loss',
                       #min_delta=0,
                       #patience=2,
                       #verbose=1,
                       #mode='auto',
                       #restore_best_weights=True)

    step_size_train = len(train_batches)
    step_size_valid = len(valid_batches)
    model = compile_model()
    history = model.fit(train_batches,
                        epochs=100,
                        steps_per_epoch=step_size_train,
                        validation_data=valid_batches,
                        validation_steps=step_size_valid,
                        #callbacks=[es],  # נמצא שכאן עשוי להיות הסיבה לבעיה
                        verbose=1)

    step_size_test = len(test_batches)
    result = model.evaluate(test_batches, steps=step_size_test)

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'vgg16-middlePart.h5'
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)

    print("Test set classification accuracy: {0:.2%}".format(result[1]))

    return history, model


if __name__ == "__main__":
    history,model=fit_model()


    predictions = model.predict(test_batches)
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = test_batches.classes

    cm = confusion_matrix(true_classes, predicted_classes)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    # model.save(r'F:\ללי שנה ב\פרויקטים ללי\Project-Bina\python\codes\saved_models\vgg16-middlePart.h5')



