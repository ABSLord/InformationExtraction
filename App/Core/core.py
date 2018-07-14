from .tools import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from InformationExtraction.settings import BASE_DIR
from ..models import ExtractionModel
from .LabelEncoder import CustomLabelEncoder


def bound_predict(tess_data, bound):
    word_pos = list(map(lambda x: float(x), tess_data[:4]))
    if word_pos[0] >= bound[0] \
            and word_pos[1] >= bound[1] \
            and word_pos[0] + word_pos[2] <= bound[0] + bound[2] \
            and word_pos[1] + word_pos[3] <= bound[1] + bound[3]:
        return True
    else:
        return False


def train_and_save_model(model_name, images, coords):
    frames = []
    Y = []
    coords = dict({coords[0]: list(map(lambda x: float(x), coords[1].split(';'))) for coords in coords.items()})
    for img in images.items():
        frame = image_to_dataframe(img[1])
        frame = frame[['left', 'top', 'width', 'height', 'text']][frame['text'] != ''].dropna()
        frames.append(frame)
        y = [1 if bound_predict(val, coords[img[0]]) else 0 for val in frame.values]
        Y += y
    frame = pd.concat(frames)
    words = frame[['text']]
    le = CustomLabelEncoder(new_labels="update")
    words = le.fit_transform(words)
    n = frame.columns[4]
    frame.drop(n, axis=1, inplace=True)
    frame[n] = words
    clf = KNeighborsClassifier()
    s = StandardScaler()
    clf.fit(frame, Y)
    joblib.dump(clf, os.path.join(BASE_DIR, 'App', 'static', 'models', model_name + '.joblib'), compress=9)
    joblib.dump(le, os.path.join(BASE_DIR, 'App', 'static', 'models', model_name + '_le.joblib'), compress=9)
    ExtractionModel.objects.get_or_create(name=model_name)


def predict(model, image):
    frame_test = image_to_dataframe(image[1])
    frame_test = frame_test[['left', 'top', 'width', 'height', 'text']][frame_test['text'] != ''].dropna()
    clf = joblib.load(os.path.join(BASE_DIR, 'App', 'static', 'models', model + '.joblib'))
    le = joblib.load(os.path.join(BASE_DIR, 'App', 'static', 'models', model + '_le.joblib'))
    words_test = le.transform(frame_test[['text']])
    n = frame_test.columns[4]
    frame_test.drop(n, axis=1, inplace=True)
    frame_test[n] = words_test
    y = clf.predict(frame_test)
    words = le.inverse_transform(words_test)
    return [words[i] for i in range(len(words)) if y[i]]
