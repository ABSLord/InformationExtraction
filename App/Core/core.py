from .tools import *
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


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
    le = preprocessing.LabelEncoder()
    words = le.fit_transform(words)
    n = frame.columns[4]
    frame.drop(n, axis=1, inplace=True)
    frame[n] = words
    clf = KNeighborsClassifier()
    s = StandardScaler()
    clf.fit(frame, Y)
    from InformationExtraction.settings import BASE_DIR
    from ..models import ExtractionModel
    joblib.dump(clf, os.path.join(BASE_DIR, 'App', 'static', 'models', model_name + '.joblib'), compress=9)
    ExtractionModel.objects.get_or_create(name=model_name)

def predict(model, image):
    pass



########################################################
#  тест на реальных данных
########################################################


def document_example():
    imgs = {'ticket-1.jpg':[450, 530, 760, 120], 'ticket-2.jpg':[490, 550, 760, 120], 'ticket-5.jpg':[490, 550, 760, 120], 'ticket-4.jpg':[490, 550, 760, 120]}
    test = {'ticket-4.jpg':[450, 530, 760, 120]}
    frames = []
    Y = []
    frames_gl = []
    for img in imgs.items():
        if img[0] != 'ticket-4.jpg':
            frame = image_to_dataframe(img[0])
            frame = frame[['left', 'top', 'width', 'height', 'text']][frame['text']!=''].dropna()
            frames.append(frame)
            frames_gl.append(frame)
            y = [1 if bound_predict(val, img[1]) else 0 for val in frame.values]
            Y += y
        else:
            frame_test = image_to_dataframe(img[0])
            frame_test = frame_test[['left', 'top', 'width', 'height', 'text']][frame_test['text'] != ''].dropna()
            frames_gl.append(frame_test)
            y_test = [1 if bound_predict(val, img[1]) else 0 for val in frame_test.values]

    frame = pd.concat(frames)
    frame_gl = pd.concat(frames_gl)
    words = frame[['text']]
    words_gl = frame_gl[['text']]
    from nltk.stem.snowball import SnowballStemmer
    le = preprocessing.LabelEncoder()
    le.fit(words_gl)
    words = le.transform(words)
    words_gl = frame_gl[['text']]
    words_gl = le.transform(words_gl)

    n = frame.columns[4]
    frame.drop(n, axis=1, inplace=True)
    frame[n] = words

    #frame[['text']] = words.reshape(161,)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier()
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler()

    n = frame_gl.columns[4]
    frame_gl.drop(n, axis=1, inplace=True)
    frame_gl[n] = words_gl

    s.fit(frame_gl)
    clf.fit(frame, Y)

    #frame = extract_from_img1('ticket-4.jpg')
    #frame = frame[['left', 'top', 'width', 'height', 'text']][frame['text'] != ''].dropna()
    #Y = [1 if bound_predict(val, test['ticket-4.jpg']) else 0 for val in frame.values]

    words_test = le.transform(frame_test[['text']])

    n = frame_test.columns[4]
    frame_test.drop(n, axis=1, inplace=True)
    frame_test[n] = words_test

    from sklearn.metrics import classification_report
    print(classification_report(clf.predict(frame_test), y_test))
    # frame_test['result'] = clf.predict(frame_test)
    # frame_test['result_t'] = y_test
    # frame_test[['text']] = le.inverse_transform(frame_test[['text']])
    # frame_test[['result']] = clf.predict(frame_test)
    pass

def words_to_nums():
    pass


if __name__ == '__main__':
    document_example()