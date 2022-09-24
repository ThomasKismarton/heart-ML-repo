import sys
import pandas as pd
import numpy as np
from sklearn import svm

def csv_conv(filename):
    tf = pd.read_csv(filename, sep = ',')
    x_frame = tf.iloc[0:-1, 0:17]
    y_frame = tf.iloc[0:-1, -1]
    tx = np.asarray(x_frame).astype(float) # Converts the dataframes into workable arrays of values
    ty = np.asarray(y_frame)
    tx = tx.tolist()
    return tx, ty


def main(argv):
    train_x, train_y = csv_conv(argv[0])
    test_x, test_y = csv_conv(argv[1])

    samplesvm = svm.SVC(C = 0.7, max_iter = 1000, kernel='rbf')
    samplesvm.fit(train_x, train_y)
    res = np.asarray(samplesvm.predict(test_x)).tolist()

    datalen = len(res)
    correct = 0
    for el in range(datalen):
        if res[el] == test_y[el]:
            correct += 1
    print(correct/datalen)


if __name__ == "__main__":
    main(sys.argv[1:])
