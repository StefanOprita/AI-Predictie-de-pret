import keras.models
from sklearn.preprocessing import MinMaxScaler
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
from pyqtgraph.examples.ExampleApp import QFont
import pandas as pd
import numpy as np
import sys


def do_predictions(model, path_to_csv):
    df = pd.read_csv(path_to_csv)
    indexes = [i for i in range(0, len(df)) if i % 30 == 0]
    first_timestamp = df["Unix Timestamp"][len(df["Unix Timestamp"]) - 1]

    temp = df.iloc[indexes]
    df = temp
    df = df[len(df) - 50:]
    df = df['Open'].values
    df = df.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df = np.array(df)
    scaler.fit(df)
    values_to_model = scaler.transform(df)
    predictions = model.predict(values_to_model)

    rez = []
    predictions = scaler.inverse_transform(predictions)
    for pred in predictions:
        rez.append(pred[0])

    final_timestamps = []
    for i in range(len(rez)):
        final_timestamps.append(first_timestamp)
        first_timestamp += 1800000
    return rez, final_timestamps


class MyWindow(QMainWindow):

    def __init__(self):
        super(MyWindow, self).__init__()
        self.setGeometry(200, 200, 1000, 600)
        self.setWindowTitle("Bitcoin Prediction")
        self.setWindowIcon((QtGui.QIcon("bitcoin_logo")))
        self.chosen_file = ""
        self.drop_down = QtWidgets.QComboBox(self)
        self.graphWidget = pg.PlotWidget(self)
        self.graphWidget.hide()
        self.label = QtWidgets.QLabel(self)
        self.btn_show_graphic = QtWidgets.QPushButton(self)
        self.btn_choose_file = QtWidgets.QPushButton(self)
        self.init_ui()

    def show_graphic(self):
        self.label.hide()
        styles = {'color': 'white', 'font-size': '15px'}
        # self.graphWidget.setAxisItems({'bottom': DateAxisItem()})
        self.graphWidget.setLabel('left', 'Value(BTC)', **styles)
        self.graphWidget.setLabel('bottom', 'Time', **styles)
        self.graphWidget.setTitle("Bitcoin Prediction")
        self.graphWidget.setGeometry(40, 50, 930, 430)
        pen = pg.mkPen(color=(255, 0, 0), width=1)

        bitcoin_value, res_time = self.get_results()
        print(res_time)
        self.graphWidget.plot(res_time, bitcoin_value, pen=pen)

        self.graphWidget.show()

    def choose_file(self):
        self.chosen_file = QtWidgets.QFileDialog.getOpenFileName(self, 'Upload CSV', 'D:\\', 'CSV files (*.csv)')

    # return function call
    def get_results(self):
        value = self.drop_down.currentText()
        if value == "Model 1":
            model = keras.models.load_model("C:\\Users\\strat\\Documents\\GitHub\\AI-Predictie-de-pret\\ProcessingData\\Models\\price_only_prediction_50_future_10_epochs")
            return do_predictions(model, self.chosen_file[0])
        elif value == "Model 2":
            model = keras.models.load_model("D:\\Facultate\\an 3\\sem 1\\AI\\AI-Predictie-de-pret\\LSTM\\Models\\price_only_prediction_50_future_20_epochs")
            return do_predictions(model, self.chosen_file[0])
        else:
            model = keras.models.load_model("D:\\Facultate\\an 3\\sem 1\\AI\\AI-Predictie-de-pret\\LSTM\\Models\\all_prediction_50_future_v2_10_epochs")
            return do_predictions(model, self.chosen_file[0])

    def init_ui(self):
        self.label.setText("There is nothing to show yet!")
        self.label.move(380, 240)
        self.label.setFont(QFont('Times font', 15))
        self.label.adjustSize()

        self.btn_show_graphic.setText("Show graph")
        self.btn_show_graphic.clicked.connect(self.show_graphic)
        self.btn_show_graphic.move(600, 520)

        self.btn_choose_file.setText("Upload CSV")
        self.btn_choose_file.clicked.connect(self.choose_file)
        self.btn_choose_file.move(300, 520)

        font = QtGui.QFont()
        font.setPointSize(10)
        self.drop_down.setFont(font)
        self.drop_down.setGeometry(20, 20, 30, 10)
        self.drop_down.addItem("Model 1")
        self.drop_down.addItem("Model 2")
        self.drop_down.addItem("Model 3")
        self.drop_down.adjustSize()


def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())


window()
