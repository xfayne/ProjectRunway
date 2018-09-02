import sys, os
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import scipy.io as sio
import threading, subprocess
from random import randint
import sched, time as Time
import pickle
from subprocess import Popen, PIPE
#import matlab.engine

# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')

count1 = 0

FILEPATH = os.path.abspath(__file__)

ECGdata = []
GSRdata = []

ECG_cleard_data = []
GSR_cleard_data = []

eventsData = []
eventsTimes = []

blocks = []

data = [GSRdata, ECGdata]

ECGtime = []
GSRtime = []

#For simulting Real Time data
RTStreamTIME =[]
RTStreamECG = []
RTStreamGSR = []
RTStreamEVENTS = []

time = [GSRtime, ECGtime]

# represents the addition of the current data stream
# lock = threading.RLock()
# condition = threading.Condition(lock)
read = threading.Event()
write = threading.Event()

#Stream
DataStream = np.array([[],[],[]]) #The data stream comes here: Data[0] = ecg val, Data[1] = gsr val, Data[2] = event/notEvent

#Pointer to the current data added
global index
index = 0

#Events
start_block = 1 #'run_start'
end_block = 2 #'run_end'
end_data = 3 #'finish'
events = [start_block, end_block, end_data]

#ML
NETWORK = 0 #Use cllasifier or neural network
filename = 'finalized Gradient Boosting Classifier.sav'

# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)

seg_len = 2200

ecg_features = []
gsr_features = []

tags_times = []

##############################

# Run .mat scripts
#eng = matlab.engine.start_matlab()


def clearSignal(signal):
    clearedSignal = []
    return clearedSignal


########## methods for real time grahps plotting and predictions ############

#Simulates Real Time streaming data
#Takes from global variables: RTStreamECG, RTStreamGSR, RTStreamEVENTS, RTStreamTIME, DataStream
def RTStream():
    global RTStreamECG, RTStreamGSR, RTStreamEVENTS, RTStreamTIME, DataStream, end_data
    i = 0
    finish = False
    #''Acquire the condition (and thus the related lock),
    # and can then attempt to stream new data
    #condition.acquire()''
    while finish == False:

        #Update streaming new data: halt reading
        write.clear()
        print('DataStream', DataStream)
        print('DataStream_s', DataStream.shape)
        #Streaming new data
        DataStream = np.append(DataStream, np.array(RTStreamECG[i]),np.array( RTStreamGSR[i]), np.arrat(RTStreamEVENTS[i])
        print('DataStream', DataStream)
        print('DataStream', DataStream.shape)
        #Update we finish write current data
        write.set()

        #''condition.wait()

        #Finish reading all data
        if RTStreamEVENTS[i] == end_data:
            finish = True

        #Update the index
        i += 1

        #Wait for reading current data stream
        read.wait()

    # ''condition.release()
    #**Add return?




#input: block of GSR snd ECG signals.
#output: median of all the seg_len size predictions in the block
def predictBlock(GSRsignal,ECGsignal):
    i = 0
    predictions = []
    while (i+seg_len < len(GSRsignal)):
        predict = predictSeg([ GSRsignal[i:i+seg_len],ECGsignal[i:i+seg_len] ])
        predictions.extend(predict)
        i+= seg_len
    return int(np.median(predictions))


#input: event data list and event samples time.
#output: list of block time in the session: [(block i Start,block i End)]
def makeBlocksList(tags, tags_time):
    lst = []
    for i in range(len(tags)):
        if (tags[i] == start_block): lst.append([tags_times[i], ])
        if (tags[i] == end_block): lst[-1].append(tags_times[i])
    return lst


#input: signal and flag specify the data type.
#flag = 0 <=> is ecg, flag = 1 <=> is gsr
#Excecute .mat script to extract features from signal flag: 0 if ecg. 1 if gsr
def extractSeg(signal, flag):
    print('extractSeg')

    # if NETWORK == 1:
    #     input_seg = eng.mat_signals(signal, flag)  # Run .mat script to filter signals
    #     X = input_seg
    #     return input_seg
    # else:
    #     #Extract features from signal: ECG or GSR
    #     input_features = eng.mat_features(signal, flag)  # Run .mat script to extract features from signals
    #     if flag == 0: #ecg
    #         ecg_features = input_features
    #
    #     else: #flag == 1: gsr
    #         gsr_features = input_features

#Gets inputs and loaded model
#inputes : array's length = 3. inputes[0] = ecg, inputes[1] = gsr, inputes[2] = events
#loaded_model for prediction
#Predict Cl of segment and return result.
#The function excecute two threads for calculate features of gsr and ecg parallel
def predictSeg(inputs):

    # start extracting features from ECG on new thread
    t0 = threading.Timer(0, extractSeg, (inputs[0], 0), )
    t0.start()

    # start extracting features from GSR on new thread
    t1 = threading.Timer(0, extractSeg, (inputs[1], 1), )
    t1.start()

    # Wait for extracting all features: ECG and GSR
    t0.join()
    t1.join()

    #Concat the features
    features = [ecg_features + gsr_features]

    #Clear the features
    ecg_features.clear()
    gsr_features.clear()

    # predict the result of the segment
    #features = features.reshape(1, -1)  #Maybe need
    result = loaded_model.predict(features)
    print(result) #For check

    return result


#Update in RT the signals plots and calling the predictSeg when needed
#Calculate the final predict of the block
def RTupdateSignals(start_time, self):
    # getting paramater 'start_time' = time of the clock when this method had been called
    # every 0.2 sec the plot is being updating with more 200 data value
    #global GSRdata, GSRtime, GSRcurve, count1, RTdataGSR, RTtimeGSR

    global DataStream, index, ECGcurve, ECGcurveClear, GSRcurve, GSRcurveClear
    # ''condition.notify()  # signal that a new item is available
    # ''condition.release()
    RTtime = np.array([])
    RTdata = np.array([[],[]])
    blockPredictions = []

    #When we update the plots of the data streaming
    display_jump = 10

    #Counting any seg_len reading data in the same block
    counter_seg = 0

    #Flag if the current data belongs to a bloack
    reading_block = 0

    #flag end of streaming data
    finish = False
    
    #time indexes for start and end of CURRENT block
    blockStart = blockEnd = 0

    while finish == False:
        #''Flag to RTStream() to wait
        #''condition.acquire()
        #Wait to RTStream() stream data
        write.wait()

        # Flag to RTStream() to wait
        read.clear()

        # Insert to RTData array ecg and gsr values from the data stream
        RTdata[0] = np.append(RTdata[0], DataStream[0])
        RTdata[1] = np.append(RTdata[1], DataStream[1])
        RTtime = np.append(RTtime, index)
        print('data', RTdata)
        print('time', RTtime)
        if counter_seg == seg_len: #Finish reading segment
            GSRseg = RTdata[0][index-counter_seg+1:index+1]
            ECGseg = RTdata[1][index-counter_seg+1:index+1]
            GSRsegClear = clearSignal(GSRseg)
            ECGsegClear = clearSignal(ECGseg)
            #updating the proceesed data arrays for displays later
            ECG_cleard_data.extend(ECGsegClear)
            GSR_cleard_data.extend(GSRsegClear)
            #like it was before, but now with the cleared signal
            segment = [GSRsegClear,ECGsegClear]
            result = predictSeg(segment)
            '''blockPredictions.extend(result) #Add predict to list of segmentations' predicts in the block
            ##### update if only in block? #####'''
            if (reading_block == 1): blockPredictions.extend(result)
            #only if the segment is in block segment update the block predictions list
            self.lcdNumber.display(result)
            self.progressBar.setValue(result)
            #graphic display of thr segments prediction
            counter_seg = 0 #Reset counter

        if DataStream[2] == end_block: #Finish reading block data. Need to predict the block
            blockEnd = RTtime
            block_predict = int(np.median(blockPredictions))
            print('Predict is: ', block_predict) #For check
            #Clear predictions
            blockPredictions.clear()
            #Reset counter
            counter_seg = 0
            lr = pg.LinearRegionItem(values=[blockStart, blockEnd], brush=pg.intColor(index=block_predict, alpha=50), movable=False)
            cleardPlot.addItem(lr)
            label = pg.InfLineLabel(lr.lines[1], "oveload " + str(block_predict), position=0.85, rotateAxis=(1, 0),anchor=(1, 1))
            #graphic display of the block prediction

        elif DataStream[2] == start_block: #Starting reading new block's data
            blockStart = RTtime
            reading_block = 1 #Change flag

        if  reading_block == 1:
            #Update the counter_seg
            
            result
            counter_seg += 1

        #Display plot
        if index % display_jump == 0 and index > 0:
            print('data',RTdata[0])
            print('time',RTtime)
            RTdata = np.asarray(RTdata)
            #print(type(RTdata))
            print('shape',RTdata[0].shape)
            ECGcurve.setData( np.asarray(RTdata[0]))
            ECGcurveClear.setData(ECG_cleard_data)
            GSRcurveClear.setData(GSR_cleard_data)
            #GSRcurve.setData( np.asarray(RTdata[1]))
            
            QtGui.QApplication.processEvents()

        if DataStream[2] == end_data:
            finish = True
        #Time.sleep(0.2 - ((Time.time() - start_time) % 0.2))  # Time.sleep(x) hold the system for x time
        #Set Flag to RTStream() to stream new data
        read.set()
        # Update index
        index += 1
    #''Need to add return?

############## methods for GUI interfence ###################


class runBlack(QDialog):
    def __init__(self):
        super(runBlack, self).__init__()
        loadUi('runBlack.ui', self)
        self.setWindowTitle('hello world')
        self.loadGSR.clicked.connect(self.loadGSR_clicked)
        self.loadECG.clicked.connect(self.loadECG_clicked)
        self.start.clicked.connect(self.start_clicked)
        self.reset.clicked.connect(self.reset_clicked)
        self.loadEvents.clicked.connect(self.loadEvents_clicked)
        self.RTinterface.clicked.connect(self.RTinterface_clicked)
        p = self.rawDataView.addPlot()
        p2 = self.rawDataView.addPlot()
        ECGcurve = p2.plot(pen=(1, 2))
        GSRcurve = p.plot(pen=(0, 2))


    @pyqtSlot()


#Real Time mode: load data
    def RTinterface_clicked(self):
        if self.rtMode.isChecked() == True:

            #global ECGdata, data, GSRdata, ECGtime, GSRtime, time, GSRcurve, ECGcurve, p
            global ECGcurve, GSRcurve
            global RTStreamTIME, RTStreamECG, RTStreamGSR, RTStreamEVENTS
            fstream = QFileDialog.getOpenFileName(self, 'Open file', "mat files (.mat)")[0]
            m = sio.loadmat(fstream)
            RTStreamTIME = m['time_index'][0]
            RTStreamECG = m['gsr'][0]
            RTStreamGSR = m['ecg'][0]
            RTStreamEVENTS =  m['events'][0]

            print(RTStreamGSR.shape)
            print(type(RTStreamGSR))

            #print('RTStreamTIME', RTStreamTIME)
            # del m
            # data = [GSRdata, ECGdata]
            # time = [GSRtime, ECGtime]

#######################################################

#Static mode: load data
            
    def loadEvents_clicked(self): 
        
        global eventsData, eventsTimes, blocks
        blocks = []
        fevents = QFileDialog.getOpenFileName(self, 'Open file', "mat files (.mat)")[0]
        m = sio.loadmat(fevents)
        eventsTimes = m['time_index'][0]
        eventsData = m['data'][0]
        for i in range(len(eventsData)):
            if (eventsData[i] == start_block): blocks.append([eventsTimes[i], ])
            if (eventsData[i] == end_block): blocks[-1].append(eventsTimes[i])
        del m
    
    def loadECG_clicked(self):

        # self.rawDataView.clear()
        global ECGdata, data, GSRdata, ECGtime, GSRtime, time, ECGcurve, start_time
        fecg = QFileDialog.getOpenFileName(self, 'Open file', "mat files (.mat)")[0]
        m= sio.loadmat(fecg)
        ECGtime = m['time_index'][0]
        ECGdata = m['data'][0]
        del m
        '''
        ECGtime = GSRtime
        ECGdata = np.random.normal(loc=6.5, scale=0.3, size=len(GSRtime))'''
        data = [GSRdata, ECGdata]
        time = [GSRtime, ECGtime]
        ECGcurve.setData(ECGtime, ECGdata)


    def loadGSR_clicked(self):

        global ECGdata, data, GSRdata, ECGtime, GSRtime, time, GSRcurve, ECGcurve, p, p2
        fgsr = QFileDialog.getOpenFileName(self, 'Open file', "mat files (.mat)")[0]
        m = sio.loadmat(fgsr)
        GSRtime = m['time_index'][0]
        GSRdata = m['data'][0]
        del m
        data = [GSRdata, ECGdata]
        time = [GSRtime, ECGtime]
        p.setXRange(-1.06, 878.15)
        p.setYRange(4.82, 9.3)
        GSRcurve.setData(GSRtime, GSRdata)
#######################################################


    #START
    def start_clicked(self):

        global data, ECGdata, GSRdata, ECGcurveClear,  GSRcurveClear, cleardPlot 

        #RT variables for simulating streaming
        global RTStreamTIME, RTStreamECG, RTStreamGSR, RTStreamEVENTS, ECGcurveClear, GSRcurveClear

        # Real Time mode
        if self.rtMode.isChecked() == True:
                # preapre procceced data curves
                cleardPlot = self.modDataView.addPlot()
                ECGcurveClear = cleardPlot.plot(pen=(1, 2))
                GSRcurveClear = cleardPlot.plot(pen=(0, 2))
            # The data that will be used to simulate streaming is loaded
            #if ((RTStreamTIME != []) and (RTStreamECG != []) and (RTStreamGSR != []) and (RTStreamEVENTS != [])):

                #Thread for input streaming
                start_time = Time.time()
                thread_stream = threading.Timer(0, RTStream ).start()

                # Thread for update signals and calculations
                start_time = Time.time()
                thread_predict = threading.Timer(0, RTupdateSignals, (start_time,self,)).start()

                #Waiting for processes to finish
                thread_stream.join()
                thread_predict.join()

        
        else:
            # system is on static test mode #
            if ((ECGdata!= []) and (GSRdata!= [])):
            # both GSR & ECG files has been loaded
                data = [GSRdata, ECGdata]
                p1 = self.modDataView.addPlot(x=time[1], y=data[1], pen=(1, 2))
                p2 = pg.PlotCurveItem(x=time[0], y=data[0], pen=(0, 2))
                p1.addItem(p2)
                for i in range(len(blocks)):
                #     # add the network workeloads predictons of each data block to the graphic view
                    blockStart = blocks[i][0]
                    blockEnd = blocks[i][1]
                    level = predictBlock(GSRdata[blocks[i][0]:blocks[i][1]], ECGdata[blocks[i][0]:blocks[i][1]])
                    lr = pg.LinearRegionItem(values=[blockStart, blockEnd], brush=pg.intColor(index=level, alpha=50), movable=False)
                    p1.addItem(lr)
                    label = pg.InfLineLabel(lr.lines[1], "oveload " + str(level), position=0.85, rotateAxis=(1, 0),anchor=(1, 1))
    
            else:
                # either GSR or ECG files has not been loaded
                 print('error')


    def reset_clicked(self):
        try:
            subprocess.Popen([sys.executable, FILEPATH])
        except OSError as exception:
            print('ERROR: could not restart aplication:')
            print('  %s' % str(exception))
        else:
            QApplication.quit()


#Start App
app = QApplication(sys.argv)
widget = runBlack()
widget.show()
sys.exit(app.exec_())


