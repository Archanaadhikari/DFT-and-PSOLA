import sys
import numpy
import pydub
import matplotlib
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import misc_util
import sounddevice

class AudioProcessor:
    
    def __init__(self, audio_filename):
        seg = pydub.AudioSegment.from_file(audio_filename)
        self.samples = numpy.array(seg.get_array_of_samples()).reshape((-1, seg.channels))
        self.sampleRate = seg.frame_rate
        self.nWindowSize = 4096
        self.nFrames = int(self.samples.shape[0] / self.nWindowSize)
    
    def calcFrame(self, i):
        frameSamples = self.samples[i * self.nWindowSize : (i + 1) * self.nWindowSize, 0]
        t0 = i * self.nWindowSize / self.sampleRate
        noteList = numpy.arange(24, 132, 0.25)
        timeList = numpy.linspace(t0, t0 + self.nWindowSize / self.sampleRate, self.nWindowSize)
        win = numpy.blackman(self.nWindowSize)
        f = misc_util.noteToFreq(noteList).reshape((-1, 1))
        t = timeList.reshape((-1, 1))
        ph = -2 * numpy.pi * f * t.T
        sh = numpy.exp(1j * ph)
        sw = (frameSamples * win * sh).sum(axis=1)
        octaveTicks = noteList[noteList % 12 == 0]
        return {
            "t0": t0,
            "ts": timeList,
            "ns": noteList,
            "lw": 10 * numpy.log10(numpy.maximum(numpy.absolute(sw), 1e-10)),
            "octaves": octaveTicks,
            "samples": frameSamples,
        }

class MainPlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent, audioProcessor, *args, **kwargs):
        self.fig = Figure(dpi=300, constrained_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.MainWindow = parent
        self.ap = audioProcessor
        super(MainPlotCanvas, self).__init__(self.fig)        
        self.frameIdx = 0
        self.initData()
        self.calcData()        
        self.mpl_connect("motion_notify_event", self.mouseMove)
        self.mpl_connect("button_release_event", self.mouseClick)     
    def mouseMove(self, event):
        if event.xdata is not None:
            self.harmonicsVisible = True;
            self.n0 = event.xdata;
            self.calcUI();
            self.updatePlot();
        elif self.harmonicsVisible:
            self.harmonicsVisible = False;
            self.updatePlot();            
    def mouseClick(self,event):
        if event.xdata is not None:
            probeSamples = misc_util.generateNote(event.xdata);
            sounddevice.play(probeSamples, 44100);            
    def playSound(self):
        sounddevice.play(self.frameResult["samples"], self.ap.sampleRate);    
    def initData(self):
        nHarmonics = 12
        self.data = {}        
        self.data["harmonics"] = {
            "x": numpy.zeros((4 * nHarmonics,)),
            "y": numpy.zeros((4 * nHarmonics,))
        }        
        self.n0 = 0
        self.harmonicsVisible = False        
    def calcData(self):
        self.frameResult = self.ap.calcFrame(self.frameIdx)        
        self.calcUI()        
    def calcUI(self):
        lw = self.frameResult["lw"]        
        ytop = lw.max() * 1.1
        ymid = lw.min()
        ybot = (ytop - ymid) * -0.1 + ymid        
        self.data["harmonics"]["y"][0::4] = ymid
        self.data["harmonics"]["y"][1::4] = ytop
        self.data["harmonics"]["y"][2::4] = ybot
        self.data["harmonics"]["y"][3::4] = ymid        
        f0 = misc_util.noteToFreq(self.n0)        
        for i in range(self.data["harmonics"]["x"].shape[0] // 4):
            nMult = i + 1
            f = nMult * f0
            n = misc_util.freqToNote(f)            
            self.data["harmonics"]["x"][4 * i : 4 * i + 4] = n            
    def makePlot(self):
        self.refs = {}      
        ax = self.ax
        ax.cla()
        ax.set_title("Spectrum, t = {:.2f}".format(self.frameResult["t0"]))
        ax.set_xlabel("MIDI Note #")
        ax.set_ylabel("Log Intensity (dB)")
        ax.set_xticks(self.frameResult["octaves"])
        ax.set_xlim(self.frameResult["ns"][0], self.frameResult["ns"][-1])  
        self.refs["harmonics"], = self.ax.plot(
            self.data["harmonics"]["x"],
            self.data["harmonics"]["y"],
            "r",
            label="harmonics"
        )     
        ax.plot(self.frameResult["ns"], self.frameResult["lw"], label="Samples")
        self.updatePlot()
    def updatePlot(self):
        self.refs["harmonics"].set_xdata(self.data["harmonics"]["x"])
        self.refs["harmonics"].set_ydata(self.data["harmonics"]["y"])
        if self.harmonicsVisible:
            self.refs["harmonics"].set_label(
                "Harmonics, n0 = {:.1f}".format(self.n0)
            )
        else:
            self.refs["harmonics"].set_label(None)
        self.refs["harmonics"].set_visible(self.harmonicsVisible)
        self.ax.legend(loc="best")
        self.draw()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, audioProcessor, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.ap = audioProcessor
        self.sc = MainPlotCanvas(self, self.ap)
        self.sc.makePlot()
        self.sc.playSound();
        self.setCentralWidget(self.sc)
        self.show()
    def seekFrames(self, nSeek):
        frameIdx = self.sc.frameIdx;
        frameIdx+= nSeek;
        while frameIdx < 0:
            frameIdx += self.ap.nFrames;    
        while frameIdx >= self.ap.nFrames:
            frameIdx -= self.ap.nFrames;            
        self.sc.frameIdx = frameIdx;        
        self.sc.calcData();
        self.sc.makePlot();
        self.sc.playSound();        
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Right:
            self.seekFrames(+1);
        elif event.key() == QtCore.Qt.Key_Left:
            self.seekFrames(-1);
        elif event.key() == QtCore.Qt.Key_Down:
            self.seekFrames(+10);
        elif event.key() == QtCore.Qt.Key_Up:
            self.seekFrames(-10);

def noteToFreq(note):
    return 440 * 2 ** ((note - 69) / 12)

def freqToNote(freq):
    return 12 * numpy.log2(freq / 440) + 69

some_harmonics = numpy.array([1, 1, 0.1, 0.1, 0.1])

def generateNote(noteNumber, sampleRate=44100, nSamples=44100, harmonics=some_harmonics):
    n0 = int(numpy.round(noteNumber))
    samples = numpy.zeros((nSamples,), dtype=numpy.int16)
    times = numpy.linspace(0, samples.size / sampleRate, samples.size)
    scaleFactor = 16383 / harmonics.sum()
    for i in range(len(harmonics)):
        nMul = i + 1
        freq = nMul * noteToFreq(n0)
        phase = 2 * numpy.pi * freq * times
        wave = numpy.exp(1j * phase) * harmonics[i]
        samples[:] += (scaleFactor * wave.real).astype(numpy.int16)
    return samples

def main(args):
    audio_filename = "E:/7thSemProject/NewFinal/riffify/test_audio.mp3"
    if len(args) > 1:
        audio_filename = args[1]
    audioProcessor = AudioProcessor(audio_filename)
    app = QtWidgets.QApplication(args)
    w = MainWindow(audioProcessor)
    app.exec_()

if __name__ == "__main__":
    main(sys.argv)