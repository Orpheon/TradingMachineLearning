import datetime
import json
import os

def interpolate(a, b, t):
    return a + t*(b-a)

def readcsv(filename, skipfirst=0, onlyfirstvalue=False):
    data = []
    with open(filename+".csv", "r") as f:
        for line in f.readlines()[skipfirst:]:
            entries = line.split(",")
            if len(entries) == 2 or onlyfirstvalue:
                data.append([int(entries[0]), float(entries[1])])
            else:
                data.append([int(entries[0]), [float(x) for x in entries[1:]]])
    return data

def splitdata(data, proportion):
    if not (0 <= proportion <= 1):
        raise ValueError
    return (data[:int(len(data)*proportion)], data[int(len(data)*proportion):])

def resampledata(rawdata, period):
    idx = 0
    data = [rawdata[0]]
    starttime, timelength = rawdata[0][0], rawdata[-1][0] - rawdata[0][0]
    try:
        while True:
            t = data[-1][0] + period
            if (t-starttime) % (int(timelength/100)) == 0:
                print("Resampling: {}%".format(int(100*(t-starttime)/timelength)))
            while rawdata[idx+1][0] <= t:
                idx += 1
            v = functions.interpolate(rawdata[idx][1], rawdata[idx+1][1], (t-rawdata[idx][0]) / (rawdata[idx+1][0]-rawdata[idx][0]))
            data.append([t, v])
    except IndexError:
        pass
    return data

def dump(name, data):
    with open("{0}.csv".format(name), "w") as f:
        text = ""
        for d in data:
            text += ",".join([str(x) for x in d]) + "\n"
        f.write(text)

def firstload(csvname):
    fulldata = readcsv(csvname, skipfirst=100, onlyfirstvalue=True)
    traindata, testdata = splitdata(resampledata(fulldata, 5), 6/7)
    dump(csvname + "_training", traindata)
    dump(csvname + "_test", testdata)
    return traindata, testdata

def load(csvname):
    if os.path.exists(csvname + "_training.csv") and os.path.exists(csvname + "_test.csv"):
        traindata = readcsv(csvname + "_training")
        testdata = readcsv(csvname + "_test")
    else:
        traindata, testdata = firstload(csvname)
    traindata = [x[1] for x in traindata]
    testdata = [x[1] for x in testdata]
    return traindata, testdata
