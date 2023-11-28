import time
import random
import numpy as np
import torch
import pandas as pd
from numpy import inf
from sklearn import preprocessing
from sklearn.model_selection import KFold
import time

class cover:
    cnt = 1
    def __init__(self, w, r, label,featureNum):
        self.w = w
        self.r = r
        self.label = label
        self.featureNum =featureNum
        self.coverd = torch.cat((w, label.reshape(1)), dim=0)

    def addPoint(self, coverd, label):
        t = torch.cat((coverd, label.reshape(1)), dim=0)
        self.coverd = torch.cat((self.coverd, t), dim=0)
        self.cnt += 1

    def show(self):
        print("W is ",self.w, "\nR is ",self.r,"\nLabel is ", self.label.item(),"\n","Points:", self.coverd.reshape(-1,self.featureNum+2),"\n\n",self.cnt," points in total\n\n")

    def ifInCover(self, point):
        if point.dot(self.w) >= self.r:
            return True
        else:
            return False
# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


file_path = "dataset/iris/iris.data"              # get data from the set
data = pd.read_csv(file_path, header=None)
feature = data.iloc[:, :-1].values                    # get the features
label = data.iloc[:, -1].values                     # get the labels



# #
# file_path = "dataset/liver+disorders/bupa.data"              # get data from the set
# data = pd.read_csv(file_path, header=None)
# feature = data.iloc[:, :-1].values                    # get the features
# label = data.iloc[:, -1].values                     # get the labels

#
# file_path = "dataset/soybean+large/soybean-small.data"              # get data from the set
# data = pd.read_csv(file_path, header=None)
# feature = data.iloc[:, :-1].values                    # get the features
# label = data.iloc[:, -1].values                     # get the labels


# file_path = "dataset/waveform+database+generator+version+1/waveform-+noise.data/waveform-+noise.data"
# data = pd.read_csv(file_path, header=None)
# feature = data.iloc[:, :-1].values                    # get the features
# label = data.iloc[:,-1].values


# file_path = "dataset/waveform+database+generator+version+1/waveform.data/waveform.data"
# data = pd.read_csv(file_path, header=None)
# feature = data.iloc[:, :-1].values                    # get the features
# label = data.iloc[:,-1].values

# file_path = "dataset/zoo/zoo.data"
# data = pd.read_csv(file_path, header=None)
# feature = data.iloc[:, 1:].values                    # get the features
# label = data.iloc[:,-1].values

# file_path = "dataset/wine/wine.data"
# data = pd.read_csv(file_path, header=None)
# feature = data.iloc[:, 1:].values                    # get the features
# label = data.iloc[:,0].values


enc = preprocessing.LabelEncoder()
label = enc.fit_transform(label)                        # trans y from string to int with LabelEncoder

feature = torch.tensor(feature, dtype=torch.float32, device=device)
label = torch.tensor(label, dtype=torch.long, device=device)
# print(label)
# print(feature.shape[1])
featureNum = feature.shape[1]
# Work begin

# Normalization
featureMin = feature.min(dim=0)[0]
featureMax = feature.max(dim=0)[0] #[0] for get value instead of index
feature = (feature - featureMin) / (featureMax - featureMin)


# Dimensionality Expansion
featureLenSquare = feature.square().sum(dim=1) # vector len square
R = featureLenSquare.sqrt().max()

tempTensor = (R*R - featureLenSquare).sqrt().reshape(-1, 1) # -1 could let that dimension be calculated automatically
# tempTensor = torch.where(torch.isnan(tempTensor), torch.full_like(tempTensor, 0), tempTensor)# deal with nan
feature = torch.cat((feature, tempTensor), dim=1)
feature= torch.where(torch.isnan(feature), torch.full_like(feature, 0), feature)
# print(feature)
# print(feature)
# print(feature)

# Construc hidden neuron

# K-Fold Cross Validation

# 数据统计
coverNum = 0
testPointCanBeDetect = 0
detectedPointRight = 0
unableDetectPoint = 0
unableDetectPointRight = 0
#
a = time.time()
splitsNum = 10
kf = KFold(n_splits=splitsNum, shuffle=True)
for trainIndex, testIndex in kf.split(feature):# 循环十次
    # print("\nTRAIN:", trainIndex, "\nTEST:", testIndex)
    featureTrain, featureTest = feature[trainIndex], feature[testIndex]
    labelTrain, labelTest = label[trainIndex], label[testIndex]
    visited = torch.zeros(labelTrain.shape[0], dtype=torch.bool, device=device)
    #print(visited)
    # print(featureTrain, featureTest, labelTrain, labelTest)
    # W = torch.zeros((0, feature.shape[-1]), dtype= ,device=device)
    # find the nearest different point and the farthest same point
    # the distance is featureTrain[i] dot featureTrain[j]
    # 找到一个点 以这个点来计算 找最近的不一样的点 这个点的距离定义为D 再找<D的一样的点 将里面的所有点都标记为visited
    Cover = []
    idx = 0
    accurence = 0
    voteNum = 5#投票轮数
    # Train Start
    for i in range(voteNum):
        indices = list(range(len(featureTrain)))
        shuffledFeatureTrain = featureTrain[indices]
        shuffledLabelTrain = labelTrain[indices]
        for current in range(shuffledLabelTrain.shape[0]):#idx

            nearestDifferentPointDistance = -inf  # d1
            farthestSamePointDistance = inf  # d2
            if visited[current]:# if visited, pass
                continue
            visited[current] = True# mark visited

            for compared in range(shuffledLabelTrain.shape[0]):# 找d1
                if visited[compared]:# if visited, pass
                    continue
                if shuffledLabelTrain[current] != shuffledLabelTrain[compared]:  # if different label, calculate the distance
                    #visited[compared] = True  # mark visited
                    differentDistance = shuffledFeatureTrain[current].dot(shuffledFeatureTrain[compared])
                    if differentDistance > nearestDifferentPointDistance:  # 最近距离就是找最大的内积
                        nearestDifferentPointDistance = differentDistance
                        #print("d1:",nearestDifferentPointDistance)
            # print("Final d1 of {}".format(current),nearestDifferentPointDistance)
            for compared in range(shuffledLabelTrain.shape[0]):#find d2
                if visited[compared]:  # if visited, pass
                     continue
                if shuffledLabelTrain[current] == shuffledLabelTrain[compared]:# if same label, pass
                    sameDistance = shuffledFeatureTrain[current].dot(shuffledFeatureTrain[compared])
                    if farthestSamePointDistance > sameDistance > nearestDifferentPointDistance:#最小距离就是找最大的内积
                        farthestSamePointDistance = sameDistance
                        #print("d2:",farthestSamePointDistance)
            # print("Final d2 of {}".format(current),farthestSamePointDistance)
            # print(labelTrain.shape)
            # print(labelTrain)
            # print(shuffledFeatureTrain.shape)
            # print(shuffledFeatureTrain)
            # print(labelTrain[current].reshape(1).shape)
            # print(shuffledFeatureTrain[current].shape)
            # 最小半径法取最远同类点距离farthestSamePointDistance
            # 最大半径法取最近异类点距离nearestDifferentPointDistance
            # 折中半径法取上述两类距离平均值

            # 最小半径法
            theta = farthestSamePointDistance
            if farthestSamePointDistance == inf:
                theta = shuffledFeatureTrain[current].dot(shuffledFeatureTrain[current])
            # 现在d1 d2 都有了 主要用d2 d2是半径

            # # 最大半径法
            # theta = nearestDifferentPointDistance
            # if nearestDifferentPointDistance == -inf:
            #     theta = shuffledFeatureTrain[current].dot(shuffledFeatureTrain[current])
            #
            # # 折中半径法
            # if nearestDifferentPointDistance == -inf:
            #     nearestDifferentPointDistance = shuffledFeatureTrain[current].dot(shuffledFeatureTrain[current])
            # if farthestSamePointDistance == inf:
            #     farthestSamePointDistance = shuffledFeatureTrain[current].dot(shuffledFeatureTrain[current])
            # theta = (nearestDifferentPointDistance + farthestSamePointDistance) / 2


            Cover.append(cover(shuffledFeatureTrain[current], theta, shuffledLabelTrain[current],featureNum=featureNum))
            # 遍历一遍 把所有在d2内的点都标记为visited 并将覆盖生成
            for compared in range(shuffledLabelTrain.shape[0]):
                if visited[compared]:
                    continue
                Distance = shuffledFeatureTrain[current].dot(shuffledFeatureTrain[compared])
                if Distance >= theta:
                    visited[compared] = True
                    Cover[idx].addPoint(shuffledFeatureTrain[compared], shuffledLabelTrain[compared]) # 把该currentPoint形成的覆盖里的点加进去
            # print(len(Cover))
            #Cover[idx].show()

            idx += 1
        coverNum += len(Cover)
# Train End

# Test Start
        labelDetect = np.zeros(max(label)+1)
        print(labelDetect)
        #labelDetect = None
        for current in range(labelTest.shape[0]):
            numInHowManyCover = 0
            pointCouldBeInclude = False
            for compared in range(idx):
                if Cover[compared].ifInCover(featureTest[current]):
                    # 如果在覆盖里
                    # accurence += 1
                    pointCouldBeInclude = True
                    numInHowManyCover += 1
            minDistance = -inf
            minDistanceIdx = None
            if pointCouldBeInclude:
                testPointCanBeDetect += 1
                if numInHowManyCover == 1:
                    for compared in range(idx):
                        if Cover[compared].ifInCover(featureTest[current]):
                            labelDetect = Cover[compared].label
                elif numInHowManyCover > 1:  # 延迟决策
                    for compared in range(idx):
                        Distance = featureTest[current].dot(Cover[compared].w)
                        if Distance > minDistance:  # 找最近距离 最大内积 距中心最近
                            minDistanceIdx = compared
                            minDistance = Distance
                    labelDetect = Cover[minDistanceIdx].label
                # print(labelDetect,labelTest[current])
                if labelDetect == labelTest[current]:
                    detectedPointRight += 1
            if not pointCouldBeInclude: # 不可识别
                unableDetectPoint += 1
                for compared in range(idx):
                    Distance = featureTest[current].dot(Cover[compared].w)
                    if Distance > minDistance:  # 找最近距离 最大内积 距中心最近
                        minDistanceIdx = compared
                        minDistance = Distance
                labelDetect = Cover[minDistanceIdx].label
                if labelDetect == labelTest[current]:
                    unableDetectPointRight += 1
print("覆盖数:",coverNum/splitsNum,"\n可识别的样本数:", testPointCanBeDetect/splitsNum, "\n可识别样本的正确数:", detectedPointRight/splitsNum,
        "\n可识别的样本的正确率:", (detectedPointRight / testPointCanBeDetect) * 100, "%\n不可识别的样本数:", unableDetectPoint/splitsNum,
        "\n不可识别样本的正确数:", unableDetectPointRight/splitsNum, "\n不可识别样本的正确率:", (unableDetectPointRight / unableDetectPoint) * 100,
        "%\n总正确率:", (detectedPointRight + unableDetectPointRight) / (testPointCanBeDetect + unableDetectPoint) * 100,
        "%")
b=time.time()
print("Time:",b-a)
# Test End



