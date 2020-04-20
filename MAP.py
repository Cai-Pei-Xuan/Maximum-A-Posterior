# -*-coding:UTF-8 -*-
# Maximum A Posterior
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det

# 載入資料，第一筆是label，其他是feature。
def loadData(file_name, label_list, feature_list):
    fp = open(file_name, "r", encoding="utf-8")
    line = fp.readline()
    while line:
        feature = []
        line_list = line.replace("\n", "").split(",")
        for index, element in enumerate(line_list):
            if index == 0:
                label_list.append(element)
            else:
                feature.append(float(element))
        feature_list.append(feature)
        line = fp.readline()
    fp.close()


# 將train_data中各個相同label的資料放在一起。
def getEachLabelDict(EachLabelDict, label_list, feature_list):
    for i, element in enumerate(label_list):
        if element not in EachLabelDict:
            EachLabelDict[element] = {}
            EachLabelDict[element]["count"] = 1
            EachLabelDict[element]["feature_list"] = []
            EachLabelDict[element]["feature_list"].append(feature_list[i])
        else:
            EachLabelDict[element]["count"] += 1
            EachLabelDict[element]["feature_list"].append(feature_list[i])


# 計算每個label的參數(probability, mean, covariance matrix, inverse of covariance matrix, determinant of covariance matrix)，並將其存成dict，避免日後重複計算。
def calculateEachLabelParameters(EachLabelDict, EachLabelParametersDict):

    AllDataNumber = 0
    for key in EachLabelDict.keys():
        AllDataNumber += EachLabelDict[key]["count"]

    for key in EachLabelDict.keys():
        EachLabelParametersDict[key] = {}
        # calculate probability
        EachLabelParametersDict[key]["probability"] = (
            EachLabelDict[key]["count"] / AllDataNumber
        )
        # calculate mean
        EachLabelParametersDict[key]["mean"] = np.mean(
            EachLabelDict[key]["feature_list"], axis=0
        )
        # calculate covariance matrix
        EachLabelParametersDict[key]["cov"] = np.cov(
            np.transpose(EachLabelDict[key]["feature_list"])
        )
        # calculate inverse of covariance matrix
        EachLabelParametersDict[key]["inv_of_cov"] = inv(
            EachLabelParametersDict[key]["cov"]
        )
        # calculate determinant of covariance matrix
        EachLabelParametersDict[key]["det_of_cov"] = det(
            EachLabelParametersDict[key]["cov"]
        )


# 在給定的特徵時，計算每個label可以得到的機率。
def calculateEachLabelProbability(x, LabelParametersDict):
    log_p = np.log(LabelParametersDict["probability"])
    var = x - LabelParametersDict["mean"]
    matrices_dot = np.transpose(var).dot(LabelParametersDict["inv_of_cov"]).dot(var)
    log_det_of_cov = np.log(LabelParametersDict["det_of_cov"])

    return -log_p + (matrices_dot / 2) + (log_det_of_cov / 2)


# 計算測試資料的正確性。
def predict(label_test, feature_test, EachLabelParametersDict):
    correct = 0
    for i, element in enumerate(feature_test):
        predict_label = ""
        min_probability = np.finfo(np.float).max  # numpy可以得到的最大數值
        for key in EachLabelParametersDict:
            Probability = calculateEachLabelProbability(
                element, EachLabelParametersDict[key]
            )
            if Probability < min_probability:
                predict_label = key
                min_probability = Probability
        if predict_label == label_test[i]:
            correct += 1

    return correct / len(feature_test)


if __name__ == "__main__":
    file_name = "wine.data"
    label_list = []
    feature_list = []
    EachLabelDict = {}
    EachLabelParametersDict = {}
    test_size = 0.5
    random_state = 1

    # 載入資料
    loadData(file_name, label_list, feature_list)
    # 使用sklearn分割資料，用法參考網站:https://blog.csdn.net/CherDW/article/details/54881167
    label_train, label_test, feature_train, feature_test = train_test_split(
        label_list, feature_list, test_size=test_size, random_state=random_state
    )
    # 將train_data中各個相同label的資料放在一起
    getEachLabelDict(EachLabelDict, np.array(label_train), np.array(feature_train))
    # 在給定的特徵時，計算每個label可以得到的機率
    calculateEachLabelParameters(EachLabelDict, EachLabelParametersDict)
    # 計算測試資料的正確性
    predict_accuracy = predict(label_test, feature_test, EachLabelParametersDict)

    print("data set name:" + file_name)
    print("predict accuracy:" + str(predict_accuracy * 100) + "%")
