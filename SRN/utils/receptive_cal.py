import torch
import numpy as np
import torch.nn as nn
import functools


import math

def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k
    pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out


def printLayer(layer, layer_name):
    print(layer_name + ":")
    print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (
    layer[0], layer[1], layer[2], layer[3]))


def weights_matrix(patch, img, n_f_h, n_f_w, jump, rf, start):
    # B, C, H, W = patch.shape
    wm = np.zeros([img.shape[0], 1, img.shape[2], img.shape[3]])
    for i in range(n_f_h):
        for j in range(n_f_w):
            val = patch[:, :, i, j]
            hf, ht = int(max(0, start + i*jump - rf//2)), int(start + i*jump + rf - rf//2)
            wf, wt = int(max(0, start + j*jump - rf//2)), int(start + j*jump + rf - rf//2)
            wm[:, :, hf:ht, wf:wt] += val
    return wm


def receptive_cal(imsize):
    convnet = [[4, 2, 1], [4, 2, 1], [4, 1, 1], [4, 1, 1]]
    layer_names = ['conv1', 'conv2', 'conv3', 'conv4']
    currentLayer = [imsize, 1, 1, 0.5]
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
    return currentLayer

def getWeights(patch, img, currentLayer_h, currentLayer_w):
    n_f_h, jump, rf, start = currentLayer_h[0], currentLayer_h[1], currentLayer_h[2], currentLayer_h[3]
    n_f_w, jump, rf, start = currentLayer_w[0], currentLayer_w[1], currentLayer_w[2], currentLayer_w[3]
    s = weights_matrix(patch, img, n_f_h, n_f_w, jump, rf, start)
    count = weights_matrix(np.ones_like(patch), img, n_f_h, n_f_w, jump, rf, start)
    return s / count


layerInfos = []
if __name__ == '__main__':
    patch = np.ones([1,1,19,12])*2
    img = np.ones([1,1,86,56])*2
    currentLayer_h, currentLayer_w = receptive_cal(img.shape[2]), receptive_cal(img.shape[3])
    res1 = getWeights(patch, img, currentLayer_h, currentLayer_w)
    print(res1.shape)



    # imsize = 344
    # currentLayer = [imsize, 1, 1, 0.5]
    # # first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
    # print("-------Net summary------")
    # # currentLayer = [imsize, 1, 1, 0.5]
    # convnet = [[4, 2, 1], [4, 2, 1], [4, 1, 1], [4, 1, 1]]
    # layer_names = ['conv1', 'conv2', 'conv3', 'conv4']
    # printLayer(currentLayer, "input image")
    # for i in range(len(convnet)):
    #     currentLayer = outFromIn(convnet[i], currentLayer)
    #     layerInfos.append(currentLayer)
    #     printLayer(currentLayer, layer_names[i])
    # print("------------------------")
    # n_f, jump, rf, start = currentLayer[0], currentLayer[1], currentLayer[2], currentLayer[3]
    # ipt = np.ones([1,1,10,10])
    # img = np.ones([1,1,48,48])
    # weights_matrix(ipt, img, n_f, jump, rf, start)



    # print(start, receptive)
    # layer_name = input("Layer name where the feature in: ")
    # layer_idx = layer_names.index(layer_name)
    # idx_x = int(input("index of the feature in x dimension (from 0)"))
    # idx_y = int(input("index of the feature in y dimension (from 0)"))
    #
    # n = layerInfos[layer_idx][0]
    # j = layerInfos[layer_idx][1]
    # r = layerInfos[layer_idx][2]
    # start = layerInfos[layer_idx][3]
    # assert (idx_x < n)
    # assert (idx_y < n)
    #
    # print("receptive field: (%s, %s)" % (r, r))
    # print("center: (%s, %s)" % (start + idx_x * j, start + idx_y * j))