
# coding: utf-8

from chainer import Chain
from chainer import Variable, optimizers
import chainer.functions as F
import chainer.links as L

import matplotlib.pyplot as plt
import numpy as np


class Generator(Chain):
    '''ランダムなベクトルから画像を生成する画像作成機
    '''

    def __init__(self, z_dim):
        super(Generator, self).__init__(
            l1=L.Linear(z_dim, 3 * 3 * 512),

            dc1=L.Deconvolution2D(512, 256, 2, stride=2, pad=1,),
            dc2=L.Deconvolution2D(256, 128, 2, stride=2, pad=1,),
            dc3=L.Deconvolution2D(128, 64, 2, stride=2, pad=1,),
            dc4=L.Deconvolution2D(64, 1, 3, stride=3, pad=1),

            # Convolution, Deconvolutionともに値は画像の大きさに合わせて変化させる
            # 必要がある。

            # 今回はMNISTをターゲットにするので、元の大きさは28×28
            # 元の512チャンネルから1チャンネル（MNISTは白黒なためチャンネルが無い）に変換するまでに
            # 3→4→5→10→28となるようにstride,pad,windowの大きさを選んでいる

            #             bn0 = L.BatchNormalization(6*6*512),
            bn1=L.BatchNormalization(512),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(128),
            bn4=L.BatchNormalization(64),
        )
        self.z_dim = z_dim

    def __call__(self, z, test=False):
        h = self.l1(z)
        # 512チャンネルをもつ、3*3のベクトルに変換する
        h = F.reshape(h, (z.data.shape[0], 512, 3, 3))

        h = F.relu(self.bn1(h, test=test))
        h = F.relu(self.bn2(self.dc1(h), test=test))
        h = F.relu(self.bn3(self.dc2(h), test=test))
        h = F.relu(self.bn4(self.dc3(h), test=test))
        x = self.dc4(h)
        return x


class Discriminator(Chain):
    '''入力された画像が偽物かどうかを判定する判別器
    '''

    def __init__(self,):
        super(Discriminator, self).__init__(
            c1=L.Convolution2D(1, 64, 3, stride=3, pad=1, ),
            c2=L.Convolution2D(64, 128, 2, stride=2, pad=1,),
            c3=L.Convolution2D(128, 256, 2, stride=2, pad=1,),
            c4=L.Convolution2D(256, 512, 2, stride=2, pad=1,),

            l1=L.Linear(3 * 3 * 512, 2),

            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(512),
        )

    def __call__(self, x, test=False):
        '''判別関数．
        return 二次元のVariable
        '''
        h = F.relu(self.c1(x))
        h = F.relu(self.bn1(self.c2(h), test=test))
        h = F.relu(self.bn2(self.c3(h), test=test))
        h = F.relu(self.bn3(self.c4(h), test=test))
        y = self.l1(h)
        return y
