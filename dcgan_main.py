from dcgan.trainer import Trainer
from dcgan.dcgan import Generator,Discriminator

from sklearn.datasets import fetch_mldata
import numpy as np


if __name__ == '__main__':
    
    gen = Generator(100)
    dis = Discriminator()

    data = fetch_mldata('MNIST original')
    X = data['data']
    n_train = X.shape[0]
    X = np.array(X, dtype=np.float32)
    X /= 256.
    X = X.reshape(n_train,1, 28,28)

    x_selected = np.random.permutation(X)[:20000]

    trainer = Trainer(gen,dis)

    trainer.fit(x_selected,batchsize=1000)
