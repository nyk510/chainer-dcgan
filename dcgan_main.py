from dcgan.trainer import Trainer
from dcgan.dcgan import Generator,Discriminator

from sklearn.datasets import fetch_mldata
import numpy as np
import pandas as pd

import pickle


if __name__ == '__main__':

    gen = Generator(100)
    dis = Discriminator()

    data = fetch_mldata('MNIST original')
    X = data['data']
    n_train = X.shape[0]
    X = np.array(X, dtype=np.float32)
    X /= 255.
    X = X.reshape(n_train,1, 28,28)

    trainer = Trainer(gen,dis)

    trainer.fit(X,batchsize=1000,epochs=1000)

    df_loss = pd.DataFrame(trainer.loss)
    df_loss.to_csv('loss.csv')


    gen.to_cpu()
    dis.to_cpu()

    with open('generator.model','wb') as w:
        pickle.dump(gen,w)

    with open('discriminator.model','wb') as w:
        pickle.dump(dis,w)
