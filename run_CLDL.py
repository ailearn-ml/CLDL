import os
import argparse
import gpytorch
from model import CLDL
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import minmax_scale
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='sample_data')
parser.add_argument('--outDim', type=int, default=30)
parser.add_argument('--max_iter', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--kernel_type', type=str, default='matern',
                    choices=['poly', 'rbf', 'linear', 'matern', 'cosine', 'rq'])
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--neighbor_num', type=int, default=30)

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.dataset
    outDim = args.outDim
    max_iter = args.max_iter
    lr = args.lr
    kernel_type = args.kernel_type
    device = args.device
    neighbor_num = args.neighbor_num

    os.makedirs(os.path.join('save', 'encoding'), exist_ok=True)
    os.makedirs(os.path.join('save', 'result'), exist_ok=True)

    data = loadmat(os.path.join('data', f'{dataset}.mat'))
    x_train = data['train_feature']
    x_test = data['test_feature']
    y_train = data['train_labels']
    y_test = data['test_labels']

    print(
        f'dataset: {dataset}, outDim: {outDim}, kernel_type:{kernel_type}, max_iter: {max_iter}, lr: {lr}, neighbor_num: {neighbor_num}')

    x_train = minmax_scale(x_train)
    x_test = minmax_scale(x_test)

    if not os.path.exists(os.path.join('save', 'result',
                                       f'{dataset}_{outDim}_{kernel_type}_{max_iter}_{lr}_Z_test.npy')):
        Z = loadmat(os.path.join('save', 'encoding', f'{dataset}_{outDim}.mat'))['encoding']
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Z.shape[1]).to(device)
        model = CLDL(x_train=x_train, encoding_train=Z,
                     num_tasks=Z.shape[1], max_iter=max_iter,
                     lr=lr, kernel_type=kernel_type, device=device)
        model.fit(x_train, Z)
        Z_train = model.predict(x_train)
        Z_test = model.predict(x_test)
        np.save(os.path.join('save', 'result',
                             f'{dataset}_{outDim}_{kernel_type}_{max_iter}_{lr}_Z_train.npy'),
                Z_train)
        np.save(os.path.join('save', 'result',
                             f'{dataset}_{outDim}_{kernel_type}_{max_iter}_{lr}_Z_test.npy'),
                Z_test)
    else:
        Z_train = np.load(os.path.join('save', 'result',
                                       f'{dataset}_{outDim}_{kernel_type}_{max_iter}_{lr}_Z_train.npy'))
        Z_test = np.load(os.path.join('save', 'result',
                                      f'{dataset}_{outDim}_{kernel_type}_{max_iter}_{lr}_Z_test.npy'))
    if not os.path.exists(os.path.join('save', 'result',
                                       f'{dataset}_{outDim}_{kernel_type}_{max_iter}_{lr}_{neighbor_num}.npy')):
        knn = KNeighborsRegressor(np.minimum(neighbor_num, y_train.shape[0]), weights='distance',
                                  metric='cosine', n_jobs=-1)
        knn.fit(Z_train, y_train)
        y_pred = knn.predict(Z_test)
        np.save(os.path.join('save', 'result',
                             f'{dataset}_{outDim}_{kernel_type}_{max_iter}_{lr}_{neighbor_num}.npy'),
                y_pred)
    else:
        y_pred = np.load(os.path.join('save', 'result',
                                      f'{dataset}_{outDim}_{kernel_type}_{max_iter}_{lr}_{neighbor_num}.npy'))

    print('Chebyshev: %.4f' % (chebyshev(y_test, y_pred)))
    print('Clark: %.4f' % (clark(y_test, y_pred)))
    print('Canberra: %.4f' % (canberra(y_test, y_pred)))
    print('KLD: %.4f' % (kld(y_test, y_pred)))
    print('Cosine: %.4f' % (cosine(y_test, y_pred)))
    print('Intersection: %.4f' % (intersection(y_test, y_pred)))
