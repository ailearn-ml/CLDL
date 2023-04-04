import gpytorch
from gpytorch.kernels import *
import torch
from sklearn.preprocessing import normalize


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood, num_tasks, kernel_type='rbf'):
        super(MultitaskGPModel, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=num_tasks)
        if kernel_type == 'rbf':
            self.covar_module = MultitaskKernel(RBFKernel(), num_tasks=num_tasks, rank=1)
        elif kernel_type == 'linear':
            self.covar_module = MultitaskKernel(LinearKernel(), num_tasks=num_tasks, )
        elif kernel_type == 'matern':
            self.covar_module = MultitaskKernel(MaternKernel(), num_tasks=num_tasks, rank=1)
        elif kernel_type == 'cosine':
            self.covar_module = MultitaskKernel(CosineKernel(), num_tasks=num_tasks, rank=1)
        elif kernel_type == 'poly':
            self.covar_module = MultitaskKernel(PolynomialKernel(2), num_tasks=num_tasks, rank=1)
        elif kernel_type == 'rq':
            self.covar_module = MultitaskKernel(RQKernel(), num_tasks=num_tasks, rank=1)
        else:
            raise ValueError('kernel type error!')

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class MultitaskGPRegressor:
    def __init__(self, x_train, y_train, num_tasks=None, max_iter=50, lr=0.1, kernel_type='rbf', device='cuda:0'):
        self.max_iter = max_iter
        self.lr = lr
        self.kernel_type = kernel_type
        self.device = device
        X = torch.from_numpy(x_train).float().to(self.device)
        Y = torch.from_numpy(y_train).float().to(self.device)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(self.device)
        self.model = MultitaskGPModel(x_train=X, y_train=Y, likelihood=self.likelihood,
                                      num_tasks=y_train.shape[1] if num_tasks is None else num_tasks,
                                      kernel_type=kernel_type).to(self.device)
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.lr}])

    def fit(self, x_train, y_train):
        self.model.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        X = torch.from_numpy(x_train).float().to(self.device)
        Y = torch.from_numpy(y_train).float().to(self.device)
        self.model.set_train_data(X, Y)
        for i in range(self.max_iter):
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, Y)
            loss.backward()
            self.optimizer.step()
            # print('Iter %d/%d - Loss: %.3f ' % (i + 1, self.max_iter, loss.item()))

    def predict(self, x_test):
        self.model.eval()
        self.likelihood.eval()
        X = torch.from_numpy(x_test).float().to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_pred = self.likelihood(self.model(X)).mean
        return y_pred.detach().cpu().numpy()

    def set_train_data(self, inputs, targets):
        inputs = torch.from_numpy(inputs).float().to(self.device)
        targets = torch.from_numpy(targets).float().to(self.device)
        self.model.set_train_data(inputs, targets)

    def save(self, path='model'):
        torch.save(self.model.state_dict(), path + '_model.pkl')

    def load(self, path='model'):
        self.model.load_state_dict(torch.load(path + '_model.pkl'))


class CLDL:
    def __init__(self, x_train, encoding_train,
                 num_tasks=None, max_iter=50, lr=0.1, kernel_type='rbf', device='cuda:0'):
        self.regressor = MultitaskGPRegressor(x_train=x_train, y_train=normalize(encoding_train),
                                              num_tasks=encoding_train.shape[1] if num_tasks is None else num_tasks,
                                              max_iter=max_iter, lr=lr, kernel_type=kernel_type, device=device)

    def fit(self, x_train, encoding_train):
        self.regressor.fit(x_train, normalize(encoding_train))

    def predict(self, x_test):
        encoding_test = self.regressor.predict(x_test)
        return encoding_test

    def save(self, path='model.pkl'):
        self.regressor.save(path)

    def load(self, path='model.pkl'):
        self.regressor.load(path)
