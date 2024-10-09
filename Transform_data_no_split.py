import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from neuromancer.psl import plot
from neuromancer import psl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from neuromancer.system import Node, System
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks

# Load your CSV data
# Load your CSV data
df = pd.read_csv("CSV_files/dataframe_output_jan.csv")
df = df[:-16]
# Extract the relevant columns for states (X), outputs (Y), and control inputs (U)
X_columns = ['zn_soft1_temp', 'Indoor_CO2_zn0']  # replace with relevant columns
U_columns = ['air_loop_fan_mass_flow']  # replace with relevant control columns
Y_columns = X_columns  # Assuming the same state variables are treated as outputs

# Convert to numpy arrays
X = df[X_columns].values    #Internal or observed variables - states
U = df[U_columns].values    #Control actions - inputs
Y = df[Y_columns].values    # Can be the same as the state - Output


# Split into train, dev, test (assume 70/15/15 split)
nsim = X.shape[0]

nsteps = 10   # number of prediction horizon steps in the loss function
bs = 100      # minibatching batch size


train_sim = {'X': X, 'Y': Y, 'U': U, 'Time': np.linspace(0, nsim * 0.1, nsim)}
# dev_sim = train_sim  # Use the same dataset for dev
# test_sim = train_sim

def split_data(X, U, Y):
    trainX = X[:X.shape[0]//2, :]
    trainU = U[:U.shape[0]//2, :]
    trainY = Y[:Y.shape[0]//2, :]

    devX = X[:X.shape[0]//4, :]
    devU = U[:U.shape[0]//4, :]
    devY = Y[:Y.shape[0]//4, :]

    testX = X[:X.shape[0]//4, :]
    testU = U[:U.shape[0]//4, :]
    testY = Y[:Y.shape[0]//4, :]
nsim = 2000   # number of simulation steps in the dataset

nx = train_sim['X'].shape[2] 
nu = train_sim['U'].shape[2]  
nbatch = nsim//nsteps
length = (nsim//nsteps) * nsteps

mean_x = train_sim['X'].mean(axis=0)  
std_x = train_sim['X'].std(axis=0)    
mean_u = train_sim['U'].mean(axis=0)
std_u = train_sim['U'].std(axis=0)

#Normalisation function
def normalize(x, mean, std):
        return (x - mean) / std

trainX = normalize(train_sim['X'], mean_x, std_x)
# trainX = train_sim['X'][:length].reshape(nbatch, nsteps, nx)
print(f"trainX shape before reshaping: {trainX.shape}")
trainX = trainX.reshape(nbatch, nsteps, nx)
print(f"trainX shape after reshaping: {trainX.shape}")
trainX = torch.tensor(trainX, dtype=torch.float32)
trainU = normalize(train_sim['U'][:length], mean_u, std_u)
trainU = trainU.reshape(nbatch, nsteps, nu)
trainU = torch.tensor(trainU, dtype=torch.float32)
train_data = DictDataset({'X': trainX, 'xn': trainX[:, 0:1, :],
                            'U': trainU}, name='train')
train_loader = DataLoader(train_data, batch_size=bs,
                            collate_fn=train_data.collate_fn, shuffle=True)

devX = normalize(dev_sim['X'][:length], mean_x, std_x)
devX = devX.reshape(nbatch, nsteps, nx)
devX = torch.tensor(devX, dtype=torch.float32)
devU = normalize(dev_sim['U'][:length], mean_u, std_u)
devU = devU[:length].reshape(nbatch, nsteps, nu)
devU = torch.tensor(devU, dtype=torch.float32)
dev_data = DictDataset({'X': devX, 'xn': devX[:, 0:1, :],
                        'U': devU}, name='dev')
dev_loader = DataLoader(dev_data, batch_size=bs,
                        collate_fn=dev_data.collate_fn, shuffle=True)

testX = normalize(test_sim['X'][:length], mean_x, std_x)
testX = testX.reshape(1, nbatch*nsteps, nx)
testX = torch.tensor(testX, dtype=torch.float32)
testU = normalize(test_sim['X'][:length], mean_u, std_u)
testU = testU.reshape(1, nbatch*nsteps, nu)
testU = torch.tensor(testU, dtype=torch.float32)
test_data = {'X': testX, 'xn': testX[:, 0:1, :],
                'U': testU}


print(train_loader, dev_loader, test_data)
# Now you can use `train_sim`, `dev_sim`, `test_sim` just like in part_5 script

class SSM(nn.Module):

    def __init__(self, fx, fu, nx, nu):
        super().__init__()
        self.fx, self.fu = fx, fu
        self.nx, self.nu = nx, nu
        self.in_features, self.out_features = nx+nu, nx

    def forward(self, x, u, d=None):
       
        # state space model
        x = self.fx(x) + self.fu(u)
        return x

# instantiate neural nets
A = blocks.MLP(nx, nx, bias=True,
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[40, 40])
B = blocks.MLP(nu, nx, bias=True,
                linear_map=torch.nn.Linear,
                nonlin=torch.nn.ReLU,
                hsizes=[40, 40])
# construct NSSM model in Neuromancer
ssm = SSM(A, B, nx, nu)

# create symbolic system model in Neuromancer
model = Node(ssm, ['xn', 'U'], ['xn'], name='NODE')
dynamics_model = System([model], name='system', nsteps=nsteps)

# visualize the system
# dynamics_model.show()

# %% Constraints + losses:
x = variable("X")
xhat = variable('xn')[:, :-1, :]

# trajectory tracking loss
reference_loss = 5.*(xhat == x)^2
reference_loss.name = "ref_loss"

# one step tracking loss
onestep_loss = 1.*(xhat[:, 1, :] == x[:, 1, :])^2
onestep_loss.name = "onestep_loss"



# aggregate list of objective terms and constraints
objectives = [reference_loss, onestep_loss]
constraints = []
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem([dynamics_model], loss)

# plot computational graph
# problem.show()



optimizer = torch.optim.Adam(problem.parameters(),
                             lr=0.003)
# trainer
trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_data,
    optimizer,
    patience=100,
    warmup=100,
    epochs=1000,
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
)

# %% train
best_model = trainer.train()
problem.load_state_dict(best_model)

"""## Parameter estimation results"""

# update the rollout length based on the test data
dynamics_model.nsteps = test_data['X'].shape[1]

# Test set results
test_outputs = dynamics_model(test_data)

pred_traj = test_outputs['xn'][:, :-1, :].detach().numpy().reshape(-1, nx)
true_traj = test_data['X'].detach().numpy().reshape(-1, nx)
input_traj = test_data['U'].detach().numpy().reshape(-1, nu)
pred_traj, true_traj = pred_traj.transpose(1, 0), true_traj.transpose(1, 0)

# plot rollout
figsize = 25
fig, ax = plt.subplots(nx + nu, figsize=(figsize, figsize))
labels = [f'$y_{k}$' for k in range(len(true_traj))]
for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
    if nx > 1:
        axe = ax[row]
    else:
        axe = ax
    axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
    axe.plot(t1, 'c', linewidth=4.0, label='True')
    axe.plot(t2, 'm--', linewidth=4.0, label='Pred')
    axe.tick_params(labelbottom=False, labelsize=figsize)
axe.tick_params(labelbottom=True, labelsize=figsize)
axe.legend(fontsize=figsize)
ax[-1].plot(input_traj, 'c', linewidth=4.0, label='inputs')
ax[-1].legend(fontsize=figsize)
ax[-1].set_xlabel('$time$', fontsize=figsize)
ax[-1].set_ylabel('$u$', rotation=0, labelpad=20, fontsize=figsize)
ax[-1].tick_params(labelbottom=True, labelsize=figsize)
plt.tight_layout()