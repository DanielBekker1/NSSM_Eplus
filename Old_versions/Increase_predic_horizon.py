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
df = pd.read_csv("CSV_files/dataframe_output_jan_Max_speed.csv")
df2 = pd.read_csv("CSV_files/dataframe_output_jan_half_speed.csv")
df3 = pd.read_csv("CSV_files/dataframe_output_jan_zero_speed.csv")

df = df[:-16]
df2 = df2[:-16]
df3 = df3[:-16]
# Extract the relevant columns for states (X), outputs (Y), and control inputs (U)
X_columns = ['zn_soft1_temp','Indoor_CO2_zn0' ]  # should contain relevant columns
U_columns = ['air_loop_fan_mass_flow']  # Should contain relevant control columns
Y_columns = X_columns  # Assuming the same state variables are treated as outputs

# Convert to numpy arrays
X = df[X_columns].values    #Internal or observed variables - states
U = df[U_columns].values    #Control actions - inputs
Y = df[Y_columns].values    # Can be the same as the state - Output

def extract_columns(df):
    X = df[X_columns].values    # State variables (temperature, CO2 levels)
    U = df[U_columns].values    # Control inputs (fan mass flow rate)
    Y = df[Y_columns].values    # Outputs (in this case, same as states)
    return X, U, Y

X1, U1, Y1 = extract_columns(df)
X2, U2, Y2 = extract_columns(df2)
X3, U3, Y3 = extract_columns(df3)
# Split into train, dev, test (assume 70/15/15 split)
nsim = X.shape[0]

nsteps = 10   # number of prediction horizon steps in the loss function
bs = 100      # minibatching batch size



def split_data(X, U, Y):
    train_size = X.shape[0] // 2
    dev_size = X.shape[0] // 4
    test_size = X.shape[0] - train_size - dev_size

    # Split states, inputs, and outputs into train, dev, test
    trainX, trainU, trainY = X[:train_size], U[:train_size], Y[:train_size]
    devX, devU, devY = X[train_size:train_size+dev_size], U[train_size:train_size+dev_size], Y[train_size:train_size+dev_size]
    testX, testU, testY = X[train_size+dev_size:], U[train_size+dev_size:], Y[train_size+dev_size:]

    return trainX, trainU, trainY, devX, devU, devY, testX, testU, testY

trainX1, trainU1, trainY1, devX1, devU1, devY1, testX1, testU1, testY1 = split_data(X1, U1, Y1)
trainX2, trainU2, trainY2, devX2, devU2, devY2, testX2, testU2, testY2 = split_data(X2, U2, Y2)
trainX3, trainU3, trainY3, devX3, devU3, devY3, testX3, testU3, testY3 = split_data(X3, U3, Y3)



trainX = np.concatenate([trainX1, trainX2, trainX3], axis=0)
trainU = np.concatenate([trainU1, trainU2, trainU3], axis=0)
trainY = np.concatenate([trainY1, trainY2, trainY3], axis=0)

# Combine dev sets
devX = np.concatenate([devX1, devX2, devX3], axis=0)
devU = np.concatenate([devU1, devU2, devU3], axis=0)
devY = np.concatenate([devY1, devY2, devY3], axis=0)

# Combine test sets
testX = np.concatenate([testX1, testX2, testX3], axis=0)
testU = np.concatenate([testU1, testU2, testU3], axis=0)
testY = np.concatenate([testY1, testY2, testY3], axis=0)
nsim = 2000   # number of simulation steps in the dataset

nx = trainX.shape[1] 
nu = trainU.shape[1]  

length_train = (trainX.shape[0]//nsteps) * nsteps
length_dev = (devX.shape[0]//nsteps) * nsteps
length_test = (testX.shape[0]//nsteps) * nsteps

nbatch_train = length_train//nsteps
nbatch_dev = length_dev//nsteps
nbatch_test = length_test//nsteps

mean_x = trainX.mean(axis=0)  
std_x = trainX.std(axis=0)    
mean_u = trainU.mean(axis=0)
std_u = trainU.std(axis=0)

#Normalisation function
def normalize(x, mean, std):
        return (x - mean) / std

def normalise_data(trainX, trainU, devX, devU, testX, testU, length_train, length_dev, length_test):
    trainX = normalize(trainX[:length_train], mean_x, std_x)
    # trainX = train_sim['X'][:length].reshape(nbatch, nsteps, nx)
    trainX = trainX.reshape(nbatch_train, nsteps, nx)
    trainX = torch.tensor(trainX, dtype=torch.float32)
    trainU = normalize(trainU[:length_train], mean_u, std_u)
    trainU = trainU[:length_train].reshape(nbatch_train, nsteps, nu)
    trainU = torch.tensor(trainU, dtype=torch.float32)
    train_data = DictDataset({'X': trainX, 'xn': trainX[:, 0:1, :],
                                'U': trainU}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                                collate_fn=train_data.collate_fn, shuffle=False)        #Shuffle Is changed to False

    devX = normalize(devX[:length_dev], mean_x, std_x)
    devX = devX.reshape(nbatch_dev, nsteps, nx)
    devX = torch.tensor(devX, dtype=torch.float32)
    devU = normalize(devU[:length_dev], mean_u, std_u)
    devU = devU[:length_dev].reshape(nbatch_dev, nsteps, nu)        #Not sure if the length should be used here and the corresponding places^^
    devU = torch.tensor(devU, dtype=torch.float32)
    dev_data = DictDataset({'X': devX, 'xn': devX[:, 0:1, :],
                            'U': devU}, name='dev')
    dev_loader = DataLoader(dev_data, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=False)              #Shuffle Is changed to False

    testX = normalize(testX[:length_test], mean_x, std_x)
    testX = testX.reshape(1, nbatch_test*nsteps, nx)
    testX = torch.tensor(testX, dtype=torch.float32)
    testU = normalize(testU[:length_test], mean_u, std_u)
    testU = testU[:length_test].reshape(1, nbatch_test*nsteps, nu)
    testU = torch.tensor(testU, dtype=torch.float32)
    test_data = {'X': testX, 'xn': testX[:, 0:1, :],
                    'U': testU}
    return (trainX, trainU, train_data, train_loader,
            devX, devU, dev_data, dev_loader,
            testX, testU, test_data)

trainX, trainU, train_data, train_loader, devX, devU, dev_data, dev_loader, testX, testU, test_data = normalise_data(trainX, trainU, devX, devU, testX, testU, length_train, length_dev, length_test)


##### NSSM model training ######


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

# # instantiate neural nets
# Models how the state (x) evovles over time
A = blocks.MLP(nx, nx, bias=True,
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[40, 40])

#models how the control input (u) affect the states
B = blocks.MLP(nu, nx, bias=True,
                linear_map=torch.nn.Linear,
                nonlin=torch.nn.ReLU,
                hsizes=[40, 40])


# construct NSSM model in Neuromancer
ssm = SSM(A, B, nx, nu)
# ssm = SSM()

# create symbolic system model in Neuromancer
nssm_node = Node(ssm, ['xn', 'U'], ['xn'], name='NSSM')
dynamics_model = System([nssm_node], name='system', nsteps=nsteps)

# visualize the system
# dynamics_model.show()

# %% Constraints + losses:
x = variable("X")
xhat = variable('xn')[:, :-1, :]

# trajectory tracking loss
reference_loss = 10.*(xhat == x)^2
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
                             lr=0.001)
#trainer
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

def update_data_loaders_with_new_horizon(nsteps, trainX, trainU, devX, devU, testX, testU, nx, nu, bs):
    # Calculate length_train and nbatch_train before using them
    length_train = (trainX.shape[0] // nsteps) * nsteps  # Ensure length is divisible by nsteps
    nbatch_train = length_train // nsteps

    print(f"trainX.shape: {trainX.shape}")  # Shape of the trainX array
    print(f"length_train: {length_train}, nbatch_train: {nbatch_train}, nsteps: {nsteps}, nx: {nx}")
    print(f"Expected elements: {length_train * nsteps * nx}, Actual elements: {trainX[:length_train].numel()}")

    # If the shape is already correct, you don't need to reshape it
    if trainX.shape == (nbatch_train, nsteps, nx):
        print("trainX already has the correct shape, no need to reshape.")
        train_data = DictDataset({'X': trainX, 'U': trainU}, name='train')  # Keep the existing shape
    else:
        # Reshape train data
        assert length_train * nsteps * nx == trainX[:length_train].numel(), "Invalid reshaping for trainX."
        trainX_reshaped = trainX[:length_train].reshape(nbatch_train, nsteps, nx)
        trainU_reshaped = trainU[:length_train].reshape(nbatch_train, nsteps, nu)

        train_data = DictDataset({'X': torch.tensor(trainX_reshaped, dtype=torch.float32),
                                  'U': torch.tensor(trainU_reshaped, dtype=torch.float32)}, name='train')

    train_loader = DataLoader(train_data, batch_size=bs, collate_fn=train_data.collate_fn, shuffle=True)

    # Adjust dev data
    length_dev = (devX.shape[0] // nsteps) * nsteps
    nbatch_dev = length_dev // nsteps
    assert length_dev * nsteps * nx == devX[:length_dev].numel(), "Invalid reshaping for devX."

    devX_reshaped = devX[:length_dev].reshape(nbatch_dev, nsteps, nx)
    devU_reshaped = devU[:length_dev].reshape(nbatch_dev, nsteps, nu)

    dev_data = DictDataset({'X': torch.tensor(devX_reshaped, dtype=torch.float32),
                            'U': torch.tensor(devU_reshaped, dtype=torch.float32)}, name='dev')
    dev_loader = DataLoader(dev_data, batch_size=bs, collate_fn=dev_data.collate_fn, shuffle=True)

    # Adjust test data
    length_test = (testX.shape[0] // nsteps) * nsteps
    nbatch_test = length_test // nsteps
    assert length_test * nsteps * nx == testX[:length_test].numel(), "Invalid reshaping for testX."

    testX_reshaped = testX[:length_test].reshape(1, nbatch_test * nsteps, nx)
    testU_reshaped = testU[:length_test].reshape(1, nbatch_test * nsteps, nu)

    test_data = {'X': torch.tensor(testX_reshaped, dtype=torch.float32),
                 'U': torch.tensor(testU_reshaped, dtype=torch.float32)}

    return train_loader, dev_loader, test_data

iterations = 5  # or any number of iterations
#Recap of nsteps = 10 and bs = 100
for i in range(iterations):
    print(f'Training iteration {i+1} with nsteps = {nsteps}')

    # Train the model
    best_model = trainer.train()
    problem.load_state_dict(best_model)

    # Increase prediction horizon for the next iteration
    nsteps += 2  # Adjust this increment as needed

    # Adjust dataloaders for the new horizon length (re-create data with updated nsteps)
    train_loader, dev_loader, test_data = update_data_loaders_with_new_horizon(
        nsteps, trainX, trainU, devX, devU, testX, testU, nx, nu, bs)
    
    trainer.train_data = train_loader
    trainer.dev_data = dev_loader
    trainer.test_data = test_data

    trainer.badcount = 0            # early stopping - if needed




torch.save(nssm_node.state_dict(), 'nssm_model_node.pth') 

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
figsize = 10
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
plt.show()