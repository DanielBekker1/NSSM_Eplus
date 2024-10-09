import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
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

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load your CSV data
df = pd.read_csv("CSV_files/dataframe_output_jan_Max_speed.csv")
df2 = pd.read_csv("CSV_files/dataframe_output_jan_half_speed.csv")
df3 = pd.read_csv("CSV_files/dataframe_output_jan_zero_speed.csv")

df = df[:-16]
df2 = df2[:-16]
df3 = df3[:-16]
# Extract the relevant columns for states (X), outputs (Y), and control inputs (U)
# X_columns = ['zn_soft1_temp','air_loop_fan_mass_flow']
X_columns = ['zn_soft1_temp', 'zn_finance1_temp',
             'Indoor_CO2_zn0', 'Indoor_CO2_zn1',
             'air_loop_fan_electric_power', 'Occupancy_schedule'
                ]                                               # should contain relevant columns
U_columns = [# 'Occupancy_schedule',
             'air_loop_fan_mass_flow']                                  # Should contain relevant control columns
# U_columns = ['Indoor_CO2_zn0']
Y_columns = X_columns                                           # Assuming the same state variables are treated as outputs

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

nsteps = 50   # number of prediction horizon steps in the loss function
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
print("After the split_data")
print(f"Shape of X (states): {X.shape}")  # Should be (number_of_samples, number_of_features_for_X)
print(f"Shape of U (control inputs): {U.shape}")  # Should be (number_of_samples, number_of_features_for_U)
print(f"Shape of Y (outputs): {Y.shape}")  # Should match the shape of X


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

print("Before the normalisation")
print(f"Train X shape after reshaping: {trainX.shape}")  # Should be (number_of_batches, nsteps, nx)
print(f"Train U shape after reshaping: {trainU.shape}")  # Should be (number_of_batches, nsteps, nu)
print(f"Dev X shape after reshaping: {devX.shape}")
print(f"Test X shape after reshaping: {testX.shape}")

#Normalisation function
def normalize(x, mean, std):
        return (x - mean) / std

def normalise_data(trainX, trainU, devX, devU, testX, testU, length_train, length_dev, length_test):
    trainX = normalize(trainX[:length_train], mean_x, std_x)
    # trainX = train_sim['X'][:length].reshape(nbatch, nsteps, nx)
    trainX = trainX.reshape(nbatch_train, nsteps, nx)
    trainX = torch.tensor(trainX, dtype=torch.float32).clone().detach()
    trainU = normalize(trainU[:length_train], mean_u, std_u)
    trainU = trainU[:length_train].reshape(nbatch_train, nsteps, nu)
    trainU = torch.tensor(trainU, dtype=torch.float32).clone().detach()
    train_data = DictDataset({'X': trainX, 'xn': trainX[:, 0:1, :],
                                'U': trainU}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                                collate_fn=train_data.collate_fn, shuffle=False)        #Shuffle Is changed to False

    devX = normalize(devX[:length_dev], mean_x, std_x)
    devX = devX.reshape(nbatch_dev, nsteps, nx)
    devX = torch.tensor(devX, dtype=torch.float32).clone().detach()
    devU = normalize(devU[:length_dev], mean_u, std_u)
    devU = devU[:length_dev].reshape(nbatch_dev, nsteps, nu)        #Not sure if the length should be used here and the corresponding places^^
    devU = torch.tensor(devU, dtype=torch.float32).clone().detach()
    dev_data = DictDataset({'X': devX, 'xn': devX[:, 0:1, :],
                            'U': devU}, name='dev')
    dev_loader = DataLoader(dev_data, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=False)              #Shuffle Is changed to False

    testX = normalize(testX[:length_test], mean_x, std_x)
    testX = testX.reshape(nbatch_test, nsteps, nx)
    testX = torch.tensor(testX, dtype=torch.float32).clone().detach()
    testU = normalize(testU[:length_test], mean_u, std_u)
    testU = testU[:length_test].reshape(nbatch_test, nsteps, nu)
    testU = torch.tensor(testU, dtype=torch.float32).clone().detach()
    test_data = {'X': testX, 'xn': testX[:, 0:1, :],
                    'U': testU}
    return (trainX, trainU, train_data, train_loader,
            devX, devU, dev_data, dev_loader,
            testX, testU, test_data)

trainX, trainU, train_data, train_loader, devX, devU, dev_data, dev_loader, testX, testU, test_data = normalise_data(trainX, trainU, devX, devU, testX, testU, length_train, length_dev, length_test)
print("After the normalisation of the data")
print(f"Train X shape after reshaping: {trainX.shape}")  # Should be (number_of_batches, nsteps, nx)
print(f"Train U shape after reshaping: {trainU.shape}")  # Should be (number_of_batches, nsteps, nu)
print(f"Dev X shape after reshaping: {devX.shape}")
print(f"Test X shape after reshaping: {testX.shape}")

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
A = blocks.MLP(nx, nx, bias=True,          #Bias is intial True
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[60, 60, 60])

#models how the control input (u) affect the states
B = blocks.MLP(nu, nx, bias=True,          #Bias is intial True
                linear_map=torch.nn.Linear,
                nonlin=torch.nn.ReLU,
                hsizes=[60, 60, 60])


def model_loading():
    ssm = SSM(A, B, nx, nu)
# ssm = SSM()

    # create symbolic system model in Neuromancer
    nssm_node = Node(ssm, ['xn', 'U'], ['xn'], name='NSSM')

    if os.path.exists('nssm_model_node.pth'):
        nssm_node.load_state_dict(torch.load('nssm_model_node.pth'))
        print("Model loaded successfully.")
    else:
        print("Model file not found. Skipping model loading.")
    return nssm_node

if os.path.exists('nssm_model_node.pth'):
    nssm_node = model_loading()
else:
    nssm_node = model_loading()
    print("Skipping model loading as 'nssm_model_node.pth' is not available.")
# construct NSSM model in Neuromancer
# ssm = SSM(A, B, nx, nu)
# ssm = SSM()

# # create symbolic system model in Neuromancer
# nssm_node = Node(ssm, ['xn', 'U'], ['xn'], name='NSSM')
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

# plot computational graph
# # problem.show()
# s'etpoint_temperature = 20

# target_co2 = 600

# fan_speed_cost = 1.0 * torch.sum(U ** 2)  # Penalize higher fan speeds
# temperature_deviation_cost = 5.0 * torch.sum((X[:, :, 0] - setpoint_temperature) ** 2)  # Penalize deviation from temperature setpoint
# co2_cost = 2.0 * torch.sum((X[:, :, 1] - target_co2) ** 2)

# total_cost = fan_speed_cost + temperature_deviation_cost + co2_cost
# total_cost.name = "total_cost"

# aggregate list of objective terms and constraints
objectives = [reference_loss, onestep_loss, 
            #   total_cost
              ]
constraints = []
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem([dynamics_model], loss)


def forward(self, x, u, d=None):
    print(f"Shape of x (states) inside forward: {x.shape}")
    print(f"Shape of u (inputs) inside forward: {u.shape}")
    
    # state space model
    x = self.fx(x) + self.fu(u)
    print(f"Shape of output x after forward pass: {x.shape}")
    return x

optimizer = torch.optim.Adam(problem.parameters(),
                             lr=0.006)
#trainer
trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_data,
    optimizer,
    patience=100,   #initial 100
    warmup=100,     # Initial 100
    epochs=1000,     # Initial 1000
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
)


iterations = 1  # or any number of iterations
#Recap of nsteps = 10 and bs = 100
for i in range(iterations):
    print(f'Training iteration {i+1} with nsteps = {nsteps}')

    # Train the model
    best_model = trainer.train()
    problem.load_state_dict(best_model)

    # Reset early stopping counter if needed
    trainer.badcount = 0         # early stopping - if needed




torch.save(nssm_node.state_dict(), 'nssm_model_node.pth') 
# normalization_params = (nx, nu, mean_u, std_u)
# with open('nssm_model_params.pkl', 'wb') as f:
#     pickle.dump(normalization_params, f)
# with open('test_data.pkl', 'wb') as f:
#     pickle.dump(test_data, f)
'''
Have to create a new object, SSM same way as the original
'''


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



### Evaluation of model
rmse = np.sqrt(mean_squared_error(true_traj.flatten(), pred_traj.flatten()))
print(f'RMSE: {rmse}')

mae = mean_absolute_error(true_traj.flatten(), pred_traj.flatten())
print(f'MAE: {mae}')

r2 = r2_score(true_traj.flatten(), pred_traj.flatten())
print(f'R² score: {r2}')

rmse_states = np.sqrt(mean_squared_error(true_traj.flatten(), pred_traj.flatten()))
rmse_inputs = np.sqrt(mean_squared_error(test_data['U'].detach().numpy().flatten(), input_traj.flatten()))

print(f'RMSE for States: {rmse_states}')
print(f'RMSE for Inputs: {rmse_inputs}')



print(test_outputs['xn'])
print(f'Shape of the test data [xn]: {test_data["xn"].shape}')
print(test_data['xn'])
print(f'Shape of the test data [U]: {test_data["U"].shape}')
print(test_data['U'])

plt.show()
'''
Add state co2 for rooms to control. Occupancy as a state. 
Main fan speed as U.
Electricity consumption as a state.
Orginase current model to have everything in the states and U as input. Actual input is one sample X and prediction horison samples of U(Could be 50 samples).
Want to generate testdata with the correct shape (Equal to prediction horison (Xn[1,1,(nx)]   U(1,50,1) 50 equal to prediction horizon))

Model will look for key x in the input data. The model needs to know for how long the prediction.
I might want to change the imput the key. Specify the key! -> dynamic_model.nstep_key = 'U'

Inputs to model:
initial X, All U for the time horizon. 
Geneate new optimisation predicting
'''