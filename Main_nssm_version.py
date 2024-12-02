'''
This script import data from three different simulations of the energyplus model. 
The no_agent_baseline.py is used to run energyPlus model. The simulations contain different settings of the fan speed, (zero, half and full speed)
First step is to preprocess the data into the right format, normalise and spilt the data into train dev and test group (0.5 / 0.25 / 0.25)
The nssm model is build and trained on the data. The model is then validated on the test data.
Thhe model is saved and imported to another file.

The different functions of the script:
Data_loading: 
split_data:
Data_preparation:
Normailse
Class SSM:
create_ssm_model
Model_loading
loss_func_and_problem
train_nssm:
save_model_and_data
predict_trajectories:
plot_results:
'''
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import DataLoader

from neuromancer.system import Node, System
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks

#Global values
nsteps = 50   # number of prediction horizon steps in the loss function
bs = 32      # minibatching batch size
nsim = 2000   # number of simulation steps in the dataset

#Data loading and extractions of states (X), outputs (Y) and control inputs (U)
def data_loading():   
    df = pd.read_csv("CSV_files/dataframe_output_jan_Max_speed.csv")
    df2 = pd.read_csv("CSV_files/dataframe_output_jan_half_speed.csv")
    df3 = pd.read_csv("CSV_files/dataframe_output_jan_zero_speed.csv")
    df4 = pd.read_csv("CSV_files/electricity_price_15min_intervals.csv")

    df = df[:-16]
    df2 = df2[:-16]
    df3 = df3[:-16]
    df4 = df4[1:-15]


    X_columns = ['zn_soft1_temp', 'zn_finance1_temp',
                'Indoor_CO2_zn0', 'Indoor_CO2_zn1',
                'air_loop_fan_electric_power' 
                    ]                                               # should contain relevant columns
    U_columns = ['air_loop_fan_mass_flow']                          # Should contain relevant control columns
    # Y_columns = X_columns                                           # Assuming the same state variables are treated as outputs
    D_columns = ['Occupancy_schedule']                              #Contain the disturbance variables from energy+
    E_columns = ['electricity_price']
    E = np.tile(df4[E_columns].values, (3, 1)).reshape(-1, 1)                           #Contain the electricity price for the disturbance signal. Increase size with 3 to be equal other df.

    def extract_columns(df):
        X = df[X_columns].values    
        U = df[U_columns].values    
        # Y = df[Y_columns].values  
        D = df[D_columns].values
        return X, U, D
    
    X1, U1, D1 = extract_columns(df)
    X2, U2, D2 = extract_columns(df2)
    X3, U3, D3 = extract_columns(df3)



    return X1, U1, D1, X2, U2, D2, X3, U3, D3, E


def split_data(X1, U1, D1, X2, U2, D2, X3, U3, D3, E):                 #Split the data into train, dev, test (assume 50/25/25 split)
    train_size = 1000
    dev_size = 500
    test_size = 500

    #Scenario 1 (df1)
    trainX1, devX1, testX1 = X1[:train_size], X1[train_size:train_size + dev_size], X1[train_size + dev_size:]
    trainU1, devU1, testU1 = U1[:train_size], U1[train_size:train_size + dev_size], U1[train_size + dev_size:]
    trainD1, devD1, testD1 = D1[:train_size], D1[train_size:train_size + dev_size], D1[train_size + dev_size:]
    #Scenario 2 (df2)
    trainX2, devX2, testX2 = X2[:train_size], X2[train_size:train_size + dev_size], X2[train_size + dev_size:]
    trainU2, devU2, testU2 = U2[:train_size], U2[train_size:train_size + dev_size], U2[train_size + dev_size:]
    trainD2, devD2, testD2 = D2[:train_size], D2[train_size:train_size + dev_size], D2[train_size + dev_size:]
    #Scenario 3 (df3)
    trainX3, devX3, testX3 = X3[:train_size], X3[train_size:train_size + dev_size], X3[train_size + dev_size:]
    trainU3, devU3, testU3 = U3[:train_size], U3[train_size:train_size + dev_size], U3[train_size + dev_size:]
    trainD3, devD3, testD3 = D3[:train_size], D3[train_size:train_size + dev_size], D3[train_size + dev_size:]

    # Combine data from all scenarios
    trainX = np.concatenate([trainX1, trainX2, trainX3], axis=0)
    trainU = np.concatenate([trainU1, trainU2, trainU3], axis=0)
    trainD = np.concatenate([trainD1, trainD2, trainD3], axis=0)

    devX = np.concatenate([devX1, devX2, devX3], axis=0)
    devU = np.concatenate([devU1, devU2, devU3], axis=0)
    devD = np.concatenate([devD1, devD2, devD3], axis=0)

    testX = np.concatenate([testX1, testX2, testX3], axis=0)
    testU = np.concatenate([testU1, testU2, testU3], axis=0)
    testD = np.concatenate([testD1, testD2, testD3], axis=0)

    #Combine the two disturbance signal 
    trainD = np.hstack((trainD, E[:trainD.shape[0]]))
    devD = np.hstack((devD, E[trainD.shape[0]:trainD.shape[0] + devD.shape[0]]))
    testD = np.hstack((testD, E[trainD.shape[0] + devD.shape[0]:]))
    return trainX, trainU, trainD, devX, devU, devD, testX, testU, testD

def Data_Preparation():

    X1, U1, D1, X2, U2, D2, X3, U3, D3, E = data_loading()

    trainX, trainU, trainD, devX, devU, devD, testX, testU, testD = split_data(X1, U1, D1, X2, U2, D2, X3, U3, D3, E)
    
    # Calculate lengths and batch sizes (dimensions)
    nx = trainX.shape[1]                                        # Number of states in the model (nx)
    nu = trainU.shape[1]                                        # Number of inputs in the model (nu)
    nd = trainD.shape[1]

    length_train = (trainX.shape[0] // nsteps) * nsteps
    length_dev = (devX.shape[0] // nsteps) * nsteps
    length_test = (testX.shape[0] // nsteps) * nsteps
    

    nbatch_train = length_train // nsteps
    nbatch_dev = length_dev // nsteps
    nbatch_test = length_test // nsteps

    # Calculate means and standard deviations for normalization
    mean_x, std_x = trainX.mean(axis=0), trainX.std(axis=0) 
    mean_u, std_u = trainU.mean(axis=0), trainU.std(axis=0)
    mean_d, std_d = trainD.mean(axis=0), trainD.std(axis=0)


    return (trainX, trainU, trainD, devX, devU, devD, testX, testU, testD, 
            nx, nu, nd, length_train, length_dev, length_test, 
            nbatch_train, nbatch_dev, nbatch_test, 
            mean_x, std_x, mean_u, std_u, mean_d, std_d)


#Normalisation function
def normalise(x, mean, std):
        return (x - mean) / std

def denormalise(data, mean, std):
    return data * std + mean

trainX, trainU, trainD, devX, devU, devD, testX, testU, testD, nx, nu, nd, length_train, length_dev, length_test, nbatch_train, nbatch_dev, nbatch_test, mean_x, std_x, mean_u, std_u, mean_d, std_d = Data_Preparation()

def normalise_data(trainX, trainU, trainD, devX, devU, devD, testX, testU, testD, nx, nu, nd, length_train, length_dev, length_test, 
                    nbatch_train, nbatch_dev, nbatch_test, mean_x, std_x, mean_u, std_u, mean_d, std_d):
    trainX = normalise(trainX[:length_train], mean_x, std_x)
    trainX = trainX[:length_train].reshape(nbatch_train, nsteps, nx)
    trainX = torch.tensor(trainX, dtype=torch.float32).clone().detach()
    trainU = normalise(trainU[:length_train], mean_u, std_u)
    trainU = trainU[:length_train].reshape(nbatch_train, nsteps, nu)                    #Ensure the correct length of data (no additional data)
    trainU = torch.tensor(trainU, dtype=torch.float32).clone().detach()
    trainD = normalise(trainD[:length_train], mean_d, std_d)
    trainD = trainD[:length_train].reshape(nbatch_train, nsteps, nd)
    trainD = torch.tensor(trainD, dtype=torch.float32).clone().detach()
    train_data = DictDataset({'X': trainX, 'xn': trainX[:, 0:1, :],
                                'U': trainU, 'd': trainD}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                                collate_fn=train_data.collate_fn, shuffle=False)        #Shuffle Is changed to False

    devX = normalise(devX[:length_dev], mean_x, std_x)
    devX = devX[:length_dev].reshape(nbatch_dev, nsteps, nx)
    devX = torch.tensor(devX, dtype=torch.float32).clone().detach()
    devU = normalise(devU[:length_dev], mean_u, std_u)
    devU = devU[:length_dev].reshape(nbatch_dev, nsteps, nu)        
    devU = torch.tensor(devU, dtype=torch.float32).clone().detach()
    devD = normalise(devD[:length_dev], mean_d, std_d)
    devD = devD[:length_dev].reshape(nbatch_dev, nsteps, nd)
    devD = torch.tensor(devD, dtype=torch.float32).clone().detach()
    dev_data = DictDataset({'X': devX, 'xn': devX[:, 0:1, :],
                            'U': devU, 'd': devD}, name='dev')
    dev_loader = DataLoader(dev_data, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=False)              #Shuffle Is changed to False

    testX = normalise(testX[:length_test], mean_x, std_x)
    testX = testX[:length_test].reshape(nbatch_test, nsteps, nx)
    testX = torch.tensor(testX, dtype=torch.float32).clone().detach()
    testU = normalise(testU[:length_test], mean_u, std_u)
    testU = testU[:length_test].reshape(nbatch_test, nsteps, nu)
    testU = torch.tensor(testU, dtype=torch.float32).clone().detach()
    testD = normalise(testD[:length_test], mean_d, std_d)
    testD = testD[:length_test].reshape(nbatch_test, nsteps, nd)
    testD = torch.tensor(testD, dtype=torch.float32).clone().detach()
    test_data = {'X': testX, 'xn': testX[:, 0:1, :],
                    'U': testU, 'd': testD}
    return (trainX, trainU, trainD, train_data, train_loader,
            devX, devU, devD, dev_data, dev_loader,
            testX, testU, testD, test_data)

trainX, trainU, trainD, train_data, train_loader, devX, devU, devD, dev_data, dev_loader, testX, testU, testD, test_data = normalise_data(trainX, trainU, trainD, devX, devU, devD, testX, testU, testD, 
                                                                                                                     nx, nu, nd, length_train, length_dev, length_test, 
                                                                                                                     nbatch_train, nbatch_dev, nbatch_test, mean_x, std_x, mean_u, std_u, mean_d, std_d)



#Creation of NSSM structure
class SSM(nn.Module):

    def __init__(self, fx, fu, fd, nx, nu, nd):
        super().__init__()
        self.fx, self.fu, self.fd = fx, fu, fd
        self.nx, self.nu, self.nd = nx, nu, nd
        self.in_features, self.out_features = nx + nu + nd, nx
        

    def forward(self, x, u, d): 
        x_next = self.fx(x) + self.fu(u) + self.fd(d)

        return x_next

    
# # instantiate neural nets
# Models how the state (x) evovles over time

    # Define the state evolution model
A = blocks.MLP(nx, nx, bias=True,          #Bias is intial True
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[64, 64, 64])

#models how the control input (u) affect the states
B = blocks.MLP(nu, nx, bias=True,          #Bias is intial True
                linear_map=torch.nn.Linear,
                nonlin=torch.nn.ReLU,
                hsizes=[64, 64, 64])

C = blocks.MLP(nd, nx, bias=True,
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[64, 64, 64])


def model_loading():
    ssm = SSM(A, B, C, nx, nu, nd)
    # create symbolic system model in Neuromancer
    nssm_node = Node(ssm, ['xn', 'U', 'd'], ['xn'], name='NSSM')

    if os.path.exists('nssm_model_node.pth'):
        nssm_node.load_state_dict(torch.load('nssm_model_node.pth'), strict=False)
        print("Model loaded successfully.")
    
    return nssm_node


# nssm_node = model_loading()

def loss_func_and_problem(dynamics_model):
    print("Shape of test_data['X'] (reference trajectory):", test_data["X"].shape)
    x = variable("X")
    xhat = variable('xn')[:, :-1, :]


    # trajectory tracking loss
    reference_loss = 10.*(xhat == x)^2
    reference_loss.name = "ref_loss"

    # one step tracking loss
    onestep_loss = 1.*(xhat[:, 1, :] == x[:, 1, :])^2
    onestep_loss.name = "onestep_loss"

    print("Reference loss objective:", reference_loss)
    print("One-step loss objective:", onestep_loss)
    objectives = [reference_loss, onestep_loss]
    loss = PenaltyLoss(objectives, []) 
    # construct constrained optimization problem
    problem = Problem([dynamics_model], loss)
    problem.show()
    return problem

def setup_optimizer(problem):
    return torch.optim.Adam(problem.parameters(), lr=0.006)


#Training of NSSM
def train_nssm(problem, train_loader, dev_loader, test_data, optimizer):
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_data,
        optimizer,
        patience=100,                          #initial 100
        warmup=100,                            # Initial 100
        epochs=1000,                           # Initial 1000
        eval_metric="dev_loss",
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="dev_loss",
    )


    return trainer.train()


#Predictions of states
def predict_trajectories(dynamics_model, test_data):
    test_outputs = dynamics_model(test_data)
    test_data["xn"] = test_outputs["xn"]
    return test_data, test_outputs

def denormalise_outputs(test_data, test_outputs, mean_x, std_x, mean_u, std_u, mean_d, std_d):
    """
    Denormalizes test data and test outputs.
    """
    denorm_test_data = {
        "X": denormalise(test_data["X"].detach().numpy(), mean_x, std_x),
        "U": denormalise(test_data["U"].detach().numpy(), mean_u, std_u),
        "xn": denormalise(test_data["xn"].detach().numpy(), mean_x, std_x),
        "d": denormalise(test_data["d"].detach().numpy(), mean_d, std_d)
    }
    denorm_test_outputs = {
        "X": denormalise(test_outputs["X"].detach().numpy(), mean_x, std_x),
        "U": denormalise(test_outputs["U"].detach().numpy(), mean_u, std_u),
        "xn": denormalise(test_outputs["xn"].detach().numpy(), mean_x, std_x),
        "d": denormalise(test_outputs["d"].detach().numpy(), mean_d, std_d)
    }
    return denorm_test_data, denorm_test_outputs

#Saving of NSSM model and test_data
def save_model_and_data(nssm_node, testX, testU, testD, mean_x, std_x, mean_u, std_u, mean_d, std_d, denorm_test_outputs, model_path='nssm_model_node.pth', data_path='CL_data.pkl'):
    torch.save(nssm_node.state_dict(), model_path)
    CL_data = {
    "testX": (testX),
    "testU": (testU),
    "testD": (testD),
    # "testX": denormalise(testX, mean_x, std_x),
    # "testU": denormalise(testU, mean_u, std_u),
    # "testD": denormalise(testD, mean_d, std_d),
    "U": denorm_test_outputs["U"],
    "xn": denorm_test_outputs["xn"],
    "mean_x": mean_x,
    "mean_u": mean_u,
    "mean_d": mean_d,
    "std_x": std_x,
    "std_u": std_u,
    "std_d": std_d
    }
    with open(data_path, 'wb') as f:
        pickle.dump(CL_data, f)


def plot_results(denorm_test_data, denorm_test_outputs, nx, nu, nd, figsize=25):
    pred_traj = denorm_test_outputs['xn'][:, :-1, :].reshape(-1, nx)
    true_traj = denorm_test_data['X'].reshape(-1, nx)
    input_traj = denorm_test_data['U'].reshape(-1, nu)
    dist_traj = denorm_test_data['d'].reshape(-1, nd)
    pred_traj, true_traj = pred_traj.transpose(1, 0), true_traj.transpose(1, 0)
    dist_traj = dist_traj.transpose(1, 0)

    fig1, ax = plt.subplots(nx + nu, figsize=(figsize, figsize))
    labels = [f'$y_{k}$' for k in range(nx)]
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
        axe = ax[row] if nx > 1 else ax
        axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        axe.plot(t1, 'c', linewidth=4.0, label='True')
        axe.plot(t2, 'm--', linewidth=4.0, label='Pred')
        axe.tick_params(labelbottom=False, labelsize=figsize)
    ax[-1].plot(input_traj, 'c', linewidth=4.0, label='inputs')
    ax[-1].legend(fontsize=figsize)
    ax[-1].set_xlabel('$time$', fontsize=figsize)
    ax[-1].set_ylabel('$u$', rotation=0, labelpad=20, fontsize=figsize)
    ax[-1].tick_params(labelbottom=True, labelsize=figsize)

    fig2, ax2 = plt.subplots(nd, figsize=(figsize, figsize))
    ax2 = ax2 if nd > 1 else [ax2]  # Ensure ax2 is always iterable
    dist_labels = [f'$d_{k}$' for k in range(nd)]
    
    for i in range(nd):
        ax2[i].plot(dist_traj[i], 'orange', linewidth=2.0, label=f'Disturbance {dist_labels[i]}')
        ax2[i].set_ylabel(dist_labels[i], rotation=0, labelpad=20, fontsize=figsize)
        ax2[i].tick_params(labelbottom=True, labelsize=figsize)
        ax2[i].legend(fontsize=figsize)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    print("Starting script...")
    #Data preparation
    trainX, trainU, trainD, devX, devU, devD, testX, testU, testD, \
        nx, nu, nd, length_train, length_dev, length_test, nbatch_train, \
            nbatch_dev, nbatch_test, mean_x, std_x, mean_u, std_u, mean_d, std_d= Data_Preparation()

    #Model creation and loading
    nssm_node = model_loading()

    dynamics_model = System([nssm_node], name='system', nsteps=nsteps)
    dynamics_model.nstep_key = 'U'
   

    #Loss function and problem setup
    problem = loss_func_and_problem(dynamics_model)

#Optimizer setup
    optimizer = setup_optimizer(problem)
#Train NSSM model
    best_model = train_nssm(problem, train_loader, dev_loader, test_data, optimizer)


#Prediction, saving, and plotting
    test_data, test_outputs = predict_trajectories(dynamics_model, test_data)

    denorm_test_data, denorm_test_outputs =  denormalise_outputs(test_data, test_outputs, mean_x, std_x, mean_u, std_u, mean_d, std_d)
    
    save_model_and_data(dynamics_model, testX, testU, testD, mean_x, std_x, mean_u, std_u, mean_d, std_d, denorm_test_outputs)
    plot_results(denorm_test_data, denorm_test_outputs, nx, nu, nd)

