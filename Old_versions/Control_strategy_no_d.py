'''
The structure of the script is the following:

Load the trained NSSM model.
Define the control policy network.
Integrate the cost function and constraints.
Solve the control problem at each timestep.
Graphviz is needed in system PATH to run the simulation. Important with problem.show()
'''
import numpy as np
import torch
import pickle
import torch.nn as nn
from neuromancer.modules import blocks
from matplotlib import pyplot as plt
from neuromancer.system import System, Node
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer

from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from Main_nssm_version import model_loading
from torch.utils.data import DataLoader
from neuromancer.dataset import DictDataset
from neuromancer.plot import pltCL, pltPhase

def load_closed_loop_system(nx, nu, show=False):

    nssm_node = model_loading()
    net = blocks.MLP_bounds(
        insize=nx, 
        outsize=nu,  
        hsizes=[64, 64, 64],                
        nonlin=nn.GELU,  
        min=0,  # Fan speed minimum
        max=1,  # Fan speed maximum
    )

    policy = Node(net, ["xn"], ["U"], name="control_policy")  # Control policy node
    nssm_node.freeze()
    cl_system = System([policy, nssm_node], name="cl_system", nsteps=nsteps)
    cl_system.nstep_key = "U"

    # if show:
    #     cl_system.show()

    return cl_system

def load_training_data():

    with open("test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    test_data["xn"] = test_data["xn"][:, 0:1, :]
    CL_dataset = DictDataset({"xn": test_data["xn"], "U": test_data["U"]}, name="CL_dataset") # The name argument is important
    
    # Splitting the data into train, dev and test. 60% train, 20 % for dev and test.
    train_size = int((3/5) * len(CL_dataset))
    dev_size = test_size = int((1/5) * len(CL_dataset))

    #squential split of the CL data.
    CL_train_data = CL_dataset[:train_size]
    CL_dev_data = CL_dataset[train_size:train_size + dev_size]
    CL_test_data = CL_dataset[train_size + dev_size:train_size + dev_size + test_size]

    CL_train_dataset = DictDataset(CL_train_data, name='train')
    CL_dev_dataset = DictDataset(CL_dev_data, name='dev')
    CL_test_dataset = DictDataset(CL_test_data, name= 'test')

    bs = 32

    train_loader = DataLoader(CL_train_dataset, 
                              collate_fn=CL_train_dataset.collate_fn, batch_size=bs, shuffle=False)
    dev_loader = DataLoader(CL_dev_dataset, 
                            collate_fn=CL_dev_dataset.collate_fn, batch_size=bs, shuffle=False)
    test_loader = DataLoader(CL_test_dataset, 
                             collate_fn=CL_test_dataset.collate_fn, batch_size=bs, shuffle=False)
   
    nx = test_data["X"].shape[2]                   #Number of states
    nu = test_data["U"].shape[2]                   #Number of inputs


    return train_loader, dev_loader, nx, nu


def train_control_policy(cl_system : System, nsteps, train_loader, dev_loader, show=True):

    '''
    Below the cost function, constrains and optimisation of problem
    '''
    ############### Cost function ###################
    # cost_function = (CO2 - CO2_setpoint)**2 + alpha * P_fan * (electricity_cost[0] / 1000)  # Only using first electricity cost value for now
    #Define the symbolic variables for the control policy

    for batch in train_loader:
        print(f"Batch 'xn' shape (should be [batch_size, 1, nx]): {batch['xn'].shape}")
        print(f"Batch 'U' shape: {batch['U'].shape}")
        break  # Only print for the first batch to avoid clutter

    for batch in train_loader:
        x_tensor = batch["xn"]
        u_tensor = batch["U"]
        print("Shape of x_tensor (initial condition):", x_tensor.shape)  # Expected to be [batch_size, 1, nx]
        print("Shape of u_tensor:", u_tensor.shape)  # Expected shape for control input

        # (Assuming you run `cl_system` on `x_tensor` here, or later in the code)
        break  # Only print for the first batch

    u = variable("U")
    xn = variable("xn")
    xhat = variable("xn")[:, :-1, :]
    CO2 = variable("xn")[:, :-1, 2] 
    P_fan = variable("xn")[:, :-1, 4]
    temp_soft = variable("xn")[:, :-1, 0]
    
    obj_1 = ((CO2 - CO2_setpoint)**2).minimize()
    obj_1.name = "CO2_loss"

    obj_2 =  ((electricity_cost[0]/1000)*P_fan).minimize()
    obj_2.name = "Energy_loss"


    
    objectives = [obj_1, obj_2]

    # Define the constrains
    U_min_constraint = (u >= U_min)
    U_min_constraint.name = "Input_Min"

    U_max_constraint = (u <= U_max)
    U_max_constraint.name = "Input_Max"

    CO2_min_constraint = (CO2 >= CO2_min)
    CO2_min_constraint.name = "CO2_min"

    CO2_max_constraint = (CO2 <= CO2_max)
    CO2_max_constraint.name = "CO2_Max"

    T_min_constraint = (temp_soft >= T_min)
    T_min_constraint.name = "Temp_min"

    T_max_constraint = (temp_soft <= T_max)
    T_max_constraint.name = "Temp_Max"


    constraints = [U_min_constraint, U_max_constraint, 
                   CO2_min_constraint, CO2_max_constraint, 
                   T_min_constraint, T_max_constraint]

    loss = PenaltyLoss(objectives, constraints)
    problem = Problem([cl_system], loss)
    
    # if show:
    #     problem.show()

    cl_system.nsteps = nsteps # Coming from the constant set in the main block

    # for batch in train_loader:
    #     print(f"Batch 'xn' shape: {batch['xn'].shape}, 'U' shape: {batch['U'].shape}")
    #     break



    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        optimizer=optimizer,
        patience=100,
        epochs=1000,
        warmup=100,
        eval_metric="dev_loss",
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="dev_loss",
    )

    # Train control policy
    best_model = trainer.train()
    # load best trained model
    trainer.model.load_state_dict(best_model)
    torch.save(cl_system.state_dict(), "cl_system.pth")

    problem.load_state_dict(best_model)

    """
    You also need to reformulate the data to be in the same format as the training data.
    There is no "X" in the test data.
    """

    data = {'xn': torch.ones(1, 1, nx, dtype=torch.float32)}        #Changed from X to xn
    # nsteps = 50
    cl_system.nsteps = nsteps
    trajectories = cl_system(data)
    
    print("Shape of trajectories['xn'] before reshape:", trajectories['xn'].shape)
    print("Shape of trajectories['U'] before reshape:", trajectories['U'].shape)
    print("Expected nsteps:", nsteps)
    print("Expected nx:", nx)

    pltCL(Y=trajectories['xn'][:, :, :2].detach().reshape(nsteps+1, 2), U=trajectories['U'].detach().reshape(nsteps, 1), figname='cl.png')    #Changed to xn
    pltPhase(X=trajectories['xn'][:, :, :2].detach().reshape(nsteps+1, 2), figname='phase.png')
    # plt.show()
    return cl_system, trajectories


def KPI_calculations(trajectories, CO2_setpoint, electricity_cost):

   
    CO2_trajectories = trajectories['xn'][:, :, 2].detach()

    mae_CO2 = torch.mean(torch.abs(CO2_trajectories - CO2_setpoint))
    rmse_CO2 = torch.sqrt(torch.mean((CO2_trajectories - CO2_setpoint) ** 2))

    print(f"Mean Absolute Error (MAE) for CO2: {mae_CO2.item():.2f}")
    print(f"Root Mean Square Error (RMSE) for CO2: {rmse_CO2.item():.2f}")
    

    P_fan_trajectories = trajectories['xn'][:, :, 4].detach()
    electricity_cost_tensor = torch.tensor(electricity_cost, dtype=torch.float32)
    fan_energy_cost = torch.sum(torch.mean(P_fan_trajectories, dim=1) * (electricity_cost_tensor / 1000))

    print(f"Total Energy Cost for Fan Power (in DKK): {fan_energy_cost.item():.2f}")
    
    return mae_CO2.item(), rmse_CO2.item(), fan_energy_cost.item()

def plots(CO2_trajectories, CO2_setpoint, P_fan_trajectories, electricity_cost):
    plt.figure(figsize=(12, 12))

    # CO₂ Concentration Over Time
    plt.subplot(3, 1, 1)
    plt.plot(CO2_trajectories.flatten().detach().numpy())
    plt.axhline(y=CO2_setpoint)
    plt.ylabel("CO₂ Concentration (ppm)")

    # Fan Power Usage Over Time
    plt.subplot(3, 1, 2)
    plt.plot(P_fan_trajectories.flatten().detach().numpy())
    plt.ylabel("Fan Power (W)")

    # Cumulative Energy Cost Over Time
    P_fan_mean = torch.mean(P_fan_trajectories, dim=1).detach()
    electricity_cost_tensor = torch.tensor(electricity_cost, dtype=torch.float32).detach()
    cumulative_cost = torch.cumsum(P_fan_mean * (electricity_cost_tensor / 1000), dim=0).detach().numpy()

    # cumulative_cost = torch.cumsum(torch.mean(P_fan_trajectories, dim=1) * (electricity_cost / 1000), dim=0).detach().numpy()
    plt.subplot(3, 1, 3)
    plt.plot(cumulative_cost)
    plt.ylabel("Cumulative Cost (DKK)")
    plt.xlabel("Time Step")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # Define constants 
    CO2_setpoint = 500                                  # Setpoint for CO2
    alpha = 0.1                                         # Weighting factor for energy consumption ??
    electricity_cost = np.array([602.79, 620.69, 611.67, 620.54, 639.86, 675.95, 780.95, 1025.33, 1205.73, 957.69, 
                                776.33, 693.70, 600.93, 574.75, 562.45, 657.83, 696.23, 1009.67, 1428.86, 1312.52, 
                                833.08, 754.85, 767.60, 694.74])                  # 24-hour electricity prices (in DKK/MWh)
    CO2_min, CO2_max = 400, 1000                        # CO2 concentration bounds
    T_min, T_max = 18, 25                               # Temperature bounds
    U_min, U_max = 0, 1.35                                # Fan speed bounds
    nsteps = 50



    #### Main program execution ####

    train_loader, dev_loader, nx, nu = load_training_data()

    for batch in train_loader:
        print("Initial shape of 'xn' in training batch:", batch['xn'].shape)
        print("Initial shape of 'U' in training batch:", batch['U'].shape)
        break  # Only inspect the first batch
    cl_system = load_closed_loop_system(nx , nu, show=True)
    cl_system, trajectories = train_control_policy(cl_system=cl_system, nsteps=nsteps, 
                                                   train_loader=train_loader, dev_loader=dev_loader, show=True)


    mae_CO2, rmse_CO2, total_fan_energy_cost = KPI_calculations(trajectories, CO2_setpoint, electricity_cost)

    plots(CO2_trajectories=trajectories['xn'][:, :, 2], 
                         CO2_setpoint=CO2_setpoint, 
                         P_fan_trajectories=trajectories['xn'][:, :, 4], 
                         electricity_cost=electricity_cost)

