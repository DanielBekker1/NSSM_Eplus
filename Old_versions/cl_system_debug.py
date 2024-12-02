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


def load_closed_loop_system(nx, nu, nd, show=False):

    nssm_node = model_loading()
    net = blocks.MLP_bounds(
        insize=nx + nd, 
        outsize=nu,  
        hsizes=[64, 64, 64],                
        nonlin=nn.GELU,  
        min=0,      # Fan speed minimum
        max=1.35,   # Fan speed maximum
    )

    dist_model = lambda d: d
    dist_obs = Node(dist_model, ["d"], ["d_obs"], name='dist_obs')

    
    policy = Node(net, ["xn", "d_obs"], ["U"], name="control_policy")  # Control policy node
    nssm_node.freeze()
    cl_system = System([dist_obs, policy, nssm_node], name="cl_system", nsteps=nsteps)
    cl_system.nstep_key = "U"

    if show:
        cl_system.show()

    return cl_system

def create_sliding_windows(data, window_size=50):
    """
    Convert data into sliding windows. The window size is set to 50.
    """
    num_sequences = len(data) // window_size
    return data[:num_sequences * window_size].reshape(num_sequences, window_size, -1)

def normalise(data, mean, std):
        return (data - mean) / std

def denormalise(data, mean, std):
    result = data * std + mean
    # print(f"Denormalising: mean={mean}, std={std}, result_shape={result.shape}")
    return result

def load_training_data(data_path = "test_data.pkl"):           #The data loading of this should be changed to 

    with open(data_path, "rb") as f:
        CL_data = pickle.load(f)

    #Calculating mean and standard deviation
    
    mean_xn, std_xn = CL_data["mean_x"], CL_data["std_x"]
    mean_u, std_u = CL_data["mean_u"], CL_data["std_u"]
    mean_d, std_d = CL_data["mean_d"], CL_data["std_d"]

    

    testD = normalise(CL_data["testD"], mean_d, std_d)
    xn = normalise(CL_data["xn"], mean_xn, std_xn)
    u = normalise(CL_data["testU"], mean_u, std_u)
    testD = torch.tensor(create_sliding_windows(testD, window_size=50), dtype=torch.float32)
    xn = torch.tensor(xn[:, 0:1, :], dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)

    debug_data_statistics(testD, "Normalized Disturbances (testD)")
    debug_data_statistics(xn, "Normalized States (xn)")
    debug_data_statistics(u, "Normalized Controls (u)")

    # Splitting the data into train, dev and test. 50% train, 25 % for dev and test.
    total_length = xn.shape[0]
    train_size = int(0.5 * total_length)
    dev_size = test_size = int(0.25 * total_length)

    CL_train_data = DictDataset({"xn": xn[:train_size], "d": testD[:train_size]}, name="train")
    CL_dev_data = DictDataset({"xn": xn[train_size:train_size + dev_size], "d": testD[train_size:train_size + dev_size]}, name="dev")
    CL_test_data = {"xn": xn[train_size + dev_size:],  "d": testD[train_size + dev_size:]}


    bs = 32

    train_loader = DataLoader(CL_train_data, 
                              collate_fn=CL_train_data.collate_fn, batch_size=bs, shuffle=False)
    dev_loader = DataLoader(CL_dev_data, 
                            collate_fn=CL_dev_data.collate_fn, batch_size=bs, shuffle=False)
    

    nx = xn.shape[2]                                   #Number of states
    nu = 1
    nd = testD.shape[2]


    return train_loader, dev_loader, nx, nu, nd, CL_test_data, mean_d, std_d, mean_xn, std_xn, mean_u, std_u

def debug_data_statistics(data, name):
    """
    Print mean and standard deviation of a dataset.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    print(f"Debugging {name} Statistics:")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Std: {np.std(data):.2f}")
    print(f"  Min: {np.min(data):.2f}")
    print(f"  Max: {np.max(data):.2f}\n")


def con_and_Obj(cl_system: System, nsteps, show=False):
    """
    Define cost functions, constraints, and optimization problem.
    """

    # Define symbolic variables for control policy
    u = variable("U")  # Control input (Fan speed)
    xn = variable("xn")  # States
    d = variable("d")  # Disturbances

    # Define specific symbolic variables from states and disturbances
    CO2 = variable("xn")[:, :-1, 2]  # CO2 concentration
    P_fan = variable("xn")[:, :-1, 4]  # Fan power
    temp_soft = variable("xn")[:, :-1, 0]  # Temperature
    electricity_price = d[:, :, 1]  # Electricity price from disturbances

    # Debug symbolic variables
    print("Symbolic Variables Debugging:")
    print(f"  U (Fan Speed): {u}")
    print(f"  xn (States): {xn}")
    print(f"  d (Disturbances): {d}")
    print(f"  CO2: {CO2}")
    print(f"  P_fan (Fan Power): {P_fan}")
    print(f"  Temp_soft (Temperature): {temp_soft}")
    print(f"  Electricity Price: {electricity_price}")

    # Normalization factors (if needed)
    norm_CO2 = 1000.0
    norm_energy = 1e6
    print(f"Normalization Factors: norm_CO2={norm_CO2}, norm_energy={norm_energy}")

    # Define objectives
    obj_1 = (max(CO2 - CO2_setpoint, 0) ** 2).minimize()  # CO2 deviation penalty
    obj_1.name = "CO2_loss"

    obj_2 = ((electricity_price / norm_energy) * P_fan).minimize()  # Energy cost
    obj_2.name = "Energy_loss"

    objectives = [obj_1, obj_2]

    # Debug objectives
    print("Defined Objectives:")
    for obj in objectives:
        print(f"  Objective: {obj.name} = {obj}")

    # Define constraints
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

    constraints = [
        U_min_constraint,
        U_max_constraint,
        CO2_min_constraint,
        CO2_max_constraint,
        T_min_constraint,
        T_max_constraint,
    ]

    # Debug constraints
    print("Defined Constraints:")
    for constraint in constraints:
        print(f"  {constraint}")

    # Create penalty loss
    loss = PenaltyLoss(objectives, constraints)

    # Define problem
    problem = Problem([cl_system], loss)

    if show:
        problem.show()

    cl_system.nsteps = nsteps  # Set the number of steps for closed-loop system

    return problem

def train_control_policy(problem, train_loader, dev_loader, CL_test_data):
    """
    Train the control policy using the provided problem and data loaders.
    """

    # Define optimizer
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.0001)

    # Initialize training parameters
    patience = 100
    epochs = 200
    warmup = 100
    eval_metric = "dev_loss"
    train_losses, dev_losses = [], []

    print("Training Initialization:")
    print(f"  Number of Epochs: {epochs}")
    print(f"  Warmup Steps: {warmup}")
    print(f"  Evaluation Metric: {eval_metric}")

    best_model = None
    best_dev_loss = float("inf")

    # Training loop with debugging
    for epoch in range(epochs):
        # Training phase
        problem.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = problem(batch)
            batch_loss = outputs["train_loss"]
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
        train_loss /= len(train_loader)

        # Validation phase
        problem.eval()
        dev_loss = 0
        with torch.no_grad():
            for batch in dev_loader:
                outputs = problem(batch)
                batch_loss = outputs["dev_loss"]
                dev_loss += batch_loss.item()
        dev_loss /= len(dev_loader)

        # Collect losses for debugging
        train_losses.append(train_loss)
        dev_losses.append(dev_loss)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Dev Loss: {dev_loss:.4f}")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_model = problem.state_dict()

    # Plot training and validation losses
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(dev_losses, label="Dev Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Trend")
    plt.legend()
    plt.show()

    # Load the best model
    print("Loading the best model based on validation performance...")
    problem.load_state_dict(best_model)
    torch.save(problem.state_dict(), "best_cl_system.pth")

    return problem

def predict_trajectories(cl_system, CL_test_data, nsteps):

    test_data = {
        'xn': CL_test_data["xn"],  
        'd': CL_test_data["d"],    
        # No "U" since it's generated by the control policy
    }       
    # nsteps = 50
    cl_system.nsteps = nsteps
    trajectories = cl_system(test_data)

    xn_single = trajectories['xn'][0, :, :2].detach()  
    U_single = trajectories['U'][0, :, :].detach()  
    
    pltCL(
        Y=xn_single.reshape(nsteps+1, 2),
        U=U_single.reshape(nsteps, 1),
        figname='cl_trajectory.png'
    )
    pltPhase(
        X=xn_single.reshape(nsteps+1, 2),
        figname='cl_phase.png'
    )
    # plt.show()
    return trajectories, test_data

def denormalise_outputs(test_data, trajectories, mean_xn, std_xn, mean_d, std_d, mean_u, std_u):

    denorm_test_data = {
        "xn": denormalise(test_data["xn"].detach().numpy(), mean_xn, std_xn),
        "d": denormalise(test_data["d"].detach().numpy(), mean_d, std_d)
    }

    denorm_trajectories = {
        "xn": denormalise(trajectories["xn"].detach().numpy(), mean_xn, std_xn),
        "d": denormalise(trajectories["d"].detach().numpy(), mean_d, std_d),
        "d_obs": denormalise(trajectories["d_obs"].detach().numpy(), mean_d, std_d),
        "U": denormalise(trajectories["U"].detach().numpy(), mean_u, std_u)
    }

    return denorm_test_data, denorm_trajectories

def KPI_calculations(denorm_trajectories, desired_CO2_setpoint):

    CO2_trajectories = torch.tensor(denorm_trajectories["xn"][:, :, 2])

    mae_CO2 = torch.mean(torch.abs(CO2_trajectories - desired_CO2_setpoint))
    rmse_CO2 = torch.sqrt(torch.mean((CO2_trajectories - desired_CO2_setpoint) ** 2))

    print(f"Mean Absolute Error (MAE) for CO2: {mae_CO2.item():.2f}")
    print(f"Root Mean Square Error (RMSE) for CO2: {rmse_CO2.item():.2f}")
    
    P_fan_trajectories = torch.tensor(denorm_trajectories['xn'][:, :, 4])
    electricity_price = torch.tensor(denorm_trajectories['d'][:, :, 1])
    
    print("Shape of P_fan_trajectories[:, :-1]:", P_fan_trajectories[:, :-1].shape)
    print("Shape of electricity_price:", electricity_price.shape)

    fan_energy_cost = torch.sum(P_fan_trajectories[:, :-1] * (electricity_price / 1e6))

    print(f"Total Energy Cost for Fan Power (in DKK): {fan_energy_cost.item():.2f}")
    
    return mae_CO2.item(), rmse_CO2.item(), fan_energy_cost.item(), CO2_trajectories, P_fan_trajectories

def plots(CO2_trajectories, desired_CO2_setpoint, P_fan_trajectories, d):
    plt.figure(figsize=(12, 16))

    d = torch.tensor(d)

    #CO₂ Concentration
    plt.subplot(4, 1, 1)
    plt.plot(CO2_trajectories.flatten(), label="CO₂ Concentration")
    plt.axhline(y=desired_CO2_setpoint, color='r', linestyle='--', label="Setpoint")
    plt.ylabel("CO₂ Concentration (ppm)")

    #Fan Power
    plt.subplot(4, 1, 2)
    plt.plot(P_fan_trajectories.flatten(), label="Fan Power")
    plt.ylabel("Fan Power (W)")
    

    #Cumulative Energy
    P_fan_mean = torch.mean(P_fan_trajectories, dim=1)
    electricity_price = d[:, :, 1]
    electricity_price_mean = torch.mean(electricity_price, dim=1)
    cumulative_cost = torch.cumsum(P_fan_mean * (electricity_price_mean / 1000), dim=0)

    plt.subplot(4, 1, 3)
    plt.plot(cumulative_cost, label="Cumulative Energy Cost")
    plt.ylabel("Cumulative Cost (DKK)")
    plt.xlabel("Time Step")

    #Electricity 
    plt.subplot(4, 1, 4)
    plt.plot(electricity_price.flatten(), label="Electricity Price")
    plt.ylabel("Price (DKK/MWh)")
    plt.xlabel("Time Step")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    train_loader, dev_loader, nx, nu, nd, CL_test_data, mean_d, std_d, mean_xn, std_xn, mean_u, std_u = load_training_data(data_path = "test_data.pkl")

    # Define constants 
    CO2_setpoint =  (1000 - mean_xn[2]) / std_xn[2]                                  # Setpoint for CO2
    desired_CO2_setpoint = 500
    CO2_min, CO2_max = 400, 2000                        # CO2 concentration bounds
    T_min, T_max = 18, 25                               # Temperature bounds
    U_min, U_max = 0, 1.35                                # Fan speed bounds
    nsteps = 50


    #### Main program execution ####

    

    cl_system = load_closed_loop_system(nx , nu, nd, show=True)
    
    problem = con_and_Obj(cl_system=cl_system, nsteps=nsteps, show=True)

    problem = train_control_policy(problem=problem, 
                                                   train_loader=train_loader, dev_loader=dev_loader, CL_test_data=CL_test_data)

    trajectories, test_data = predict_trajectories(cl_system, CL_test_data, nsteps)

    denorm_test_data, denorm_trajectories = denormalise_outputs(test_data, trajectories, mean_xn, std_xn, mean_d, std_d, mean_u, std_u)

    # print("Electricity price in control policy:", denorm_trajectories['d'][:, :, 1])
    mae_CO2, rmse_CO2, total_fan_energy_cost, CO2_trajectories, P_fan_trajectories = KPI_calculations(denorm_trajectories, desired_CO2_setpoint)

    plots(CO2_trajectories=CO2_trajectories, 
                         desired_CO2_setpoint=desired_CO2_setpoint, 
                         P_fan_trajectories=P_fan_trajectories, 
                         d=denorm_trajectories['d'])


