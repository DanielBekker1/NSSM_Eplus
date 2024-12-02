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

def discretize_control(U_continuous, mean_u, std_u):
    levels = [0, 0.675, 1.35]
    U_continuous = denormalise(U_continuous, mean=mean_u, std=std_u)
    U_continuous = torch.clamp(U_continuous, 0, 1.35)
    discrete_signal = np.array([min(levels, key= lambda x: abs(x - val)) for val in U_continuous])
    
    return discrete_signal


def load_closed_loop_system(nx, nu, nd, U_min, U_max, discrete_values, training=True, show=True):

    nssm_node = model_loading()
    net = blocks.MLP_bounds(
        insize=nx + nd, 
        outsize=nu,  
        hsizes=[64, 64, 64],                
        nonlin=nn.GELU,  
        min=U_min,      # Fan speed minimum
        max=U_max,   # Fan speed maximum
    )

    dist_model = lambda d: d
    dist_obs = Node(dist_model, ["d"], ["d_obs"], name='dist_obs')

    
    policy = Node(net, ["xn", "d_obs"], ["U_continuous"], name="control_policy")  # Control policy node
     
    discretization = lambda U_continuous: discretize_control(U_continuous, mean_u, std_u)
 
    discretize_node = Node(discretization, ["U_continuous"], ["U"], name="discretize_policy")

    nssm_node.freeze()
    cl_system = System([dist_obs, policy, discretize_node, nssm_node], name="cl_system", nsteps=nsteps)
    cl_system.nstep_key = "U"

    # if show:
    #     cl_system.show()

    return cl_system

def create_sliding_windows(data, window_size=50):

    num_sequences = len(data) // window_size
    return data[:num_sequences * window_size].reshape(num_sequences, window_size, -1)

def normalise(data, mean, std):
        return (data - mean) / std

def denormalise(data, mean, std):
    with torch.no_grad():
        result = data * std + mean

    return result

def load_training_data(data_path = "test_data.pkl"):           #The data loading of this should be changed to 

    with open(data_path, "rb") as f:
        CL_data = pickle.load(f)

    #Calculating mean and standard deviation
    
    mean_xn, std_xn = CL_data["mean_x"], CL_data["std_x"]
    mean_u, std_u = CL_data["mean_u"], CL_data["std_u"]
    mean_d, std_d = CL_data["mean_d"], CL_data["std_d"]

    # plt.figure(figsize=(12, 8))
    # plt.subplot(4, 1, 1)
    # plt.plot(CL_data["testD"][:, 0], label="Disturbance Feature 1")
    # plt.subplot(4, 1, 2)
    # plt.plot(CL_data["testD"][:, 1], label="Disturbance Feature 2")

    # plt.subplot(4, 1, 3)
    # for i in range(5):  # Iterate over the 5 state variables
    #     plt.plot(torch.tensor(CL_data["xn"][:, :, i], dtype=torch.float32).flatten(), label=f"State Variable {i+1}")
    
    # plt.subplot(4, 1, 4)
    # plt.plot(CL_data["testU"][:, 0], label="Fan Speed (u)")
    # plt.tight_layout()
    # plt.show()


    testD = normalise(CL_data["testD"], mean_d, std_d)
    xn = normalise(CL_data["xn"], mean_xn, std_xn)
    u = normalise(CL_data["testU"], mean_u, std_u)

    
    testD = torch.tensor(create_sliding_windows(testD, window_size=50), dtype=torch.float32)
    xn = torch.tensor(xn[:, 0:1, :], dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
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

def con_and_Obj (cl_system : System, nsteps , show=False):
    '''
    Below the cost function, constrains and optimisation of problem
    '''
    ############### Cost function ###################
    # cost_function = (CO2 - CO2_setpoint)**2 + alpha * P_fan * (electricity_cost[0] / 1000)  # Only using first electricity cost value for now
    #Define the symbolic variables for the control policy

    u = variable("U")
    xn = variable("xn")
    d = variable("d")
    CO2 = variable("xn")[:, :-1, 2] 
    P_fan = variable("xn")[:, :-1, 4]
    temp_soft = variable("xn")[:, :-1, 0]
    electricity_price = variable("d")[:, :, 1]

    obj_1 = (max(CO2 - CO2_setpoint, 0)**2).minimize()      #If CO2 > CO2_setpoint, returns CO2 - CO2_setpoint, else 0
    obj_1.name = "CO2_loss"

    # electricity_price = d[:, :, 1]

    obj_2 =  ((electricity_price)*(P_fan/1e6)).minimize()
    obj_2.name = "Energy_loss"

    # Penalise odd fan speed (anything else than 0, 0.675 and 1.35) 
    # obj_3 = ((u - U_min)**2).minimize() 
    # obj_3.name = "fan_spd_low"
    # obj_4 = ((u - U_medium)**2).minimize() 
    # obj_4.name = "fan_spd_medium"
    # obj_5 = ((u - U_max)**2).minimize()
    # obj_5.name = "fan_spd_high"    

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

    return problem

def train_control_policy(problem, train_loader, dev_loader, CL_test_data):
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.0001)
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        CL_test_data,
        optimizer=optimizer,
        patience=100,
        epochs=20,
        warmup=100,
        eval_metric="dev_loss",
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="dev_loss",
    )

    best_model = trainer.train()
    # Train control policy
    
    # load best trained model
    trainer.model.load_state_dict(best_model)
    torch.save(cl_system.state_dict(), "cl_system.pth")

    problem.load_state_dict(best_model)

    return cl_system, problem

def predict_trajectories(cl_system, CL_test_data, discrete_values, nsteps):

    test_data = {
        'xn': CL_test_data["xn"],  
        'd': CL_test_data["d"],    
        # No "U" since it's generated by the control policy
    }       
    # nsteps = 50
    cl_system.nsteps = nsteps
    trajectories = cl_system(test_data)

    # U_continuous = trajectories["U"]
    # print("U_continuous:", U_continuous.detach().numpy())
    # U_discrete = discrete_values[torch.argmin(torch.abs(U_continuous - discrete_values), dim=-1)]
    
    # trajectories["U"] = U_discrete.unsqueeze(-1)

    xn_single = trajectories['xn'][0, :, :2].detach()  
    U_single = trajectories['U'][0, :, :].detach()  
    
    pltCL(
        Y=xn_single.reshape(nsteps+1, 2),
        U=U_single.reshape(nsteps, 1),
        figname='cl_trajectory.png'
    )
    # pltPhase(
    #     X=xn_single.reshape(nsteps+1, 2),
    #     figname='cl_phase.png'
    # )
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

    CO2_trajectories = torch.tensor(denorm_trajectories["xn"][:, :, 2], dtype=torch.float32)

    mae_CO2 = torch.mean(torch.abs(CO2_trajectories - desired_CO2_setpoint))
    rmse_CO2 = torch.sqrt(torch.mean((CO2_trajectories - desired_CO2_setpoint) ** 2))

    print(f"Mean Absolute Error (MAE) for CO2: {mae_CO2.item():.2f}")
    print(f"Root Mean Square Error (RMSE) for CO2: {rmse_CO2.item():.2f}")
    
    P_fan_trajectories = torch.tensor(denorm_trajectories['xn'][:, :, 4], dtype=torch.float32)
    electricity_price = torch.tensor(denorm_trajectories['d'][:, :, 1], dtype=torch.float32)
    
 

    fan_energy_cost = torch.sum(P_fan_trajectories[:, :-1] * (electricity_price / 1e6))

    print(f"Total Energy Cost for Fan Power (in DKK): {fan_energy_cost.item():.2f}")
    
    return mae_CO2.item(), rmse_CO2.item(), fan_energy_cost.item(), CO2_trajectories, P_fan_trajectories

def plots(CO2_trajectories, desired_CO2_setpoint, P_fan_trajectories, denorm_trajectories, d):
    plt.figure(figsize=(12, 16))

    d = torch.tensor(d, dtype=torch.float32)

    fan_speed = torch.tensor(denorm_trajectories['U'], dtype=torch.float32)

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
    elec_kwh = torch.tensor(electricity_price, dtype=torch.float32) / 1000
    electricity_price_mean = torch.mean(electricity_price, dim=1)
    cumulative_cost = torch.cumsum(P_fan_mean * (electricity_price_mean / 1e6), dim=0)

    # plt.subplot(4, 1, 3)
    # plt.plot(cumulative_cost, label="Cumulative Energy Cost")
    # plt.ylabel("Cumulative Cost (DKK)")
    # plt.xlabel("Time Step")

    plt.subplot(4, 1, 3)
    plt.plot(fan_speed.flatten(), label="Fan speed")
    plt.ylabel("Fan speed")
    plt.xlabel("Time Step")

    #Electricity 
    plt.subplot(4, 1, 4)
    plt.plot(elec_kwh.flatten(), label="Electricity Price")
    plt.ylabel("Price (DKK/KWh)")
    plt.xlabel("Time Step")

    plt.tight_layout()
    plt.show()

def plot_predicted_vs_test(denorm_trajectories, denorm_test_data):

    pred_co2 = torch.tensor(denorm_trajectories["xn"][0, :, 2], dtype=torch.float32)
    test_c02 = torch.tensor(denorm_test_data["xn"][0, :, 2], dtype=torch.float32)

    # Predicted and test control inputs
    # pred_u = denorm_trajectories["U"]
    pred_u = torch.tensor(denorm_trajectories['U'], dtype=torch.float32)    
    # Plot for CO₂ concentration (State 3)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(test_c02.flatten(), linestyle="--", color="red")  # Test CO₂
    plt.plot(pred_co2.flatten(), linestyle="-", color="blue")  # Predicted CO₂
    plt.xlabel("Steps")
    plt.ylabel("CO₂ Concentration (ppm)")

    # Plot for control inputs (Fan Speed)
    plt.subplot(2, 1, 2)
    plt.plot(pred_u.flatten(), linestyle="-", color="blue")  # Predicted Fan Speed
    plt.xlabel("Steps")
    plt.ylabel("Fan Speed (Input)")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    train_loader, dev_loader, nx, nu, nd, CL_test_data, mean_d, std_d, mean_xn, std_xn, mean_u, std_u = load_training_data(data_path = "test_data.pkl")

    # Define constants - Normalised
    CO2_setpoint =  (1000 - mean_xn[2]) / std_xn[2]                                     # Setpoint for CO2
    desired_CO2_setpoint = 500                                                          #Only for plotting and KPI calculations. Used after denormalisation
    CO2_min = (400 - mean_xn[2]) / std_xn[2]                                            # CO2 concentration bounds - Normalised
    CO2_max = (2200 - mean_xn[2]) / std_xn[2]
    T_min = (18 - mean_xn[0]) / std_xn[0]                                               # Temperature bounds - Normalised
    T_max = (25 - mean_xn[0]) / std_xn[0]
    U_min = (0 - mean_u[0]) / std_u[0]                                                  # Fan speed bounds - Normalised
    U_medium = (0.675 - mean_u[0]) / std_u[0]
    U_max = (1.35 - mean_u[0]) / std_u[0]

    nsteps = 50

    discrete_values = torch.tensor([U_min, U_medium, U_max], dtype=torch.float32)
    #### Main program execution ####

    
    cl_system = load_closed_loop_system(nx , nu, nd, U_min, U_max, discrete_values=discrete_values, training=True, show=True)
    
    problem = con_and_Obj(cl_system=cl_system, nsteps=nsteps, show=True)

    cl_system, problem = train_control_policy(problem=problem, 
                                                   train_loader=train_loader, dev_loader=dev_loader, CL_test_data=CL_test_data)

    trajectories, test_data = predict_trajectories(cl_system, CL_test_data, discrete_values, nsteps)

    denorm_test_data, denorm_trajectories = denormalise_outputs(test_data, trajectories, mean_xn, std_xn, mean_d, std_d, mean_u, std_u)

    # print("Electricity price in control policy:", denorm_trajectories['d'][:, :, 1])
    mae_CO2, rmse_CO2, total_fan_energy_cost, CO2_trajectories, P_fan_trajectories = KPI_calculations(denorm_trajectories, desired_CO2_setpoint)

    plots(CO2_trajectories=CO2_trajectories, 
                         desired_CO2_setpoint=desired_CO2_setpoint, 
                         P_fan_trajectories=P_fan_trajectories, denorm_trajectories=denorm_trajectories, 
                         d=denorm_trajectories['d'])
    plot_predicted_vs_test(denorm_trajectories, denorm_test_data)


