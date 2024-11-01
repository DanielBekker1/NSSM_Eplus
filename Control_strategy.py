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
from torch.utils.data import DataLoader, TensorDataset, random_split
from neuromancer.dataset import DictDataset


def load_closed_loop_system(nx, nu, show=False):

    nssm_node = model_loading()
    net = blocks.MLP_bounds(
        insize=nx, 
        outsize=nu,  
        hsizes=[32, 32],                
        nonlin=nn.GELU,  
        min=torch.tensor([[0.0]]),  # Fan speed minimum
        max=torch.tensor([[1.0]]),  # Fan speed maximum
    )

    policy = Node(net, ["xn"], ["U"], name="control_policy")  # Control policy node
    nssm_node.freeze()
    cl_system = System([policy, nssm_node], name="cl_system", nsteps=nsteps)

    if show:
        cl_system.show()

    return cl_system

def load_training_data():

    """
    31/10/2024: Only two more comments: The DictDataset object should be created after the split (you want to create 3 of them)
    They should contain the "name" argument to avoid the KeyError you were having, it is specific to the neuromancer library.

    A random split is not a good idea since the data is sequential. 
    You should split the data in a way that the train data is before the dev data and the dev data is before the test data.
    Load the data and split it into train, dev and test sets.
    Then create the dict datasets and dataloaders for each set. With a name for each DictDataset.

    You can check the "Learning to stabilize a linear dynamical system." notebook for a similar example, not exactly the same but similar.

    """


    with open("test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    test_data["xn"] = test_data["xn"][:, :-1, :]
    CL_dataset = DictDataset({"xn": test_data["xn"], "U": test_data["U"]}, name="CL dataset") # The name argument is important
    
    # Splitting the data into train, dev and test. 60% train, 20 % for dev and test.
    train_size = int((3/5) * len(CL_dataset))
    dev_size = test_size = int((1/5) * len(CL_dataset))

    #squential split of the CL data.
    CL_train_data = CL_dataset[:train_size]
    CL_dev_data = CL_dataset[train_size:train_size + dev_size]
    CL_test_data = CL_dataset[train_size + dev_size:train_size + dev_size + test_size]

    CL_train_dataset = DictDataset(CL_train_data, name="Train_data")
    CL_dev_dataset = DictDataset(CL_dev_data, name="Dev_data")
    CL_test_dataset = DictDataset(CL_test_data, name="Test_data")

    train_loader = DataLoader(CL_train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(CL_dev_dataset, batch_size=32)
    test_loader = DataLoader(CL_test_dataset, batch_size=32)
   
    nx = test_data["X"].shape[2]                   #Number of states
    nu = test_data["U"].shape[2]                   #Number of inputs

    return train_loader, dev_loader, nx, nu


def train_control_policy(cl_system : System, nsteps, train_loader, dev_loader, show=False):

    '''
    Below the cost function, constrains and optimisation of problem
    '''
    ############### Cost function ###################
    # cost_function = (CO2 - CO2_setpoint)**2 + alpha * P_fan * (electricity_cost[0] / 1000)  # Only using first electricity cost value for now
    #Define the symbolic variables for the control policy
    u = variable("U")
    x = variable("xn")
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
    
    if show:
        problem.show()

    cl_system.nsteps = nsteps # Coming from the constant set in the main block

    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        optimizer=optimizer,
        patience=100,
        epochs=1000,
        warmup=100,
        train_metric="train_loss",
        eval_metric="dev_loss",
        dev_metric="dev_loss",
        test_metric="dev_loss"
        
    )

    # Train control policy
    best_model = trainer.train()
    # load best trained model
    trainer.model.load_state_dict(best_model)
    torch.save(cl_system.state_dict(), "cl_system.pth")

    return cl_system

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
    cl_system = load_closed_loop_system(nx , nu, show=True)
    cl_system = train_control_policy(cl_system=cl_system, nsteps=nsteps, train_loader=train_loader, dev_loader=dev_loader, show=True)



