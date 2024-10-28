'''
The structure of the script is the following:

Load the trained NSSM model.
Define the control policy network.
Integrate the cost function and constraints.
Solve the control problem at each timestep.
Graphviz is needed in system PATH to run the simulation. Important with problem.show()
'''
import sys
sys.stdout.reconfigure(encoding='utf-8')

# -*- coding: utf-8 -*-

import numpy as np
import torch
import pickle
import torch.nn as nn
from neuromancer.modules import blocks
# import matplotlib.pyplot as plt

from neuromancer.system import System, Node
from neuromancer.problem import Problem

from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from Main_nssm_version import model_loading



## Data loading
nsteps = 50

with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

nssm_node = model_loading()
# nssm_node.load_state_dict(torch.load('nssm_model_node.pth'))

#Variables
# x = variable("X")
u = variable("U")

x = variable("X")
xhat = variable('xn')[:, :-1, :]

CO2 = variable('xn')[:, :-1, 2] 
P_fan = variable('xn')[:, :-1, 4]



nx = test_data['X'].shape[2]                   #Number of states
nu = test_data['U'].shape[2]                   #Number of inputs

# Define constants 
CO2_setpoint = 500                                  # Setpoint for CO2
alpha = 0.1                                         # Weighting factor for energy consumption ??
electricity_cost = np.array([602.79, 620.69, 611.67, 620.54, 639.86, 675.95, 780.95, 1025.33, 1205.73, 957.69, 
                             776.33, 693.70, 600.93, 574.75, 562.45, 657.83, 696.23, 1009.67, 1428.86, 1312.52, 
                             833.08, 754.85, 767.60, 694.74])                  # 24-hour electricity prices (in DKK/MWh)
CO2_min, CO2_max = 400, 1000                        # CO2 concentration bounds
T_min, T_max = 18, 25                               # Temperature bounds
U_min, U_max = 0, 1.35                                # Fan speed bounds

# CO2 = test_data['X'][:, :, 2]
Temp_soft = test_data['X'][:, :, 0]
U = test_data['U'][:, :, 0]



#Control policy 

net = blocks.MLP_bounds(
    insize=nx,  # Size of system state 
    outsize=nu,  
    hsizes=[32, 32],                
    nonlin=nn.GELU,  
    min=torch.tensor([[0.0]]),  # Fan speed minimum
    max=torch.tensor([[1.0]]),  # Fan speed maximum
)



policy = Node(net, ['X'], ['U'], name='control_policy')  # Control policy node

nssm_node.freeze()
cl_system = System([policy, nssm_node], name='cl_system', nsteps=nsteps)

# cl_system.show()



'''
Below the cost function, constrains and optimisation of problem
'''
############### Cost function ###################
# cost_function = (CO2 - CO2_setpoint)**2 + alpha * P_fan * (electricity_cost[0] / 1000)  # Only using first electricity cost value for now

CO2_loss= (CO2 - CO2_setpoint)**2
energy_loss =  P_fan * (electricity_cost[0] / 1000)


# CO2 deviation minimization
# co2_loss = (x[:, :, 2] - CO2_setpoint)**2  

# objectives = [co2_loss.sum()]
# co2_loss_sum = CO2_loss.sum()
# energy_loss_sum = energy_loss.sum()

obj_1 = CO2_loss.minimize()
obj_1.name = 'CO2_loss'
obj_2 = energy_loss.minimize()
obj_1.name = 'Energyloss'

objectives = [obj_1, obj_2]

U_min_constraint = ("U" >= U_min)
U_min_constraint.name = 'Input_Min'

U_max_constraint = ("U" <= U_max)
U_max_constraint.name = 'Input_Max'

CO2_min_constraint = ('xn'[:, :-1, 2] >= CO2_min)
CO2_min_constraint.name = 'CO2_min'

CO2_max_constraint = ('xn'[:, :-1, 2] <= CO2_max)
CO2_max_constraint.name = 'CO2_Max'

T_min_constraint = ('xn'[:, :-1, 0] >= T_min)
T_min_constraint.name = 'Temp_min'

T_max_constraint = ('xn'[:, :-1, 0] <= T_max)
T_max_constraint.name = 'Temp_Max'


constraints = [U_min_constraint, U_max_constraint, CO2_min_constraint, CO2_max_constraint, T_min_constraint, T_max_constraint]


# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem([cl_system], loss)


with open("problem_graph.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    problem.show()

cl_system.nsteps = test_data['X'].shape[1]
test_outputs = cl_system(test_data)

# import networkx as nx
# # Instead of problem.show()
# graph = problem.get_graph()  # Get the computational graph
# nx.write_gml(graph, "problem_graph.gml")  # Write the graph to a file (GML format)


# optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
# #  Neuromancer trainer
# trainer = Trainer(
#     problem,
#     train_loader,
#     dev_loader,
#     optimizer=optimizer,
#     epochs=200,
#     train_metric='train_loss',
#     eval_metric='dev_loss',
#     warmup=200,
# )


# # Train control policy
# best_model = trainer.train()
# # load best trained model
# trainer.model.load_state_dict(best_model)



### Evaluation of model
# rmse = np.sqrt(mean_squared_error(true_traj.flatten(), pred_traj.flatten()))
# print(f'RMSE: {rmse}')

# mae = mean_absolute_error(true_traj.flatten(), pred_traj.flatten())
# print(f'MAE: {mae}')

# r2 = r2_score(true_traj.flatten(), pred_traj.flatten())
# print(f'R² score: {r2}')

# rmse_states = np.sqrt(mean_squared_error(true_traj.flatten(), pred_traj.flatten()))
# rmse_inputs = np.sqrt(mean_squared_error(test_data['U'].detach().numpy().flatten(), input_traj.flatten()))

# print(f'RMSE for States: {rmse_states}')
# print(f'RMSE for Inputs: {rmse_inputs}')



# print(test_outputs['xn'])
# print(f'Shape of the test data [xn]: {test_data["xn"].shape}')
# # print(test_data['xn'])
# print(f'Shape of the test data [U]: {test_data["U"].shape}')
# # print(test_data['U'])

