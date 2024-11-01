import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from neuromancer.system import Node, System
from neuromancer.dataset import DictDataset
from neuromancer.modules import blocks
from Laptop_version import SSM
from Laptop_version import normalise_data, model_loading
from Laptop_version import trainX, trainU, devX, devU, testX, testU, length_train, length_dev, length_test


def denormalize_disturbance(d, mean_d, std_d):
    return d * std_d + mean_d

def denormalize(y, mean_y, std_y):
    return y * std_y + mean_y

trainX, trainU, train_data, train_loader, devX, devU, dev_data, dev_loader, testX, testU, test_data = normalise_data(trainX, trainU, devX, devU, testX, testU, length_train, length_dev, length_test)


with open('nssm_model_params.pkl', 'rb') as f:
    nx, nu, mean_u, std_u = pickle.load(f)
    A = blocks.MLP(nx, nx, bias=True,
               linear_map=torch.nn.Linear,
               nonlin=torch.nn.ReLU,
               hsizes=[40, 40])  # Use the same dimensions as in your training script
B = blocks.MLP(nu, nx, bias=True,
               linear_map=torch.nn.Linear,
               nonlin=torch.nn.ReLU,
               hsizes=[40, 40])
ssm = SSM(A, B, nx, nu)

nssm_node = Node(ssm, ['xn', 'U'], ['xn'], name='NSSM')
dynamics_model = System([nssm_node], name='system')
nssm_node.load_state_dict(torch.load('nssm_model_node.pth', weights_only=True))

# dynamics_model.show()

with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

test_outputs = dynamics_model(test_data)

#denormalise the output
# Denormalize input control trajectory if needed
input_traj = denormalize(test_data['U'].detach().numpy(), mean_u, std_u)  # Assuming you have mean_u, std_u for control inputs

# Predicted trajectory (state variables)
pred_traj = test_outputs['xn'][:, :-1, :].detach().numpy().reshape(-1, nx)

# True trajectory (state variables)
true_traj = test_data['X'].detach().numpy().reshape(-1, nx)

# Transpose for plotting (to align the dimensions)
pred_traj, true_traj = pred_traj.transpose(1, 0), true_traj.transpose(1, 0)

# Plot rollout
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
    
input_traj = input_traj.squeeze()
# For input trajectory
ax[-1].plot(range(input_traj.shape[0]), input_traj, 'c', linewidth=4.0, label='inputs')
ax[-1].legend(fontsize=figsize)
ax[-1].set_xlabel('$time$', fontsize=figsize)
ax[-1].set_ylabel('$u$', rotation=0, labelpad=20, fontsize=figsize)
ax[-1].tick_params(labelbottom=True, labelsize=figsize)

# Final formatting and display
plt.tight_layout()
# plt.show()

plt_nsteps = test_data['X'].shape[1]
variables_to_plot = [
    {"name": "zn_soft1_temp", "true_label": "True Zone Temperature", "pred_label": "Predicted Zone Temperature"},
    {"name": "Indoor_CO2_zn0", "true_label": "True Zone CO2", "pred_label": "Predicted Zone CO2"}
]

def plot_variable(true_traj, pred_traj, var_index, plt_nsteps, true_label, pred_label, ylabel='Value'):
    fig, ax = plt.subplots()
    ax.plot(range(plt_nsteps), true_traj[:, var_index], label=true_label, color='b')
    ax.plot(range(plt_nsteps), pred_traj[:, var_index], label=pred_label, color='r', linestyle='--')
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time steps")
    ax.legend()
    plt.show()


for var in variables_to_plot:
    var_index = variables_to_plot.index(var)  # Get the index of the current variable
    plot_variable(
        true_traj=true_traj,
        pred_traj=pred_traj,
        var_index=var_index,
        plt_nsteps=plt_nsteps,
        true_label=var["true_label"],
        pred_label=var["pred_label"]
    )