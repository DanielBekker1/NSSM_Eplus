import sys

import pandas as pd
sys.path.insert(0, r'C:\Users\Bruger\OneDrive\Dokumenter\MasterThesis\Test\energy-plus-DRL\RL-EmsPy\emspy')
sys.path.insert(0, r'C:\EnergyPlusV24-1-0')


from pyenergyplus import api #Importing from folder, therefore a warning may show
from pyenergyplus.api import EnergyPlusAPI
import numpy as np
# from emspy import BcaEnv
from emspy import EmsPy
from bca import BcaEnv
import datetime
import matplotlib.pyplot as plt
from Control_strategy import load_closed_loop_system
from Control_strategy import nx, nu, nd, U_min, U_max, nsteps, mean_d, std_d, xn
from torch.utils.data import DataLoader
from neuromancer.dataset import DictDataset
import torch

# -- FILE PATHS --""
# * E+ Download Path *
ep_path = r'C:\EnergyPlusV24-1-0'  # path to E+ on system
# IDF File / Modification Path
# idf_file_name = r"C:\Users\danie\OneDrive\Dokumenter\Neuromancer\EnergyPlus\ReMoni_OS_Model_jan.idf"  # building energy model (BEM) IDF file
# # Weather Path
# ep_weather_path = r"C:\Users\danie\OneDrive\Dokumenter\Neuromancer\EnergyPlus\DNK_MJ_Aarhus_jan_2007-2021.epw" #  EPW weather file entire year
# # Output .csv Path (optional)
# cvs_output_path = r'C:\Users\danie\OneDrive\Dokumenter\Neuromancer\CSV_files\dataframe_output_jan.csv'
# #Data input from CSV files
# elec_price = r"C:\Users\danie\OneDrive\Dokumenter\Neuromancer\CSV_files\ElectricityPrice_Jan.csv"
# occupancy_signal = r"C:\Users\danie\OneDrive\Dokumenter\Neuromancer\CSV_files/Occupancy.csv"
# #Model Path
# model_path = r"C:\Users\danie\OneDrive\Dokumenter\Neuromancer\cl_system.pth"

#For non-laptop simulation
idf_file_name = r"C:\Users\Bruger\OneDrive\Dokumenter\Neuromancer\EnergyPlus\ReMoni_OS_Model_jan.idf"  # building energy model (BEM) IDF file
ep_weather_path = r"C:\Users\Bruger\OneDrive\Dokumenter\Neuromancer\EnergyPlus\DNK_MJ_Aarhus_jan_2007-2021.epw" #  EPW weather file entire year
cvs_output_path = r'C:\Users\Bruger\OneDrive\Dokumenter\Neuromancer\CSV_files\CL_simulation.csv'
elec_price = r"C:\Users\Bruger\OneDrive\Dokumenter\Neuromancer\CSV_files\ElectricityPrice_Jan.csv"
occupancy_signal = r"C:\Users\Bruger\OneDrive\Dokumenter\Neuromancer\CSV_files/Occupancy.csv"
model_path = r"C:\Users\Bruger\OneDrive\Dokumenter\Neuromancer\cl_system.pth"

# STATE SPACE (& Auxiliary Simulation Data)

zn0 = 'Thermal Zone: Software_Office_1' #name of the zone to control 
zn1 = 'Thermal Zone: Finance_Office_1'
zn2 = 'Thermal Zone: Hardware_Corridor'

# zn2 = ''
tc_intvars = {}  # empty, don't need any

########### Fetching the variables that I will need. Find the names of the available variables in the .edd file.

tc_vars = {
    # Building
    #'hvac_operation_sched': ('Schedule Value', 'HtgSetp 1'),  # is building 'open'/'close'?
    # -- Zone 0 (Core_Zn) --
    'zn_soft1_temp': ('Zone Air Temperature', zn0),
    'zn_finance1_temp': ('Zone Air Temperature', zn1),
    'zn_hardware_corri_temp': ('Zone Air Temperature', zn2),
    'air_loop_fan_electric_power': ('Fan Electricity Rate', 'Const Spd Fan'),    # Electricity usage of the fan in HVAC system 
    'air_loop_fan_mass_flow': ('Fan Air Mass Flow Rate', 'Const Spd Fan'),
    'Indoor_CO2_zn0' : ('Zone Air CO2 Concentration',zn0),  #Indoor CO2 concentration. affected by ventilation and infil.
    'Indoor_CO2_zn1' : ('Zone Air CO2 Concentration',zn1),  #Indoor CO2 concentration. affected by ventilation and infil.
    'Occupancy_schedule' : ('Schedule Value', 'Office Occupancy'), 
    # 'ventil_zn0' : ('Zone Ventilation Mass Flow Rate',zn0),
        # deg C
}

tc_meters = {} # empty, don't need any

tc_weather = {
    'oa_rh': ('outdoor_relative_humidity'),  # %RH
    'oa_db': ('outdoor_dry_bulb'),  # deg C
    'oa_pa': ('outdoor_barometric_pressure'),  # Pa
    'sun_up': ('sun_is_up'),  # T/F
    'rain': ('is_raining'),  # T/F
    'snow': ('is_snowing'),  # T/F
    'wind_dir': ('wind_direction'),  # deg
    'wind_speed': ('wind_speed')  # m/s
}

# ACTION SPACE
tc_actuators = {
    # HVAC Control Setpoints
    #'zn0_CO2_con': ('Zone Temperature Control', 'Cooling Setpoint', zn0),  # deg C
    'air_loop_fan_mass_flow_actuator' : ('Fan','Fan Air Mass Flow Rate', 'CONST SPD FAN'),  # kg/s
    # 'air_loop_Ventil_flow_rate_actuator' : ('Sizing:System', 'Main Supply Volume Flow Rate','REMONI_VENTILATION'),
}
# -- Simulation Params --
calling_point_for_callback_fxn = EmsPy.available_calling_points[7]  # 6-16 valid for timestep loop during simulation
sim_timesteps = 4  # every 60 / sim_timestep minutes (e.g 10 minutes per timestep)

# -- Create Building Energy Simulation Instance --
sim = BcaEnv(
    ep_path=ep_path,
    ep_idf_to_run=idf_file_name,
    timesteps=sim_timesteps,
    tc_vars=tc_vars,
    tc_intvars=tc_intvars,
    tc_meters=tc_meters,
    tc_actuator=tc_actuators,
    tc_weather=tc_weather
)

class Agent:
    """
    Create agent instance, which is used to create actuation() and observation() functions (both optional) and maintain
    scope throughout the simulation.
    Since EnergyPlus' Python EMS using callback functions at calling points, it is helpful to use a object instance
    (Agent) and use its methods for the callbacks. * That way data from the simulation can be stored with the Agent
    instance.
    """
    def __init__(self, bca: BcaEnv):
        self.bca = bca
# Should be same varialbes as obersevation_function

        # simulation data state
        self.zn0_temp = None  # deg C                       # self nr. 0
        self.zn1_temp = None  # deg C                       # self nr. 1
        self.zn2_temp = None  # deg C                       # Self nr. 2
        self.fan_electric_power = None  # W                 # self nr. 3
        self.fan_mass_flow = None   #kg/s                   # self nr. 4
        
        self.CO2_indoor_zn0 = None #ppm                     # self nr. 5
        self.CO2_indoor_zn2 = None #ppm                     # self nr. 6
        self.Occupancy_schedule = None #ppm                 # self nr. 7
        # self.ventil_zn0 = None                              # self nr. 8

    #   self.zn2_temp = None  # deg C                       # self nr. 5
        self.state_size = (10,1)
        self.action_size = 10                               #Should this be adjusted to 3?

    
        self.Actor = load_closed_loop_system(nx, nu, nd, U_min, U_max, nsteps)
        self.occupancy_profile = pd.read_csv(occupancy_signal)
        self.occupancy_profile['Datetime'] = pd.to_datetime(self.occupancy_profile['Datetime'])
        self.price_profile = pd.read_csv(elec_price)
        self.price_profile['Datetime'] = pd.to_datetime(self.price_profile['Datetime'])
        self.state = None
        self.time_of_day = None


    def observation_function(self):
        # -- FETCH/UPDATE SIMULATION DATA --
        self.time = self.bca.get_ems_data(['t_datetimes'])
        #check that self.time is less than current time
        #Load the occupancy dataframe

        #Load the energy price dataframe

        if self.time < datetime.datetime.now():

            # Get data from simulation at current timestep (and calling point) using ToC names
            var_data = self.bca.get_ems_data(list(self.bca.tc_var.keys()))

            # Save one set of simulation results PER controller over the 21 days of simulation data
            # Have also separately one set of results without any custom controller to compare (baseline)
            self.zn0_temp = var_data[0]                            
            self.zn1_temp = var_data[1]
            self.zn2_temp = var_data[2]
            self.fan_electric_power = var_data[3]               # W
            self.fan_mass_flow = var_data[4]                    # kg/s
            self.CO2_in_zn0_con = var_data[5]                      # ppm
            self.CO2_in_zn2_con = var_data[6]                      # ppm


            self.state = self.get_state()

            
          
# Define thresholds - Can find example in model_test.py.

    def actuation_function(self):
        """
        Compute the control action, ensure compatibility with EnergyPlus, and apply it.
        """
        # Get the formatted disturbance signal
        disturbance_loader = self.get_state()

        # Fetch a batch from the DataLoader
        for batch in disturbance_loader:
            output = self.Actor(batch)  # Pass the disturbance signal to the Actor
            action = output['U'][0, 0, 0].item()  # Extract control signal (fan speed)

            # Rescale and validate the action
            # action_rescaled = action * (self.fan_max - self.fan_min) + self.fan_min
            # action = np.clip(float(action_rescaled), self.fan_min, self.fan_max)  # Ensure bounds

            # Apply the control action to EnergyPlus
            actuator_name = 'air_loop_fan_mass_flow_actuator'
            print(f"Final control signal: {action} (kg/s)")
            return {actuator_name: action}

    def get_state(self):
        """
        Prepare the disturbance signal (`d`) in the correct format for the control strategy.
        """
        xn_tensor = xn[0:1, :, :]
        self.time_of_day = self.bca.get_ems_data(['t_hours'])

        # Convert hours and fractional minutes to timestamp
        hours = int(self.time_of_day)
        minutes = int((self.time_of_day - hours) * 60)
        current_time = pd.Timestamp(f"2019-01-01 {hours:02d}:{minutes:02d}:00")

        # Ensure both DataFrames have datetime columns
        if not pd.api.types.is_datetime64_any_dtype(self.occupancy_profile['Datetime']):
            self.occupancy_profile['Datetime'] = pd.to_datetime(self.occupancy_profile['Datetime'])
        if not pd.api.types.is_datetime64_any_dtype(self.price_profile['Datetime']):
            self.price_profile['Datetime'] = pd.to_datetime(self.price_profile['Datetime'])

        # Find matching indices
        occupancy_index = self.occupancy_profile['Datetime'].searchsorted(current_time)
        price_index = self.price_profile['Datetime'].searchsorted(current_time)

        # Ensure there are enough samples for the prediction horizon
        if occupancy_index + 50 <= len(self.occupancy_profile) and price_index + 50 <= len(self.price_profile):
            # Fetch the next 50 samples
            occupancy_samples = self.occupancy_profile.iloc[occupancy_index:occupancy_index + 50]['Occupancy_schedule'].values
            price_samples = self.price_profile.iloc[price_index:price_index + 50]['Electricity_price'].values

            # Stack and normalize the disturbance data
            samples = np.stack((occupancy_samples, price_samples), axis=1).astype(np.float32)
            disturbance = (samples - mean_d) / std_d  # Normalize with mean and std

        else:
            # Fallback for insufficient data
            disturbance = np.zeros((50, 2), dtype=np.float32)

        # Convert to torch.tensor and wrap in a DictDataset
        disturbance_tensor = torch.tensor(disturbance, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 50, 2)
        dataset = DictDataset({"xn": xn_tensor, 'd': disturbance_tensor}, name="state_data")

        # Wrap in a DataLoader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        return dataloader
        #I want one tensor of [1,50,2]. Find the matching datetime in the dataframes. Take in the next 50 samples. 
        # Has to be tensor with dtype float32
        #Will feed the model with one intial state (from the predition of 50 - prediction horizon)
        #Could rename the return state here to not get confused by "self.state = self.get_state()".

#  --- Create agent instance ---

my_agent = Agent(sim)

# --- Set your callback function (observation and/or actuation) function for a given calling point ---
sim.set_calling_point_and_callback_function(
    calling_point=calling_point_for_callback_fxn,
    observation_function=my_agent.observation_function,  # optional function
    actuation_function= my_agent.actuation_function,  # optional function
    # actuator_names=['air_loop_fan_mass_flow_actuator'], # Ensure the correct actuator name is used
    update_state=True,  # use this callback to update the EMS state
    update_observation_frequency=1,  # linked to observation update
    update_actuation_frequency=1  # linked to actuation update
)

# -- RUN BUILDING SIMULATION --

sim.run_env(ep_weather_path)
sim.reset_state()  # reset when done


# -- Sample Output Data --
output_dfs = sim.get_df(to_csv_file=cvs_output_path)  # LOOK at all the data collected here, custom DFs can be made too, possibility for creating a CSV file (GB in size)

# -- Plot Results --

df_var = output_dfs['var']

Start_period = 0
num_data_points = Start_period + (24 * 6 * 7)

week_data = df_var.iloc[Start_period:num_data_points]


fig, (ax1, ax2, ax5) = plt.subplots(ncols=3, figsize=(12, 12))  # Remember to change the ncols, number ax and figsize.

week_data.plot(y='zn_soft1_temp', ax=ax1, color='red')
week_data.plot(y='zn_finance1_temp', ax=ax2, color='red')
week_data.plot(y='zn_hardware_corri_temp', ax=ax5, color='red')


# ax1, ax2,
# output_dfs['var'].plot(y='zn_soft1_temp', use_index=True, ax=ax1)
ax1.set_title('zn_soft1_temp')

# output_dfs['var'].plot(y='zn_finance1_temp', use_index=True, ax=ax2)
ax2.set_title('zn_finance1_temp')

ax1.set_xlabel('Time')
ax2.set_xlabel('Time')

fig, (ax3, ax4) = plt.subplots(ncols=2, figsize=(12, 12))

week_data.plot(y='air_loop_fan_mass_flow', ax=ax3, color='green')
week_data.plot(y='air_loop_fan_electric_power', ax=ax4, color='blue')

ax3.set_ylabel('Fan Mass Flow Rate (kg/s)', color='green')
ax3.set_xlabel('Time')

ax4.set_ylabel('Fan Electricity power [W]', color='blue')
ax4.set_xlabel('Time')
ax3.legend(loc='upper left')
ax4.legend(loc='upper left')

plt.show()
# Analyze results in "out" folder, DView, or directly from your Python variables and Pandas Dataframes