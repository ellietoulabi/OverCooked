from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai.src.overcooked_ai_py.agents.agent import Agent,AgentPair,StayAgent, GreedyHumanModel
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator, LayoutGenerator
import numpy as np
import random
from collections import defaultdict, deque, namedtuple
from overcooked_ai.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
# import wandb
from utils import ImageAnimator , game_runs_info
from ql import QlearningAgent
from aif import AIFAgent
import matplotlib.pyplot as plt

# from pymdp.agent import Agent as pymdpAgent
from pymdp import utils, maths
# from pymdp.envs import Env

# wandb.init(project="Overcooked_Qlearning")




class CustomRandomAgent(Agent):
    def action(self, state):
        action_probs = np.zeros(Action.NUM_ACTIONS)
        legal_actions = Action.ALL_ACTIONS
        legal_actions_indices = np.array([Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
        action_probs[legal_actions_indices] = 1 / len(legal_actions_indices)
        print(state)
        return Action.sample(action_probs), {"action_probs": action_probs}

    def actions(self, states, agent_indices):
        print('hiiiiiiii');
        return [self.action(state) for state in states]
    



grid_dims = [2, 3]
num_grid_points = np.prod(grid_dims) 
grid = np.arange(num_grid_points).reshape(grid_dims)
it = np.nditer(grid, flags=["multi_index"])
loc_list = []

while not it.finished:
    loc_list.append(it.multi_index)
    it.iternext()

holding_objects = ['EMPTY', 'ONION', 'DISH', 'SOUP']
oven_states = ['EMPTY'] + ['SOUP-'+ str(i) for i in range(1,3)] + ['SOUP-3-' + str(i) for i in range(21)]
# num_states = [num_grid_points, len(Direction.ALL_DIRECTIONS), ]
num_states = [num_grid_points, len(Direction.ALL_DIRECTIONS), len(holding_objects), len(oven_states)]



reward_obs = ['NONE', 'SERVE', 'DISH_PICKUP', 'SOUP_PICKUP', 'ONION_PICKUP', 'DROP_ONION']
# reward_obs = ['NONE', 'SERVE']

num_obs = [num_grid_points, len(Direction.ALL_DIRECTIONS), len(holding_objects), len(oven_states), len(reward_obs)]

A_m_shapes = [ [o_dim] + num_states for o_dim in num_obs] # list of shapes of modality-specific A[m] arrays



A = utils.obj_array_zeros(A_m_shapes) # initialize A array to an object array of all-zero subarrays

# Position
for i in range(num_grid_points):
   A[0][i,i,:,:] = 1


# A[0] = np.tile(np.expand_dims(np.eye(num_grid_points), (-2, -1)), (1, 1, num_states[1], num_states[2], num_states[3]))



#Direction
for i in range(len(Direction.ALL_DIRECTIONS)):
   A[1][i,:,i,:,:] = 1
# A[1] = np.tile(np.expand_dims(np.eye(len(Direction.ALL_DIRECTIONS)), (-2, -1)), (1, 1, num_states[1], num_states[2], num_states[3]))


for i in range(len(holding_objects)):
    A[2][i,:,:,i,:] = 1




for i in range(len(oven_states)):
    A[3][i,:,:,:,i] = 1

A[4][reward_obs.index('ONION_PICKUP'), 0, Direction.ALL_DIRECTIONS.index(Direction.WEST), 0, :] = 1
A[4][reward_obs.index('ONION_PICKUP'), 2, Direction.ALL_DIRECTIONS.index(Direction.EAST), 0, :] = 1
A[4][reward_obs.index('SOUP_PICKUP'), 1, Direction.ALL_DIRECTIONS.index(Direction.NORTH), 2, 23] = 1
A[4][reward_obs.index('DROP_ONION'), 1, Direction.ALL_DIRECTIONS.index(Direction.NORTH), 1, 0] = 1
A[4][reward_obs.index('DROP_ONION'), 1, Direction.ALL_DIRECTIONS.index(Direction.NORTH), 1, 1] = 1
A[4][reward_obs.index('DROP_ONION'), 1, Direction.ALL_DIRECTIONS.index(Direction.NORTH), 1, 2] = 1
A[4][reward_obs.index('DISH_PICKUP'), 3, Direction.ALL_DIRECTIONS.index(Direction.SOUTH), 0, :] = 1
A[4][reward_obs.index('SERVE'), 5, Direction.ALL_DIRECTIONS.index(Direction.SOUTH), 3, :] = 1
A[4][reward_obs.index('NONE'), 0, 0, :, :] = 1
A[4][reward_obs.index('NONE'), 0, 1, :, :] = 1
A[4][reward_obs.index('NONE'), 0, 2, :, :] = 1
A[4][reward_obs.index('NONE'), 1, 1, :, :] = 1
A[4][reward_obs.index('NONE'), 1, 2, :, :] = 1
A[4][reward_obs.index('NONE'), 1, 3, :, :] = 1
A[4][reward_obs.index('NONE'), 2, 0, :, :] = 1
A[4][reward_obs.index('NONE'), 2, 1, :, :] = 1
A[4][reward_obs.index('NONE'), 2, 3, :, :] = 1
A[4][reward_obs.index('NONE'), 3, 0, :, :] = 1
A[4][reward_obs.index('NONE'), 3, 2, :, :] = 1
A[4][reward_obs.index('NONE'), 3, 3, :, :] = 1
A[4][reward_obs.index('NONE'), 4, :, :, :] = 1
A[4][reward_obs.index('NONE'), 5, 0, :, :] = 1
A[4][reward_obs.index('NONE'), 5, 2, :, :] = 1
A[4][reward_obs.index('NONE'), 5, 3, :, :] = 1

A[4][reward_obs.index('NONE'), 0, 3, 1, :] = 1
A[4][reward_obs.index('NONE'), 0, 3, 2, :] = 1
A[4][reward_obs.index('NONE'), 0, 3, 3, :] = 1
A[4][reward_obs.index('NONE'), 1, 0, 0, :] = 1
A[4][reward_obs.index('NONE'), 1, 0, 1, 3:] = 1
A[4][reward_obs.index('NONE'), 1, 0, 2, :23] = 1
A[4][reward_obs.index('NONE'), 1, 0, 3, :] = 1
A[4][reward_obs.index('NONE'), 2, 2, 1, :] = 1
A[4][reward_obs.index('NONE'), 2, 2, 2, :] = 1
A[4][reward_obs.index('NONE'), 2, 2, 3, :] = 1
A[4][reward_obs.index('NONE'), 3, 1, 1, :] = 1
A[4][reward_obs.index('NONE'), 3, 1, 2, :] = 1
A[4][reward_obs.index('NONE'), 3, 1, 3, :] = 1
A[4][reward_obs.index('NONE'), 5, 1, 0, :] = 1
A[4][reward_obs.index('NONE'), 5, 1, 1, :] = 1
A[4][reward_obs.index('NONE'), 5, 1, 2, :] = 1







num_controls = [Action.NUM_ACTIONS, Action.NUM_ACTIONS, Action.NUM_ACTIONS, Action.NUM_ACTIONS]
control_fac_idx = [0,1,2,3]
# initialize the shapes of each sub-array `B[f]`
B_f_shapes = [ [ns, ns, num_controls[f]] for f, ns in enumerate(num_states)]

# create the `B` array and fill it out
B = utils.obj_array_zeros(B_f_shapes)

for action_id, action_label in enumerate(Action.ALL_ACTIONS):

  for curr_state, grid_location in enumerate(loc_list):

    y, x = grid_location

    if action_label == Direction.NORTH:
      next_y = y - 1 if y > 0 else y 
      next_x = x
    elif action_label == Direction.SOUTH:
      next_y = y + 1 if y < (grid_dims[0]-1) else y 
      next_x = x
    elif action_label == Direction.WEST:
      next_x = x - 1 if x > 0 else x 
      next_y = y
    elif action_label == Direction.EAST:
      next_x = x + 1 if x < (grid_dims[1]-1) else x 
      next_y = y
    elif action_label == Action.STAY or action_label == Action.INTERACT:
      next_x = x
      next_y = y

    new_location = (next_y, next_x)
    next_state = loc_list.index(new_location)
    B[0][next_state, curr_state, action_id] = 1.0


#Direction
for curr_direction_id, curr_direction in enumerate(Direction.ALL_DIRECTIONS):
    for action_id, action_label in enumerate(Action.ALL_ACTIONS):   
        if action_label != Action.STAY and action_label != Action.INTERACT:
            next_direction = action_label
        else:
            next_direction = curr_direction
        

        next_direction_id = Direction.ALL_DIRECTIONS.index(next_direction)
        B[1][next_direction_id,curr_direction_id,action_id] = 1

#holding Objects
for action_id, action_label in enumerate(Action.ALL_ACTIONS):
    for current_id, current_object in enumerate(holding_objects):
        if action_label != Action.INTERACT:
            B[2][current_id,current_id,action_id] = 1
        else:
            values = {
               'EMPTY': np.array([5, 0.54, 0.3, 0.3]),
               'ONION': np.array([1.5, 4.5, 0, 0]),
               'DISH': np.array([1.25, 0, 4.7, 0.04]),
               'SOUP': np.array([1.5, 0, 0, 4.5]),
            }
            probs = maths.softmax(values[current_object])
            for i in range(len(holding_objects)):
                B[2][i, current_id, action_id] = probs[i]


# Oven 
B[3][
   oven_states.index('EMPTY'),
   oven_states.index('SOUP-3-20'),
   Action.ALL_ACTIONS.index(Action.INTERACT)
] = 1
B[3][
   oven_states.index('SOUP-1'),
   oven_states.index('EMPTY'),
   Action.ALL_ACTIONS.index(Action.INTERACT)
] = 1
B[3][
   oven_states.index('SOUP-2'),
   oven_states.index('SOUP-1'),
   Action.ALL_ACTIONS.index(Action.INTERACT)
] = 1
B[3][
   oven_states.index('SOUP-3-0'),
   oven_states.index('SOUP-2'),
   Action.ALL_ACTIONS.index(Action.INTERACT)
] = 1
for i in range(20):
    B[3][
        oven_states.index(f'SOUP-3-{i+1}'),
        oven_states.index(f'SOUP-3-{i}'),
        :
    ] = 1

for i in range(5):
    B[3][:,:,i] = 1/24   
# B[3][0:25,0,0] = 1/24
# B[3][0:25,0,1] = 1/24
# B[3][0:25,0,2] = 1/24
# B[3][0:25,0,3] = 1/24
# B[3][0:25,0,4] = 1/24
# B[3][0:25,1,0] = 1/24


# for i in range(6):
#     B[3][0,0,i] = 1
# oven_states = ['EMPTY'] + ['SOUP-'+ str(i) for i in range(1,3)] + ['SOUP-3-' + str(i) for i in range(21)]

# print(B[3][0,0,1])
# print(B[3][1,0,0])
# print(B[3][2,0,0])
# print(B[3][3,0,0])
# print(B[3][4,0,0])
# print(B[3][5,0,0])
# with np.printoptions(threshold=np.inf):
    
#     print(B[3].sum(axis=0))
# exit()

C = utils.obj_array_zeros(num_obs)
# C[4][0] = 0
C[4][1] = 20 #[TODO] should this be c4??
# C[4][2] = 3
# C[4][3] = 5
# C[4][4] = 3
# C[4][5] = 3



D = utils.obj_array_uniform(num_states)
D[0] = utils.onehot(loc_list.index((1,0)), num_grid_points)
D[1] = utils.onehot(Direction.ALL_DIRECTIONS.index(Direction.NORTH), len(Direction.ALL_DIRECTIONS))
D[2] = utils.onehot(0, len(holding_objects))
D[3] = utils.onehot(0, len(oven_states))





if __name__ == "__main__":
    
    # agent_pair = AgentPair(CustomRandomAgent(), CustomRandomAgent())
    
    
    actions = Action.ALL_ACTIONS
    
    q_table_1 = defaultdict(lambda: np.zeros(len(actions)))  # Initialize Q-table
    q_table_2 = defaultdict(lambda: np.zeros(len(actions)))  # Initialize Q-table


    
    for i in range (1):
    
    
        # Initialize two Q-learning agents
        q_agent_1 = QlearningAgent("QL1",q_table_1, actions=actions, learning_rate=0.1, discount_factor=0.99, epsilon=1)
        q_agent_2 = QlearningAgent("QL2", q_table_2, actions=actions, learning_rate=0.1, discount_factor=0.99, epsilon=1)
        aif_agent = AIFAgent(actions, A, B, C, D, control_fac_idx, policy_len=8)
        # Create an AgentPair with the two Q-learning agents
        agent_pair = AgentPair(aif_agent, q_agent_2)


        mdp_gen_params = {"layout_name": 'cramped_room'}
        mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
        env_params = {"horizon":5000}
        agent_eval = AgentEvaluator(env_params=env_params,mdp_fn=mdp_fn)
        
        trajectory_random_pair = agent_eval.evaluate_agent_pair(agent_pair,num_games=1)
        # trajectory_random_pair = agent_eval.evaluate_random_pair(num_games=1000)
        print("######EvalDone####")
        # action_probs = [ [q_agent_1.action(state)[1]["action_probs"]]*2 for state in trajectory_random_pair["ep_states"][0]]
        StateVisualizer().display_rendered_trajectory(trajectory_random_pair, ipython_display=False,img_directory_path='./images/',img_extension='.png',img_prefix='r0-')
        
    # animator = ImageAnimator('./images/')
    # animator.create_gif('./gifs/output_imageio.gif', duration=0.05)

   
    print("Random pair rewards: \n",trajectory_random_pair['ep_returns'])
    sr , shr = game_runs_info(trajectory_random_pair,print_details=True)
    agent1_rewards = [reward[0] for reward in shr]
    agent2_rewards = [reward[1] for reward in shr]

    # Create a plot for each agent's rewards over time
    plt.figure(figsize=(10, 5))
    plt.plot(agent1_rewards, label="Agent 1 Reward")
    plt.plot(agent2_rewards, label="Agent 2 Reward")

    # Add labels and title
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.title("Rewards for Each Agent Over Time")
    plt.legend()
    plt.show()    
    
    
    