from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai.src.overcooked_ai_py.agents.agent import Agent,AgentPair,StayAgent
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator, LayoutGenerator
import numpy as np
from collections import defaultdict
import random
import dill
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pymdp.agent import Agent as pymdpAgent
from pymdp import utils, maths
from pymdp.envs import Env

import wandb



# class CoockingEnv(Env):

#     UP = 0
#     RIGHT = 1
#     DOWN = 2
#     LEFT = 3
#     STAY = 4
#     INTERACT = 5

#     factor_names = ["POSITION", ""] #NOTE: only learning position?
     

#     CONTROL_NAMES = ["UP", "RIGHT", "DOWN", "LEFT", "STAY", "INTERACT"]

#     def __init__(self, shape=[2, 2], init_state=None):
#         """
#         Initialization function for 2-D grid world

#         Parameters
#         ----------
#         shape: ``list`` of ``int``, where ``len(shape) == 2``
#             The dimensions of the grid world, stored as a list of integers, storing the discrete dimensions of the Y (vertical) and X (horizontal) spatial dimensions, respectively.
#         init_state: ``int`` or ``None``
#             Initial state of the environment, i.e. the location of the agent in grid world. If not ``None``, must be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the initial location of the agent in grid world.
#             If ``None``, then an initial location will be randomly sampled from the grid.
#         """
        
#         self.shape = shape
#         self.n_states = np.prod(shape) # product of array elements over a specified axis
#         self.n_observations = self.n_states #NOTE: only observe location?!
#         self.n_control = 5 #NOTE: what about the 6th control? 5 because intract doesn't change location?
#         self.max_y = shape[0]
#         self.max_x = shape[1]
#         self._build()
#         self.set_init_state(init_state)
#         self.last_action = None

#     def reset(self, init_state=None):
#         """
#         Reset the state of the 2-D grid world. In other words, resets the location of the agent, and wipes the current action.

#         Parameters
#         ----------
#         init_state: ``int`` or ``None``
#             Initial state of the environment, i.e. the location of the agent in grid world. If not ``None``, must be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the initial location of the agent in grid world.
#             If ``None``, then an initial location will be randomly sampled from the grid.

#         Returns
#         ----------
#         self.state: ``int``
#             The current state of the environment, i.e. the location of the agent in grid world. Will be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the location of the agent in grid world.
#         """
#         self.set_init_state(init_state)
#         self.last_action = None
#         return self.state

#     def set_state(self, state):
#         """
#         Sets the state of the 2-D grid world.

#         Parameters
#         ----------
#         state: ``int`` or ``None``
#             State of the environment, i.e. the location of the agent in grid world. If not ``None``, must be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the location of the agent in grid world.
#             If ``None``, then a location will be randomly sampled from the grid.

#         Returns
#         ----------
#         self.state: ``int``
#             The current state of the environment, i.e. the location of the agent in grid world. Will be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the location of the agent in grid world.
#         """
#         self.state = state
#         return state

#     def step(self, action):
#         """
#         Updates the state of the environment, i.e. the location of the agent, using an action index that corresponds to one of the 5 possible moves.

#         Parameters
#         ----------
#         action: ``int`` 
#             Action index that refers to which of the 5 actions the agent will take. Actions are, in order: "UP", "RIGHT", "DOWN", "LEFT", "STAY". NOTE: Why exclude intract?

#         Returns
#         ----------
#         state: ``int``
#             The new, updated state of the environment, i.e. the location of the agent in grid world after the action has been made. Will be discrete index in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the location of the agent in grid world.
#         """
#         state = self.P[self.state][action]
#         self.state = state
#         self.last_action = action
#         return state

#     def render(self, title=None):
#         """
#         Creates a heatmap showing the current position of the agent in the grid world.

#         Parameters
#         ----------
#         title: ``str`` or ``None``
#             Optional title for the heatmap.
#         """
#         values = np.zeros(self.shape)
#         values[self.position] = 1.0
#         _, ax = plt.subplots(figsize=(3, 3))
#         if self.shape[0] == 1 or self.shape[1] == 1:
#             ax.imshow(values, cmap="OrRd")
#         else:
#             _ = sns.heatmap(values, cmap="OrRd", linewidth=2.5, cbar=False, ax=ax)
#         plt.xticks(range(self.shape[1]))
#         plt.yticks(range(self.shape[0]))
#         if title != None:
#             plt.title(title)
#         plt.show()

#     def set_init_state(self, init_state=None):
#         if init_state != None:
#             if init_state > (self.n_states - 1) or init_state < 0:
#                 raise ValueError("`init_state` is greater than number of states")
#             if not isinstance(init_state, (int, float)):
#                 raise ValueError("`init_state` must be [int/float]")
#             self.init_state = int(init_state)
#         else:
#             self.init_state = np.random.randint(0, self.n_states)
#         self.state = self.init_state

#     def _build(self):
#         P = {} # a dictionary where each state (s) is mapped to a set of actions (a).
#         grid = np.arange(self.n_states).reshape(self.shape)
#         it = np.nditer(grid, flags=["multi_index"]) #iterate through the grid efficiently while keeping track of both the current state index (s) and its multi-dimensional indices (y, x)

#         while not it.finished:
#             s = it.iterindex
#             y, x = it.multi_index
#             P[s] = {a: [] for a in range(self.n_control)}

#             next_up = s if y == 0 else s - self.max_x
#             next_right = s if x == (self.max_x - 1) else s + 1
#             next_down = s if y == (self.max_y - 1) else s + self.max_x
#             next_left = s if x == 0 else s - 1
#             next_stay = s

#             P[s][self.UP] = next_up
#             P[s][self.RIGHT] = next_right
#             P[s][self.DOWN] = next_down
#             P[s][self.LEFT] = next_left
#             P[s][self.STAY] = next_stay

#             it.iternext()

#         self.P = P

#     def get_init_state_dist(self, init_state=None):
#         init_state_dist = np.zeros(self.n_states)
#         if init_state == None:
#             init_state_dist[self.init_state] = 1.0
#         else:
#             init_state_dist[init_state] = 1.0

#     def get_transition_dist(self):
#         B = np.zeros([self.n_states, self.n_states, self.n_control])
#         for s in range(self.n_states):
#             for a in range(self.n_control):
#                 ns = int(self.P[s][a])
#                 B[ns, s, a] = 1
#         return B

#     def get_likelihood_dist(self):
#         A = np.eye(self.n_observations, self.n_states)
#         return A

#     def sample_action(self):
#         return np.random.randint(self.n_control)

#     @property
#     def position(self):
#         """ @TODO might be wrong w.r.t (x & y) """
#         return np.unravel_index(np.array(self.state), self.shape)




### States
# grid_dims = [4, 5]
# num_grid_points = np.prod(grid_dims) 
# grid = np.arange(num_grid_points).reshape(grid_dims)
# it = np.nditer(grid, flags=["multi_index"])
# loc_list = []

# while not it.finished:
#     loc_list.append(it.multi_index)
#     it.iternext()

# num_states = [num_grid_points]
# num_obs = [num_grid_points]

# A_m_shapes = [ [o_dim] + num_states for o_dim in num_obs] # list of shapes of modality-specific A[m] arrays
# A = utils.obj_array_zeros(A_m_shapes) # initialize A array to an object array of all-zero subarrays

# A[0] = np.eye(num_grid_points)

# num_controls = [Action.NUM_ACTIONS]
# control_fac_idx = [0]
# # initialize the shapes of each sub-array `B[f]`
# B_f_shapes = [ [ns, ns, num_controls[f]] for f, ns in enumerate(num_states)]

# # create the `B` array and fill it out
# B = utils.obj_array_zeros(B_f_shapes)

# for action_id, action_label in enumerate(Action.ALL_ACTIONS):

#   for curr_state, grid_location in enumerate(loc_list):

#     y, x = grid_location

#     if action_label == Direction.NORTH:
#       next_y = y - 1 if y > 1 else y 
#       next_x = x
#     elif action_label == Direction.SOUTH:
#       next_y = y + 1 if y < (grid_dims[0]-2) else y 
#       next_x = x
#     elif action_label == Direction.WEST:
#       next_x = x - 1 if x > 1 else x 
#       next_y = y
#     elif action_label == Direction.EAST:
#       next_x = x + 1 if x < (grid_dims[1]-2) else x 
#       next_y = y
#     elif action_label == Action.STAY or action_label == Action.INTERACT:
#       next_x = x
#       next_y = y

#     new_location = (next_y, next_x)
#     next_state = loc_list.index(new_location)
#     B[0][next_state, curr_state, action_id] = 1.0


# C = utils.obj_array_zeros(num_obs)
# C[0][13] = 10

# D = utils.obj_array_uniform(num_states)
# D[0] = utils.onehot(loc_list.index((2,1)), num_grid_points)
# agent = pymdpAgent(A=A, B=B, C=C, D=D, control_fac_idx=control_fac_idx
#                    ,policy_len = 1
#                    )

class AIFAgent(Agent):
    def __init__(self, actions, A, B, C, D, control_fac_idx, policy_len):
        """
        AIFAgent class for an agent that learns to play Overcooked using the AIF algorithm.
        Parameters
        ----------
        actions : list
            List of possible actions the agent can take.
        A : numpy.ndarray
            Observation likelihood matrix.
        B : numpy.ndarray
            Transition likelihood matrix.
        C : numpy.ndarray
            Prior preferences over observations.
        D : numpy.ndarray
            Prior beliefs over states.
        control_fac_idx : int
            Index of the control factor.
        policy_len : int
            Length of the policy.
        """
        super().__init__()
        self.agent_index = None
        self.actions = actions
        self.aif_agent = pymdpAgent(A=A, B=B, C=C, D=D, control_fac_idx=control_fac_idx ,policy_len = policy_len, inference_algo="MMP")
        grid_dims = [2, 3]
        num_grid_points = np.prod(grid_dims) 
        grid = np.arange(num_grid_points).reshape(grid_dims)
        it = np.nditer(grid, flags=["multi_index"])
        self.loc_list = []
        self.C = C

        while not it.finished:
            self.loc_list.append(it.multi_index)
            it.iternext()

    def reset(self):
        """Resets agent-specific attributes."""
        
        super().reset()  # Call the base reset to clear trajectory-specific attributes
        self.last_state = None
        self.last_action = None

    def determineOvenState(self, objects):
        oven_states = ['EMPTY'] + ['SOUP-'+ str(i) for i in range(1,3)] + ['SOUP-3-' + str(i) for i in range(21)]
        oven_state_id = 0
        for pos, obj in objects.items():
            print("***",obj.__dict__)
            
            if obj.name == 'soup':
                num_onions = len(list(filter(lambda x : x == 'onion', obj.ingredients)))
                if num_onions == 0:
                    oven_state_id = 0
                elif num_onions < 3:
                    oven_state_id = oven_states.index(f'SOUP-{num_onions}')
                else:
                    oven_state_id = oven_states.index(f'SOUP-{num_onions}-{obj.cooking_tick}')
                return oven_state_id
        return oven_state_id
                    
    def determineReward(self, state, oven_state_id):
        idx = 0
        hold_index = 0
        agent = state.players[self.agent_index]
        if agent.held_object:
            hold_index = ['EMPTY', 'ONION', 'DISH', 'SOUP'].index(agent.held_object.name.upper())
        pos = agent.position
        if pos[0] == 2 and pos[1] == 3 and agent.orientation == Direction.SOUTH and self.last_action == Action.INTERACT and hold_index == 3:
            idx = 1
        elif pos[0] == 2 and pos[1] == 1 and agent.orientation == Direction.SOUTH and self.last_action == Action.INTERACT and hold_index == 0:
            idx = 2
        elif pos[0] == 1 and pos[1] == 2 and agent.orientation == Direction.NORTH and self.last_action == Action.INTERACT and hold_index == 2:
            idx = 3
        elif ((pos[0] == 1 and pos[1] == 1 and agent.orientation == Direction.WEST) or (pos[0] == 1 and pos[1] == 3 and agent.orientation == Direction.EAST)) and self.last_action == Action.INTERACT and hold_index == 0:
            idx = 4
        elif pos[0] == 1 and pos[1] == 2 and agent.orientation == Direction.NORTH and hold_index == 1 and self.last_action == Action.INTERACT and oven_state_id <= 3:
            idx=5
        
        return idx

    def state_representation(self, state):
        """
        Simplifies the state representation for the 'cramped_room' layout.

        Args:
            state (OvercookedState): The current state of the environment.

        Returns:
            tuple: A simplified representation of the state.
        """
        # agent_pos = state.players[self.agent_index].position  # Agent's position
        # pot_state = tuple(state.get_pot_states())  # Tuple of pot states (filled, cooking, empty, etc.)
        # return (agent_pos, pot_state)
        agent = state.players[self.agent_index]
        agent_pos = agent.position
        if agent.held_object:
            hold_index = ['EMPTY', 'ONION', 'DISH', 'SOUP'].index(agent.held_object.name.upper())
        else:
            hold_index = 0
        oven_state_id = self.determineOvenState(state.objects)
        reward_id = self.determineReward(state, oven_state_id)
        # print([ #NOTE: log
        #     self.loc_list.index(tuple([agent_pos[1]-1, agent_pos[0]-1])),
        #     Direction.ALL_DIRECTIONS.index(agent.orientation),
        #     hold_index,
        #     oven_state_id,
        #     reward_id
        # ])
        print("###",self.C[4][reward_id])

        wandb.log({"aif_reward": self.C[4][reward_id]})
        print(f"aif_reward:{self.C[4][reward_id]}")

        return [
            self.loc_list.index(tuple([agent_pos[1]-1, agent_pos[0]-1])),
            Direction.ALL_DIRECTIONS.index(agent.orientation),
            hold_index,
            oven_state_id,
            reward_id
        ]

    def choose_action(self, obs):
        """
        Select an action based on epsilon-greedy policy.

        Args:
            state (tuple): The simplified state representation.

        Returns:
            int: The chosen action.
        """
        import datetime
        # print("infer state starts: ",datetime.datetime.now())
        elstart = datetime.datetime.now()
        #TODO: time fix :)
        
        self.aif_agent.infer_states(obs)
        
        elend= datetime.datetime.now()
        # print("Infer state Duration:", elend - elstart)
        
        elstart = datetime.datetime.now()
   

        self.aif_agent.infer_policies()
       
        elend= datetime.datetime.now()
        # print("Infer policy Duration:", elend - elstart)
        elstart = datetime.datetime.now()

        chosen_action_id = self.aif_agent.sample_action()
        elend= datetime.datetime.now()
        # print("Sample Action Duration:", elend - elstart)
        # movement_id = int(chosen_action_id[0])#NOTE:Removed
        movement_id = int(chosen_action_id)
        
        choice_action = self.actions[movement_id]

        print(f'choosed action: {choice_action}')
        # print("C: ",self.aif_agent.C)
        return choice_action
        if random.random() < self.epsilon:
            return random.choice(self.actions)  # Exploration: random action
        else:
            best_action_index = np.argmax(self.q_table[state])  # Exploitation
            return self.actions[best_action_index]

    def update(self, info):
        """
        Update the Q-value for the given state-action pair based on observed reward and next state.

        Args:
            state (tuple): The current state.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_state (tuple): The next state after taking the action.
        """
        # action_idx = self.actions.index(action)
        # best_next_action = np.argmax(self.q_table[next_state])  # Get best action for next state
        # td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        # td_error = td_target - self.q_table[state][action_idx]
        
        # # Update Q-value using temporal-difference error
        # self.q_table[state][action_idx] += self.learning_rate * td_error
        
        
        
        
        
    # def action(self, state):
    #     action_probs = np.zeros(Action.NUM_ACTIONS)
    #     legal_actions = Action.ALL_ACTIONS
    #     legal_actions_indices = np.array([Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
    #     action_probs[legal_actions_indices] = 1 / len(legal_actions_indices)
    #     return Action.sample(action_probs), {"action_probs": action_probs}

    def actions(self, states, agent_indices):
        return [self.action(state) for state in states]
    def action(self, state):
        """
        Returns an action based on the current state and updates the agent's internal state.

        Args:
            state (OvercookedState): The current state in Overcooked.

        Returns:
            tuple: The selected action and Q-value info.
        """
        # Simplify state representation for Q-learning
        simplified_state = self.state_representation(state)
        
        # Choose an action
        action = self.choose_action(simplified_state)
       
        
        # Update internal state for learning
        self.last_state = simplified_state
        self.last_action = action
        
        # Return the action and Q-values as action info for logging
        return action, {}

    
    # def actions(self, states, agent_indices):
    #     """
    #     Choose actions for multiple states in a batch.

    #     Args:
    #         states (list): List of OvercookedStates for each agent.
    #         agent_indices (list): List of indices indicating which agent in the state list.

    #     Returns:
    #         list: List of (action, action_info) tuples for each state.
    #     """
    #     actions_and_infos = []
    #     for state, agent_idx in zip(states, agent_indices):
    #         self.set_agent_index(agent_idx)  # Set correct agent index
    #         actions_and_infos.append(self.action(state))
    #     return actions_and_infos

    def save(self, path):
        """Saves the Q-table to a file for future use."""
        if not os.path.exists(path):
            os.makedirs(path)
        pickle_path = os.path.join(path, self.agent_file_name)
        with open(pickle_path, "wb") as f:
            dill.dump(dict(self.q_table), f)

    @classmethod
    def load(cls, path, actions):
        """Loads a saved Q-table from a file and initializes a new agent."""
        with open(path, "rb") as f:
            q_table = dill.load(f)
        agent = cls(actions)
        agent.q_table = defaultdict(lambda: np.zeros(len(actions)), q_table)
        return agent


 