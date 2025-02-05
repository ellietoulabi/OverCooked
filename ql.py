from overcooked_ai.src.overcooked_ai_py.agents.agent import Agent
from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, PlayerState, ObjectState, SoupState
from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action, Direction
import random
import numpy as np
import wandb

class QLearningAgent(Agent):
    def __init__(self, q_table, actions, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1, seed=42):
        super().__init__()
        self.all_actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = q_table
        np.random.seed(seed)

    def reset(self):
        super().reset()
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.1



    def _argmax_rand(self, a):
        indices = np.where(np.array(a) == np.max(a))[0]
        return np.random.choice(indices)
    
    def _parse_state(self, state: OvercookedState):
        player_state = state.players[self.agent_index] # type: PlayerState
        
        agent_position_state = (player_state.position[0]-1) + (player_state.position[1]-1) * 3
        opponent_position_state = (state.players[1].position[0]-1) + (state.players[1].position[1]-1) * 3

        orientation_state = Direction.DIRECTION_TO_INDEX[player_state.orientation]

        if player_state.has_object():
            held_object = player_state.get_object() # type: ObjectState

            # match held_object.name:
            #     case 'soup':
            #         held_object_state = 3
            #     case 'onion':
            #         held_object_state = 2
            #     case 'dish':
            #         held_object_state = 1
            
            if held_object.name == 'soup':
                held_object_state = 3
            elif held_object.name == 'onion':
                held_object_state = 2
            elif held_object.name == 'dish':
                held_object_state = 1
        else:
            held_object_state = 0
        
        oven_state = 0
        for pos, obj in state.objects.items():
            if isinstance(obj, SoupState):
                if obj.cook_time_remaining == 0:
                    oven_state = 2
                else:
                    oven_state = 1

        
        return (agent_position_state, opponent_position_state, orientation_state, held_object_state, oven_state)
    
    @staticmethod
    def generate_q_table(actions):
        states = {
            'agent_position': list(range(6)),
            'opponent_position': list(range(6)),
            'orientation': list(range(4)),
            'hold_object': ['NONE', 'DISH', 'ONION', 'SOUP'],
            'oven_state': ['IDLE', 'BUSY', 'READY'],
        }
        state_shape = list(map(lambda x: len(x), states.values()))
        return np.zeros( ( len(actions), *state_shape ) )

    def action(self, state):
        parsed_state = self._parse_state(state)

        if random.random() < self.epsilon:
            action = self.all_actions[np.random.choice(len(self.all_actions))]
        else:
            best_action_index = self._argmax_rand(self.q_table[:, *parsed_state])
            action = self.all_actions[best_action_index]
        
        return action, {}
    
    def update(self, info_dict):
        """
        Updates Q table

        Args:
        info_dict (dict): A dictionary with the following keys and value types:
            - 'actions' (list[Action]): List of actions for all agents.
            - 'prev_state' (OvercookedState): State before the actions.
            - 'next_state' (OvercookedState): State after the actions.
            - 'total_sparse_reward' (int): Sum of sparse reward for both of agents
            - 'done' (bool): True if game is over, either by the end of time steps or terminate of mdp which is always False
            - 'sparse_r_by_agent' (list[int]):  
            - 'shaped_r_by_agent' (list[int]): 
        Returns:
        None
        """
        choice_action_index = self.all_actions.index(info_dict['actions'][self.agent_index])
        state = self._parse_state(info_dict['prev_state'])
        new_state = self._parse_state(info_dict['next_state'])
        s_reward = info_dict['sparse_r_by_agent'][self.agent_index]
        sh_reward = info_dict['shaped_r_by_agent'][self.agent_index]
        wandb.log({f"agent{self.agent_index}_sparse_reward": s_reward})
        wandb.log({f"agent{self.agent_index}_shaped_reward": sh_reward})
        reward = s_reward
        self.q_table[choice_action_index, *state] = (1-self.learning_rate)*self.q_table[choice_action_index, *state] + self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[:, *new_state]) )
        