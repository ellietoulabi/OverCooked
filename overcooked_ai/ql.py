from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai.src.overcooked_ai_py.agents.agent import Agent,AgentPair,StayAgent
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator, LayoutGenerator
import numpy as np
from collections import defaultdict
import random
import dill
import os


class QlearningAgent(Agent):
    def __init__(self, name, qtable, actions, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1):
        """
        Q-Learning Agent for the Overcooked 'cramped_room' layout.

        Args:
            actions (list): List of possible actions ('up', 'down', 'left', 'right', 'stay', 'interact').
            learning_rate (float): Rate at which the agent learns from new experiences.
            discount_factor (float): Factor by which future rewards are discounted.
            epsilon (float): Initial exploration rate for epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays after each episode.
            min_epsilon (float): Minimum value of epsilon for exploration.
        """
        super().__init__()
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        # self.q_table = defaultdict(lambda: np.zeros(len(actions)))  # Initialize Q-table
        self.q_table = qtable
        self.agent_index = None
        self.name = name

    def reset(self):
        """Resets agent-specific attributes."""
        
        super().reset()  # Call the base reset to clear trajectory-specific attributes
        self.last_state = None
        self.last_action = None

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
        return state

    def choose_action(self, state):
        """
        Select an action based on epsilon-greedy policy.

        Args:
            state (tuple): The simplified state representation.

        Returns:
            int: The chosen action.
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)  # Exploration: random action
        else:
            best_action_index = np.argmax(self.q_table[state])  # Exploitation
            return self.actions[best_action_index]

    def update(self,info):
        """
        Update the Q-value for the given state-action pair based on observed reward and next state.

        Args:
            state (tuple): The current state.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_state (tuple): The next state after taking the action.
        """
        
        # (s_tp1, s_t, a_t, r_t, done, info)       
        
        action_idx = self.actions.index(info[2][self.agent_index])
        next_state = info[0]
        best_next_action = np.argmax(self.q_table[next_state])  # Get best action for next state
        state = info[1]
        sparse_r= info[5]['sparse_r_by_agent'][self.agent_index]
        shaped_r = info[5]['shaped_r_by_agent'][self.agent_index]
        
        td_target = shaped_r + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action_idx]
        
        # Update Q-value using temporal-difference error
        self.q_table[state][action_idx] += self.learning_rate * td_error
        
        if shaped_r > 0:
            print(state)
            
            
        
        
        
        
        
    # def action(self, state):
    #     action_probs = np.zeros(Action.NUM_ACTIONS)
    #     legal_actions = Action.ALL_ACTIONS
    #     legal_actions_indices = np.array([Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
    #     action_probs[legal_actions_indices] = 1 / len(legal_actions_indices)
    #     return Action.sample(action_probs), {"action_probs": action_probs}

    def actions(self, states, agent_indices):
        # print(self.name)
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
        return action, {"q_values": self.q_table[simplified_state]}

    def observe_transition(self, state, action, reward, next_state):
        """
        Stores a transition and updates the Q-values.

        Args:
            state (OvercookedState): The state before taking the action.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (OvercookedState): The state after taking the action.
        """
        # Simplify state representations
        simplified_state = self.state_representation(state)
        simplified_next_state = self.state_representation(next_state)
        
        # Update Q-table
        self.update(simplified_state, action, reward, simplified_next_state)
        
        # Decay epsilon for exploration-exploitation balance
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

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


 