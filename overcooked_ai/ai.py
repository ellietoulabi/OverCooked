import numpy as np
from pymdp.agent import Agent as PymdpAgent
from collections import defaultdict
import dill
import os

class ActiveInferenceAgent(PymdpAgent):
    def __init__(self, name, actions, state_dim, obs_dim, transition_matrix, observation_matrix, policies, prior_preferences, prior_belief=None, precision=1.0):
        """
        Active Inference Agent for Overcooked using `pymdp`.

        Args:
            actions (list): List of possible actions.
            state_dim (int): Number of possible states.
            obs_dim (int): Number of possible observations.
            transition_matrix (np.ndarray): State transition probabilities.
            observation_matrix (np.ndarray): Observation likelihood.
            policies (list): List of possible policy sequences (action sequences).
            prior_preferences (np.ndarray): Agent's preferences over observations.
            prior_belief (np.ndarray, optional): Initial belief distribution over states.
            precision (float): Precision for softmax policy selection.
        """
        super().__init__(num_states=state_dim, num_obs=obs_dim, num_controls=len(actions))
        
        self.actions = actions
        self.name = name
        self.precision = precision
        self.agent_index = None
        
        # Define agent's generative model parameters
        self.A = observation_matrix  # Observation likelihood matrix
        self.B = transition_matrix  # Transition probability matrix
        self.C = prior_preferences  # Prior preferences over observations
        self.policies = policies  # List of policies

        # Set prior belief over states, if provided
        if prior_belief is not None:
            self.D = prior_belief

    def reset(self):
        """Resets the agent's beliefs and internal state."""
        self.infer_states()
        
    def choose_action(self, observation):
        """
        Select an action based on observation and belief updating.

        Args:
            observation (int): Observed state index.
        
        Returns:
            int: The chosen action.
        """
        # Update beliefs based on observation
        self.infer_states(observation)

        # Compute the expected free energy for each policy and select action
        self.infer_policies()
        
        # Select the action from the best policy based on free energy minimization
        chosen_policy = self.sample_action()  # Sample policy according to free energy

        return self.actions[chosen_policy[0]]

    def update_belief(self, observation, action):
        """
        Update belief based on the observation and chosen action.
        
        Args:
            observation (int): Observed state index.
            action (int): Index of the action taken.
        """
        # Update state belief using action and observation
        self.infer_states(observation, action)

    def action(self, state):
        """
        Returns an action based on current beliefs.

        Args:
            state (OvercookedState): The current state.
        
        Returns:
            tuple: The chosen action and belief info.
        """
        observation = self.state_representation(state)
        action = self.choose_action(observation)
        self.last_action = action
        return action, {"belief": self.qs}

    def save(self, path):
        """Saves the model parameters and belief to a file."""
        if not os.path.exists(path):
            os.makedirs(path)
        pickle_path = os.path.join(path, f"{self.name}_model.pkl")
        with open(pickle_path, "wb") as f:
            dill.dump({"A": self.A, "B": self.B, "C": self.C, "D": self.D, "policies": self.policies}, f)

    @classmethod
    def load(cls, path, actions, state_dim, obs_dim, precision=1.0):
        """Loads saved model parameters and initializes a new agent."""
        with open(path, "rb") as f:
            model = dill.load(f)
        agent = cls(
            name="LoadedAgent",
            actions=actions,
            state_dim=state_dim,
            obs_dim=obs_dim,
            transition_matrix=model["B"],
            observation_matrix=model["A"],
            policies=model["policies"],
            prior_preferences=model["C"],
            prior_belief=model.get("D"),
            precision=precision
        )
        return agent

    def state_representation(self, state):
        """Encodes the environment state into an observation index (simplified)."""
        # Example of a simplified observation mapping; customize as needed for Overcooked
        return hash(state) % len(self.A)  # Hash the state to map into observation indices
