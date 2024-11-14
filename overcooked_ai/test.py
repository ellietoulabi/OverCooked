from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai.src.overcooked_ai_py.agents.agent import Agent,AgentPair,StayAgent
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator, LayoutGenerator
import numpy as np
import random
from collections import defaultdict, deque, namedtuple
from overcooked_ai.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import wandb
from utils import ImageAnimator , game_runs_info
from ql import QlearningAgent
import matplotlib.pyplot as plt



# wandb.init(project="Overcooked_Qlearning")




class CustomRandomAgent(Agent):
    def action(self, state):
        action_probs = np.zeros(Action.NUM_ACTIONS)
        legal_actions = Action.ALL_ACTIONS
        legal_actions_indices = np.array([Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
        action_probs[legal_actions_indices] = 1 / len(legal_actions_indices)
        return Action.sample(action_probs), {"action_probs": action_probs}

    # def actions(self, states, agent_indices):
    #     return [self.action(state) for state in states]
    
    


if __name__ == "__main__":
    
    # agent_pair = AgentPair(CustomRandomAgent(), CustomRandomAgent())
    
    
    actions = Action.ALL_ACTIONS
    
    q_table_1 = defaultdict(lambda: np.zeros(len(actions)))  # Initialize Q-table
    q_table_2 = defaultdict(lambda: np.zeros(len(actions)))  # Initialize Q-table

    
    

    
    for i in range (1):
    
    
        # Initialize two Q-learning agents
        q_agent_1 = QlearningAgent("A1", q_table_1, actions=actions, learning_rate=0.1, discount_factor=0.99, epsilon=0.4)
        q_agent_2 = QlearningAgent("A2", q_table_2, actions=actions, learning_rate=0.1, discount_factor=0.99, epsilon=0.4)

        # Create an AgentPair with the two Q-learning agents
        agent_pair = AgentPair(q_agent_1, q_agent_2)


        mdp_gen_params = {"layout_name": 'cramped_room'}
        mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
        env_params = {"horizon":1000}
        agent_eval = AgentEvaluator(env_params=env_params,mdp_fn=mdp_fn)
        
        # print(agent_eval.env.state)

        # print("\n", agent_eval.env.display_states)
        # exit()
        
        trajectory_random_pair = agent_eval.evaluate_agent_pair(agent_pair,num_games=200)
        # trajectory_random_pair = agent_eval.evaluate_random_pair(num_games=1)
        # print("######EvalDone####")
        # print(len(q_table_1.keys()))
        # print(len(q_table_2.keys()))
        # print("###################")

        # action_probs = [ [q_agent_1.action(state)[1]["action_probs"]]*2 for state in trajectory_random_pair["ep_states"][0]]
        StateVisualizer().display_rendered_trajectory(trajectory_random_pair, ipython_display=False,img_directory_path='./images/',img_extension='.png',img_prefix='r0-')
        
    # animator = ImageAnimator('./images/')
    # animator.create_gif('./gifs/output_imageio.gif', duration=0.05)

   
    print("Random pair rewards: \n",trajectory_random_pair['ep_returns'])
    sr , shr = game_runs_info(trajectory_random_pair,print_details=True)
    agent1_rewards = [reward[0] for reward in shr]
    agent2_rewards = [reward[1] for reward in shr]
    
    
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column

    # Plot Agent 1's rewards on the first subplot
    axs[0].plot(agent1_rewards, label="Agent 1 Reward")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("Reward")
    axs[0].set_title("Agent 1 Rewards Over Time")
    axs[0].legend()

    # Plot Agent 2's rewards on the second subplot
    axs[1].plot(agent2_rewards, label="Agent 2 Reward")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Reward")
    axs[1].set_title("Agent 2 Rewards Over Time")
    axs[1].legend()

    # Display the plots
    plt.tight_layout()
    plt.show()
        
        
        
    
    
    

    # Create a plot for each agent's rewards over time
    # plt.figure(figsize=(10, 5))
    # plt.plot(agent1_rewards, label="Agent 1 Reward")
    # plt.plot(agent2_rewards, label="Agent 2 Reward")

    # # Add labels and title
    # plt.xlabel("Time Step")
    # plt.ylabel("Reward")
    # plt.title("Rewards for Each Agent Over Time")
    # plt.legend()
    # plt.show()    
    
    
    