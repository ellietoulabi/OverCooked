from overcooked_ai.src.overcooked_ai_py.agents.agent import AgentPair, StayAgent
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action
from overcooked_ai.src.overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import matplotlib.pyplot as plt
import numpy as np
import time


from ql import QLearningAgent
# from aif import ActiveInferenceAgent
from utils import ImageAnimator , game_runs_info, plot_rewards
import wandb

num_games = 100
horizon = 500



wandb.init(
    # set the wandb project where this run will be logged
    project="QLQLOverCooked",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.1, 
    "discount_factor": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.99, 
    "min_epsilon": 0.1, 
    "seed": 42,
    "env": "Overcooked(cramped_room)",
    "num_games": num_games,
    "horizon": horizon
    }
)

wandb.run.log_code(".")

stay_agent = StayAgent()

qtable = QLearningAgent.generate_q_table(Action.ALL_ACTIONS)
ql_agent = QLearningAgent(
    q_table = qtable,
    actions = Action.ALL_ACTIONS,
)

qtable2 = QLearningAgent.generate_q_table(Action.ALL_ACTIONS)
ql_agent2 = QLearningAgent(
    q_table = qtable2,
    actions = Action.ALL_ACTIONS,
)

# aif_agent = ActiveInferenceAgent(
#     Action.ALL_ACTIONS, 
#     ActiveInferenceAgent.generate_matrix_A(Action.ALL_ACTIONS),
#     ActiveInferenceAgent.generate_matrix_B(Action.ALL_ACTIONS),
#     ActiveInferenceAgent.generate_matrix_C(),
#     ActiveInferenceAgent.generate_matrix_D(),
#     2
# )

agent_pair = AgentPair(ql_agent,ql_agent2)


mdp_gen_params = {"layout_name": 'cramped_room'}
mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
env_params = {"horizon":horizon}
agent_eval = AgentEvaluator(env_params=env_params,mdp_fn=mdp_fn)

trajectory_random_pair = agent_eval.evaluate_agent_pair(agent_pair,num_games=num_games)

print("######EvalDone####")
StateVisualizer().display_rendered_trajectory(trajectory_random_pair, ipython_display=False,img_directory_path='./images/',img_extension='.png',img_prefix='qlql-')
animator = ImageAnimator('/Users/el/Desktop/Workspace/OverCooked/images')
timestamp = time.strftime("%Y%m%d-%H%M%S")

gifname= f"./gifs/qlql_{timestamp}.gif"
animator.create_gif(gifname, duration=0.05)

wandb.log({f"qlql_gameplay": wandb.Image(gifname)})


print("Random pair rewards: \n",trajectory_random_pair['ep_returns'])
sr , shr = game_runs_info(trajectory_random_pair,print_details=False)
agent1_s_rewards = [reward[0] for reward in sr]
agent2_s_rewards = [reward[1] for reward in sr]

agent1_sh_rewards = [reward[0] for reward in shr]
agent2_sh_rewards = [reward[1] for reward in shr]


plot_rewards(agent1_s_rewards, agent2_s_rewards, num_games, horizon, "qlql_sparse_rewards", "Agent 1 Sparse Reward", "Agent 2 Sparse Reward", "Rewards for Each QL Agent Over Time", "Sparse Reward",timestamp)
plot_rewards(agent1_sh_rewards, agent2_sh_rewards, num_games, horizon, "qlql_shaped_rewards", "Agent 1 Shaped Reward", "Agent 2 Shaped Reward", "Rewards for Each QL Agent Over Time", "Shaped Reward",timestamp)

plot_rewards(np.cumsum(agent1_s_rewards), np.cumsum(agent2_s_rewards), num_games, horizon, "qlql_cum_sparse_rewards", "Agent 1 Cumulative Sparse Reward", "Agent 2 Cumulative Sparse Reward", "Cumulative Rewards for Each QL Agent Over Time", "Cumulative Sparse Reward",timestamp)
plot_rewards(np.cumsum(agent1_sh_rewards), np.cumsum(agent2_sh_rewards), num_games, horizon, "qlql_cum_shaped_rewards", "Agent 1 Cumulative Shaped Reward", "Agent 2 Cumulative Shaped Reward", "Cumulative Rewards for Each QL Agent Over Time", "Cumulative Shaped Reward",timestamp)


timesteps = np.array(range(0, num_games * horizon))
# wandb.log({"timesteps": timesteps, "agent1_cum_sparse_rewards": np.cumsum(agent1_s_rewards),"agent2_cum_sparse_rewards": np.cumsum(agent2_s_rewards)})
# wandb.log({"timesteps": timesteps, "agent1_cum_shaped_rewards": np.cumsum(agent1_sh_rewards),"agent2_cum_shaped_rewards": np.cumsum(agent2_sh_rewards)})

wandb.log({
    "qlql_cum_sparse_reward": wandb.plot.line_series(
        xs=timesteps, ys=[np.cumsum(agent1_s_rewards), np.cumsum(agent2_s_rewards)], 
        # xlabel="Timestep", ylabel="Cumulative Reward", title="Cumulative Sparse Reward",
        keys=["agent1_cum_sparse_rewards", "agent2_cum_sparse_rewards"]
    )
})

wandb.log({
    "qlql_cum_shaped_reward": wandb.plot.line_series(
        xs=timesteps, ys=[np.cumsum(agent1_sh_rewards), np.cumsum(agent2_sh_rewards)], 
        # xlabel="Timestep", ylabel="Cumulative Reward", title="Cumulative Shaped Reward",
        keys=["agent1_cum_shaped_rewards", "agent2_cum_shaped_rewards"]
    )
})



# # Create a plot for each agent's rewards over time
# plt.figure(figsize=(10, 5))
# plt.plot(agent1_s_rewards, label="Agent 1 Sparse Reward")
# plt.plot(agent2_s_rewards, label="Agent 2 Sparse Reward")

# # Add labels and title
# plt.xlabel("Time Step")
# plt.ylabel("Reward")
# plt.title("Rewards for Each QL Agent Over Time")
# plt.legend()
# plt.savefig('./qlql-rewards.png')   

