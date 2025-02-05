from overcooked_ai.src.overcooked_ai_py.agents.agent import AgentPair, StayAgent
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action
from overcooked_ai.src.overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import matplotlib.pyplot as plt


from ql import QLearningAgent
# from aif import ActiveInferenceAgent
from utils import ImageAnimator , game_runs_info





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
env_params = {"horizon":5000}
agent_eval = AgentEvaluator(env_params=env_params,mdp_fn=mdp_fn)

trajectory_random_pair = agent_eval.evaluate_agent_pair(agent_pair,num_games=20)

print("######EvalDone####")
StateVisualizer().display_rendered_trajectory(trajectory_random_pair, ipython_display=False,img_directory_path='./images/',img_extension='.png',img_prefix='r0-')
# animator = ImageAnimator('/home/user/Workspace/eli-tests/single-qlearning/images')
# animator.create_gif('./gifs/output_imageio.gif', duration=0.05)
# end for

print("Random pair rewards: \n",trajectory_random_pair['ep_returns'])
sr , shr = game_runs_info(trajectory_random_pair,print_details=False)
agent1_rewards = [reward[0] for reward in sr]
agent2_rewards = [reward[1] for reward in sr]

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