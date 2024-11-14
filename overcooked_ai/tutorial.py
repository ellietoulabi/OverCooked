
from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai.src.overcooked_ai_py.agents.agent import Agent,AgentPair,StayAgent
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator, LayoutGenerator
import numpy as np
import random
from collections import defaultdict, deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, OvercookedGridworld
from itertools import product







layout_name = "cramped_room"
mdp = OvercookedGridworld.from_layout_name(layout_name)
env = OvercookedEnv.from_mdp(mdp, horizon=400)


print(env.game_stats)

exit()

print(3 * "\n" + 50 * "_" + "Action" + 50 * "_" +  1 * "\n")

print(Action.ALL_ACTIONS)
'''
[(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), 'interact']
'''

print(Action.ACTION_TO_CHAR)
'''
{(0, -1): '↑', (0, 1): '↓', (1, 0): '→', (-1, 0): '←', (0, 0): 'stay', 'interact': 'interact'}
'''

print(Action.ACTION_TO_INDEX)
'''
{(0, -1): 0, (0, 1): 1, (1, 0): 2, (-1, 0): 3, (0, 0): 4, 'interact': 5}
'''

print(Action.sample([0.01,0.01,.01,.01,.01,.95]))
'''
interact
'''

print(Action.uniform_probs_over_actions())
'''
[0.16666667 0.16666667 0.16666667 0.16666667 0.16666667 0.16666667]
'''

# ##########################################################################################
print(3 * "\n" + 50 * "_" + "Direction" + 50 * "_" +  1 * "\n")

print(Direction.ALL_DIRECTIONS)
'''
[(0, -1), (0, 1), (1, 0), (-1, 0)]
'''

print(Direction.DIRECTION_TO_NAME)
'''
{(0, -1): 'NORTH', (0, 1): 'SOUTH', (1, 0): 'EAST', (-1, 0): 'WEST'}
'''

print(Direction.OPPOSITE_DIRECTIONS)
'''
{(0, -1): (0, 1), (0, 1): (0, -1), (1, 0): (-1, 0), (-1, 0): (1, 0)}
'''

print(Direction.get_adjacent_directions((0,-1)))
'''
[(1, 0), (-1, 0)]
'''

# ##########################################################################################
print(3 * "\n" + 50 * "_" + "Agent" + 50 * "_" +  1 * "\n")

print(Agent.a_probs_from_action(Action.INTERACT))
'''
[0. 0. 0. 0. 0. 1.] ?
'''

# print(Agent.action())

# ##########################################################################################
print(3 * "\n" + 50 * "_" + "AgentEvaluator" + 50 * "_" +  1 * "\n")

mdp_gen_params = {"layout_name":"cramped_room"}
mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
env_params = {"horizon": 1000}
agent_eval = AgentEvaluator(env_params=env_params, mdp_fn=mdp_fn)

print(agent_eval.env.state)
'''
Players: ((1, 2) facing (0, -1) holding None, (3, 1) facing (0, -1) holding None), Objects: [], Bonus orders: [] All orders: [('onion', 'onion', 'onion')] Timestep: 0
'''

print("\n", agent_eval.env.display_states)
'''
X       X       P       X       X       

O                       ↑1      O       

X       ↑0                      X       

X       D       X       S       X       
'''

print(agent_eval.env.game_stats)
'''
{'tomato_pickup': [[], []], 'useful_tomato_pickup': [[], []], 'tomato_drop': [[], []], 'useful_tomato_drop': [[], []], 'potting_tomato': [[], []], 'onion_pickup': [[], []], 'useful_onion_pickup': [[], []], 'onion_drop': [[], []], 'useful_onion_drop': [[], []], 'potting_onion': [[], []], 'dish_pickup': [[], []], 'useful_dish_pickup': [[], []], 'dish_drop': [[], []], 'useful_dish_drop': [[], []], 'soup_pickup': [[], []], 'soup_delivery': [[], []], 'soup_drop': [[], []], 'optimal_onion_potting': [[], []], 'optimal_tomato_potting': [[], []], 'viable_onion_potting': [[], []], 'viable_tomato_potting': [[], []], 'catastrophic_onion_potting': [[], []], 'catastrophic_tomato_potting': [[], []], 'useless_onion_potting': [[], []], 'useless_tomato_potting': [[], []], 'cumulative_sparse_rewards_by_agent': array([0, 0]), 'cumulative_shaped_rewards_by_agent': array([0, 0])}
'''

print('\n\n',agent_eval.env.step((Action.STAY, Action.INTERACT)))
'''
(<overcooked_ai_py.mdp.overcooked_mdp.OvercookedState object at 0x1205c7890>, 0, False, {'agent_infos': [{}, {}], 'sparse_r_by_agent': [0, 0], 'shaped_r_by_agent': [0, 0], 'phi_s': None, 'phi_s_prime': None})
'''




