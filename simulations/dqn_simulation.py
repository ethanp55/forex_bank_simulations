from agents.dqn import DQNAgent
from environment.state import State
from environment.trade import TradeType
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np


def _filter_state(state: np.array, is_bank: bool) -> np.array:
    if is_bank:
        return state

    # Make sure the regular agents don't have access to how many buys and sells there are
    slice = state[:-3, ]
    slice = np.append(slice, state[-1, ])

    return slice


num_agents = 11
state = State(num_agents)
state_dim, starting_balance, bank_starting_balance = \
    state.vectorize().shape[-1], state.starting_balance, state.bank_starting_balance
agents = [DQNAgent(i, state_dim, 3, is_bank=True) if i == num_agents - 1 else
          DQNAgent(i, state_dim - 2, is_bank=False) for i in range(num_agents)]
num_episodes = 50
training_profits = {}

for episode in range(num_episodes):
    print(f'EPISODE {episode + 1}')

    done, state = False, State(num_agents)
    curr_state_matrix = state.vectorize()
    n_iterations = 0

    while not done:
        trades = []
        curr_price = state.curr_price()

        for i, agent in enumerate(agents):
            agent_state = _filter_state(curr_state_matrix[i], agent.is_bank)
            # if agent.is_bank:
            #     print(f'BANK STATE: {agent_state}')
            # else:
            #     print(f'AGENT STATE: {agent_state}')
            agent_trade = agent.place_trade(agent_state, curr_price)
            trades.append(agent_trade)

        prev_state_matrix = curr_state_matrix
        curr_state_matrix, rewards, done = state.step(trades)
        n_iterations += 1

        for i, reward in enumerate(rewards):
            agent = agents[i]
            trade = trades[i]
            agent_state = _filter_state(prev_state_matrix[i], agent.is_bank)
            agent_next_state = _filter_state(curr_state_matrix[i], agent.is_bank)
            # if agent.is_bank:
            #     action = 0 if trade.trade_type is TradeType.BUY else 1
            # else:
            #     action = 0 if trade is None else (1 if trade.trade_type is TradeType.BUY else 2)
            action = 0 if trade is None else (1 if trade.trade_type is TradeType.BUY else 2)
            agent.add_experience(agent_state, action, reward, agent_next_state, done)
            agent.train()

            if reward != 0:
                agent.trade_finished(reward)

    print(f'Finished in {n_iterations} iterations\n')
    print('Profits: ')

    for i in range(num_agents):
        balance = state.agent_balances[i]
        agent_name = f'Agent {i}' if i != num_agents - 1 else 'Bank'
        starting_balance_adjusted = starting_balance if agent_name != 'Bank' else bank_starting_balance
        profit = balance - starting_balance_adjusted
        print(f'{agent_name}\'s profit: {profit}')
        training_profits[agent_name] = training_profits.get(agent_name, []) + [profit]

    # final_price = state.curr_price()
    # print(f'\nFinal price: {final_price}')

    if episode % 1 == 0:
        print('\nUpdating target networks')

        for agent in agents:
            agent.update_networks()

    print()

# Save the trained models (for later use, if needed)
for agent in agents:
    agent.save()

# Generate training profits plot
cmap = get_cmap('tab20')
line_colors = [cmap(i % 20) for i in range(len(training_profits))]
x = range(0, num_episodes)
plt.grid()
i = 0
for name, profits in training_profits.items():
    color = line_colors[i]
    plt.plot(x, profits, label=name, color=color)
    i += 1
plt.xlabel('Training Episode')
plt.ylabel('Profit')
plt.legend(loc='best')
plt.title(f'Total Profit Achieved During Each Training Episode')
plt.savefig(f'../results/dqn_training_profits', bbox_inches='tight')
plt.clf()

test_episodes = 5

for episode in range(test_episodes):
    print(f'TEST EPISODE {episode + 1}')

    done, state = False, State(num_agents)
    curr_state_matrix = state.vectorize()
    n_iterations = 0

    while not done:
        trades = []
        curr_price = state.curr_price()

        for i, agent in enumerate(agents):
            agent_state = _filter_state(curr_state_matrix[i], agent.is_bank)
            agent_trade = agent.place_trade(agent_state, curr_price)
            trades.append(agent_trade)

        prev_state_matrix = curr_state_matrix
        curr_state_matrix, rewards, done = state.step(trades)
        n_iterations += 1

        for i, reward in enumerate(rewards):
            agent = agents[i]

            if reward != 0:
                agent.trade_finished(reward)

    print(f'Finished in {n_iterations} iterations\n')
    print('Profits: ')

    for i in range(num_agents):
        balance = state.agent_balances[i]
        print(f'Agent {i}\'s balance: {balance}')

    # final_price = state.curr_price()
    # print(f'\nFinal price: {final_price}\n')
