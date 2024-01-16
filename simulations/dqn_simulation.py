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


num_agents = 50
num_agents += 1  # An extra agent that represents the bank
state = State(num_agents)
state_dim, starting_balance, bank_starting_balance = \
    state.vectorize().shape[-1], state.starting_balance, state.bank_starting_balance
agents = [DQNAgent('Bank', state_dim, 2, is_bank=True) if i == num_agents - 1 else
          DQNAgent(f'DQN_{i}', state_dim - 2, is_bank=False) for i in range(num_agents)]
num_episodes = 300
training_profits, test_profits = {}, {}

for episode in range(num_episodes):
    print(f'EPISODE {episode + 1}')

    done, state = False, State(num_agents)
    curr_state_matrix = state.vectorize()
    n_iterations = 0

    while not done:
        trades = []
        curr_price = state.curr_price()
        n_buys, n_sells = curr_state_matrix[0, -3], curr_state_matrix[0, -2]

        for i, agent in enumerate(agents):
            agent_state = _filter_state(curr_state_matrix[i], agent.is_bank)
            agent_trade = agent.place_trade(agent_state, curr_price, n_buys, n_sells)
            trades.append(agent_trade)

            if not agent.is_bank:
                n_buys += 1 if (agent_trade is not None and agent_trade.trade_type is TradeType.BUY) else 0
                n_sells += 1 if (agent_trade is not None and agent_trade.trade_type is TradeType.SELL) else 0

        curr_state_matrix, rewards, done = state.step(trades)
        n_iterations += 1

        for i, reward in enumerate(rewards):
            agent = agents[i]
            trade = trades[i]
            agent_next_state = _filter_state(curr_state_matrix[i], agent.is_bank)
            if agent.is_bank:
                action = 0 if trade.trade_type is TradeType.BUY else 1
            else:
                action = 0 if trade is None else (1 if trade.trade_type is TradeType.BUY else 2)
            # action = 0 if trade is None else (1 if trade.trade_type is TradeType.BUY else 2)
            agent.add_experience(action, reward, agent_next_state, done)
            agent.train()

            if reward != 0:
                agent.trade_finished(reward)

    print(f'Finished in {n_iterations} iterations\n')
    print('Profits: ')

    for i in range(num_agents):
        balance = state.agent_balances[i]
        agent_name = agents[i].name
        starting_balance_adjusted = starting_balance if agent_name != 'Bank' else bank_starting_balance
        profit = balance - starting_balance_adjusted
        print(f'{agent_name}\'s profit: {profit}')
        training_profits[agent_name] = training_profits.get(agent_name, []) + [profit]

    if episode % 1 == 0:
        print('\nUpdating target networks')

        for agent in agents:
            agent.update_networks()

    print()

# Generate training profits plot
cmap = get_cmap('tab20')
line_colors = [cmap(i % 20) for i in range(len(training_profits))]
x = range(0, num_episodes)
plt.grid()
i = 0
for name, profits in training_profits.items():
    color = line_colors[i]
    if name == 'Bank':
        plt.plot(x, profits, label=name, color=color)
    else:
        plt.plot(x, profits, color=color)
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
        n_buys, n_sells = curr_state_matrix[0, -3], curr_state_matrix[0, -2]

        for i, agent in enumerate(agents):
            agent_state = _filter_state(curr_state_matrix[i], agent.is_bank)
            agent_trade = agent.place_trade(agent_state, curr_price, n_buys, n_sells)
            trades.append(agent_trade)

            if not agent.is_bank:
                n_buys += 1 if (agent_trade is not None and agent_trade.trade_type is TradeType.BUY) else 0
                n_sells += 1 if (agent_trade is not None and agent_trade.trade_type is TradeType.SELL) else 0

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
        agent_name = agents[i].name
        starting_balance_adjusted = starting_balance if agent_name != 'Bank' else bank_starting_balance
        profit = balance - starting_balance_adjusted
        print(f'{agent_name}\'s profit: {profit}')
        test_profits[agent_name] = test_profits.get(agent_name, []) + [profit]

# Generate test profits plot
cmap = get_cmap('tab20')
line_colors = [cmap(i % 20) for i in range(len(test_profits))]
x = range(0, test_episodes)
plt.grid()
i = 0
for name, profits in test_profits.items():
    color = line_colors[i]
    if name == 'Bank':
        plt.plot(x, profits, label=name, color=color)
    else:
        plt.plot(x, profits, color=color)
    i += 1
plt.xlabel('Test Episode')
plt.ylabel('Profit')
plt.legend(loc='best')
plt.title(f'Total Profit Achieved During Each Test Episode')
plt.savefig(f'../results/dqn_test_profits', bbox_inches='tight')
plt.clf()

