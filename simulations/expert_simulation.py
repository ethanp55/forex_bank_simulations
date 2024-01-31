from agents.dqn import DQNAgent
from agents.macd import MACD
from agents.macd_stochastic import MACDStochastic
from agents.rsi import RSI
from agents.stochastic import Stochastic
from environment.state import State
from environment.trade import TradeType
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import pickle


num_agents = 50
num_agents += 1  # An extra agent that represents the bank
bank_balance_multiplier = 1.0
state = State(num_agents, bank_balance_multiplier=bank_balance_multiplier)
state_dim, starting_balance, bank_starting_balance = \
    state.vectorize().shape[-1], state.starting_balance, state.bank_starting_balance
n_agent_types = 4
per_agent_type = (num_agents - 1) // n_agent_types
n_last_agent_type = num_agents - 1 - (per_agent_type * (n_agent_types - 1))
macd_agents = [MACD(f'MACD_{i}') for i in range(per_agent_type)]
rsi_agents = [RSI(f'RSI_{i}') for i in range(per_agent_type)]
macd_stoch_agents = [MACDStochastic(f'MACDStoch_{i}') for i in range(per_agent_type)]
stochastic_agents = [Stochastic(f'Stoch_{i}') for i in range(n_last_agent_type)]
bank_agents = [DQNAgent(f'Bank', state_dim, is_bank=True)]
agents = macd_agents + rsi_agents + macd_stoch_agents + stochastic_agents + bank_agents
assert len(agents) == num_agents
num_episodes = 1000
training_profits, test_profits = {}, {}

for episode in range(num_episodes):
    print(f'EPISODE {episode + 1}')

    done, state = False, State(num_agents, bank_balance_multiplier=bank_balance_multiplier)
    curr_state_matrix = state.vectorize()
    n_iterations = 0

    while not done:
        trades = []
        curr_price = state.curr_price()

        for i, agent in enumerate(agents):
            agent_state = curr_state_matrix[i]
            agent_trade = agent.place_trade(agent_state, curr_price)
            trades.append(agent_trade)

        curr_state_matrix, rewards, done = state.step(trades)
        n_iterations += 1

        for i, reward in enumerate(rewards):
            agent = agents[i]

            if isinstance(agent, DQNAgent):
                trade = trades[i]
                agent_next_state = curr_state_matrix[i]
                action = 0 if trade is None else (1 if trade.trade_type is TradeType.BUY else 2)
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
            if isinstance(agent, DQNAgent):
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
plt.savefig(f'../results/expert_training_profits_0_{int(bank_balance_multiplier * 10)}', bbox_inches='tight')
plt.clf()

test_episodes = 5

for episode in range(test_episodes):
    print(f'TEST EPISODE {episode + 1}')

    done, state = False, State(num_agents, bank_balance_multiplier=bank_balance_multiplier)
    curr_state_matrix = state.vectorize()
    n_iterations = 0

    while not done:
        trades = []
        curr_price = state.curr_price()

        for i, agent in enumerate(agents):
            agent_state = curr_state_matrix[i]
            agent_trade = agent.place_trade(agent_state, curr_price)
            trades.append(agent_trade)

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
plt.savefig(f'../results/expert_test_profits_0_{int(bank_balance_multiplier * 10)}', bbox_inches='tight')
plt.clf()

# Save data (for later processing, if needed)
with open(f'../results/data/expert_training_profits_0_{int(bank_balance_multiplier * 10)}.pickle', 'wb') as f:
    pickle.dump(training_profits, f)

with open(f'../results/data/expert_test_profits_0_{int(bank_balance_multiplier * 10)}.pickle', 'wb') as f:
    pickle.dump(test_profits, f)

