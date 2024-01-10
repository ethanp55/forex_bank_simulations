from agents.ucb import UCB
from environment.state import State

num_agents = 11
agents = [UCB() for _ in range(num_agents)]
agents[-1].is_bank = True
done, state = False, State(num_agents)
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
print('Balances: ')

for i in range(num_agents):
    balance = state.agent_balances[i]
    print(f'Agent {i}\'s balance: {balance}')


final_price = state.curr_price()
print(f'\nFinal price: {final_price}')

