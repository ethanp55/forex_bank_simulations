from agents.ucb import UCB
from environment.state import State

NUM_AGENTS = 11
agents = [UCB() for _ in range(NUM_AGENTS)]
agents[-1].is_bank = True
done, state = False, State(NUM_AGENTS)
curr_state_matrix = state.vectorize()
n_iterations = 0


while not done:
    curr_price = state.curr_price()
    trades = []

    for i, agent in enumerate(agents):
        agent_trade = agent.place_trade(curr_price)
        trades.append(agent_trade)

    curr_state_matrix, rewards, done = state.step(trades)
    n_iterations += 1

    for i, reward in enumerate(rewards):
        agent = agents[i]

        if reward != 0:
            agent.trade_finished(reward)


print(f'Finished in {n_iterations} iterations\n')
print('Balances: ')

for i in range(NUM_AGENTS):
    balance = curr_state_matrix[i, -1]
    print(f'Agent {i}\'s balance: {balance}')


final_price = curr_state_matrix[0, -2]
print(f'\nFinal price: {final_price}')

