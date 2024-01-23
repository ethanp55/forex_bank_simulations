from agents.agent import Agent
from collections import deque
from copy import deepcopy
from environment.trade import Trade, TradeType
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from typing import Optional


class DynamicMinMaxScaler:
    def __init__(self, num_features: int) -> None:
        self.num_features = num_features
        self.min_vals = np.inf * np.ones(num_features)
        self.max_vals = -np.inf * np.ones(num_features)

    def update(self, state: np.array) -> None:
        self.min_vals = np.minimum(self.min_vals, state)
        self.max_vals = np.maximum(self.max_vals, state)

    def scale(self, state: np.array) -> np.array:
        return (state - self.min_vals) / (self.max_vals - self.min_vals + 0.00001)


class DQN(tf.keras.Model):
    def __init__(self, state_dim: int,  action_dim: int) -> None:
        super(DQN, self).__init__()
        self.dense1 = layers.Dense(state_dim, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(action_dim, activation='linear')

    def call(self, state: np.array) -> tf.Tensor:
        x = self.dense1(state)
        x = self.dense2(x)

        return self.output_layer(x)


class DQNAgent(Agent):
    def __init__(self, name: str, state_dim: int, action_dim: int = 3, learning_rate: float = 0.001,
                 discount_factor: float = 0.99, epsilon: float = 0.1, replay_buffer_size: int = 10000,
                 batch_size: int = 64, is_bank: bool = False) -> None:
        super().__init__(name, is_bank=is_bank)
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size

        # DQN model and target model
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.set_weights(self.model.get_weights())

        # Optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        # Replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Episode experiences (to add to the replay buffer)
        self.current_episode_experiences = []

        # Keeps track of whether there is a trade on or not
        self.curr_action = None

        # State scaler
        self.scaler = DynamicMinMaxScaler(state_dim)

        self.training_started = False
        self.state = None

    def trade_finished(self, net_profit: float) -> None:
        self.curr_action = None

    def place_trade(self, state: np.array, curr_price: float) -> Optional[Trade]:
        self.state = deepcopy(state)

        # If there is already an existing trade, return
        if self.curr_action is not None and not self.is_bank:
            return None

        # Epsilon-greedy policy for action selection
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_dim)

        else:
            state_adjusted = deepcopy(state)
            scaled_state = self.scaler.scale(state_adjusted)
            q_values = self.model(np.expand_dims(scaled_state, 0))

            action = np.argmax(q_values.numpy())

        # Action representing "do nothing"
        if action == 0 and not self.is_bank:
            return None

        # Place the trade
        self.curr_action = action
        action_modifier = 0 if self.is_bank else 1
        trade_type = TradeType.BUY if action == action_modifier else TradeType.SELL
        open_price = curr_price
        stop_loss = (open_price - self.pips_to_risk) if action == action_modifier else (open_price + self.pips_to_risk)
        stop_gain = (open_price + self.pips_to_risk * self.risk_reward_ratio) if action == action_modifier else \
            (open_price - self.pips_to_risk * self.risk_reward_ratio)
        trade = Trade(trade_type, open_price, stop_loss, stop_gain, self.percent_to_risk)

        return trade

    def update_networks(self) -> None:
        # Update target network weights periodically
        if self.training_started:
            self.target_model.set_weights(self.model.get_weights())

    def train(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return

        self.training_started = True

        # Sample a batch of experiences from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = map(np.array, zip(*batch))

        # Q-learning update using the DQN loss
        next_q_values = self.target_model(batch_next_states)
        max_next_q_values = np.max(next_q_values.numpy(), axis=1)

        targets = batch_rewards + (1 - batch_dones) * self.discount_factor * max_next_q_values

        with tf.GradientTape() as tape:
            q_values = self.model(batch_states)
            selected_action_values = tf.reduce_sum(tf.one_hot(batch_actions, self.action_dim) * q_values, axis=1)
            loss = tf.losses.mean_squared_error(targets, selected_action_values)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def add_experience(self, action: int, reward: float, next_state: np.array, done: bool):
        # Accumulate experiences over multiple time steps
        scaled_state = self.scaler.scale(self.state)
        scaled_next_state = self.scaler.scale(next_state)
        self.scaler.update(next_state)

        self.current_episode_experiences.append((scaled_state, action, reward, scaled_next_state, done))

        # If the episode is done, add the accumulated experiences to the replay buffer
        if done:
            self.replay_buffer.extend(self.current_episode_experiences)
            self.current_episode_experiences = []
