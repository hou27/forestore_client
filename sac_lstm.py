import pandas as pd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

# 코드의 목적은 주어진 데이터에서 누락된 시간의 재고 데이터를 보완하는 것입니다.
# 예를 들어, 타임스탬프가 3시간 간격으로 주어진 경우,
# 그 사이의 2시간 동안의 재고 변화를 선형적으로 추정하여 데이터를 보완합니다.

device = torch.device(
    "cuda:1" if torch.cuda.is_available() else "cpu"
)  # Check if GPU is available


def predict(df: pd.DataFrame) -> list:
    # 하이퍼파라미터 설정
    state_dim = 5  # 상태의 차원
    replay_buffer_size = 100000  # 리플레이 버퍼 크기
    batch_size = 64  # 배치 크기
    gamma = 0.99  # 할인율
    tau = 0.005  # 타겟 네트워크를 위한 소프트 업데이트 비율
    actor_lr = 3e-4  # 액터 네트워크의 학습률
    critic_lr = 3e-4  # 크리틱 네트워크의 학습률
    policy_freq = 2  # 정책 업데이트 빈도

    def parsing_hour(timestamp):
        return timestamp.split(" ")[1].split(":")[0]

    def add_hour(timestamp, hour=1):
        return timestamp + timedelta(hours=hour)

    df_for_preprocessing = df.reset_index(drop=True)
    data_len = len(df_for_preprocessing)
    result_df = df_for_preprocessing.copy()

    df_for_preprocessing["timestamp"] = pd.to_datetime(
        df_for_preprocessing["timestamp"], format="%Y-%m-%d %H:%M:%S"
    )

    i = 1

    while i < data_len:
        if i == 1:
            print(df_for_preprocessing.iloc[i])
        current_timestamp = df_for_preprocessing.iloc[i]["timestamp"]
        previous_timestamp = df_for_preprocessing.iloc[i - 1]["timestamp"]

        current_hour = int(
            parsing_hour(current_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        )
        previous_hour = int(
            parsing_hour(previous_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        )

        if current_hour == previous_hour:
            i += 1
            continue

        term = current_hour - previous_hour
        stock_apart = float(df_for_preprocessing.loc[i - 1, "remaining_stock"]) - float(
            df_for_preprocessing.loc[i, "remaining_stock"]
        )
        minus = stock_apart / term

        item_id = df_for_preprocessing.iloc[0]["item_id"]
        prev_timestamp = previous_timestamp
        prev_remaining_stock = float(df_for_preprocessing.loc[i - 1, "remaining_stock"])

        for j in range(1, term):
            timestamp = prev_timestamp + timedelta(hours=j)
            prev_remaining_stock -= minus
            temp_df = pd.DataFrame(
                {
                    "item_id": item_id,
                    "timestamp": timestamp,
                    "remaining_stock": prev_remaining_stock,
                },
                index=[i - 1 + j / (term + 1)],
            )
            result_df = pd.concat([result_df, temp_df])

        i += 1

    test_result_df = result_df.sort_index().reset_index(drop=True)

    # timestamp 일괄 편집
    test_result_df["timestamp"] = pd.to_datetime(
        result_df["timestamp"], format="%Y-%m-%d %H:%M:%S"
    )  # ['timestamp'] = result_df['timestamp'].apply(lambda x: x.strftime('%m-%d %H:%M:%S'))

    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data = test_result_df["remaining_stock"].values
    stock_data = scaler.fit_transform(stock_data.reshape(-1, 1))
    print("stock_data", stock_data)

    test_stock_df = pd.DataFrame(stock_data, columns=["saled"])

    class SaleForecastEnv:
        def __init__(self, data, state_size=5):
            self.data = data
            self.state_size = state_size
            self.reset()

        def reset(self):
            self.current_step = 0
            self.timestep = self.state_size  # Here we initialize the timestep
            self.done = False
            return torch.tensor(
                self.data[
                    self.current_step : self.current_step + self.state_size
                ].values.flatten(),
                dtype=torch.float32,
            )

        def step(self, action):
            self.timestep += 1

            actual_value = self.data[self.timestep - 1]

            reward = -abs(self.data[self.timestep - 1] - action)
            start_index = max(
                0, self.timestep - self.state_size
            )  # Ensure that the state always has `state_size` length
            self.state = self.data[start_index : self.timestep].values

            if self.timestep >= len(self.data):
                self.done = True
            else:
                self.done = False

            return self.state, reward, self.done, actual_value

    test_env = SaleForecastEnv(test_stock_df["saled"], state_size=5)

    class LSTMActor(nn.Module):
        def __init__(self, input_dim, hidden_dim=128, output_dim=1):
            super(LSTMActor, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Tanh(),
            )

        def forward(self, state):
            if len(state.shape) == 2:
                state = state.unsqueeze(1)
            _, (h_n, _) = self.lstm(state)
            h_n = h_n.squeeze(0)
            return self.fc(h_n)

    class LSTMCritic(nn.Module):
        def __init__(self, input_dim, action_dim, hidden_dim=128):
            super(LSTMCritic, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, state, action):
            if len(state.shape) == 2:
                state = state.unsqueeze(1)
            _, (h_n, _) = self.lstm(state)
            h_n = h_n.squeeze(0)
            return self.fc(torch.cat([h_n, action], dim=1))

    class ReplayBuffer:
        def __init__(self, capacity):
            self.capacity = capacity
            self.buffer = []
            self.position = 0

        def push(self, state, action, reward, next_state, done):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done

        def __len__(self):
            return len(self.buffer)

    # 모델 객체 생성
    actor = LSTMActor(state_dim).to(device)
    critic1 = LSTMCritic(state_dim, action_dim=1).to(device)
    critic2 = LSTMCritic(state_dim, action_dim=1).to(device)

    # 타겟 크리틱 네트워크 생성 (소프트 업데이트를 위해)
    target_critic1 = LSTMCritic(state_dim, action_dim=1).to(device)
    target_critic2 = LSTMCritic(state_dim, action_dim=1).to(device)

    # 타겟 네트워크의 가중치를 크리틱 네트워크의 가중치로 초기화
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    actor, critic1, critic2

    # 저장한 모델을 불러오는 코드
    actor = LSTMActor(state_dim).to(device)
    actor.load_state_dict(torch.load("./actor_model.pth"))

    critic1 = LSTMCritic(state_dim, action_dim=1).to(device)
    critic1.load_state_dict(torch.load("./critic1_model.pth"))

    critic2 = LSTMCritic(state_dim, action_dim=1).to(device)
    critic2.load_state_dict(torch.load("./critic2_model.pth"))

    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic1_optimizer = optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2_optimizer = optim.Adam(critic2.parameters(), lr=critic_lr)

    predictions = []
    targets = []
    losses = []
    state = test_env.reset()
    max_timesteps = len(test_stock_df) - state_dim

    criterion = nn.MSELoss()  # 평균 제곱 오차 손실 함수

    with torch.no_grad():  # Disable gradient computation
        for t in range(max_timesteps):
            state_tensor = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            )
            action = actor(state_tensor).item()
            # predictions.append(action)

            next_state, reward, done, actual_value = test_env.step(action)

            # 손실 계산
            predicted_tensor = torch.tensor([action], dtype=torch.float32).to(device)
            actual_tensor = torch.tensor([actual_value], dtype=torch.float32).to(device)

            # 손실 계산 전에 역 정규화
            predicted_denorm = scaler.inverse_transform(
                predicted_tensor.cpu().numpy().reshape(-1, 1)
            ).squeeze()
            actual_denorm = scaler.inverse_transform(
                np.array([actual_value]).reshape(-1, 1)
            ).squeeze()

            predictions.append(predicted_denorm)  # 예측값 추가
            targets.append(actual_denorm)  # 실제값 추가

            # 역 정규화된 값들로 MSE 손실 계산
            mse_loss = criterion(
                torch.tensor(predicted_denorm), torch.tensor(actual_denorm)
            )

            # RMSE 손실 계산
            rmse_loss = torch.sqrt(mse_loss)

            losses.append(rmse_loss.item())

            if done:
                break

            state = next_state

    average_loss = sum(losses) / len(losses)
    print(f"Average RMSE: {average_loss}")

    return predictions


def save_predictions():
    prediction_list = [[] for _ in range(302)]
    for idx in range(1, 302):
        predictions = all(idx)
        prediction_list[idx] = np.array(predictions)

    pd.DataFrame(prediction_list).to_excel("prediction_list.xlsx")