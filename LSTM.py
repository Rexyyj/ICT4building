import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import sys
from sklearn.metrics import mean_squared_error as mse

# torch.cuda.set_device(0)

# train_epoch_num, hidden_neuron_num = sys.argv[1:3]
# train_epoch_num, hidden_neuron_num = int(train_epoch_num), int(hidden_neuron_num)
# print(train_epoch_num, hidden_neuron_num)
train_epoch_num, hidden_neuron_num = 180, 18

late_winter_period = (0, 24 * 105)
spring_period = (24 * 105, 24 * 157)
summer_period = (24 * 157, 24 * 243)
fall_period = (24 * 243, 24 * 303)
early_winter_period = (24 * 303, 24 * 365)

season_begin, season_end = early_winter_period
roll_size=24*7
test_size = 24
train_size = 24*7*5
assert train_size<=season_end - test_size - season_begin
start_idx = 0


df = pd.read_csv('C:\\Users\\maqly\\Desktop\\ict4bd lab\\data.csv')
# df['prev_heating'] = df['DistrictHeating:Facility [J](Hourly)'].shift(1)
# df['prev_cooling'] = df['DistrictHeating:Facility [J](Hourly)'].shift(1)
# df.drop(0, inplace=True)
df = df.iloc[season_begin:season_end]
df.reset_index(drop=True,inplace=True)

x_col_names = [
    'hour',
    'is_weekday',
    # 'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)',
    # 'Indoor Mean Air Temperature [C](Hourly)',
    # 'Environment:Site Wind Speed [m/s](Hourly)',
    # 'Environment:Site Direct Solar Radiation Rate per Area [W/m2](Hourly)',
    # 'Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](Hourly)',
    # 'Environment:Site Solar Altitude Angle [deg](Hourly)',
    # 'MAINXOFFICE:ZONE2:Zone Air Relative Humidity [%](Hourly:ON)',
    # 'Electricity:Facility [J](Hourly)',
    # 'DistrictCooling:Facility [J](Hourly)',
    # 'DistrictHeating:Facility [J](Hourly)',
    # 'prev_heating',
    # 'prev_cooling'
]
y_col_names = [
    # 'Electricity:Facility [J](Hourly)',
    # 'DistrictCooling:Facility [J](Hourly)',
    # 'DistrictHeating:Facility [J](Hourly)',
    'Indoor Mean Air Temperature [C](Hourly)'
]


def to_variable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)


hidden_size = 10
batch_size = 1
input_size = len(x_col_names)
final_output_size = len(y_col_names)
learning_rate = 1e-4
print(len(df))
# assert len(df) % batch_size == 0

# seq_len = len(df) // batch_size
# lstm = torch.nn.LSTM(input_size=len(x_col_names), hidden_size=hidden_size)
input = np.array(df[x_col_names])
input = torch.FloatTensor(input.reshape((-1, batch_size, input_size)))
# print(len(input))
y_train = np.array(df[y_col_names]).reshape((-1, batch_size, final_output_size))


# input = torch.randn(1, batch_size, len(x_col_names))
# h0 = torch.randn(1, batch_size, hidden_size)
# c0 = torch.randn(1, batch_size, hidden_size)
# output = lstm(input, (h0, c0))
# regress = torch.nn.Linear(hidden_size, len(y_col_names))
# hidden_neuron_num = 15


class LSTMPred(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, final_output_size):
        super(LSTMPred, self).__init__()

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.dense = torch.nn.Linear(hidden_size, final_output_size)
        # self.dense = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_size, hidden_neuron_num),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_neuron_num, final_output_size)  # 最后一个时序的输出接一个全连接层
        # )
        self.h = torch.randn(1, batch_size, hidden_size)
        self.c = torch.randn(1, batch_size, hidden_size)

    def forward(self, x):  # x是输入数据集
        lstm_out, (h, c) = self.lstm(x, (self.h, self.c))  # 如果不导入h_s和h_c，默认每次都进行0初始化
        # print(f'lstm_out.shape: {lstm_out.shape}')
        dense_out = self.dense(lstm_out.view(len(x), -1))
        return dense_out


model = LSTMPred(input_size, hidden_size, batch_size, final_output_size)
loss_f = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(train_epoch_num):
    # print(f'epoch: {epoch}')
    for i in range(train_size - 1):
        seq = input[start_idx + i:start_idx + i + batch_size]
        # print(seq.size())
        seq = seq.view(len(seq), batch_size, input_size)
        # print(f'seq.shape: {seq.shape}')

        y_test = torch.FloatTensor(y_train[start_idx + i:start_idx + i + batch_size].reshape(-1, 1))

        optimizer.zero_grad()
        try:
            y_hat = model(seq)
        except RuntimeError:
            print(f'seq: {seq}')
            print(f'input shape: {input.shape}')
            print(start_idx + i)
            print(f'start index: {start_idx}')
            print(f'train size: {train_size}')
            print('error')
            sys.exit()

        y_hat = y_hat.view(-1, 1)

        # print(y_hat.shape)
        # print(y_test.shape)

        loss = loss_f(y_hat, y_test)
        loss.backward()
        optimizer.step()
        # print(loss)

torch.save(model, 'model/lstm_model.pth')

plt.figure()
y_pred = []
for i in range(start_idx+train_size, start_idx+train_size+test_size):
    seq = input[i]
    seq = seq.view(len(seq), batch_size, input_size)
    # y_test = torch.FloatTensor(y_train[i + 1].reshape(-1))
    y_hat = model(seq)

    y_pred.append((y_hat.view(-1)).detach().numpy())

y_pred = np.array(y_pred)
y_pred = y_pred.reshape(-1)
# plt.title(f'{train_epoch_num}  {hidden_neuron_num}')
plt.plot(y_pred, label='predict')
y_real = df.loc[start_idx+train_size: start_idx+train_size+test_size-1, y_col_names]
y_real.reset_index(drop=True, inplace=True)
assert len(y_real) == len(y_pred)
plt.plot(y_real, label='real')
plt.legend()
plt.xlabel('Hour [h]')
plt.ylabel('Internal Temperature [℃]')
plt.title('Internal Temperature prediction with LSTM (winter)')

# plt.savefig('lstm winter temp')
plt.show()
# plt.savefig(f'C:\\Users\\maqly\\Desktop\\ict4bd lab\\img2\\{train_epoch_num} {hidden_neuron_num}.png')
print(f'mse: {mse(y_real, y_pred)}')
