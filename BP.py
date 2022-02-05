import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

# torch.cuda.set_device(0)

df = pd.read_csv('C:\\Users\\maqly\\Desktop\\ict4bd lab\\data.csv')
# (24*157,24*243)
df = df.iloc[:24*105]
# df1 = df.iloc[:24*105]
# df2 = df.iloc[24*303:24*365]
# df = df1.append(df2)
df.reset_index(drop=True,inplace=True)
print(len(df))
# indoor_temp_mean = df['MAINXGROUND:ZONE1:Zone Mean Air Temperature [C](Hourly)'].mean()
# df['MAINXGROUND:ZONE1:Zone Mean Air Temperature [C](Hourly)'] -= indoor_temp_mean
# print(df['MAINXGROUND:ZONE1:Zone Mean Air Temperature [C](Hourly)'].describe())
df['Date/Time'] = (list(range(24))*(int(len(df)/24)))[:len(df)]
# df.loc[1:, 'previous_temp'] = df[:-1,'MAINXGROUND:ZONE1:Zone Mean Air Temperature [C](Hourly)']
df['prev_temp'] = df['Indoor Mean Air Temperature [C](Hourly)'].shift(1)
df['prev_heat_energy'] = df['DistrictHeating:Facility [J](Hourly)'].shift(1)
# df['Indoor Mean Air Temperature [C](Hourly) diff'] = df['Indoor Mean Air Temperature [C](Hourly)'].diff()
df.drop(0, inplace=True)
# print(df[['prev_temp', 'Indoor Mean Air Temperature [C](Hourly)']])
# print(df['Date/Time'])
# df = df.sample(frac=1)

x_col_names = [
                'Date/Time',
                'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)',
                'Indoor Mean Air Temperature [C](Hourly)',
                'Environment:Site Wind Speed [m/s](Hourly)',
                'Environment:Site Direct Solar Radiation Rate per Area [W/m2](Hourly)',
                'Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](Hourly)',
                'Environment:Site Solar Altitude Angle [deg](Hourly)',
                # 'MAINXGROUND:ZONE1:Zone Air Relative Humidity [%](Hourly)',
                # 'MAINXGROUND:ZONE5:Zone Air Relative Humidity [%](Hourly)',
                'prev_temp',
                # 'prev_heat_energy',
                # 'Electricity:Facility [J](Hourly)',
                # 'DistrictCooling:Facility [J](Hourly)',
                # 'DistrictHeating:Facility [J](Hourly)',
]
y_col_names = [
                # 'Electricity:Facility [J](Hourly)',
                # 'DistrictCooling:Facility [J](Hourly)',
                # 'DistrictHeating:Facility [J](Hourly)',
                'Indoor Mean Air Temperature [C](Hourly)'
]
for column in y_col_names:
    print(df[column].describe())
x = torch.tensor(df.loc[:, x_col_names].values).float()
y = torch.tensor(df.loc[:, y_col_names].values).float()
print(y)
print(x.shape)
print(y.shape)
hidden_layer_neuron_num = 15

model = torch.nn.Sequential(
    torch.nn.Linear(len(x_col_names), hidden_layer_neuron_num),
    torch.nn.ReLU(True),
    torch.nn.Linear(hidden_layer_neuron_num, len(y_col_names)),
    # torch.nn.ReLU(True)
    # torch.nn.Flatten(0, 3)
)
# model.cuda()
loss_fn = torch.nn.MSELoss()
learning_rate = 0.001
optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
for step in range(1000000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    # if loss.item() < 0.012:
    #     break
    if step % 100 == 1:
        print(step, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# rand_num = np.random.randint(0,len(df)-200)
rand_num = 24*12
df_test = df.iloc[len(df)-24*7:]
x_test = torch.FloatTensor(df_test.loc[:, x_col_names].values)
y_test = torch.FloatTensor(df_test.loc[:, y_col_names].values)
y_hat = model(x_test)
# print(y_hat)
loss = loss_fn(y_hat, y_test)
print(loss)
plt.figure()
plt.plot(y_hat.detach().numpy(), label='predict')
# plt.plot(np.array(df_test['prev_temp']), label='prev_temp')
plt.plot(y_test.detach().numpy(), label='real')
plt.legend()
plt.xlabel('Hour [h]')
plt.ylabel('Internal Temperature [℃]')
plt.title('Internal Temperature prediction with NN (winter)')
plt.savefig('bp winter heat')
plt.show()

MODEL_PATH = 'model/bp_heat.pth'
torch.save(model, MODEL_PATH)

late_winter_period = (0,24*105)
spring_period = (24*105,24*157)
summer_period = (24*157,24*243)
fall_period = (24*243,24*303)
early_winter_period = (24*303,24*365)