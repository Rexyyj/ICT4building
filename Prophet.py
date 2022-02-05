import pandas as pd
from prophet import Prophet
import plotly
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

rolling_window_size = 24 * 7*2
train_size = rolling_window_size
start_idx = 24*7*3

late_winter_period = (0,24*105)
spring_period = (24*105,24*157)
summer_period = (24*157,24*243)
fall_period = (24*243,24*303)
early_winter_period = (24*303,24*365)

def stan_init(m):
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res

df = pd.read_csv('C:\\Users\\maqly\\Desktop\\ict4bd lab\\data.csv')
y_name = 'Indoor Mean Air Temperature [C](Hourly)'
ds_name = 'Date/Time'
df = df[[y_name, ds_name]]
df.rename(columns={y_name: 'y', ds_name: 'ds'}, inplace=True)
df['weekday_num'] = pd.to_datetime(df['ds'])
df['weekday_num'] = df['weekday_num'].dt.dayofweek
df['is_weekend'] = df['weekday_num'].apply(lambda x: 0 if x <= 4 else 1)
df['is_weekday'] = df['weekday_num'].apply(lambda x: 1 if x <= 4 else 0)
df.drop(columns=['weekday_num'], inplace=True)

print(df)

m = Prophet()
# m.add_seasonality(name='daily', period=24, fourier_order=24 * 7 * 2, condition_name='daily_1')
m.add_seasonality(name='weekend', period=7, fourier_order=24 * 7 * 2, condition_name='is_weekend')
# m.add_seasonality(name='weekday', period=7 * 24, fourier_order=24*7*2, condition_name='is_weekday')
train_df = df.loc[start_idx:start_idx+train_size-1]
# train_df = df
m.fit(train_df)

# yhats = []
# pred_periods = 24
# for i in range(pred_periods):
#     train_df = df.iloc[start_idx + i: start_idx + i + rolling_window_size - 1]
#     if i > 0:
#         # print(init_params)
#         # m = Prophet().fit(train_df, init=stan_init(m))
#         m = Prophet().fit(train_df)
#     else:
#         m.fit(train_df)
#     future = df.iloc[start_idx + i + len(train_df)].to_frame().T
#     print(future)
#     forecast = m.predict(future)
#     assert len(forecast) == 1
#     yhats.append(forecast.loc[0, 'yhat'])
#
# assert len(yhats) == pred_periods
# plt.figure()
# plt.plot(yhats, label='pred')
# ys = df.loc[start_idx + rolling_window_size: start_idx + rolling_window_size + pred_periods-1, 'y']
# ys.index = range(len(ys))
# plt.plot(ys, label='real')
# plt.legend()
# plt.show()


# df_cv = cross_validation(m, initial=f'{24*7*4} hours', period='1 hour', horizon = '1 hour')

pred_period = 24
future = m.make_future_dataframe(periods=pred_period, freq='H')

# future = future.iloc[-periods:]
future['weekday_num'] = pd.to_datetime(future['ds'])
future['weekday_num'] = future['weekday_num'].dt.dayofweek
future['is_weekend'] = future['weekday_num'].apply(lambda x: 0 if x <= 4 else 1)
future['is_weekday'] = future['weekday_num'].apply(lambda x: 1 if x <= 4 else 0)
future.drop(columns=['weekday_num'], inplace=True)
forecast = m.predict(future)
# print(forecast)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast, xlabel='Time', ylabel='Internal Temperature [℃]')
fig1.savefig('forecast')
fig1.show()

figure = plt.figure()
y_hat = forecast.tail(pred_period)['yhat']
print(y_hat)
y_hat.reset_index(inplace=True,drop=True)
plt.plot(y_hat,label='predict')
y = df.loc[start_idx+train_size:start_idx + train_size+pred_period-1,'y']
y.reset_index(inplace=True,drop=True)
assert len(y)==pred_period
plt.plot(y,label='real')
plt.legend()
plt.xlabel('Hour (h)')
plt.ylabel('Internal Temperature (℃)')
plt.savefig('prophet两周预测一天 winter')
plt.title('Internal Temperature in winter')
plt.show()
print(f'mse: {mse(y,y_hat)}')
print(f'r2: {r2_score(y,y_hat)}')

# Python
# from prophet.plot import add_changepoints_to_plot
# fig = m.plot(forecast)
# a = add_changepoints_to_plot(fig.gca(), m, forecast)
# fig.show()

fig = m.plot_components(forecast)
fig.savefig('trend')
fig.show()
