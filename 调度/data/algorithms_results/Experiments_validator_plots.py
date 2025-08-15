import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.linalg
from sklearn.linear_model import LinearRegression
#from IPython.display import Image

results_data = pd.read_csv('algorithms_results_table/EV_experiments_validating.csv')
results_data = results_data.iloc[: , 1:]
results_data['underconsume'].astype(float)
results_data['overconsume'].astype(float)
results_data['overcost'].astype(float)
results_data['av_EV_energy_left'].astype(float)
results_data['cumulative_reward'].astype(float)
results_data.info()
results_data.tail(6)

results_data_=results_data.drop("Name", axis=1)
results_data_=results_data_.drop("overcost", axis=1)
results_data_=results_data_.drop("cumulative_reward", axis=1)
data = results_data_.values

underc_min = results_data_["underconsume"].values[np.argmin(results_data_["underconsume"].values)]
overc_min = results_data_["overconsume"].values[np.argmin(results_data_["overconsume"].values)]
underc_max = results_data_["underconsume"].values[np.argmax(results_data_["underconsume"].values)]
overc_max = results_data_["overconsume"].values[np.argmax(results_data_["overconsume"].values)]

# regular grid covering the domain of the data
X,Y = np.meshgrid(np.arange(underc_min, underc_max, 100), np.arange(overc_min, overc_max, 100))
XX = X.flatten()
YY = Y.flatten()
# best-fit quadratic curve
A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
# evaluate it on a grid
Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

# results_data_complete = results_data
# results_data = results_data.iloc[:-6]
# results_data.info()
fig = px.scatter_3d(results_data, x='underconsume', y='overconsume', z='av_EV_energy_left',
                    #size='scaled_cumulative_reward',
                    hover_data=['cumulative_reward'],
                    color='Name',
                    labels={
                     "av_EV_energy_left": "Av.EVs battery at departure(kWh)",
                     "Name": "number of weekly EVs arrival test",
                     "underconsume": "Unused RE-to-vehicle energy(kWh)",
                     "overconsume": "Grid energy used(kWh)"
                    })
#fig.update_layout(scene_zaxis_type="log")
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
#fig.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.5))
fig.update_layout(title_text='EVs arrivals experiment results scatter plot',  width=1400,height=700,)
# fig.update_layout(
#     scene = dict(
#         xaxis = dict(nticks=4, range=[-100,100],),
#                      yaxis = dict(nticks=4, range=[-50,100],),
#                      zaxis = dict(nticks=4, range=[-100,100],),),
#     width=700,
#     margin=dict(r=20, l=10, b=10, t=10))
#fig.show()

min_rew = results_data["cumulative_reward"].values[np.argmin(results_data["cumulative_reward"].values)]
max_rew = results_data["cumulative_reward"].values[np.argmax(results_data["cumulative_reward"].values)]
results_data["scaled_cumulative_reward"] = np.interp(results_data["cumulative_reward"].values, [min_rew, max_rew], [1, 50] )

fig = px.scatter_3d(results_data, x='underconsume', y='overconsume', z='av_EV_energy_left',
                    size='scaled_cumulative_reward',
                    hover_data=['cumulative_reward'],
                    color='Name',
                    labels={
                     "av_EV_energy_left": "Av.EVs battery at departure(kWh)",
                     "Name": "number of weekly EVs arrival test",
                     "underconsume": "Unused RE-to-vehicle energy(kWh)",
                     "overconsume": "Grid energy used(kWh)"
                    })
#fig.update_layout(scene_zaxis_type="log")
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
#fig.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.3))
fig.update_layout(title_text='EVs arrivals experiment results scatter plot (size being the scaled_cumulative_reward)',  width=1400,height=700,)

fig = px.scatter_3d(results_data, x='underconsume', y='overconsume', z='av_EV_energy_left',
                    #size='scaled_cumulative_reward',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    hover_data=['cumulative_reward','Name'],
                    color='cumulative_reward',
                    labels={
                     "av_EV_energy_left": "Av.EVs battery at departure(kWh)",
                     "Name": "Algorithm type",
                     "underconsume": "Unused RE-to-vehicle energy(kWh)",
                     "overconsume": "Grid energy used(kWh)"
                    })
#fig.update_layout(scene_zaxis_type="log")
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
#fig.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.3))
fig.update_layout(title_text='EVs arrivals experiment results scatter plot',  width=1400,height=700,)

from sklearn.linear_model import LinearRegression
min_overc = results_data["overconsume"].values[np.argmin(results_data["overconsume"].values)]
max_overc = results_data["overconsume"].values[np.argmax(results_data["overconsume"].values)]
# Get index for the second highest value.
min_overc2 = results_data["overconsume"].values[results_data["overconsume"].values.argsort()[1]]
#print(min_overc2)
results_data["inverted_overconsume"] = np.interp(results_data["overconsume"].values, [min_overc, min_overc2, max_overc], [1, 1, 0] )

fig = px.scatter(results_data, x='underconsume', y='av_EV_energy_left',
                    #color = 'scaled_cumulative_reward',
                    #color = 'inverted_overconsume',
                    #color_continuous_scale=px.colors.sequential.Viridis,
                    #size = 'scaled_cumulative_reward',
                    size='inverted_overconsume',
                    hover_data=['cumulative_reward','overconsume'],
                    color='Name',
                    #text="Name",
                    labels={
                     "av_EV_energy_left": "Av.EVs battery at departure(kWh)",
                     "Name": "number of weekly EVs arrival test",
                     "underconsume": "Unused RE-to-vehicle energy(kWh)",
                     "overconsume": "Grid energy used(kWh)",
                     "inverted_overconsume": "inverted Grid energy used[0-1]"
                    })

fig.update_layout(title_text=f'EVs arrivals experiment results scatter plot (size being the mapped Grid energy used [{round(min_overc/1000,1)}, {round(max_overc/1000,3)} MWh]-->[1,0])',  width=1600,height=800,)

fig = px.scatter(results_data, x='underconsume', y='av_EV_energy_left',
                    color = 'cumulative_reward',
                    #color = 'inverted_overconsume',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    #size = 'scaled_cumulative_reward',
                    size='inverted_overconsume',
                    hover_data=['cumulative_reward','overconsume'],
                    #color='Name',
                    #text="Name",
                    labels={
                     "av_EV_energy_left": "Av.EVs battery at departure(kWh)",
                     "Name": "number of weekly EVs arrival test",
                     "underconsume": "Unused RE-to-vehicle energy(kWh)",
                     "overconsume": "Grid energy used(kWh)",
                     "inverted_overconsume": "inverted Grid energy used[0-1]"
                    })

fig.update_layout(title_text=f'EVs arrivals experiment results scatter plot (size being the mapped Grid energy used [{round(min_overc/1000,1)}, {round(max_overc/1000,3)} MWh]-->[1,0])',  width=1500,height=800,)


min_overc = results_data["overconsume"].values[np.argmin(results_data["overconsume"].values)]
max_overc = results_data["overconsume"].values[np.argmax(results_data["overconsume"].values)]
# Get index for the second highest value.
min_overc2 = results_data["overconsume"].values[results_data["overconsume"].values.argsort()[1]]
#print(min_overc2)
results_data["inverted_overconsume"] = np.interp(results_data["overconsume"].values, [min_overc, min_overc2, max_overc], [1, 1, 0] )

# selecting rows based on condition
filtered_data = results_data.loc[results_data['inverted_overconsume'] > 0.98]

x = filtered_data['underconsume'].values
x = x.reshape((-1,1))
y = filtered_data['av_EV_energy_left'].values
model = LinearRegression().fit(x, y)
x_plot = np.arange(25000)
x_new = x_plot.reshape((-1, 1))
y_new = model.predict(x_new)

fig = px.scatter(results_data, x='underconsume', y='av_EV_energy_left',
                    #color = 'scaled_cumulative_reward',
                    color = 'overconsume',
                    #color_continuous_scale=px.colors.sequential.Viridis,
                    size = 'scaled_cumulative_reward',
                    #size='inverted_overconsume',
                    hover_data=['cumulative_reward','overconsume'],
                    #color='Name',
                    #text="Name",
                    labels={
                     "av_EV_energy_left": "Av.EVs battery at departure(kWh)",
                     "Name": "Algorithm type",
                     "underconsume": "Unused RE-to-vehicle energy(kWh)",
                     "overconsume": "Grid energy used(kWh)",
                     "inverted_overconsume": "inverted Grid energy used(0-1)"
                    })


#fig.update_layout(scene_zaxis_type="log")
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
#fig.add_trace(go.Scatter(x=x_plot, y=y_new, line={'color':'yellow'},
#                            #name="minimum overconsume regression line"
#                            ))
fig.update_layout(title_text='EVs arrivals experiment results scatter plot (size being the scaled_cumulative_reward)',  width=1600,height=800,)