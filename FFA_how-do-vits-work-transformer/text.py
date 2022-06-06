import numpy as np
from mpl_toolkits.mplot3d import Axes3D

############### Plot the Loss Landscape ###############
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# load losslandscape raw data of ResNet-50 or ViT-Ti
names = ["x", "y", "l1", "l2", "loss_value"]
# path = "%s/resources/results/cifar100_resnet_dnn_50_losslandscape.csv" % root  # for ResNet-50
path = r'C:\Users\pc\Desktop\how-do-vits-work-transformer\checkpoints\NH\NH_Uformer_Informer_x1_losslandscape.csv'
data = pd.read_csv(path, names=names)
data["loss"] = data["loss_value"]

path = r'C:\Users\pc\Desktop\how-do-vits-work-transformer\checkpoints\NH\middle_result.txt'
x, y = [], []
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    data_x, data_y, loss = [], [], []
    for line in lines:  # Grid:  [ 0.5 -0.9], loss_value: 1.8769
        loss.append(eval(line.split('loss_value: ')[1].strip()))
    xs = np.linspace(-1, 1, 21)
    ys = np.linspace(-1, -0.6, 5)
    ratio_grid = np.stack(np.meshgrid(xs, ys), axis=0).transpose((1, 2, 0))  # (2, 11, 11)->(11, 11, 2)
    for ratio in ratio_grid.reshape([-1, 2]):
        x.append(ratio[0])
        y.append(ratio[1])


x.extend(data["x"].tolist())
y.extend(data["y"].tolist())
loss.extend(data["loss"].tolist())
print(len(loss))

# prepare data
p = int(math.sqrt(len(loss)))
shape = [p, p]
print(shape)

xs = np.array(x).reshape(shape)
ys = np.array(y).reshape(shape)
zs = np.array(loss).reshape(shape)

zs = zs - zs[np.isfinite(zs)].min()
zs[zs > 42] = np.nan

# Normalize(vmin=None, vmax=None) 是用来把数据标准化(归一化)到[0,1]这个期间内,
# vmin是设置最小值, vmax是设置最大值，小于最小值就取最小值，大于最大值就取最大值。
norm = plt.Normalize(zs[np.isfinite(zs)].min(), zs[np.isfinite(zs)].max())  # normalize to [0,1]
colors = cm.plasma(norm(zs))
rcount, ccount, _ = colors.shape

fig = plt.figure(figsize=(4.2, 4), dpi=120)
ax = fig.gca(projection="3d")
ax.view_init(elev=15, azim=15)  # angle

# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

surf = ax.plot_surface(
    xs, ys, zs,
    rcount=rcount, ccount=ccount,
    facecolors=colors, shade=False,
)
surf.set_facecolor((0, 0, 0, 0))

# remove white spaces
adjust_lim = 0.8
ax.set_xlim(-1 * adjust_lim, 1 * adjust_lim)
ax.set_ylim(-1 * adjust_lim, 1 * adjust_lim)
ax.set_zlim(10, 32)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.axis('off')

# plt.savefig('save.jpg')
plt.show()
