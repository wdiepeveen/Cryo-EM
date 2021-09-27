from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 10000
x = np.random.standard_normal(n)
y = np.random.standard_normal(n)
z = np.random.standard_normal(n)
c = np.random.standard_normal(n)+3
inds = (0<x) *(x<1)*(-0.5<y) *(y <-0.1)*(0.8<z) *(z<1.5)
c[inds] = 20

# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False
mask = c > 1.
alphas = (c > 0.) * c +0.1
img = ax.scatter(x, y, z, c=c, cmap=plt.cool(), alpha=0.1)  #
plt.tight_layout()

# img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
fig.colorbar(img)
plt.show()

# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# import matplotlib.pyplot as plt
#
# mpl.rcParams['legend.fontsize'] = 10
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
# z = np.linspace(-2, 2, 100)
# r = z**2 + 1
# x = r * np.sin(theta)
# y = r * np.cos(theta)
# ax.plot(x, y, z, label='parametric curve')
# ax.legend()
#
# ax.set_xlabel('$X$', fontsize=20)
# ax.set_ylabel('$Y$')
# ax.yaxis._axinfo['label']['space_factor'] = 3.0
# # set z ticks and labels
# ax.set_zticks([-2, 0, 2])
# # change fontsize
# for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(10)
# # disable auto rotation
# ax.zaxis.set_rotate_label(False)
# ax.set_zlabel('$\gamma$', fontsize=30, rotation = 0)
# plt.show()