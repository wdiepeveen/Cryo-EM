import numpy as np
import matplotlib.pyplot as plt
import os


from tools.exp_tools import Exp
# from tools.dbg import dbglevel,dbg,dbglevel_atleast

exp = Exp()
exp.begin(aux_folders=["demos", "tools"], prefix="ExpToolsTest")
exp.dbglevel(4)

exp.dbg(0,"Results folder is " + exp.folder())

exp.dbg(0,"This is an experiment")

# exp_comment("Test");

exp.dbg(1, "Important message")

for i in range(10000):
    i = i + 1
    i = i - 1


exp.dbg(4, "Less important")

x = np.arange(1000)
f = (x - 50) ** 2
plt.plot(x, f)

exp.savefig("someresult",save_eps=True)


exp.save("x-and-f", x, f)