import numpy as np
import matplotlib.pyplot as plt


from tools.exp_tools import exp_begin,exp_save,exp_savefig,exp_open,exp_save_mrc,exp_folder,exp_filename,exp_cancelled,exp_reset_timer
from tools.dbg import dbglevel,dbg,dbglevel_atleast


exp_begin(aux_folders=["demos", "tools"], prefix="ExpToolsTest")
dbglevel(4)

dbg(0,"Results folder is " + exp_folder())

dbg(0,"This is an experiment")

# exp_comment("Test");

dbg(1, "Important message")

for i in range(10000):
    i = i + 1
    i = i - 1


dbg(4, "Less important")

x = np.arange(1000)
f = (x - 50) ** 2
plot = plt.plot(x, f)

exp_savefig("someresult", plot)


exp_save("x-and-f", x, f)