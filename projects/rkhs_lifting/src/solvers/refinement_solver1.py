import numpy as np

from projects.rkhs_lifting.src.integrators.multi.local_regular import Local_Regular
from projects.rkhs_lifting.src.solvers import Joint_Volume_Rots_Solver

class Refinement_Solver(Joint_Volume_Rots_Solver):
    def __init__(self,
                 vol=None,
                 squared_noise_level=None,
                 rots=None,
                 stop=None,
                 images=None,
                 filter=None,
                 amplitude=None,
                 integrator=None,
                 dtype=np.float32,
                 seed=0,
                 ):

        plan = Refinement_Plan1()

        super().__init__(plan=plan)

    def stop_solver(self):
        # TODO this one should probably go better elsewhere since it is quite default
        return self.iter == self.plan.o.stop  # TODO this now only works since we assume that this is an integer

    def step_solver(self):
        self.rots_step()
        # self.cost.append(self.plan.get_cost())

        self.volume_step()
        # self.cost.append(self.plan.get_cost())

    def finalize_solver(self):
        print("Solver has finished")

    def rots_step(self):
        L = self.plan.p.L
        N = self.plan.p.N
        dtype = self.plan.p.dtype

        # Construct sampling sets for all images
        multi_integrator = Local_Regular(quaternions=self.plan.o.quaternions, dtype=dtype)  # TODO l and sep_dist
        rots = multi_integrator.rots  # (n', N, 3, 3)

        # TODO compute error terms (n',N)
        #  - same routine as with rots density step in other solvers
        #  -- Need forward map here though that allows us to input rotstions (Plan)
        #  -- We can just reshape the rots array into a vector and do it like before, then rescale

        # TODO start loop here (first scalable, then better integration, e.g., through new sampling sets per iter)

        # TODO compute gradients (n',N, 4)
        #  - Not in KeOps this time
        #  -- ror RCK compute the matrix with weights (n',N) and multiply with logs (n', N, 4)

        # TODO reduce gradients (N, 4)
        #  - mult and sum over correct axis (=0)

        # TODO descent step (N,4)


    def volume_step(self):
        L = self.plan.p.L
        n = self.plan.p.n
        dtype = self.plan.p.dtype

        # TODO compute backprojection on rots

        # TODO the rest should be basically the same as before