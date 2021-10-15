import numpy as np
import logging

from projects.rkhs_lifting.src.integrators.multi.local_regular import Local_Regular
from projects.rkhs_lifting.src.solvers import Joint_Volume_Rots_Solver

logger = logging.getLogger(__name__)


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

        n = multi_integrator.n

        # Compute q:
        logger.info("Computing qs")
        im = self.plan.p.images.asnumpy()
        qs = np.zeros((n, N), dtype=self.plan.p.dtype)
        logger.info("Construct qs with batch size {}".format(self.plan.o.rots_batch_size))
        q3 = np.sum(im ** 2, axis=(1, 2))
        for img_ind in range(0, self.plan.p.N):
            for start in range(0, n, self.plan.o.rots_batch_size):
                logger.info(
                    "Image {} | Running through projections {}/{} = {}%".format(img_ind, start, n,
                                                                                np.round(start / n * 100, 2)))
                all_idx = np.arange(start, min(start + self.plan.o.rots_batch_size, n))
                selected_rots = rots[all_idx, img_ind, :, :].squeeze()
                rots_sampling_projections = self.plan.forward(self.plan.o.vol, selected_rots).asnumpy()

                q1 = np.sum(rots_sampling_projections ** 2, axis=(1, 2))
                q2 = - 2 * np.einsum("jk,gjk->g", im[img_ind, :, :], rots_sampling_projections)

                qs[all_idx, img_ind] = (q1 + q2 + q3) / (2 * self.plan.o.squared_noise_level * L ** 2)

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
