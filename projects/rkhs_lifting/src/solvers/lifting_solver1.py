
from projects.rkhs_lifting.src.solvers import Joint_Volume_Rots_Solver

class RKHS_Lifting_Solver1(Joint_Volume_Rots_Solver):
    def __init__(self):

        super().__init__()
        # Constuct problem
        # p = problem()

        # Construct options
        # o = options

        # https: // github.com / JuliaManifolds / Manopt.jl / blob / e0ec985b5baf177b5f7d0570899cb66d960f1199 / src / solvers / ChambollePock.jl  # L1-L60

    def stop_solver(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def step_solver(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def finalize_solver(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def volume_step(self):
        k=2

    def rots_density_step(self):
        k=2

    # TODO also forward etc in here

    def forward(self):
        """
        Apply forward image model to volume
        :param vol: A volume instance.
        :param start: Start index of image to consider
        :param num: Number of images to consider
        :return: The images obtained from volume by projecting, applying CTFs, translating, and multiplying by the
            amplitude.
        """
        # print(type(self.vol))
        im = self.vol.project(0, self.integrator.rots)
        im = self.eval_filter(im)  # Here we only use 1 filter, but might as well do one for every entry
        # im = im.shift(self.offsets[all_idx, :])  # TODO use this later on
        im *= self.amplitude  # [im.n, np.newaxis, np.newaxis]  # Here we only use 1 amplitude,
                                # but might as well do one for every entry

        return im

    def adjoint_forward(self, im):
        """
        Apply adjoint mapping to set of images
        :param im: An Image instance to which we wish to apply the adjoint of the forward model.
        :param start: Start index of image to consider
        :return: An L-by-L-by-L volume containing the sum of the adjoint mappings applied to the start+num-1 images.
        """
        weights = self.integrator.coeffs2weights(self.rots_dcoef)

        integrands = Image(np.einsum("ig,ikl->gkl", weights, im.asnumpy()))
        integrands *= self.amplitude
        # im = im.shift(-self.offsets[all_idx, :])
        integrands = self.eval_filter(integrands)

        res = integrands.backproject(self.integrator.rots)[0]

        logger.info(f"Determined adjoint mappings. Shape = {res.shape}")
        return res
