# TODO does not work!
def rots_grid(dims):

    grid = np.ceil(np.arange(-n / 2, n / 2, dtype=dtype))

    if shifted and n % 2 == 0:
        grid = np.arange(-n / 2 + 1 / 2, n / 2 + 1 / 2, dtype=dtype)

    if normalized:
        if shifted and n % 2 == 0:
            grid = grid / (n / 2 - 1 / 2)
        else:
            grid = grid / (n / 2)

    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    phi, theta, r = cart2sph(x, y, z)

    # TODO: Should this theta adjustment be moved inside cart2sph?
    theta = np.pi / 2 - theta

    return {"x": x, "y": y, "z": z, "phi": phi, "theta": theta, "r": r}

    np.random.random(n) * 2 * np.pi,
    np.arccos(2 * np.random.random(n) - 1),
    np.random.random(n) * 2 * np.pi,

    k=2