import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#

def test_data():
    return {"k": perm, "bc": bc, "source": source, "vector_source": vector_source, "tol": 1e-6}

# ------------------------------------------------------------------------------#

def perm(g, d, flow_solver):
    # set a fake permeability for the 0d grids
    if g.dim == 0:
        return np.zeros(g.num_cells)

    # cell flux
    flux_norm = np.linalg.norm(d[pp.STATE][flow_solver.P0_flux], axis=0)

    # here is the condition satisfied with a < which is \Omega_1
    if g.tags["condition"] == 1:
        return np.ones(g.num_cells)
    # here is the condition satisfied with a > which is \Omega_2
    elif g.tags["condition"] == 0:
        return 1/(0.01 + 3*flux_norm)
    else:
        import pdb; pdb.set_trace()
        raise ValueError

# ------------------------------------------------------------------------------#

def source(g, d, flow_solver):
    # set zero source term for the 0d grids
    if g.dim == 0:
        return np.zeros(g.num_cells)

    if g.tags["original_id"] == 0 or g.tags["original_id"] == 1:
        dx = 0
    else:
        dx = 1
    cond1 = g.cell_centers[dx, :] <= 0.3
    cond2 = np.logical_and(g.cell_centers[dx, :] > 0.3,
                           g.cell_centers[dx, :] < 0.7)
    cond3 = g.cell_centers[dx, :] >= 0.7

    rhs = g.cell_volumes.copy()
    rhs[cond1] *= 1
    rhs[cond2] *= -1
    rhs[cond3] *= 1

    return rhs

# ------------------------------------------------------------------------------#

def vector_source(g, d, flow_solver):

    if g.dim == 0:
        return np.zeros((g.num_cells, 3))

    cell_volumes = g.cell_volumes.copy()

    vect = np.vstack(
            (0*cell_volumes, np.zeros(g.num_cells), np.zeros(g.num_cells))
            ).ravel(order="F")
    return vect

# ------------------------------------------------------------------------------#


def bc(g, data, tol, flow_solver):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # define outflow type boundary conditions
    out_flow = np.logical_or.reduce((b_face_centers[0] > 1 - tol,
                                     b_face_centers[1] > 1 - tol,
                                     b_face_centers[1] < 0 + tol))

    # define inflow type boundary conditions
    in_flow = b_face_centers[0] < 0 + tol

    # define the labels and values for the boundary faces
    labels = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(g.num_faces)

    labels[in_flow] = "dir"
    labels[out_flow] = "dir"
    bc_val[b_faces[in_flow]] = 0
    bc_val[b_faces[out_flow]] = 0.1

    return labels, bc_val

# ------------------------------------------------------------------------------#
