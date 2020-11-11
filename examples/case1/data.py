import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#

def test_data():
    return {"k": perm, "bc": bc, "source": source, "tol": 1e-6}

# ------------------------------------------------------------------------------#

def perm(g, d):
    # set a fake permeability for the 0d grids
    if g.dim == 0:
        return np.zeros(g.num_cells)

    # here is the condition satisfied with a <
    if g.tags["condition"] == 1:
        return 1 * np.ones(g.num_cells)
    # here is the condition satisfied with a >
    elif g.tags["condition"] == 0:
        return 1e2 * np.ones(g.num_cells)
    else:
        import pdb; pdb.set_trace()
        raise ValueError

# ------------------------------------------------------------------------------#

def source(g, d):
    # set zero source term for the 0d grids
    if g.dim == 0:
        return np.zeros(g.num_cells)

    # here is the condition satisfied with a <
    if g.tags["condition"] == 1:
        return 1 * g.cell_volumes
    # here is the condition satisfied with a >
    elif g.tags["condition"] == 0:
        return 1 * g.cell_volumes
    else:
        import pdb; pdb.set_trace()
        raise ValueError

# ------------------------------------------------------------------------------#

def bc(g, data, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # define outflow type boundary conditions
    out_flow = np.logical_or(b_face_centers[0] > 1 - tol,
                             b_face_centers[1] > 1 - tol)

    # define inflow type boundary conditions
    in_flow = np.logical_or(b_face_centers[0] < 0 + tol,
                            b_face_centers[1] < 0 + tol)

    # define the labels and values for the boundary faces
    labels = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(g.num_faces)

    labels[in_flow] = "dir"
    labels[out_flow] = "dir"
    bc_val[b_faces[in_flow]] = 0
    bc_val[b_faces[out_flow]] = 0

    return labels, bc_val

# ------------------------------------------------------------------------------#
