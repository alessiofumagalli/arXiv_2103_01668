import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#

def test_data():
    return {"k": 1, "bc": bc, "source": source, "tol": 1e-6}

# ------------------------------------------------------------------------------#

def bc(g, data, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # define outflow type boundary conditions
    out_flow = b_face_centers[0] > 1 - tol

    # define inflow type boundary conditions
    in_flow = b_face_centers[0] < 0 + tol

    # define the labels and values for the boundary faces
    labels = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(g.num_faces)

    labels[in_flow] = "dir"
    labels[out_flow] = "dir"
    bc_val[b_faces[in_flow]] = 0
    bc_val[b_faces[out_flow]] = 0

    return labels, bc_val

# ------------------------------------------------------------------------------#

def source(g, data, tol):
    return np.ones(g.num_cells)

# ------------------------------------------------------------------------------#
