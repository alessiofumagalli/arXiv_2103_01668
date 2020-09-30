import numpy as np
import porepy as pp
import scipy.sparse as sps

# ------------------------------------------------------------------------------#

def detect_interface(gb, network, scheme, condition_interface):

    # new points to define the domains
    dom_pts = dict.fromkeys(network.tags["original_id"], np.empty(0))
    dom_R = dict.fromkeys(network.tags["original_id"], np.empty((3, 3)))

    # do only this computation only for the higher dimensional grids
    for g in gb.grids_of_dimension(gb.dim_max()):
        # get the data for the current grid
        d = gb.node_props(g)

        # for the current grid compute the flux at each face, 1 co-dimensional grid edge included
        flux = _get_flux(g, d, gb.edges_of_node(g), scheme.flux, scheme.mortar)

        # compute the new points according to the interface
        f_id = g.tags["original_id"]
        new_pts, dom_R[f_id] = _has_interface(g, d, scheme, flux, condition_interface)

        # save the points
        dom_pts[f_id] = np.hstack((dom_pts[f_id], new_pts))

    # for each orginal id order the points

    print(dom_pts)
    print(dom_R)
    import pdb; pdb.set_trace()

        # devo considerare tutte le griglie che originariamente si chiamavano in un modo e poi farne un eventuale merge,
        # altrimenti rischio che mi esplode il loro numero

# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#

def _get_flux(g, d, g_edges, flux_name, mortar_name):
    # we need to recover the flux from the mortar variable
    # only lower dimensional edges need to be considered.
    flux = d[pp.STATE][flux_name]
    if np.any(g.tags["fracture_faces"]):
        # recover the sign of the flux, since the mortar is assumed
        # to point from the higher to the lower dimensional problem
        _, indices = np.unique(g.cell_faces.indices, return_index=True)
        sign = sps.diags(g.cell_faces.data[indices], 0)

        for _, d_e in g_edges:
            g_m = d_e["mortar_grid"]
            if g_m.dim == g.dim:
                continue
            # project the mortar variable back to the higher dimensional
            # problem
            flux += (
                sign * g_m.master_to_mortar_avg().T * d_e[pp.STATE][mortar_name]
            )

    return flux

# ------------------------------------------------------------------------------#

def _has_interface(g, d, scheme, flux, condition_interface, evaluation_tol=1e-5, evaluation_max_iter=100):
    discr = d[pp.DISCRETIZATION][scheme.variable][scheme.discr_name]

    # detect if the condition is met, since the velocity is linear then we first check in the
    # boundaries of each grid element
    faces, cells, sign = sps.find(g.cell_faces)
    index = np.argsort(cells)
    faces, sign = faces[index], sign[index]

    # Map the domain to a reference geometry (i.e. equivalent to compute
    # surface coordinates in 1d and 2d)
    _, f_normals, f_centers, R, dim, node_coords = pp.map_geometry.map_grid(
        g, tol = 1e-5
    )
    node_coords = node_coords[: g.dim, :]

    # get the opposite face-nodes
    cell_face_to_opposite_node = d[discr.cell_face_to_opposite_node]

    # list of grid points + interface points with the condition intersection label
    interface_pts = np.empty((f_centers.shape[0]+1, 0))

    for c in np.arange(g.num_cells):
        # For the current cell retrieve its faces
        loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
        faces_loc = faces[loc]

        # get the opposite node id for each face
        node = cell_face_to_opposite_node[c, :]
        coord_loc = node_coords[:, node]

        # check what's happening in the cell-boundary, since the flux is maximum there
        check = np.zeros(faces_loc.size, dtype=np.bool)
        for idx, face_loc in enumerate(faces_loc):
            P = scheme.discr.faces_to_cell(
                    f_centers[:, face_loc],
                    coord_loc,
                    f_centers[:, faces_loc],
                    f_normals[:, faces_loc],
                    dim,
                    R,
                )
            check[idx] = condition_interface(np.dot(P, flux[faces_loc]), "<")

        # if the condition is valid for a cell-boundary we might need to compute internally
        if not(np.all(check) or np.all(np.logical_not(check))) and np.any(check):
            #NOTE: this algorithm may work only in 1d
            bd_pts = f_centers[:, faces_loc]
            not_okay = True
            it = 0
            while not_okay and it < evaluation_max_iter:
                # bisection method to get the actual value of the interface
                probe_pt = np.average(bd_pts, axis=1)
                P = scheme.discr.faces_to_cell(
                        probe_pt,
                        coord_loc,
                        f_centers[:, faces_loc],
                        f_normals[:, faces_loc],
                        dim,
                        R,
                    )
                if condition_interface(np.dot(P, flux[faces_loc]), "==", evaluation_tol):
                    not_okay = False
                else:
                    cond = condition_interface(np.dot(P, flux[faces_loc]), "<")
                    bd_pts[:, check == cond] = probe_pt
                it += 1
            # recover the 3d representation of the found point
            interface_pts = np.hstack((interface_pts, np.vstack((probe_pt, True))))

    # collect all the pts for the current 1d domain
    all_pts = np.vstack((f_centers, np.zeros(f_centers.shape[1])))
    all_pts = np.hstack((all_pts, interface_pts))

    # order the pts
    mask = np.argsort(all_pts[0, :])
    all_pts = all_pts[:, mask]

    # create the new points for the new division, we take the boundary and all the interface pts
    mask = all_pts[1, :] == 1
    mask[0] = True
    mask[-1] = True

    return all_pts[0, mask], R

    ## map back all the found pts
    #new_pts = np.zeros((3, np.sum(mask)))
    #new_pts[dim, :] = all_pts[0, mask]

    #return np.dot(R.T, new_pts)

# ------------------------------------------------------------------------------#
