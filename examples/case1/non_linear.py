import numpy as np
import porepy as pp
import scipy.sparse as sps

from non_linear_data import test_data

import sys; sys.path.insert(0, "../../src/")
from flow import Flow
from interface import detect_interface
from exporter import write_network_pvd, make_file_name

# ------------------------------------------------------------------------------#

def condition_interface(flux_threshold, flux, op, tol=0):
    norm = np.linalg.norm(flux)
    if op == "<":
        return (norm - flux_threshold) < tol
    elif op == ">":
        return (norm - flux_threshold) > - tol
    elif op == "==":
        return (norm - flux_threshold) > - tol and (norm - flux_threshold) < tol

# ------------------------------------------------------------------------------#

def main():

    # tolerance in the computation
    tol = 1e-10

    # assign the flag for the low permeable fractures
    mesh_size = 1e-2
    tol_network = mesh_size
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

    # read and mark the original fracture network, the fractures id will be preserved
    file_name = "network.csv"
    domain = {"xmin": 0, "xmax": 1, "ymin": -1, "ymax": 1}
    network = pp.fracture_importer.network_2d_from_csv(file_name, domain=domain)
    # set the original id
    network.tags["original_id"] = np.arange(network.num_frac, dtype=np.int)
    # save the original network
    network_original = network.copy()

    # set the condition, meaning if for a branch we solve problem with a < (1) or with > (0)
    # for simplicity we just set all equal
    network.tags["condition"] = np.ones(network.num_frac, dtype=np.int)

    flux_threshold = 0.15
    cond = lambda flux, op, tol=0: condition_interface(flux_threshold, flux, op, tol)

    file_name = "case1"
    folder_name = "./non_linear/"
    variable_to_export = [Flow.pressure, Flow.P0_flux, "original_id", "condition"]

    iteration = 0
    max_iteration = 50
    max_iteration_non_linear = 50
    max_err_non_linear = 1e-4
    okay = False
    while not okay:

        print("iteration", iteration)

        # create the grid bucket
        gb = network.mesh(mesh_kwargs, dfn=True, preserve_fracture_tags=["original_id", "condition"])

        # create the discretization
        discr = Flow(gb)

        # the mesh is changed as well as the interface, do not use the solution at previous step
        # initialize the non-linear algorithm by setting zero the flux which is equivalent to get
        # the Darcy solution at the first iteration
        for g, d in gb:
            d.update({pp.STATE: {}})
            d[pp.STATE].update({Flow.P0_flux: np.zeros((3, g.num_cells))})
            d[pp.STATE].update({Flow.P0_flux + "_old": np.zeros((3, g.num_cells))})

        # non-linear problem solution with a fixed point strategy
        err_non_linear = max_err_non_linear + 1
        iteration_non_linear = 0
        while err_non_linear > max_err_non_linear and iteration_non_linear < max_iteration_non_linear:

            # solve the linearized problem
            discr.set_data(test_data())
            A, b = discr.matrix_rhs()
            x = sps.linalg.spsolve(A, b)
            discr.extract(x)

            # compute the exit condition
            all_flux = np.empty((3, 0))
            all_flux_old = np.empty((3, 0))
            all_cell_volumes = np.empty(0)
            for g, d in gb:
                # collect the current flux
                flux = d[pp.STATE][Flow.P0_flux]
                all_flux = np.hstack((all_flux, flux))
                # collect the old flux
                flux_old = d[pp.STATE][Flow.P0_flux + "_old"]
                all_flux_old = np.hstack((all_flux_old, flux_old))
                # collect the cell volumes
                all_cell_volumes = np.hstack((all_cell_volumes, g.cell_volumes))
                # save the old flux
                d[pp.STATE][Flow.P0_flux + "_old"] = flux

            # compute the error and normalize the result
            err_non_linear = np.sum(all_cell_volumes * np.linalg.norm(all_flux - all_flux_old, axis=0))
            norm_flux_old = np.sum(all_cell_volumes * np.linalg.norm(all_flux_old, axis=0))
            err_non_linear = err_non_linear / norm_flux_old if norm_flux_old != 0 else err_non_linear

            print("iteration non-linear problem", iteration_non_linear, "error", err_non_linear)
            iteration_non_linear += 1

        # exporter
        save = pp.Exporter(gb, "sol_" + file_name, folder_name=folder_name)
        save.write_vtu(variable_to_export, time_step=iteration)

        # save the network points to check if we have reached convergence
        old_network_pts = network.pts

        # construct the new network such that the interfaces are respected
        network = detect_interface(gb, network, network_original, discr, cond, tol)
        # export the current network with the associated tags
        network_file_name = make_file_name(file_name, iteration)
        network.to_file(network_file_name, data=network.tags, folder_name=folder_name, binary=False)

        # check if any point in the network has changed
        all_pts = np.hstack((old_network_pts, network.pts))
        distances = pp.distances.pointset(all_pts) > tol_network
        # consider only the block between the old and new points
        distances = distances[:old_network_pts.shape[1], -network.pts.shape[1]:]
        # check if an old point has a point equal in the new set
        check = np.any(np.logical_not(distances), axis=0)

        if np.all(check) or iteration > max_iteration:
            okay = True
        iteration += 1

    save.write_pvd(np.arange(iteration), np.arange(iteration))
    write_network_pvd(file_name, folder_name, np.arange(iteration))

if __name__ == "__main__":
    main()
