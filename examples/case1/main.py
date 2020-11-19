# definizione problema 1d con interfaccia 0d, ovvero due "fratture" con un accoppiamento
# imporre le equazioni, l'accoppiamento 0d deve essere di continuita' flusso e pressione
import numpy as np
import porepy as pp
import scipy.sparse as sps

from data import test_data

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

# QUANDO AVRO' IL FORCHHAIMER, FACCIO UNA ZONA AD ALTO INFLO E UNA A BASSO INFLO.
# MI ASPETTO CHE IL FORCH PRENDA LA ZONA AD ALTO INFLO MENTRE IL DARCY QUELLA A BASSO INFO.
# CHIARAMENTE MI DIPENDE DALLA CONDIZIONE DI SOGLIA

def main():

    # tolerance in the computation
    tol = 1e-10

    # assign the flag for the low permeable fractures
    mesh_size = 1e-3
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

    # WE SET NOW THE INITIAL CONDITION BY ASSUMING A P0_FLUX KNOWN

    # set the condition, meaning if for a branch we solve problem with a < (1) or with > (0)
    # for simplicity we just set all equal
    network.tags["condition"] = np.ones(network.num_frac, dtype=np.int)

    # DA SISTEMARE IL FATTO CHE POTREBBE NON GENERARE INTERFACCE

    flux_threshold = 0.15
    cond = lambda flux, op, tol=0: condition_interface(flux_threshold, flux, op, tol)

    file_name = "case1"
    folder_name = "./solution/"
    variable_to_export = [Flow.pressure, Flow.P0_flux, "original_id", "condition"]

    iteration = 0
    max_iteration = 100
    okay = False
    while not okay:

        print("iteration", iteration)

        # create the grid bucket
        #pp.plot_fractures(network.pts, network.edges, network.domain, np.arange(network.edges.shape[1]))
        gb = network.mesh(mesh_kwargs, dfn=True, preserve_fracture_tags=["original_id", "condition"])

        # create the discretization
        discr = Flow(gb)
        discr.set_data(test_data())

        # problem solution
        A, b = discr.matrix_rhs()
        x = sps.linalg.spsolve(A, b)
        discr.extract(x)

        # exporter
        save = pp.Exporter(gb, "sol_" + file_name, folder_name=folder_name)
        save.write(variable_to_export, time_step=iteration)

        # save the network points to check if we have reached convergence
        old_network_pts = network.pts

        # construct the new network such that the interfaces are respected
        network = detect_interface(gb, network, network_original, discr, cond, tol)
        # export the current network with the associated tags
        network_file_name = make_file_name(file_name, iteration)
        network.write(network_file_name, data=network.tags, folder_name=folder_name)

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
