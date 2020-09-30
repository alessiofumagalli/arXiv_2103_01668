# definizione problema 1d con interfaccia 0d, ovvero due "fratture" con un accoppiamento
# imporre le equazioni, l'accoppiamento 0d deve essere di continuita' flusso e pressione
import numpy as np
import porepy as pp
import scipy.sparse as sps

from data import test_data

import sys; sys.path.insert(0, "../../src/")
from flow import Flow
from interface import detect_interface

# ------------------------------------------------------------------------------#

def condition_interface(flux, op, tol=0):
    flux_threshold = 1.1
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

    # assign the flag for the low permeable fractures
    mesh_size = 0.25
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

    # read and mark the original fracture network, the fractures id will be preserved
    file_name = "network.csv"
    network = pp.fracture_importer.network_2d_from_csv(file_name)
    network.tags["original_id"] = np.arange(network.num_frac)

    import pdb; pdb.set_trace()

    iteration = 0
    okay = False
    while not okay:

        # create the grid bucket
        gb = network.mesh(mesh_kwargs, dfn=True, preserve_fracture_tags="original_id")

        # create the discretization
        discr = Flow(gb)
        discr.set_data(test_data())

        # problem solution
        A, b = discr.matrix_rhs()
        x = sps.linalg.spsolve(A, b)
        discr.extract(x)

        # construct the new network such that the interfaces are respected
        network = detect_interface(gb, network, discr, condition_interface)

        # exporter
        save = pp.Exporter(gb, "case1", folder_name="solution")
        save.write_vtk([discr.pressure, discr.P0_flux], time_step=iteration)

        if iteration > -1: # exit condition to check
            okay = True
        else:
            network.pts[:, interface_node] *= np.random.rand()

        iteration += 1

    save.write_pvd(np.arange(iteration), np.arange(iteration))

if __name__ == "__main__":
    main()
