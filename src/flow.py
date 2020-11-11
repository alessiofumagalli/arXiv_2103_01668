import numpy as np
import porepy as pp

class Flow(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, gb, model="flow"):

        self.model = model
        self.gb = gb
        self.data = None
        self.data_time = None
        self.assembler = None
        self.assembler_variable = None

        # discretization operator name
        self.discr_name = self.model + "_flux"
        self.discr = pp.RT0

        # coupling operator
        self.coupling_name = self.discr_name + "_coupling"
        self.coupling = pp.FluxPressureContinuity

        # source
        self.source_name = self.model + "_source"
        self.source = pp.DualScalarSource

        # master variable name
        self.variable = self.model + "_variable"
        self.mortar = self.model + "_lambda"

        # post process variables
        self.pressure = "pressure"
        self.flux = "darcy_flux"  # it has to be this one
        self.P0_flux = "P0_darcy_flux"

    # ------------------------------------------------------------------------------#

    def set_data(self, data):
        self.data = data

        for g, d in self.gb:
            param = {}

            d["deviation_from_plane_tol"] = 1e-4
            d["is_tangential"] = True

            # assign permeability
            k = self.data["k"] * np.ones(g.num_cells)

            # no source term is assumed by the user
            param["second_order_tensor"] = pp.SecondOrderTensor(kxx=k, kyy=1, kzz=1)
            param["source"] = data["source"](g, data, data["tol"])

            # Boundaries
            b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if b_faces.size:
                labels, param["bc_values"] = data["bc"](g, data, data["tol"])
                param["bc"] = pp.BoundaryCondition(g, b_faces, labels)
            else:
                param["bc_values"] = np.zeros(g.num_faces)
                param["bc"] = pp.BoundaryCondition(g, np.empty(0), np.empty(0))

            pp.initialize_data(g, d, self.model, param)

#        for e, d in self.gb.edges():
#            mg = d["mortar_grid"]
#            g_slave, g_master = self.gb.nodes_of_edge(e)
#            check_P = mg.slave_to_mortar_avg()
#
#            if "fracture" in g_slave.name or "fracture" in g_master.name:
#                aperture = self.gb.node_props(g_slave, pp.STATE)["fracture_aperture"]
#                aperture_initial = self.gb.node_props(g_slave, pp.STATE)["fracture_aperture_initial"]
#                k_n = self.data["fracture_k_n"]
#            else:
#                aperture = self.gb.node_props(g_slave, pp.STATE)["layer_aperture"]
#                aperture_initial = self.gb.node_props(g_slave, pp.STATE)["layer_aperture_initial"]
#                k_n = self.data["layer_k_n"]
#
#            k = 2 * check_P * (np.power(aperture/aperture_initial, alpha-1) * k_n)
#            pp.initialize_data(mg, d, self.model, {"normal_diffusivity": k})
#

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self):

        # set the discretization for the grids
        for g, d in self.gb:
            discr = self.discr(self.model)
            source = self.source(self.model)

            d[pp.PRIMARY_VARIABLES] = {self.variable: {"cells": 1, "faces": 1}}
            d[pp.DISCRETIZATION] = {self.variable: {self.discr_name: discr,
                                                    self.source_name: source}}
            d[pp.DISCRETIZATION_MATRICES] = {self.model: {}}

        # define the interface terms to couple the grids
        for e, d in self.gb.edges():
            g_slave, g_master = self.gb.nodes_of_edge(e)

            # retrive the discretization of the master and slave grids
            discr_master = self.gb.node_props(g_master, pp.DISCRETIZATION)[self.variable][self.discr_name]
            discr_slave = self.gb.node_props(g_slave, pp.DISCRETIZATION)[self.variable][self.discr_name]

            coupling = self.coupling(self.model, discr_master, discr_slave)

            d[pp.PRIMARY_VARIABLES] = {self.mortar: {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                self.coupling_name: {
                    g_slave: (self.variable, self.discr_name),
                    g_master: (self.variable, self.discr_name),
                    e: (self.mortar, coupling),
                }
            }
            d[pp.DISCRETIZATION_MATRICES] = {self.model: {}}

        # assembler
        self.assembler = pp.Assembler(self.gb)
        self.assembler.discretize()
        return self.assembler.assemble_matrix_rhs()

    # ------------------------------------------------------------------------------#

    def extract(self, x):
        self.assembler.distribute_variable(x)

        discr = self.discr(self.model)
        for g, d in self.gb:
            var = d[pp.STATE][self.variable]
            d[pp.STATE][self.pressure] = discr.extract_pressure(g, var, d)
            d[pp.STATE][self.flux] = discr.extract_flux(g, var, d)

        # export the P0 flux reconstruction
        pp.project_flux(self.gb, discr, self.flux, self.P0_flux, self.mortar)

    # ------------------------------------------------------------------------------#