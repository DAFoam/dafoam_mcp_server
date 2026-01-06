#!/usr/bin/env python

# =============================================================================
# Imports
# =============================================================================
import os
import argparse
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder, OptFuncs
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP


parser = argparse.ArgumentParser()
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="IPOPT")
parser.add_argument("-task", help="type of run to do", type=str, default="run_model")
parser.add_argument("-angle_of_attack", help="angle of attack", type=float, default=3.0)
parser.add_argument("-mach_number", help="mach number", type=float, default=0.3)
parser.add_argument("-reynolds_number", help="Reynolds number", type=float, default=1000000.0)
parser.add_argument("-lift_constraint", help="The lift constraint", type=float, default=0.5)
parser.add_argument("-max_opt_iters", help="The max optimization iterations", type=int, default=20)
args = parser.parse_args()

# =============================================================================
# Input Parameters
# =============================================================================

T0 = 300.0
p0 = 101325.0
nuTilda0 = 4.5e-5

L0 = 1.0
R = 287.0
k = 1.4
C = float(np.sqrt(k * R * T0))
U0 = args.mach_number * C

rho0 = p0 / R / T0
nu = U0 * L0 / args.reynolds_number
mu = nu * rho0

solverName = "DARhoSimpleFoam"
transonicPC = 0
if args.mach_number > 0.6:
    solverName = "DARhoSimpleCFoam"
    transonicPC = 1

lift_constraint = args.lift_constraint
aoa0 = args.angle_of_attack
A0 = 0.01

# Input parameters for DAFoam
daOptions = {
    "designSurfaces": ["wing"],
    "solverName": solverName,
    "transonicPCOption": transonicPC,
    "primalMinResTol": 1.0e-7,
    "primalMinResTolDiff": 1e3,
    "primalFuncStdTol": {"tol": 1e-5, "funcName": "CD", "nSteps": 50},
    "primalMinIters": 50,
    "printInterval": 10,
    "primalInitCondition": {"U": [U0, 0.0, 0.0], "p": p0, "T": T0},
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "thermo:mu": mu,
        "useWallFunction": True,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "parallelToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "normalToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
        "CM": {
            "type": "moment",
            "source": "patchToFace",
            "patches": ["wing"],
            "axis": [0.0, 0.0, 1.0],
            "center": [0.25, 0.0, 0.0],
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0 * L0),
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-5,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
    },
    "normalizeStates": {
        "U": U0,
        "p": p0,
        "T": T0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
    },
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
        "patchV": {
            "type": "patchVelocity",
            "patches": ["inout"],
            "flowAxis": "x",
            "normalAxis": "y",
            "components": ["solver", "function"],
        },
    },
    "checkMeshThreshold": {
        "maxNonOrth": 70.0,
        "maxSkewness": 6.0,
        "maxAspectRatio": 10000.0,
    },
}

# Mesh deformation setup
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.01], [0.0, 0.0, 1.0]],
    ],
}


# Top class to setup the optimization problem
class Top(Multipoint):
    def setup(self):

        # create the builder to initialize the DASolvers
        dafoam_builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the mesh component
        self.add_subsystem("mesh", dafoam_builder.get_mesh_coordinate_subsystem())

        # add the geometry component (FFD)
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/FFD.xyz", type="ffd"))

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario("scenario1", ScenarioAerodynamic(aero_builder=dafoam_builder))

        # need to manually connect the x_aero0 between the mesh and geometry components
        # here x_aero0 means the surface coordinates of structurally undeformed mesh
        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        # need to manually connect the x_aero0 between the geometry component and the scenario1
        # scenario group
        self.connect("geometry.x_aero0", "scenario1.x_aero")

    def configure(self):

        # get the surface coordinates from the mesh component
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry.nom_add_discipline_coords("aero", points)

        # set the triangular points to the geometry component for geometric constraints
        tri_points = self.mesh.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # use the shape function to define shape variables for 2D airfoil
        pts = self.geometry.DVGeo.getLocalIndex(0)
        dir_y = np.array([0.0, 1.0, 0.0])
        shapes = []
        for i in range(1, pts.shape[0] - 1):
            for j in range(pts.shape[1]):
                # k=0 and k=1 move together to ensure symmetry
                shapes.append({pts[i, j, 0]: dir_y, pts[i, j, 1]: dir_y})
        # LE/TE shape, the j=0 and j=1 move in opposite directions so that
        # the LE/TE are fixed
        for i in [0, pts.shape[0] - 1]:
            shapes.append(
                {
                    pts[i, 0, 0]: dir_y,
                    pts[i, 0, 1]: dir_y,
                    pts[i, 1, 0]: -dir_y,
                    pts[i, 1, 1]: -dir_y,
                }
            )
        self.geometry.nom_addShapeFunctionDV(dvName="shape", shapes=shapes)

        # setup the volume and thickness constraints
        leList = [[0.03, 5.0, 1e-3], [0.03, 5.0, 0.01 - 1e-3]]
        teList = [[0.97, 5.0, 1e-3], [0.97, 5.0, 0.01 - 1e-3]]
        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=2, nChord=10)
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=2, nChord=10)
        self.geometry.nom_addLERadiusConstraints(
            "rcon",
            [[0.01, 0.0, 1e-3], [0.01, 0.0, 0.01 - 1e-3]],
            2,
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        )
        # NOTE: we no longer need to define the sym and LE/TE constraints
        # because these constraints are defined in the above shape function

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.array([0] * len(shapes)))
        self.dvs.add_output("patchV", val=np.array([U0, aoa0]))
        # manually connect the dvs output to the geometry and scenario1
        self.connect("patchV", "scenario1.patchV")
        self.connect("shape", "geometry.shape")

        # define the design variables to the top level
        self.add_design_var("shape", lower=-0.1, upper=0.1, scaler=10.0)
        # here we fix the U0 magnitude and allows the aoa to change
        self.add_design_var("patchV", lower=[U0, -10.0], upper=[U0, 10.0], scaler=0.1)

        # add objective and constraints to the top level
        self.add_objective("scenario1.aero_post.CD", scaler=1.0)
        self.add_constraint("scenario1.aero_post.CL", lower=lift_constraint, scaler=1.0)
        self.add_constraint("geometry.thickcon", lower=0.5, upper=3.0, scaler=1.0)
        self.add_constraint("geometry.volcon", lower=1.0, scaler=1.0)
        self.add_constraint("geometry.rcon", lower=0.8, scaler=1.0)


# OpenMDAO setup
prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")

# initialize the optimization function
optFuncs = OptFuncs(daOptions, prob)

# use pyoptsparse to setup optimization
prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = args.optimizer
# options for optimizers
if args.optimizer == "SNOPT":
    prob.driver.opt_settings = {
        "Major feasibility tolerance": 1.0e-4,
        "Major optimality tolerance": 1.0e-4,
        "Minor feasibility tolerance": 1.0e-4,
        "Verify level": -1,
        "Function precision": 1.0e-5,
        "Major iterations limit": args.max_opt_iters,
        "Nonderivative linesearch": None,
        "Print file": "opt_SNOPT_print.txt",
        "Summary file": "opt_SNOPT_summary.txt",
    }
elif args.optimizer == "IPOPT":
    prob.driver.opt_settings = {
        "tol": 1.0e-4,
        "constr_viol_tol": 1.0e-4,
        "max_iter": args.max_opt_iters,
        "print_level": 5,
        "output_file": "opt_IPOPT.txt",
        "mu_strategy": "adaptive",
        "limited_memory_max_history": 10,
        "nlp_scaling_method": "none",
        "alpha_for_y": "full",
        "recalc_y": "yes",
        "bound_frac": 1e-3,  # allow IPOPT to use init dv closer to upper bound
    }
elif args.optimizer == "SLSQP":
    prob.driver.opt_settings = {
        "ACC": 1.0e-4,
        "MAXIT": args.max_opt_iters,
        "IFILE": "opt_SLSQP.txt",
    }
else:
    print("optimizer arg not valid!")
    exit(1)

prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]
prob.driver.options["print_opt_prob"] = True
prob.driver.hist_file = "OptView.hst"

if args.task == "run_driver":
    # solve CL
    optFuncs.findFeasibleDesign(
        ["scenario1.aero_post.CL"],
        ["patchV"],
        targets=[lift_constraint * 1.001],
        designVarsComp=[1],
        epsFD=[1e-2],
        tol=1e-3,
    )
    # run the optimization
    prob.run_driver()
    if MPI.COMM_WORLD.rank == 0:
        os.system("touch .dafoam_run_finished")
elif args.task == "run_model":
    # just run the primal once
    prob.run_model()
    if MPI.COMM_WORLD.rank == 0:
        os.system("touch .dafoam_run_finished")
elif args.task == "compute_totals":
    # just run the primal and adjoint once
    prob.run_model()
    totals = prob.compute_totals()
    if MPI.COMM_WORLD.rank == 0:
        print(totals)
elif args.task == "check_totals":
    # verify the total derivatives against the finite-difference
    prob.run_model()
    prob.check_totals(compact_print=False, step=1e-3, form="central", step_calc="abs")
else:
    print("task arg not found!")
    exit(1)
