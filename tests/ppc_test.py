from enterprise_workflow.parameters import Params

params = Params.from_config_file("test_configs/test_simulation_config.ini")

# perform simulation
params.ppc.perform_gp_draws()
# params.ostat.run_otstat()
