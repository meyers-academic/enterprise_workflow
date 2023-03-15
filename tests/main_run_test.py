from parameters import Params

params = Params.from_config_file("test_configs/test_simulation_config.ini")
print(params)
# perform simulation
# params.injection.simulate()
# params.ppc.perform_gp_draws()
params.main_run.sample_mcmc()