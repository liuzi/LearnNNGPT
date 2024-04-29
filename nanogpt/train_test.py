wandb_log = False 
wandb_project = 'owt'
wandb_run_name = 'gpt2' 

print(wandb_log)
print(wandb_project)
print(wandb_run_name)

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
# config = {k: globals()[k] for k in config_keys} # will be useful for logging

print(wandb_log)
print(wandb_project)
print(wandb_run_name)