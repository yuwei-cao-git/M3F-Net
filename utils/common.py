import yaml
import time
import wandb

# get configs for a sweep from .yaml file
def get_configs_from_file(path_yaml):
    dict_yaml = yaml.load(open(path_yaml).read(), Loader=yaml.Loader)
    sweep_config = dict_yaml['sweep_config']
    params_config = dict_yaml['params_config']
    search_space = {}
    hash_keys = []
    for k,v in params_config.items():  
        search_space[k] = {"values":v}
        if len(v)>1:
            hash_keys.append(k)
        if k=='num_runs':
            assert int(v[0]) > 0
            search_space['runs'] = {"values":list(range(int(v[0])))}
    search_space['hash_keys'] = {"values":[hash_keys]}
    sweep_config['parameters'] = search_space
    return sweep_config

# modify some specific hyper parameters in sweep's config
def modify_sweep(sweep_config, dict_new):
    for key in dict_new.keys():
        sweep_config['parameters'][key] = {'values':dict_new[key]}
    return sweep_config

def GetRunTime(func):
    def call_func(*args, **kwargs):
        begin_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        Run_time = end_time - begin_time
        print("Execution time for func [%s] is [%s]"%(str(func.__name__), str(Run_time)))
        return ret
    return call_func

def get_timestamp():
    time.tzset()
    now = int(round(time.time()*1000))
    timestamp = time.strftime('%Y-%m%d-%H%M',time.localtime(now/1000))
    return timestamp

# calculate the size of a sweep's search space or the number of runs
def count_sweep(mode, entity, project, id):
    # mode: size_space, num_runs
    api = wandb.Api()
    sweep = api.sweep('%s/%s/%s'%(entity, project, id))
    if mode=='size_space':
        cnt = 1
        params= sweep.config['parameters']
        for key in params.keys():
            cnt *= len(params[key]['values'])
    elif mode=='num_runs':
        cnt = len(sweep.runs)
    return cnt