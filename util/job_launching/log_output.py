import json, re, sys, os, datetime

# TODO: put the name of the slurm job in the log file

job_number = sys.argv[4]
        
this_directory = sys.argv[1]               
with open(os.path.join(this_directory, sys.argv[2])) as f:
    export_dict=json.load(f)
             
benchmark = export_dict['benchmark']
self_benchmark_args_subdirs_args = export_dict['self.benchmark_args_subdirs[args]']
self_run_subdir = export_dict['self.run_subdir']
build_handle = export_dict['build_handle']
options_launch_name = export_dict['options.launch_name']
uuid_var = export_dict['uuid_var']
                    
if len(job_number) > 0:
    # print('len torque out > 0')
    # print(f'this directory: {this_directory}')
    # Dump the benchmark description to the logfile
    
    if not os.path.exists(os.path.join(this_directory, "logfiles")):
        # In the very rare case that concurrent builds try to make the directory at the same time
        # (after the test to os.path.exists -- this has actually happened...)
        try:
            os.makedirs(os.path.join(this_directory, "logfiles"))
        except:
            pass
    now_time = datetime.datetime.now()
    day_string = now_time.strftime("%y.%m.%d-%A")
    time_string = now_time.strftime("%H-%M-%S")
    log_name = "sim_log.{0}".format(options_launch_name) # TODO: need "options.launch_name"
    
    # print(f'log name is: {log_name}')
    job_name = sys.argv[3]
    
    logfile = open(
        os.path.join(this_directory, 'logfiles', log_name + "." + job_name + "." + job_number + ".txt"), "w"
    )
    print(
        "%s %6s %-22s %-100s %-25s %s %s"
        % (
            time_string,
            job_number,
            benchmark, # TODO: need "benchmark"
            self_benchmark_args_subdirs_args, # TODO: need "self.benchmark_args_subdirs[args]"
            self_run_subdir, # TODO: need "run_subdir"
            build_handle, # TODO: need "build_handle"
            uuid_var
        ),
        file=logfile,
    )
    logfile.close()