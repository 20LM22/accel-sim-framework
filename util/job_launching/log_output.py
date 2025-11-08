import json, os, re, datetime

# get the name of the torque out file that run simulations would have produced
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
torque_out_filename = this_directory + "job_number.txt"
print(f'torque_out_filename: {torque_out_filename}')

torque_out_file = open(torque_out_filename, "r")
torque_out = re.sub(r"[^\d]*(\d*).*", r"\1", torque_out_file.read().strip())
                    
with open("/scratch/gpfs/WENTZLAF/lm4677/analytical-model/util/job_launching/export_dict.json") as f:
    export_dict=json.load(f)
                    
benchmark = export_dict['benchmark']
self_benchmark_args_subdirs_args = export_dict['self.benchmark_args_subdirs[args]']
self_run_subdir = export_dict['self.run_subdir']
build_handle = export_dict['build_handle']
options_launch_name = export_dict['options.launch_name']

print(f'torque out: {torque_out}')
                    
if len(torque_out) > 0:
    print('len torque out > 0')
    print(f'this directory: {this_directory}')
    # Dump the benchmark description to the logfile
    if not os.path.exists(this_directory + "logfiles/"):
        # In the very rare case that concurrent builds try to make the directory at the same time
        # (after the test to os.path.exists -- this has actually happened...)
        try:
            os.makedirs(this_directory + "logfiles/")
        except:
            pass
    now_time = datetime.datetime.now()
    day_string = now_time.strftime("%y.%m.%d-%A")
    time_string = now_time.strftime("%H-%M-%S")
    log_name = "sim_log.{0}".format(options_launch_name) # TODO: need "options.launch_name"
    
    # print(f'log name is: {log_name}')
    
    logfile = open(
        this_directory + "logfiles/" + log_name + "." + day_string + "." + time_string + ".txt", "w",
    )
    print(
        "%s %6s %-22s %-100s %-25s %s"
        % (
            time_string,
            torque_out,
            benchmark, # TODO: need "benchmark"
            self_benchmark_args_subdirs_args, # TODO: need "self.benchmark_args_subdirs[args]"
            self_run_subdir, # TODO: need "run_subdir"
            build_handle, # TODO: need "build_handle"
        ),
        file=logfile,
    )
    logfile.close()