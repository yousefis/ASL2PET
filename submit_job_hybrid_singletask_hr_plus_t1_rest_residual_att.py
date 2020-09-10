##################################################
## {Description}
##################################################
## {License_info}
##################################################
## Author: {Sahar Yousefi}
## Copyright: Copyright {2020}, {LUMC}
## Credits: [Sahar Yousefi]
## License: {GPL}
## Version: 1.0.0
## Mmaintainer: {Sahar Yousefi}
## Email: {s.yousefi.radi[at]lumc.nl}
## Status: {Research}
##################################################
import datetime
import os
import shutil
import functions.gpu_server_manager.oge_utils as oge
import functions.gpu_server_manager.slurm_utils as slurm



def submit_job(fold):
    # Choosing the preferred setting and backup the whole code and submit the job
    # script_address= server_path+Logs
    queue = 'gpu'  # 'cpu', 'gpu', 'LKEBgpu'
    manager = 'Slurm'  # 'OGE', Slurm
    setting = dict()
    setting['never_generate_image'] = False
    TF='TF112' #version of tensorflow 1.12

    net_config = [3, 3, 5, 3, 3]
    main_script = '/run_hybrid_singletask_net_hr_plust1_rest_no_residual_att.py'


    # Slurm
    setting['cluster_MemPerCPU'] = 4200   #2200  # 6200
    setting['cluster_Partition'] = queue             # 'gpu', 'LKEBgpu'
    setting['cluster_NodeList'] = 'res-hpc-gpu02'    # None, LKEBgpu: ['res-hpc-lkeb03', 'res-hpc-lkeb02', 'res-hpc-gpu01']
    setting['cluster_NumberOfCPU'] = 5#7#10 #3               # Number of CPU per job
    setting['cluster_where_to_run'] = 'Cluster'      # 'Cluster', 'Auto'
    setting['cluster_venv_slurm'] = '/exports/lkeb-hpc/syousefi/Programs/'+TF+'/bin/activate'  # venv path

    if setting['cluster_NodeList'] == 'res-hpc-gpu02':
        setting['cluster_venv_slurm'] = '/exports/lkeb-hpc/syousefi/Programs/'+TF+'/bin/activate'  # venv path

    if setting['cluster_NodeList'] == 'res-hpc-lkeb02':
        setting['cluster_MemPerCPU'] = 15000

    folder_script = 'functions'
    # A backup from all files are created. So later if you modify the codes, this does not affect the submitted code.
    backup_script_address, backup_number = backup_script(script_address=os.path.realpath(__file__), main_script=main_script, folder_script  =folder_script,net_config=net_config)
    ext = ''.join(str(net_config[e]) for e in range(len(net_config)))
    job_name = 'not1att_'+str(fold)+'_'+ext+'_'+str(backup_number)
    write_and_submit_job(setting, manager=manager, job_name=job_name, script_address=backup_script_address,fold=fold)


def write_and_submit_job(setting, manager, job_name, script_address,fold):
    """
    Write a bashscript and submit the bashscript as a job to SGE using qsub command.
    :param setting:
    :param manager
    :param job_name:
    :param script_address:
    :return:
    """
    # backup_folder = script_address.rsplit('/', maxsplit=1)[0]
    # job_script_folder = backup_folder + '/Jobs/'
    # job_output_file = job_script_folder + 'output.txt'

    server_path = '/exports/lkeb-hpc/syousefi/Code/'
    backup_folder= 'Log_asl_pet/rest/01_cross_validation/single_plust1/single_plust1_fold_' + str(fold) + '/'
    job_script_folder = server_path+backup_folder+'Jobs/'
    job_output_file =job_script_folder+'output.txt'

    print(job_output_file)
    if not os.path.exists(job_script_folder):
        os.makedirs(job_script_folder)
    job_script_address = job_script_folder + 'jobscript_'+manager+'.sh'
    with open(job_script_address, "w") as string_file:
        if manager == 'OGE':
            string_file.write(oge.job_script(setting, job_name=job_name, script_address=script_address, job_output_folder=job_script_folder))
        elif manager == 'Slurm':
            string_file.write(slurm.job_script(setting, job_name=job_name, script_address=script_address, job_output_file=job_output_file))
        else:
            raise ValueError("manager should be in ['OGE', 'Slurm']")
        string_file.close()
    for root, dir_list, file_list in os.walk(backup_folder):
        for f in dir_list+file_list:
            os.chmod(os.path.join(root, f), 0o754)
    if manager == 'OGE':
        submit_cmd = 'qsub ' + job_script_address
    elif manager == 'Slurm':
        submit_cmd = 'sbatch ' + job_script_address
    else:
        raise ValueError("manager should be in ['OGE', 'Slurm']")
    os.system(submit_cmd)


def backup_script(script_address, main_script, folder_script, net_config):
    """
    backup the current script to the backup-1 folder. If this folder already exists it won't overwrite it, instead
    it will try to create another folder called backup-2.

    :param script_address: current script address
    :param main_script:
    :param folder_script
    :return: backup_script_address: backup script address
    :return: backup_number: backup script address
    """
    ext = ''.join(str(net_config[e]) for e in range(len(net_config)))
    script_address = script_address.replace('\\', '/')  # windows path correction
    script_folder = script_address.rsplit('/', maxsplit=1)[0] + '/'
    main_script_address = script_folder + main_script
    date_now = datetime.datetime.now()
    backup_number = '{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.\
        format(date_now.year, date_now.month, date_now.day, date_now.hour, date_now.minute, date_now.second)
    backup_root_folder = script_folder + '/Log_asl_pet/rest/net'+ext+'/'
    backup_folder = backup_root_folder + 'backup-' + str(backup_number) + '/'
    os.makedirs(backup_folder)
    shutil.copy(script_address, backup_folder)
    shutil.copy(main_script_address, backup_folder)
    shutil.copytree(script_folder + folder_script + '/', backup_folder + folder_script + '/')
    main_script_backup_script_address = backup_folder + main_script
    return main_script_backup_script_address, backup_number


if __name__ == '__main__':
    fold=8
    submit_job(fold)
