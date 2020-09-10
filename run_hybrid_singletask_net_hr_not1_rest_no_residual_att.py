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
from datetime import datetime
from functions.network_caller.hybrid_single_task_translate_hr_not1_rest_no_residual_att import net_translate
import numpy as np
import tensorflow as tf
import os

if __name__ == '__main__':
    '''
    this function calls the translation network

    '''
    fold=1
    no_averages=0
    config = [3, 3, 5, 3, 3]
    np.random.seed(1)
    tf.set_random_seed(1)

    server_path = '/exports/lkeb-hpc/syousefi/Code/'
    Logs = 'Log_asl_pet/rest/01_cross_validation/single_not1/single_not1_fold_' + str(fold) + '/'

    # use mixed precision
    mixed_precision = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # check if gpu is available
    GPU = tf.test.is_gpu_available(
        cuda_only=False,
        min_cuda_compute_capability=None
    )

    # current date and time
    now = datetime.now()
    date_time = now.strftime("%m%d%Y_%H")

    dc12 = net_translate(data_path_AMUC="/exports/lkeb-hpc/syousefi/Data/ASL2PET_high_res/AMUC_high_res/",
                         data_path_LUMC="/exports/lkeb-hpc/syousefi/Data/ASL2PET_high_res/LUMC_high_res/",
                         server_path=server_path, Logs=Logs,
                         config=config)
    dc12.run_net(no_averages,fold=fold)


