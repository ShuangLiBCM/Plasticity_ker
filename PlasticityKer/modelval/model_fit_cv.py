"""
    Fit the neural network given network parameter and resturn CV mean and std

"""
import numpy as np
import pandas as pd
from modelval import pairptl, network, trainer, dataset, perform_eval
from modelval.ArbDataGen import arb_w_gen, data_Gen
from modelval.kernel import KernelGen

def ModelFitCV(data_type = 'hippocampus', data_aug='gp_mean', test_fold_num=0, vali_split=5, save_dir_set=None,
               random_seed=0, reg_scale=(1, 1, 1, 1), init_seed=(0, 1, 2, 3, 4)):

    data = pd.read_csv('/src/Plasticity_Ker/data/kernel_training_data_auto.csv')

    data_gen_train, data_gen_vali, data_gen_test, y_train, y_vali, y_test = \
        data_Gen(data, data_type=data_type, data_aug=data_aug, test_fold_num=test_fold_num, vali_split=vali_split)

    # Visualize kernel
    vali_err = np.zeros(len(data_gen_train))
    kernel_pre = []
    kernel_post = []
    kernel_post_post = []
    fc_w = []
    bias = []
    scale = []

    for i in range(len(data_gen_train)):

        len_kernel = 101
        len_trip = 151
        ker_test = KernelGen(len_kernel=len_kernel, len_trip=len_trip)

        # Generat the spike trains and targets for STDP
        data_hippo = data[data['ptl_idx'] < 5]
        ptl_list = [1, 2, 3, 4]
        spk_len = int(data_hippo['train_len'].max() * 1000 / ker_test.reso_kernel)

        if data_aug == 'gp_mean':
            if_noise = 1
        else:
            if_noise = 0

        aug_times = [1, 1, 1, 1]
        spk_pairs_train, targets_train = arb_w_gen(df=data_gen_train[i], ptl_list=ptl_list, targets=y_train[i],
                                                   if_noise=if_noise, spk_len=spk_len, kernel=ker_test,
                                                   net_type='triplet', aug_times=aug_times, seed=723)
        spk_pairs_vali, targets_vali = arb_w_gen(df=data_gen_vali[i], ptl_list=ptl_list, targets=y_vali[i],
                                                 if_noise=if_noise, spk_len=spk_len, kernel=ker_test,
                                                 net_type='triplet', aug_times=aug_times, seed=606)

        # Create the network
        ground_truth_init = 0
        reg_scale = reg_scale
        np.random.seed(random_seed)


        toy_data_net = network.TripNet(kernel=ker_test, ground_truth_init=ground_truth_init, reg_scale=reg_scale,
                                       n_input=spk_pairs_train.shape[1], init_seed=init_seed)

        # Create the trainer
        save_dir = save_dir_set + 'cv' + str(i)
        toy_net_trainer = trainer.Trainer(toy_data_net.mse, toy_data_net.loss, input_name=toy_data_net.inputs,
                                          target_name=toy_data_net.target, save_dir=save_dir,
                                          optimizer_config={'learning_rate': toy_data_net.lr},  device_config=toy_data_net.device_config)

        # Package the data
        train_data = dataset.Dataset(spk_pairs_train, targets_train)
        vali_data = dataset.Dataset(spk_pairs_vali, targets_vali)

        # Learn the kernel from random initialization
        learning_rate = 0.001
        iterations = 5
        min_error = -1
        for _ in range(iterations):
            mini_vali_loss = toy_net_trainer.train(train_data, vali_data, batch_size=128, min_error=min_error,
                                                   feed_dict={toy_data_net.lr: learning_rate})
            learning_rate = learning_rate / 3

        vali_err[i] = mini_vali_loss
        toy_net_trainer.restore_best()
        kernel_pre.append(toy_net_trainer.evaluate(ops=toy_data_net.kernel_pre))
        kernel_post.append(toy_net_trainer.evaluate(ops=toy_data_net.kernel_post))
        kernel_post_post.append(toy_net_trainer.evaluate(ops=toy_data_net.kernel_post_post))
        fc_w.append(toy_net_trainer.evaluate(ops=toy_data_net.fc_w))
        bias.append(toy_net_trainer.evaluate(ops=toy_data_net.bias))
        scale.append(toy_net_trainer.evaluate(ops=toy_data_net.scaler))

    return vali_err, kernel_pre, kernel_post, kernel_post_post, fc_w, bias, scale