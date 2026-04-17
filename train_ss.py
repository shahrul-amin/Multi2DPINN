import os
import torch
import numpy as np
import csv

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
# mpl.use('tkagg')
# mpl.use('Agg')

import random

from src.first_stage.config import T_n_points
from src.first_stage.generator import testing_list
from src.first_stage.reader import idx_MSE, truth_MSE
from src.first_stage.validator import *
from src.first_stage.nn import mynet, MSE_loss, optimizer, MSE_batches, MSE_bz
from src.first_stage.parser import args, device

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

# Training begins
last_saved_epoch = 0
loss_list = []
for epoch in range(args.n_epochs):
    mynet.train()
    random.shuffle(MSE_batches)
    loss_average = 0
    for batch in range(args.n_batches):
        # print(f"Epoch : {epoch}, Batch: {batch}")
        MSE_btch = idx_MSE[MSE_batches[batch*MSE_bz:(batch+1)*MSE_bz]]
        truth_MSE_btch = truth_MSE[MSE_batches[batch*MSE_bz:(batch+1)*MSE_bz]]

        MSE_btch = torch.FloatTensor(MSE_btch).to(device)
        truth_MSE_btch = torch.FloatTensor(truth_MSE_btch).to(device)

        mynet.zero_grad()

        loss_mse = MSE_loss(MSE_btch, truth_MSE_btch)

        loss = args.w_mse*loss_mse

        loss_average += loss.item()

        loss.backward()
        # print(mynet.il.weight.grad)
        optimizer.step()

    loss_average /= args.n_batches
    loss_list.append([loss_average])
    with open(args.run_dir+'/image_rmse_list.csv', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(loss_list)

    print(f"Epoch {epoch}:"\
            f" mean training loss: {loss_average:.6f},"\
            f" training loss: {loss:.6f}, MSE: {loss_mse.item():.6f}")

    if epoch !=0 and (epoch % args.save_freq == 0 or epoch == args.n_epochs-1):
        if os.path.exists(args.run_dir + '/trial_function_mlp_{}.pkl'.format(last_saved_epoch)):
            os.remove(args.run_dir + '/trial_function_mlp_{}.pkl'.format(last_saved_epoch))

        torch.save(mynet.state_dict(), args.run_dir + '/trial_function_mlp_{}.pkl'.format(epoch))
        print('Model saved')
        
        last_saved_epoch = epoch

        for k in range(len(testing_list)):
            valid_X_single_case_list = valid_X_list[k]
            valid_T_single_case_list = valid_T_list[k]
            valid_L_single_case_list = valid_L_list[k]
            valid_W_single_case_list = valid_W_list[k]
            valid_G_single_case_list = valid_G_list[k]
            valid_k1_single_case_list = valid_k1_list[k]
            valid_k2_single_case_list = valid_k2_list[k]
            valid_truth_single_case_list = valid_truth_list[k]

            u_valid_list = []
            for i in range(len(valid_X_single_case_list)):
                u = mynet(valid_X_single_case_list[i],
                        valid_T_single_case_list[i],
                        valid_L_single_case_list[i],
                        valid_W_single_case_list[i],
                        valid_G_single_case_list[i],
                        valid_k1_single_case_list[i],
                        valid_k2_single_case_list[i])
                u = to_numpy(u).reshape([T_n_points, -1]).T
                u_valid_list.append(u)

            u = np.vstack(u_valid_list)
            truth = np.vstack(valid_truth_single_case_list)

            font_size = 24
            fig = plt.figure(figsize=(60, 20))
            gs = gridspec.GridSpec(3, 1)

            # Subplot 1: ground truth
            ax = plt.subplot(gs[0])
            ax.imshow(truth, cmap='rainbow', origin='upper', aspect='auto',  vmin = -1, vmax = 1)
            ax.set_title('Ground Truth',  fontsize = font_size)


            # Subplot 2: Prediction
            ax = plt.subplot(gs[1])
            ax.imshow(u, cmap='rainbow',  origin='upper', aspect='auto', vmin = -1, vmax = 1)
            ax.set_title('Prediction',  fontsize = font_size)

            # Subplot 3: Final stress
            ax = plt.subplot(gs[2])
            stress_end_truth = truth[:,-1]
            stress_end_pred = u[:,-1]
            ax.plot(stress_end_truth,'r-', label = 'Ground Truth')
            ax.plot(stress_end_pred,'b.', label='Prediction')
            ax.legend(loc='upper right')
            ax.set_title('Final stress',  fontsize = font_size)


            fig.suptitle(args.hparams)

            # plt.show()
            plt.savefig('{}/truth_vs_pred_{}_epoch_{}.png'.format(args.run_dir, str(testing_list[k]), str(epoch).zfill(4)), bbox_inches='tight')
            plt.close(fig)

