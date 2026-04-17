import torch
import numpy as np
# mpl.use('tkagg')
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
import csv
from timeit import default_timer as timer

from src.second_stage.reader import testing_list, stress_min, stress_range
from src.second_stage.validator import *
from src.second_stage.nn import mynet, flux_predictor, optimizer, scheduler
from src.second_stage.parser import args, device

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))
    
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def rmse_nan_array(array1, array2):
    diff_list = []
    for i in range(array1.shape[0]):
        for j in range(array1.shape[1]):
            if not np.isnan(array1[i,j]):
                diff_list.append(array1[i,j] - array2[i,j])

    rmse = np.sqrt(np.mean(np.square(diff_list)))
    return rmse

# Training begins
last_saved_epoch = 0
loss_list = []
if args.timer:
    total_training_time = 0
for epoch in range(args.n_epochs):
    if args.timer:
        tmr_start = timer()
    mynet.eval()
    flux_predictor.train()
    loss_average = 0
    for case in range(len(testing_list)):
        flux_predictor.zero_grad()
        # Flux prediction
        node_segments_list_single_case = node_segments_list[case]
        flux_predictor_inputs_Ni_single_case_list = flux_predictor_inputs_Ni_list[case]
        flux_predictor_inputs_Gl_single_case_list = flux_predictor_inputs_Gl_list[case]
        flux_predictor_inputs_Gu_single_case_list = flux_predictor_inputs_Gu_list[case]
        flux_predictor_inputs_Gr_single_case_list = flux_predictor_inputs_Gr_list[case]
        flux_predictor_inputs_Gd_single_case_list = flux_predictor_inputs_Gd_list[case]
        flux_predictor_inputs_T_single_case_list = flux_predictor_inputs_T_list[case]

        stress_predictor_inputs_k_single_case_list = [[torch.zeros(T_n_points,1,device=device),torch.zeros(T_n_points,1,device=device)] for _ in range(len(boundary_X_list[case]))] # initialize all [k1,k2] to zeros with shape of [T_n_points,1]
        j = 0
        for i in range(len(flux_predictor_inputs_Ni_single_case_list)): # loop over all internal junctions
            k = flux_predictor(flux_predictor_inputs_Ni_single_case_list[i],
                                flux_predictor_inputs_Gl_single_case_list[i],
                                flux_predictor_inputs_Gu_single_case_list[i],
                                flux_predictor_inputs_Gr_single_case_list[i],
                                flux_predictor_inputs_Gd_single_case_list[i],
                                flux_predictor_inputs_T_single_case_list[i])

            # Fill predicted k values into stress_predictor_inputs_k_single_case_list
            while(sum(segment is not None for segment in node_segments_list_single_case[j]) == 1): # skip boundary nodes which are not internal junctions
                j+=1
            connected_segments = node_segments_list_single_case[j]
            connected_segments_n = sum(segment is not None for segment in connected_segments)
            flux_sum = torch.zeros(T_n_points,1) # kl-ku-kr+kd=0
            flux_sum = flux_sum.to(device)
            if(connected_segments[0] is not None): # left
                stress_predictor_inputs_k_single_case_list[connected_segments[0]][1] = k[:,0:1]
                flux_sum += k[:,0:1]
                connected_segments_n -= 1
            if(connected_segments[1] is not None): # up
                if(connected_segments_n==1):
                    stress_predictor_inputs_k_single_case_list[connected_segments[1]][0] = flux_sum
                else:
                    stress_predictor_inputs_k_single_case_list[connected_segments[1]][0] = k[:,1:2]
                    flux_sum -= k[:,1:2]
                connected_segments_n -= 1
            if(connected_segments[2] is not None): # right
                if(connected_segments_n==1):
                    stress_predictor_inputs_k_single_case_list[connected_segments[2]][0] = flux_sum
                else:
                    stress_predictor_inputs_k_single_case_list[connected_segments[2]][0] = k[:,2:3]
                    flux_sum -= k[:,2:3]
                connected_segments_n -= 1
            if(connected_segments[3] is not None): # down
                stress_predictor_inputs_k_single_case_list[connected_segments[3]][1] = -flux_sum
            
            j+=1
                

            # stress_predictor_inputs_k_single_case_list.append(k)

        # stress continuity
        boundary_X_single_case_list = boundary_X_list[case]
        boundary_T_single_case_list = boundary_T_list[case]
        boundary_L_single_case_list = boundary_L_list[case]
        boundary_W_single_case_list = boundary_W_list[case]
        boundary_G_single_case_list = boundary_G_list[case]

        u_boundary_list = []
        for i in range(len(boundary_X_single_case_list)): # loop over all segments
            k1, k2 = stress_predictor_inputs_k_single_case_list[i]

            # if(i==0):
            #     k1 = torch.zeros_like(stress_predictor_inputs_k_single_case_list[0])
            #     k2 = stress_predictor_inputs_k_single_case_list[0]
            # elif(i == len(boundary_X_single_case_list)-1):
            #     k1 = stress_predictor_inputs_k_single_case_list[-1]
            #     k2 = torch.zeros_like(stress_predictor_inputs_k_single_case_list[-1])
            # else:
            #     k1 = stress_predictor_inputs_k_single_case_list[i-1]
            #     k2 = stress_predictor_inputs_k_single_case_list[i]

            # X_n_points = 2
            X_n_points = int(boundary_X_single_case_list[i].shape[0]/T_n_points)
            k1_cat = [k1 for _ in range(X_n_points)]
            k2_cat = [k2 for _ in range(X_n_points)]

            k1 = torch.cat(k1_cat,dim=1) # shape=[T_n_points, L_n_points]
            k2 =  torch.cat(k2_cat,dim=1) # shape=[T_n_points, L_n_points]

            k1 = torch.reshape(k1, [-1,1])
            k2 = torch.reshape(k2, [-1,1])

            u = mynet(boundary_X_single_case_list[i],
                    boundary_T_single_case_list[i],
                    boundary_L_single_case_list[i],
                    boundary_W_single_case_list[i],
                    boundary_G_single_case_list[i],
                    k1,
                    k2)
            u = u.reshape([T_n_points, -1])
            u_boundary_list.append(u)


        loss_single_case = []
        j = 0
        for i in range(len(flux_predictor_inputs_Ni_single_case_list)): # loop over all internal junctions
            while(sum(segment is not None for segment in node_segments_list_single_case[j]) == 1): # skip boundary nodes which are not internal junctions
                j+=1
            connected_segments = node_segments_list_single_case[j]
            junction_stress_list = []
            for l,segment in enumerate(connected_segments):
                if(segment is not None):
                    if(l==0 or l==3): # left or down, append stress on right end to the stress_list
                        junction_stress_list.append(u_boundary_list[segment][:,1])
                    elif(l==1 or l==2): # up or right, append stress on left end to the stress_list
                        junction_stress_list.append(u_boundary_list[segment][:,0])
            for l in range(len(junction_stress_list)-1):
                loss = junction_stress_list[l] - junction_stress_list[l+1]
                loss = (loss**2).mean()
                loss_single_case.append(loss)
            j+=1

        loss = loss_single_case[0]
        for l in loss_single_case[1:]:
            loss += l

        loss_average += loss.item()

        loss.backward()
        # print(mynet.il.weight.grad)

        optimizer.step()

    if args.timer:
        tmr_end = timer()
        total_training_time += tmr_end - tmr_start
        print(f" Epoch time: {str(tmr_end - tmr_start)}")
        print(f" Total time: {str(total_training_time)}")

    # lr scheduling
    step = epoch
    pct = step / args.n_epochs
    if(args.lr_pct >= 0):
        lr = scheduler.step(pct)
        adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr

    loss_average /= len(testing_list)
    loss_list.append([loss_average])
    if not args.timer:
        with open(args.run_dir+'/traning_loss_list.csv', "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(loss_list)

    print(f"Epoch {epoch}:"\
            f" mean training loss: {loss_average:.6f}, lr: {lr:.6f}")

    if epoch !=0 and (epoch % args.save_freq == 0 or epoch == args.n_epochs-1):
        for case in range(len(testing_list)):
            # Flux prediction
            node_segments_list_single_case = node_segments_list[case]
            segment_endopoints_and_direction_single_case = segment_endopoints_and_direction_list[case]
            flux_predictor_inputs_Ni_single_case_list = flux_predictor_inputs_Ni_list[case]
            flux_predictor_inputs_Gl_single_case_list = flux_predictor_inputs_Gl_list[case]
            flux_predictor_inputs_Gu_single_case_list = flux_predictor_inputs_Gu_list[case]
            flux_predictor_inputs_Gr_single_case_list = flux_predictor_inputs_Gr_list[case]
            flux_predictor_inputs_Gd_single_case_list = flux_predictor_inputs_Gd_list[case]
            flux_predictor_inputs_T_single_case_list = flux_predictor_inputs_T_list[case]

            stress_predictor_inputs_k_single_case_list = [[torch.zeros(T_n_points,1,device=device),torch.zeros(T_n_points,1,device=device)] for _ in range(len(boundary_X_list[case]))] # initialize all [k1,k2] to zeros with shape of [T_n_points,1]
            j = 0
            for i in range(len(flux_predictor_inputs_Ni_single_case_list)):
                k = flux_predictor(flux_predictor_inputs_Ni_single_case_list[i],
                                    flux_predictor_inputs_Gl_single_case_list[i],
                                    flux_predictor_inputs_Gu_single_case_list[i],
                                    flux_predictor_inputs_Gr_single_case_list[i],
                                    flux_predictor_inputs_Gd_single_case_list[i],
                                    flux_predictor_inputs_T_single_case_list[i])

                # Fill predicted k values into stress_predictor_inputs_k_single_case_list
                while(sum(segment is not None for segment in node_segments_list_single_case[j]) == 1): # skip boundary nodes which are not internal junctions
                    j+=1;
                connected_segments = node_segments_list_single_case[j]
                connected_segments_n = sum(segment is not None for segment in connected_segments)
                flux_sum = torch.zeros(T_n_points,1) # kl-ku-kr+kd=0
                flux_sum = flux_sum.to(device)
                if(connected_segments[0] is not None): # left
                    stress_predictor_inputs_k_single_case_list[connected_segments[0]][1] = k[:,0:1]
                    flux_sum += k[:,0:1]
                    connected_segments_n -= 1
                if(connected_segments[1] is not None): # up
                    if(connected_segments_n==1):
                        stress_predictor_inputs_k_single_case_list[connected_segments[1]][0] = flux_sum
                    else:
                        stress_predictor_inputs_k_single_case_list[connected_segments[1]][0] = k[:,1:2]
                        flux_sum -= k[:,1:2]
                    connected_segments_n -= 1
                if(connected_segments[2] is not None): # right
                    if(connected_segments_n==1):
                        stress_predictor_inputs_k_single_case_list[connected_segments[2]][0] = flux_sum
                    else:
                        stress_predictor_inputs_k_single_case_list[connected_segments[2]][0] = k[:,2:3]
                        flux_sum -= k[:,2:3]
                    connected_segments_n -= 1
                if(connected_segments[3] is not None): # down
                    stress_predictor_inputs_k_single_case_list[connected_segments[3]][1] = -flux_sum
                
                j+=1

            valid_X_single_case_list = valid_X_list[case]
            valid_T_single_case_list = valid_T_list[case]
            valid_L_single_case_list = valid_L_list[case]
            valid_W_single_case_list = valid_W_list[case]
            valid_G_single_case_list = valid_G_list[case]
            valid_k1_single_case_list = valid_k1_list[case]
            valid_k2_single_case_list = valid_k2_list[case]
            valid_truth_single_case_list = valid_truth_list[case]

            u_valid_list = []
            if args.timer:
                total_inference_time = 0
            for i in range(len(valid_X_single_case_list)):
                k1, k2 = stress_predictor_inputs_k_single_case_list[i]

                # X_n_points = 2
                X_n_points = int(valid_X_single_case_list[i].shape[0]/T_n_points)
                k1_cat = [k1 for _ in range(X_n_points)]
                k2_cat = [k2 for _ in range(X_n_points)]

                k1 = torch.cat(k1_cat,dim=1) # shape=[T_n_points, L_n_points]
                k2 =  torch.cat(k2_cat,dim=1) # shape=[T_n_points, L_n_points]

                k1 = torch.reshape(k1, [-1,1])
                k2 = torch.reshape(k2, [-1,1])
                if args.timer:
                    tmr_start = timer()
                u = mynet(valid_X_single_case_list[i],
                        valid_T_single_case_list[i],
                        valid_L_single_case_list[i],
                        valid_W_single_case_list[i],
                        valid_G_single_case_list[i],
                        # valid_k1_single_case_list[i],
                        # valid_k2_single_case_list[i])
                        k1,
                        k2)

                if args.timer:
                    tmr_end = timer()
                    total_inference_time += tmr_end - tmr_start
                    print(f" Segment time: {str(tmr_end - tmr_start)}")
                    print(f" Total time: {str(total_inference_time)}")

                u = to_numpy(u).reshape([T_n_points, -1]).T
                u_valid_list.append(u)

            # RMSE calculation
            u = np.vstack(u_valid_list)
            truth = np.vstack(valid_truth_single_case_list)

            u_orig = (u+1)/2 * stress_range + stress_min
            truth_orig = (truth+1)/2 * stress_range + stress_min

            rmse = np.sqrt(np.mean((u_orig-truth_orig)**2))
            with open(args.run_dir+'/testcase_rmse_list.csv', 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerows([[args.data_path, testing_list[case], rmse]])


            if('1d' in args.fig_format):
                # u = np.vstack(u_valid_list)
                # truth = np.vstack(valid_truth_single_case_list)

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


                # Subplot 3: 3 time instances stress
                ax = plt.subplot(gs[2])
                stress_end_truth = truth[:,1]
                stress_end_pred = u[:,1]
                ax.plot(stress_end_truth,'-', label = 'FEM: t=1E4')
                ax.plot(stress_end_pred,'--', label='Proposed: t=1E4')

                stress_end_truth = truth[:,10]
                stress_end_pred = u[:,10]
                ax.plot(stress_end_truth,'-', label = 'FEM: t=1E5')
                ax.plot(stress_end_pred,'--', label='Proposed: t=1E5')

                stress_end_truth = truth[:,100]
                stress_end_pred = u[:,100]
                ax.plot(stress_end_truth,'-', label = 'FEM: t=1E6')
                ax.plot(stress_end_pred,'--', label='Proposed: t=1E6')

                # ax.legend(loc='upper right')
                ax.legend(loc='lower right')
                ax.set_title('FEM vs Proposed',  fontsize = font_size)

                fig.suptitle(args.hparams)

                # plt.show()
                plt.savefig('{}/truth_vs_pred_{}_epoch_{}_1d.png'.format(args.run_dir, str(testing_list[case]), str(epoch).zfill(4)), bbox_inches='tight')
                plt.close(fig)
            
            if('2d' in args.fig_format):
                v_max = 1
                v_min = -1
                max_size = 256
                max_size = int(max_size * 2)
                true_stress_map = np.ones([max_size,max_size]) * float("nan")
                pred_stress_map = np.ones([max_size,max_size]) * float("nan")
                for segment in range(len(u_valid_list)):
                    stress_pred = u_valid_list[segment][:,-1]
                    stress_true = valid_truth_single_case_list[segment][:,-1]

                    start_point = segment_endopoints_and_direction_single_case[segment][0]
                    end_point = segment_endopoints_and_direction_single_case[segment][1]
                    
                    start_point = (np.array(start_point) * 2).astype(np.int64)
                    end_point = (np.array(end_point) * 2).astype(np.int64)
                    
                    segment_length = max(end_point-start_point)
                    segment_width = 1
                    segment_width = int(segment_width * 2)

                    x = np.linspace(0, segment_length, len(stress_true))
                    interp_points = np.linspace(0,segment_length, segment_length+1)

                    f = interpolate.interp1d(x,stress_true)
                    stress_true_interp = [f(point).item() for point in interp_points]

                    f = interpolate.interp1d(x,stress_pred)
                    stress_pred_interp = [f(point).item() for point in interp_points]

                    if(start_point[0]==end_point[0]): # wire is vertical
                        for n_width in range(segment_width):
                            true_stress_map[start_point[0] - int(segment_width/2) + n_width, start_point[1]:end_point[1]+1] = stress_true_interp
                            pred_stress_map[start_point[0] - int(segment_width/2) + n_width, start_point[1]:end_point[1]+1] = stress_pred_interp
                    else: # wire is horizontal
                        for n_width in range(segment_width):
                            true_stress_map[start_point[0]:end_point[0]+1, start_point[1] - int(segment_width/2) + n_width] = stress_true_interp
                            pred_stress_map[start_point[0]:end_point[0]+1, start_point[1] - int(segment_width/2) + n_width] = stress_pred_interp
                
                # 2D image RMSE calculation
                pred_stress_map_orig = (pred_stress_map+1)/2 * stress_range + stress_min
                true_stress_map_orig = (true_stress_map+1)/2 * stress_range + stress_min

                rmse = rmse_nan_array(pred_stress_map_orig, true_stress_map_orig)
                with open(args.run_dir+'/testcase_2d_image_rmse_list.csv', 'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows([[args.data_path, testing_list[case], rmse]])

                fig = plt.figure(figsize=(60, 60))
                ax = plt.subplot(111)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis='both', which='both', length=0)
                im = ax.imshow(pred_stress_map, cmap='rainbow', vmax = v_max, vmin = v_min)
                # im = ax.imshow(pred_stress_map, cmap='rainbow', interpolation='nearest')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.15)
                plt.colorbar(im, cax=cax)
                # plt.show()
                plt.savefig('{}/truth_vs_pred_{}_epoch_{}_pred_2d.png'.format(args.run_dir, str(testing_list[case]), str(epoch).zfill(4)), bbox_inches='tight')
                plt.close(fig)

                fig = plt.figure(figsize=(60, 60))
                ax = plt.subplot(111)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis='both', which='both', length=0)
                im = ax.imshow(true_stress_map, cmap='rainbow', vmax = v_max, vmin = v_min)
                # im = ax.imshow(true_stress_map, cmap='rainbow', interpolation='nearest')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.15)
                plt.colorbar(im, cax=cax)
                # plt.show()
                plt.savefig('{}/truth_vs_pred_{}_epoch_{}_true_2d.png'.format(args.run_dir, str(testing_list[case]), str(epoch).zfill(4)), bbox_inches='tight')
                plt.close(fig)


            if('3d' in args.fig_format):
                # t=1e6
                font_size = 24
                fig = plt.figure(figsize=(60, 20))
                ax = plt.axes(projection='3d')
                for segment in range(len(u_valid_list)):
                    # if(segment not in [6,8,10,12,14,17,16,18]):
                    #     continue
                    z_pred_value = u_valid_list[segment][:,-1]
                    z_truth_value = valid_truth_single_case_list[segment][:,-1]

                    start_point = segment_endopoints_and_direction_single_case[segment][0]
                    end_point = segment_endopoints_and_direction_single_case[segment][1]

                    x_idx = np.linspace(start_point[0], end_point[0], num=z_pred_value.shape[0], endpoint=True)
                    y_idx = np.linspace(start_point[1], end_point[1], num=z_pred_value.shape[0], endpoint=True)

                    if segment == 0:
                        label_r = 'FEM: t=1E6'
                        label_p = 'Proposed: t=1E6'
                    else:
                        label_r = None
                        label_p = None
                    # ax.plot3D(x_idx, y_idx, z_truth_value, '-', color = 'dodgerblue', label = label_r, linewidth=2)
                    ax.plot3D(x_idx, y_idx, z_truth_value, 'r-', label = label_r)
                    # ax.scatter(x_idx, y_idx, z_truth_value, c=z_pred_value, cmap='Blues', alpha=1, label='Prediction')

                    # ax.plot3D(x_idx, y_idx, z_pred_value, '--', color='orange', label = label_p, linewidth=2)
                    ax.scatter(x_idx, y_idx, z_pred_value, color = 'b', marker = 'o', alpha=0.2, label=label_p)
                    # ax.plot3D(x_idx, y_idx, z_pred_value, 'b.', alpha=0.1, label='Prediction')
                    # ax.scatter(x_idx, y_idx, z_pred_value, c=z_pred_value, marker = 'o', cmap='Blues', alpha=0.5, label='Prediction')

                ax.legend(loc='upper right', fontsize = font_size)
                ax.set_title('FEM vs Proposed', fontsize = font_size)

                fig.suptitle(args.hparams)

                # plt.show()
                plt.savefig('{}/truth_vs_pred_{}_epoch_{}_3d.png'.format(args.run_dir, str(testing_list[case]), str(epoch).zfill(4)), bbox_inches='tight')
                plt.close(fig)