#!/bin/bash
dataset_list=('ABIDE')
batch_size_list=(64 32 16)
base_lr_list=(1e-5 1e-4)
target_lr_list=(1e-4 1e-5)
wd_list=(1e-3 1e-4 1e-5)
layers_list=(1 2 3)
activation_list=('leaky_relu')
dropout_list=(0.1 0. 0.2 0.3)
pooling_list=(True)
cluster_num_list=(32 16 4)
knn_num_list=(80 160 20 40)
dae_loss_percent_list=(1.0 2.0 4.0 8.0)
noise_percent_list=(0.05 0.1 0.2)
knn_layer_list=(3 2 5 4 1)




for name in "${dataset_list[@]}"; do
  for pl in "${pooling_list[@]}"; do
    for acl in "${activation_list[@]}"; do
      for b_lrl in "${base_lr_list[@]}"; do
        for t_lrl in "${target_lr_list[@]}"; do
          for wdl in "${wd_list[@]}"; do
            for bzl in "${batch_size_list[@]}"; do
              for dl in "${dropout_list[@]}"; do
                for clus_numl in "${cluster_num_list[@]}"; do
                  for knn_num in "${knn_num_list[@]}"; do
                    for dae_loss_percent in "${dae_loss_percent_list[@]}"; do
                      for layersl in "${layers_list[@]}"; do
                        for noise_percent in "${noise_percent_list[@]}"; do
                          for knn_layer in "${knn_layer_list[@]}"; do
                                python main.py --dataset $name \
                                    --batch_size $bzl \
                                    --base_lr $b_lrl \
                                    --target_lr $t_lrl \
                                    --weight_decay $wdl \
                                    --layers $layersl \
                                    --activation $acl \
                                    --dropout $dl \
                                    --pooling $pl \
                                    --cluster_num $clus_numl \
                                    --knn_num $knn_num \
                                    --dae_loss_percent $dae_loss_percent \
                                    --noise_percent $noise_percent \
                                    --knn_layer $knn_layer
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done


