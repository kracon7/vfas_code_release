import os
import time
import numpy as np
from datetime import timedelta
from data.data_manager import DatasetManager, GraspEvaluatorData
from torch.utils.data import DataLoader
from scripts.writer import Writer
from test import run_test
from options.train_options import TrainOptions
from models import create_model


def main():
    opt = TrainOptions().parse()
    if opt == None:
        return
    #Split data and enforce ratios of positive, negative and hard negative grasps
    data_manager = DatasetManager()
    data_load_start = time.time()
    train_df, test_df = data_manager.get_train_test_dataframes(
        opt.dataset_root_folder,
        opt.test_split,
        opt.total_num_objs,
        remove_duplicates=True,
    )
    print(f"Finished loading data into training and testing dataframes with lengths {len(train_df)} and {len(test_df)}")
    
    # data_manager.test_normal_based_occlusion(
    #     train_df,
    #     np.random.choice(range(len(train_df)), size=30),
    #     #[11747, 12705, 8718, 4142, 10561, 13929, 4391],  #USE SEED=8 for these indexes
    #     angle_threshold=80,
    #     drop_probability=0.7)
    # return
    # data_manager.test_add_pcd_noise(
    #     train_df,
    #     np.random.choice(range(len(train_df)), size=10),
    #     noise=0.002)
    # return
    if len(test_df)>0:
        test_df = data_manager.enforce_data_balance(
            test_df,
            'Testing dataframe',
            pos_ratio = opt.positive_grasps_ratio,
            neg_ratio = 1.0-opt.positive_grasps_ratio-opt.hard_negative_grasps_ratio,
            hard_neg_ratio = opt.hard_negative_grasps_ratio,
        )
        model_dir = os.path.join(opt.checkpoints_dir ,opt.name)
        data_manager.save_dataframe(test_df, model_dir, "testing_df.pkl")
    test_dataset = GraspEvaluatorData(opt, test_df)
    print(f"########### Took {timedelta(seconds=time.time()-data_load_start)} to load all data ###########")


    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        batch_iter = 0
    
        # Reshuffle balanced data (balancing samples random)
        print("New Epoch, rebalancing training data")
        balanced_train_df = data_manager.enforce_data_balance(
            train_df,
            'Training dataframe',
            pos_ratio = opt.positive_grasps_ratio,
            neg_ratio = 1.0-opt.positive_grasps_ratio-opt.hard_negative_grasps_ratio,
            hard_neg_ratio = opt.hard_negative_grasps_ratio,
        )
        # data_manager.save_dataframe(train_df, model_dir, "training_df.pkl")
        train_dataset = GraspEvaluatorData(opt, balanced_train_df)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            pin_memory=True,
            drop_last=True,
        )
        train_dataset_size = len(train_dataloader)
        
        # Training main loop
        for i, data in enumerate(train_dataloader):
            iter_start_time = time.time()
            total_steps += opt.batch_size
            batch_iter += 1
            model.set_input(data)
            model.optimize_parameters()
            if batch_iter % opt.print_freq == 0:
                loss = [ model.loss ] 
                loss_types = ["classification_loss"]
                iter_t = (time.time() - iter_start_time)
                writer.print_current_losses(epoch, total_steps, loss, iter_t,
                                            loss_types)
                writer.plot_loss(loss, epoch, total_steps, train_dataset_size,
                                 loss_types)

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest', epoch)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest', epoch)
            model.save_network(str(epoch), epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay,
               time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            #Run both test and training data through network, collect results
            if len(train_dataset)>0:
                print(' ######### Running Test on training data  #########')
                train_class_stats = run_test(train_dataset,
                                             epoch, 
                                             name=opt.name,
                                             dataset_descriptor="Training data")
            if len(test_dataset)>0:
                print(' ######### Running Test on test data  #########')
                test_class_stats = run_test(test_dataset,
                                            epoch,
                                            name=opt.name,
                                            dataset_descriptor="Testing data")
            #Log info to tensorboard
            if len(test_dataset)>0:
                writer.plot_class_stats(epoch, train_class_stats, test_class_stats)
            else:
                writer.plot_class_stats(epoch, train_class_stats, None)

    writer.close()


if __name__ == '__main__':
    main()
