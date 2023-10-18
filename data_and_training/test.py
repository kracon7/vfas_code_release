import os
import numpy as np
from options.test_options import TestOptions
from torch.utils.data import DataLoader
from data.data_manager import DatasetManager
from models import create_model
from scripts.writer import Writer
from scripts.test_visualization import TestVisualization


def run_test(test_dataset, epoch=-1, name="", dataset_descriptor=""):
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.name = name
    dataset = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            drop_last=True,
    )
    model = create_model(opt)
    writer = Writer(opt, dataset_descriptor)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        tp, fp, tn, fn, nexamples = model.test()
        writer.update_counter(tp, fp, tn, fn, nexamples)
    
    writer.print_classifier_metrics(epoch,
                                    writer.acc,
                                    writer.tp_rate,
                                    writer.fp_rate,
                                    writer.tn_rate,
                                    writer.fn_rate)
    #writer.print_reg(epoch, writer.regression_stats)
    return writer.classification_stats#, writer.regression_stats


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.name = opt.model_name
    data_manager = DatasetManager()
    test_df = data_manager.load_dataframe(os.path.join('./checkpoints', opt.model_name, 'testing_df.pkl'))
    test_viz = TestVisualization(opt, test_df)

    unique_scene = np.unique(test_df["Scene_info_path"])
    for scene_path in unique_scene:
        print(f"Filtering test dataframe for scene: {scene_path}")
        obj_df = test_df[test_df["Scene_info_path"]==scene_path]
        obj_df = data_manager.enforce_data_balance(obj_df,
                                                   "Object dataframe",
                                                   pos_ratio=0.5,
                                                   neg_ratio=0.2,
                                                   hard_neg_ratio=0.3)
        test_viz.view_classification_results(obj_df, sample_n=50)
        #test_viz.visualize_grasp_quality(obj_df, sample_n=50, score_intervals=[(0.0, 0.5),(0.5, 0.75),(0.75, 1.0)])
