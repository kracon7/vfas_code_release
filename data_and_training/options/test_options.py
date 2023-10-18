from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--which_epoch',
            type=str,
            default='latest',
            help='which epoch to load? set to latest to use latest cached model'
        )

        self.parser.add_argument("--env_param_file", type=str,
                    default="config/GraspEvaNet_Dataset_Gen_SR_Gripper.yaml")
        
        self.parser.add_argument("--num_objects", type=int, default=3)

        self.is_train = False