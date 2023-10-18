def create_model(opt):
    from .grasp_model import GraspEvalModel
    model = GraspEvalModel(opt)
    return model
