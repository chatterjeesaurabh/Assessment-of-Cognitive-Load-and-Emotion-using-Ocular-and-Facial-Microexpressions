import os
import argparse

from solver_inference import solver_inference
from model_inference import model_inference         ## new run inference only
from utils import set_seed 

# TESTING / INFERENCE MAE and ResNet trained models:
# Run this 'inference.py' with the defined arguments : code: 'model_inference.py' and 'solver_inference.py'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)

# Single Model Inference only OR Test set Inference ?
parser.add_argument('--run_type', type=str, default='run', choices=['run', 'test']) 

# storage
parser.add_argument('--data_root', type=str, default='AU_Recognition/data')                       # path to your dataset on your local machine.
parser.add_argument('--ckpt_path', type=str, default='AU_Recognition/resnet_disfa_all')           # folder path for writing checkpoints.

# data
parser.add_argument('--data', type=str, default='DISFA', choices=['BP4D', 'DISFA'])               # Dataset to be used. [“DSIFA”,“BP4D”,"AffectNet","RAF-DB"]
parser.add_argument('--fold', type=str, default='all', choices=['0', '1', '2', '3', '4','all'])   # five-fold cross-validation to report performance on DISFA, and three-fold on BP4D
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--num_labels', type=int, default=12)
parser.add_argument('--sigma', type=float, default=10.0)

# model
parser.add_argument('--model_name', type=str, default='resnet', choices=['resnet_heatmap','resnet','swin','mae','emotionnet_mae','gh_feat'])    # choose: 'resnet' or 'emotionnet_mae'
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--hidden_dim', type=int, default=128) 
parser.add_argument('--half_precision', action='store_true')        # Float16
# training
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--interval', type=int, default=500)
parser.add_argument('--threshold', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=3e-5)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--loss', type=str, default='unweighted')
parser.add_argument('--clip', type=int, default=1.0)
parser.add_argument('--when', type=int, default=10, help='when to decay learning rate')
parser.add_argument('--patience', type=int, default=5, help='early stopping')
parser.add_argument('--fm_distillation', action='store_true')                                   # Use feature matching distillation for training.
                                                                                                # 'store_true': if this argument 'fm_distillation' is MENTIONED in the command, 
                                                                                                # then it will return True, else if not mentioned then it returns False.
# device
parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'])


opts = parser.parse_args()          # stores all the arguments/variables values defined above
print(opts)
# os.makedirs(opts.ckpt_path,exist_ok=True)

# Fix random seed
set_seed(opts.seed)


if (opts.run_type == 'test'):       # inference on Test set
    # Setup solver 
    solver = solver_inference(opts).cuda()

    # Start training
    solver.run()

elif (opts.run_type == 'run'):      # inference on any image
    # Setup solver 
    solver = model_inference(opts).cuda()

    # Run Model for Output Predicted Label:
    solver.run()


