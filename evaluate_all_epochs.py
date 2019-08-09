import argparse, os, sympy.series
from evaluate import eval

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", help="Name of directory of specific model in ./models parent directory, such as unet, attention-unet or segan.")
parser.add_argument("-d", "--dataset_path", required=True)
parser.add_argument("-wp", "--weights_path", help="Full path to the weight file.")
parser.add_argument("-s", "--store_imgs", default=False, action='store_true')
parser.add_argument("-i", "--exp_id", type=int, required=True)

args = parser.parse_args()

assert(args.model_name in ['unet', 'attentionUnet'])

sys.path.append('./models/unet')

if args.model_name == 'unet':
    model_dir = './models/unet'    
    from base_model import Unet
    model = Unet()

elif args.model_name == 'attentionUnet' :        
    model_dir = './models/attentionUnet'
    sys.path.append(model_dir)
    from model import AttentionalUnet
    model = AttentionalUnet()

for f in os.listdir(args.weights_path):
    print("loading weight: {}".format(f))
    eval(model, os.path.join(args.weights_path, f), args.store_imgs, args.dataset_path, args.exp_id)