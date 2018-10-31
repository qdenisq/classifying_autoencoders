"""
Classifying variational autoencoders
"""
import argparse
import pprint
import numpy as np
import numpy.random as random
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from src.pytorch_cl_vae.model import ClVaeModel



def train_MNIST(args):

    params = vars(args)
    device =torch.device(params['device'])
    # load MNIST database
    mnist = fetch_mldata('MNIST original', data_home=params['data_dir'])
    mnist.data = mnist.data / 255
    num_samples, input_dim = mnist.data.shape
    num_classes = len(np.unique(mnist.target))
    lb = preprocessing.LabelBinarizer()
    lb.fit(mnist.target)
    params['classes_dim'] = [num_classes]
    params['original_dim'] = input_dim
    print('MNIST db has been successfully loaded, stored in the: "{}"'.format(params['data_dir'] + '/mldata'))
    # split data to train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.1, random_state=0)
    print("| Train subset shape:{} | Test subset shape:{} |".format(X_train.shape, X_test.shape))

    # Initialize ClVaeModel
    model = ClVaeModel(**params)
    print("Model successfully initialized with params: ")
    pprint.PrettyPrinter(indent=4).pprint(params)

    train_losses = []
    train_accuracies = []
    # Losses = namedtuple('Losses', 'rec_loss z_dkl_loss class_loss w_dkl_loss')
    # Accuracies = namedtuple("accuracies", 'accuracy')

    # Train loop
    train_step_i = 0
    for epoch in range(params['num_epochs']):
        for i in range(X_train.shape[0] // params['batch_size']):
            # Sample batch
            idx = random.choice(np.arange(0, X_train.shape[0]), params['batch_size'])
            x_batch = torch.from_numpy(X_train[idx]).float().to(device)
            y_batch = lb.transform(y_train[idx])
            y_batch = [torch.from_numpy(y_batch).float().to(device)]

            step_losses, step_accuracies = model.train_step(x_batch, y_batch)
            train_losses.append(step_losses)
            train_accuracies.append(step_accuracies)
            train_step_i += 1

            print("\r|progress: {:.2f}% | train step: {} | rec loss: {:.4f} | z_dkl loss: {:.4f} | class loss: {:.4f}"
                  " | w_dkl loss: {:.4f} | class_accuracy: {:.4f} |".format(
                100.* train_step_i / (X_train.shape[0] // params['batch_size'] * params['num_epochs']), train_step_i, *step_losses, *step_accuracies
                ), end='')
            if train_step_i % 100 == 0:
                print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
                help='tag for current run')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='whether to use gpu or not')
    parser.add_argument('--batch_size', type=int, default=100,
                help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam-wn',
                help='optimizer name') # 'rmsprop'
    parser.add_argument('--num_epochs', type=int, default=200,
                help='number of epochs')
    parser.add_argument('--original_dim', type=int, default=88,
                help='input dim')
    parser.add_argument('--intermediate_dim', type=int, default=88,
                help='intermediate dim')
    parser.add_argument('--latent_dim', type=int, default=2,
                help='latent dim')
    parser.add_argument('--encoder_hidden_size', type=int, default=512,
                        help='encoder hidden layer size')
    parser.add_argument('--decoder_hidden_size', type=int, default=512,
                        help='decoder hidden layer size')
    parser.add_argument('--classifier_hidden_size', type=int, default=512,
                        help='classifier hidden layer size')
    parser.add_argument('--vae_learning_rate', type=float, default=0.0001,
                        help='vae learning rate')
    parser.add_argument('--classifier_learning_rate', type=float, default=0.0001,
                        help='classifier learning rate')
    parser.add_argument('--seq_length', type=int, default=1,
                help='sequence length (concat)')
    parser.add_argument('--class_weight', type=float, default=1.0,
                help='relative weight on classifying key')
    parser.add_argument('--w_log_var_prior', type=float, default=0.0,
                help='w log var prior')
    parser.add_argument('--intermediate_class_dim',
                type=int, default=88,
                help='intermediate dims for classes')
    parser.add_argument("--do_log", action="store_true", 
                help="save log files")
    parser.add_argument("--predict_next", action="store_true", 
                help="use x_t to 'autoencode' x_{t+1}")
    parser.add_argument("--use_x_prev", action="store_true",
                help="use x_{t-1} to help z_t decode x_t")
    parser.add_argument('--patience', type=int, default=5,
                help='# of epochs, for early stopping')
    parser.add_argument("--kl_anneal", type=int, default=0, 
                help="number of epochs before kl loss term is 1.0")
    parser.add_argument("--w_kl_anneal", type=int, default=0, 
                help="number of epochs before w's kl loss term is 1.0")
    parser.add_argument('--log_dir', type=str, default='../data/logs',
                help='basedir for saving log files')
    parser.add_argument('--model_dir', type=str,
                default='../../data/models',
                help='basedir for saving model weights')    
    parser.add_argument('--train_file', type=str,
                default='../data/input/JSB Chorales_Cs.pickle',
                help='file of training data (.pickle)')
    parser.add_argument('--data_dir', type=str,
                        default='../../data',
                        help='basedir for saving and loading training data')
    args = parser.parse_args()
    train_MNIST(args)
