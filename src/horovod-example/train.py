import argparse
import pathlib

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms, models
import horovod.torch as hvd


parser = argparse.ArgumentParser(description="PyTorch + Horovod distributed training benchmark")
parser.add_argument('--data-dir', type=str, help='path to ILSVR data')
_help = """
number of batches processed locally before executing allreduce across workers; 
it multiplies total batch size.
"""
parser.add_argument('--batches-per-allreduce', type=int, default=1, help=_help)

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32, help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90, help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=1.25e-2, help='learning rate for a single GPU')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=5e-5, help='weight decay')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def _partial_fit(model_fn, loss_fn, X_batch, y_batch, opt):
    # forward pass
    loss = loss_fn(model_fn(X_batch), y_batch)

    # back propagation
    loss.backward()
    opt.step()
    opt.zero_grad() # don't forget to reset the gradient after each batch!


def _validate(model_fn, loss_fn, validation_data_loader):
        model_fn.eval()
        with torch.no_grad():
            batch_losses, batch_sizes = zip(*[(loss_fn(model_fn(X), y), len(X)) for X, y in validation_data_loader])
            validation_loss = np.sum(np.multiply(batch_losses, batch_sizes)) / np.sum(batch_sizes)
            print(f"Training epoch: {epoch}, Validation loss: {validation_loss}")


def fit(model_fn, loss_fn, training_data_loader, opt, validation_data_loader=None, number_epochs=2):
    
    for epoch in range(number_epochs):
        model_fn.train()
        for X_batch, y_batch in training_data_loader:
            _partial_fit(model_fn, loss_fn, X_batch, y_batch, opt)
        
        # compute validation loss after each training epoch
        if validation_data_loader is not None:
            _validate(model_fn, loss_fn, validation_data_loader)


hvd.init()
torch.manual_seed(args.seed)
torch.cuda.set_device(hvd.local_rank()) # Horovod: pin GPU to local rank.
torch.cuda.manual_seed(args.seed)

# create data sets
data_dir = pathlib.Path(args.data_dir)
_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder(data_dir / "train", transform=_train_transform)

_val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_dataset = datasets.ImageFolder(data_dir / "val", transform=_val_transform)

# Horovod: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=hvd.size()` and `rank=hvd.rank()`.
train_sampler = (data.distributed
                     .DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank()))
val_sampler = (data.distributed
                   .DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank()))

# create data loaders 
class WrappedDataLoader:
    
    def __init__(self, data_loader, f):
        self._data_loader = data_loader
        self._f = f
        
    def __len__(self):
        return len(self._data_loader)
    
    def __iter__(self):
        for batch in iter(self._data_loader):
            yield self._f(*batch)

_data_loader_kwargs = {'num_workers': 6, "pin_memory": True}
_train_data_loader = (data.DataLoader(train_dataset,
                                      batch_size = args.batch_size * args.batches_per_allreduce,
                                      sampler=train_sampler,
                                      **_data_loader_kwargs))

_val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=args.val_batch_size,
                                               sampler=val_sampler,
                                               **_data_loader_kwargs)

_data_to_gpu = lambda X, y: (X.cuda(), y.cuda())
train_data_loader = WrappedDataLoader(_train_data_loader, _data_to_gpu)
val_data_loader = WrappedDataLoader(_val_data_loader, _data_to_gpu)

# Set up standard ResNet-50 model.
model_fn = (models.resnet50()
                  .cuda())

loss_fn = F.cross_entropy
            
# Horovod: scale learning rate by the number of GPUs.
_optimizer = optim.SGD(model_fn.parameters(),
                       lr=args.base_lr,
                       momentum=args.momentum,
                       weight_decay=args.weight_decay)

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(_optimizer,
                                     named_parameters=model_fn.named_parameters(),
                                     backward_passes_per_step=args.batches_per_allreduce)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model_fn.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

fit(model_fn, loss_fn, train_data_loader, optimizer, val_data_loader, number_epochs=args.epochs)




