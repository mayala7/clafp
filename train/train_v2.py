import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import math
from torch.utils.tensorboard import SummaryWriter
from utils.metric import AverageMeter, evaluate
from utils.model import set_seeds
from models.textencoder import TextEncoder
from models.boldencoder import BOLDEncoder
from dataset.dataset import CustomDataset
import argparse
from pca import * 

def eval(text_encoder, bold_encoder, eval_loader, device, split):
    text_encoder.eval()
    bold_encoder.eval()

    text_embeddings = None
    bold_embeddings = None

    with torch.no_grad():
        for text, bold in tqdm(eval_loader):
            text = text.to(device)
            bold = bold.to(device)

            # Forward
            text_embedding = text_encoder(text)
            bold_embedding = bold_encoder(bold)

            if text_embeddings is None:
                text_embeddings = text_embedding
                bold_embeddings = bold_embedding
            else:
                text_embeddings = torch.cat((text_embeddings, text_embedding), dim=0)
                bold_embeddings = torch.cat((bold_embeddings, bold_embedding), dim=0)

        # Compute similarity
        sim = torch.einsum('i d, j d -> i j', text_embeddings, bold_embeddings) * math.e
        labels = torch.arange(text_embeddings.shape[0]).to(device)

        # Compute accuracy
        acc1, acc2 = evaluate(sim, labels, topk=(1, 2))
        print(f'{split} Acc@1 {acc1.item():.3f}')
        print(f'{split} Acc@2 {acc2.item():.3f}')
        return acc1.item(), acc2.item()

def train(epoch, text_encoder, bold_encoder, optimizer, train_loader, writer, device):
    text_encoder.train()
    bold_encoder.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    iteration = len(train_loader) * epoch

    for text, bold in tqdm(train_loader):
        text = text.to(device)
        bold = bold.to(device)
        batch_size = text.shape[0]

        optimizer.zero_grad()

        # Forward
        text_embedding = text_encoder(text)
        bold_embedding = bold_encoder(bold)

        # Compute similarity
        sim = torch.einsum('i d, j d -> i j', text_embedding, bold_embedding) * math.e

        # Compute loss
        labels = torch.arange(batch_size).to(device)
        loss = F.cross_entropy(sim, labels)

        # Update metric
        losses.update(loss.item(), batch_size)
        acc1, acc2 = evaluate(sim, labels, topk=(1, 2))
        top1.update(acc1.item(), batch_size)
        top2.update(acc2.item(), batch_size)

        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 50 == 0:
            writer.add_scalar('Train/Loss', losses.avg, iteration)
            writer.add_scalar('Train/Acc@1', top1.avg, iteration)
            writer.add_scalar('Train/Acc@2', top2.avg, iteration)

    print(f'Epoch: {epoch}')
    print(f'Train Acc@1 {top1.avg:.3f}')
    print(f'Train Acc@2 {top2.avg:.3f}')

def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # exp_path = os.path.join('./experiments', args.exp_name)
    # ckpt_path = os.path.join(exp_path, 'checkpoints')
    # logs_path = os.path.join(exp_path, 'logs')
    # os.makedirs(exp_path, exist_ok=True)
    # os.makedirs(ckpt_path, exist_ok=True)
    # os.makedirs(logs_path, exist_ok=True)

    writer = SummaryWriter(log_dir=logs_path)

    # Load dataset
    #load_data(text_file_path, bold_file_path, batch_size, num_workers)
    train_loader, val_loader, test_loader = load_data(args.data_dir, args.batch_size, args.num_workers)

    # Initialize models
    text_encoder = TextEncoder(output_channels=args.num_output_channels).to(device)
    bold_encoder = BOLDEncoder(output_channels=args.num_output_channels).to(device)

    # Optimizer
    optimizer = optim.Adam(list(text_encoder.parameters()) + list(bold_encoder.parameters()), lr=args.lr)

    # Optionally continue from a checkpoint
    start_epoch = 0
    if args.continue_from_epoch >= 0:
        checkpoint_path = os.path.join(ckpt_path, f'ckpt_epoch_{args.continue_from_epoch}.pth')
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            text_encoder.load_state_dict(checkpoint['text_encoder'])
            bold_encoder.load_state_dict(checkpoint['bold_encoder'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = args.continue_from_epoch + 1

    # Training loop
    for epoch in range(start_epoch, args.total_epochs):
        train(epoch, text_encoder, bold_encoder, optimizer, train_loader, writer, device)
        if epoch % args.save_freq == 0:
            save_path = os.path.join(ckpt_path, f'ckpt_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'text_encoder': text_encoder.state_dict(),
                'bold_encoder': bold_encoder.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_path)

        # Validation
        eval(text_encoder, bold_encoder, val_loader, device, 'val')

    # Test and close resources
    eval(text_encoder, bold_encoder, test_loader, device, 'test')
    writer.close()

def load_data(text_file_path, bold_file_path, batch_size, num_workers):
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    import numpy as np

    # Load text embeddings
    text_embeddings = np.load(text_file_path)

    # Load BOLD embeddings
    bold_data = np.load(bold_file_path)

    #applying PCA to BOLD 
    bold_pca = apply_pca_all_tensors(bold_data)

    # Create a TensorDataset
    dataset = TensorDataset(bold_pca, text_embeddings)

    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return data_loader



def get_args():
    parser = argparse.ArgumentParser(description="Train a model to match text embeddings with BOLD embeddings.")
    
    # Add arguments
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation (default: cuda).')
    #required
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset files.')
    #required
    parser.add_argument('--text_file_path', type=str, required=True, help='Absolute path to the text embeddings npz file.')
    #required
    parser.add_argument('--bold_file_path', type=str, required=True, help='Absolute path to the BOLD embeddings npz file.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation (default: 32).')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for loading data (default: 4).')
    parser.add_argument('--num_output_channels', type=int, default=128, help='Number of output channels for the encoders (default: 128).')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer (default: 0.001).')
    parser.add_argument('--total_epochs', type=int, default=20, help='Total number of training epochs (default: 20).')
    parser.add_argument('--save_freq', type=int, default=5, help='Frequency of saving the checkpoint (default: every 5 epochs).')
    parser.add_argument('--continue_from_epoch', type=int, default=-1, help='Epoch from which to continue training, -1 means from scratch (default: -1).')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    set_seeds(42)  # Assuming set_seeds is a function you have for setting random seeds
    run(args)
