import time
import torch
import utils
import numpy as np
from sklearn.metrics import f1_score
import random
import os
import argparse
from Modules import torch_LSTM
import pandas as pd
from predict import predict

def train(model, opt, device, x = None, y = None, xt = None, yt = None):
    print(f"Device is {device}.\n")
    start = time.time()

    if opt.active_log:
        import wandb
        wandb.init(project="Lithology-"+opt.run_name, entity="2djagpal")
        wandb.config.update(opt)
        
    print('Loading Data ...\n')
    train_only = False

    if type(x) == type(None):
            train_data = utils.import_data(device,"../../Data/X_train_WELL.csv", "../../Data/Y_train.csv")
            valid_data = utils.import_data(device,"../../Data/X_test_WELL.csv", "../../Data/Y_test.csv")
    elif type(xt) != type(None):
        train_data = utils.RNN_dataset(device, x, y)
        valid_data = utils.RNN_dataset(device, xt, yt)
        #print(utils.samples(train_data), utils.samples(valid_data))
    else:
        train_only = True
        train_data = utils.RNN_dataset(device, x, y)


    # Model save path
    if opt.savemodel:
        modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.run_name)
        os.makedirs(modelsave_path, exist_ok=True)

    # Penalty Matrix for scores
    A = np.load('../penalty_matrix.npy')

    # Choice of optimiser
    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),lr = opt.lr, weight_decay=opt.weight_decay)
    if opt.optim == 'sgdm':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum= opt.momentum, dampening=0, weight_decay=opt.weight_decay, nesterov=False)
    if opt.optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr, alpha=0.99, eps=1e-08, weight_decay=opt.weight_decay, momentum=opt.momentum, centered=False)

    # If learning rate scheduler milestones set
    if opt.milestones != None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.1)

    # Loss criterion
    if opt.class_weights == 'bal':
        Criterion = torch.nn.CrossEntropyLoss(weight=torch.load('../cw').to(device))
    else: 
        Criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(np.ones(12)).to(device))

    # Set baseline best accuracy
    best_valid_accuracy = 0
    
    print('Starting training...\n')
    # Main loop over number of epochs
    for epoch in range(opt.n_epoch):
        model.train()
        # For each well
        Names = train_data.UniqueNames.copy()
        # Shuffle order being learned each epoch
        random.shuffle(Names)
        for i in Names:
            dl = torch.utils.data.DataLoader(train_data.datasets[i], shuffle = False,  batch_size=opt.batchsize)
            for x,y in dl:
                optimizer.zero_grad() # Clears existing gradients from previous epoch
                output, hidden = model(x.unsqueeze(-1).permute(2,0,1))
                loss = Criterion(output, y.view(-1).long())
                loss.backward() # Does backpropagation and calculates gradients
                optimizer.step() # Updates the weights accordingly
                
                if opt.active_log:
                    wandb.log({'train_batch_loss': loss.data})
                # print(f'Well {i}, loss {loss.data}')
                # Incase of gradient exploding
                if loss.data == torch.nan:
                    print('NAN GRADIENT DETECTED')
                    break
        
        # Loss
        train_epoch_loss = utils.calc_loss(model,train_data, Criterion)
        if opt.active_log:
            wandb.log({'train_epoch_loss': train_epoch_loss})
        if train_only == False:
            valid_epoch_loss = utils.calc_loss(model, valid_data, Criterion)
            if opt.active_log:
                wandb.log({'valid_epoch_loss': valid_epoch_loss})
        # Start evaluation
        model.eval()

        # predictions
        train_y_pred = predict(model, train_data)
        if train_only == False:
            valid_y_pred = predict(model, valid_data)

        # Accuracy
        train_epoch_acc = utils.accuracy(train_data.y, train_y_pred)
        if opt.active_log:
            wandb.log({'train_accuracy': train_epoch_acc})
        if train_only == False:
            valid_epoch_acc = utils.accuracy(valid_data.y, valid_y_pred)
            if opt.active_log:
                wandb.log({'valid_accuracy': valid_epoch_acc})

        # Score
        train_epoch_score = utils.score(train_data.y,train_y_pred,A)
        if opt.active_log:
            wandb.log({'train_score': train_epoch_score})
        if train_only == False:
            valid_epoch_score = utils.score(valid_data.y,valid_y_pred,A)
            if opt.active_log:
                wandb.log({'valid_score': valid_epoch_score})


        print('Training:')
        print(f'Epoch: {epoch+1} ... Accuracy: {train_epoch_acc} ... Score: {train_epoch_score}\n')
        if train_only == False:
            print('Validation:')
            print(f'Epoch: {epoch+1} ... Accuracy: {valid_epoch_acc} ... Score: {valid_epoch_score}\n')
        
        if train_only == False:
            if opt.savemodel:
               if valid_epoch_acc > best_valid_accuracy:
                    best_valid_accuracy = valid_epoch_acc
                    torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))

        if epoch == opt.n_epoch - 1:
            print(f'Class F1 Scores')
            train_f1 = f1_score(train_data.y.cpu().numpy(),train_y_pred.cpu().numpy(),average=None)
            print(f'Training: \n{train_f1}\n')
            if train_only == False:
                valid_f1 = f1_score(valid_data.y.cpu().numpy(),valid_y_pred.cpu().numpy(),average=None)
                print(f'Validation: \n{valid_f1}\n')

            



    end = time.time()
    print("Training Runtime: {}\n".format(end-start))

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', default=1, type=int)
    parser.add_argument('--input_size', default=26, type=int)
    parser.add_argument('--output_size', default=12, type=int)
    parser.add_argument('--hidden_dim', default=16, type=int)
    parser.add_argument('--n_layers', default=1, type=int)

    parser.add_argument('--from_file', default='.', type=str)

    parser.add_argument('--optim', default='adamw', type=str,choices = ['adamw','sgdm','RMSprop'])
    parser.add_argument('--milestones', default=[], type=list)
    parser.add_argument('--class_weights', default= 'bal' , type=str, choices = ['bal', 'unbal'])
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--n_epoch', default=100, type=int)
    parser.add_argument('--batchsize', default=128, type=int)
    parser.add_argument('--momentum', default=0.1, type=float)

    parser.add_argument('--savemodel', default=True, type=bool)
    parser.add_argument('--savemodelroot', default='./bestmodels/RNN', type=str)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--active_log', action = 'store_true')


    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    model = torch_LSTM(opt.window_size, opt.input_size, opt.output_size, opt.hidden_dim, opt.n_layers, device).to(device)
    print(f'Model Parameters: \n{model.parameters}\n')
    
    if opt.from_file != '.':
        model.load_state_dict(torch.load(opt.from_file))
    
    train(model, opt, device)

    