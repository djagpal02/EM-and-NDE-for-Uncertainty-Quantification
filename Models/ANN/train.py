import time
import torch
import utils
import numpy as np
from sklearn.metrics import f1_score
import Modules
import os
import argparse
from predict import predict
from torchsampler import ImbalancedDatasetSampler

def train(model, opt, device, x, y , xt = None, yt = None):
    print(f"Device is {device}.\n")
    start = time.time()

    if opt.active_log:
        import wandb
        wandb.init(project="Lithology-ANN_"+opt.run_name, entity="2djagpal")
        wandb.config.update(opt)
        
    print('Loading Data ...\n')
    train_only = True


    train_data = utils.dataset(x, y , device)

    if type(xt) != type(None):
        train_only = False
        valid_data = utils.dataset(xt, yt, device)

    # Create DataLoader objects, Loss Objec and Optimiser Object 
    if opt.balance == True:
        train_loader = torch.utils.data.DataLoader(train_data, sampler = ImbalancedDatasetSampler(train_data),  batch_size=opt.batchsize)
    else:
        train_loader = torch.utils.data.DataLoader(train_data,  batch_size=opt.batchsize)


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
        # Shuffle order being learned each epoch
        for x,y in train_loader:
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            z = model(x)
            loss = Criterion(z, y.view(-1).long())
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
                print(f'Training: \n{valid_f1}\n')

            



    end = time.time()
    print("Training Runtime: {}\n".format(end-start))

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../../Data/X_train_0.csv', type=str)
    parser.add_argument('--labels', default='../../Data/Y_train.csv', type=str)

    parser.add_argument('--Layers', default=[26,12], type=list)
    parser.add_argument('--activation', default=None, type=str, choices = ['relu', 'leaky_relu', 'sigmoid', 'tanh'])
    parser.add_argument('--dropout', default=0, type=float)

    parser.add_argument('--from_file', default='.', type=str)

    parser.add_argument('--optim', default='adamw', type=str,choices = ['adamw','sgdm','RMSprop'])
    parser.add_argument('--milestones', default=[], type=list)
    parser.add_argument('--class_weights', default= 'bal' , type=str, choices = ['bal', 'unbal'])
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--n_epoch', default=100, type=int)
    parser.add_argument('--batchsize', default=1024, type=int)
    parser.add_argument('--momentum', default=0.1, type=float)
    parser.add_argument('--balance', default=False, type=bool)
    
    parser.add_argument('--savemodel', default=True, type=bool)
    parser.add_argument('--savemodelroot', default='./bestmodels/ANN', type=str)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--active_log', action = 'store_true')


    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    model = Modules.Net(opt.Layers, opt.activation, opt.dropout).to(device)
    print(f'Model Parameters: \n{model.parameters}\n')
    
    if opt.from_file != '.':
        model.load_state_dict(torch.load(opt.from_file))
    
    x,y = utils.import_df(opt.data, opt.labels)

    train(model, opt, device, x, y)

    
