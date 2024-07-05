# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

"""# Import Libraries"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils.model import select_model
from utils.options import parse_args_function
from utils.dataset import Dataset

args = parse_args_function()

"""# Load Dataset"""

root = args.input_file

#mean = np.array([120.46480086, 107.89070987, 103.00262132])
#std = np.array([5.9113948 , 5.22646725, 5.47829601])

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])

if args.train:
    trainset = Dataset(root=root, load_set='train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    
    print('Train files loaded')

if args.val:
    valset = Dataset(root=root, load_set='val', transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    print('Validation files loaded')

if args.test:
    testset = Dataset(root=root, load_set='test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    print('Test files loaded')

"""# Model"""

use_cuda = False
if args.gpu:
    use_cuda = True

model = select_model(args.model_def)

if use_cuda and torch.cuda.is_available():
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=args.gpu_number)

"""# Load Snapshot"""

if args.pretrained_model != '':
    model.load_state_dict(torch.load(args.pretrained_model))
    losses = np.load(args.pretrained_model[:-4] + '-losses.npy').tolist()
    start = len(losses)
else:
    losses = []
    start = 0

"""# Optimizer"""

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
scheduler.last_epoch = start
lambda_1 = 0.01
lambda_2 = 1

"""# Train"""

if args.train:
    print('Begin training the network...')
    
    for epoch in range(start, args.num_iterations):  # loop over the dataset multiple times
    
        running_loss = 0.0
        train_loss = 0.0
        for i, tr_data in enumerate(trainloader):
            # get the inputs
            inputs, depths, labels2d, labels3d = tr_data
    
            # wrap them in Variable
            inputs = Variable(inputs)
            # depths = Variable(depths)
            labels2d = Variable(labels2d)
            labels3d = Variable(labels3d)
            
            if use_cuda and torch.cuda.is_available():
                inputs = inputs.float().cuda(device=args.gpu_number[0])
                depths = depths.float().cuda(device=args.gpu_number[0])
                labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                labels3d = labels3d.float().cuda(device=args.gpu_number[0])
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            if args.model_def.lower() == "hopenet":
                outputs2d_init, outputs2d, outputs3d = model(inputs, depths)
                loss2d_init = criterion(outputs2d_init, labels2d)
                loss2d = criterion(outputs2d, labels2d)
                loss3d = criterion(outputs3d, labels3d)
                loss = (lambda_1)*loss2d_init + (lambda_1)*loss2d + (lambda_2)*loss3d
            else:
                outputs3d = model(inputs, depths)
                loss = criterion(outputs3d[0], labels3d)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.data
            train_loss += loss.data
            if (i+1) % args.log_batch == 0:    # print every log_iter mini-batches
                print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / args.log_batch))
                running_loss = 0.0
                # print(loss2d_init, loss2d, loss3d)
                
        if args.val and (epoch+1) % args.val_epoch == 0:
            val_loss = 0.0
            for v, val_data in enumerate(valloader):
                # get the inputs
                inputs, depths, labels2d, labels3d = val_data
                
                # wrap them in Variable
                inputs = Variable(inputs)
                # depths = Variable(depths)
                labels2d = Variable(labels2d)
                labels3d = Variable(labels3d)
        
                if use_cuda and torch.cuda.is_available():
                    inputs = inputs.float().cuda(device=args.gpu_number[0])
                    depths = depths.float().cuda(device=args.gpu_number[0])
                    labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                    labels3d = labels3d.float().cuda(device=args.gpu_number[0])
                if args.model_def.lower() == "hopenet":
                    outputs2d_init, outputs2d, outputs3d = model(inputs, depths)    
                    loss2d_init = criterion(outputs2d_init, labels2d)
                    loss2d = criterion(outputs2d, labels2d)
                    loss3d = criterion(outputs3d, labels3d)
                    loss = (lambda_1)*loss2d_init + (lambda_1)*loss2d + (lambda_2)*loss3d
                else:
                    outputs3d = model(inputs, depths)
                    loss = criterion(outputs3d[0], labels3d)
                val_loss += loss.data
            print('val error: %.5f' % (val_loss / (v+1)))
        losses.append((train_loss / (i+1)).cpu().numpy())
        
        if (epoch+1) % args.snapshot_epoch == 0:
            torch.save(model.state_dict(), args.output_file+str(epoch+1)+'.pkl')
            np.save(args.output_file+str(epoch+1)+'-losses.npy', np.array(losses))

        # Decay Learning Rate
        scheduler.step()
    
    print('Finished Training')

"""# Test"""

import numpy as np

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def compute_mpjpe(estimated_poses, ground_truth_poses):
    """
    Compute the Mean Per Joint Position Error (MPJPE) between estimated and ground truth poses.
    
    Parameters:
        estimated_poses (list of arrays): List of estimated poses, where each pose is represented as an array of joint coordinates.
        ground_truth_poses (list of arrays): List of ground truth poses, where each pose is represented as an array of joint coordinates.
    
    Returns:
        float: The MPJPE value.
    """
    num_poses = len(estimated_poses)
    num_joints = len(estimated_poses[0])  # Assuming the same number of joints for all poses
    
    total_error = 0.0
    
    for i in range(num_poses):
        pose_error = 0.0
        for j in range(num_joints):
            pose_error += euclidean_distance(estimated_poses[i][j], ground_truth_poses[i][j])
        total_error += pose_error / num_joints
    
    mpjpe = total_error / num_poses
    
    return mpjpe


if args.test:
    print('Begin testing the network...')
    
    running_loss = 0.0
    hand_mpjme, object_mpjme, mpjme = 0.0, 0.0, 0.0
    img_names = []
    for i, ts_data in enumerate(testloader):
        # get the inputs
        inputs, depths, labels2d, labels3d = ts_data
        
        # wrap them in Variable
        inputs = Variable(inputs)
        # depths = Variable(depths)
        labels2d = Variable(labels2d)
        labels3d = Variable(labels3d)

        if use_cuda and torch.cuda.is_available():
            inputs = inputs.float().cuda(device=args.gpu_number[0])
            depths = depths.float().cuda(device=args.gpu_number[0])
            labels2d = labels2d.float().cuda(device=args.gpu_number[0])
            labels3d = labels3d.float().cuda(device=args.gpu_number[0])

        # outputs = model(inputs)
        if args.model_def.lower() == "hopenet":
            outputs2d_init, outputs2d, outputs3d = model(inputs, depths)
        else:
            outputs3d, _ = model(inputs, depths)
        
        # outputs3d = outputs[0]
        loss = criterion(outputs3d, labels3d)
        hand_mpjme += compute_mpjpe(outputs3d[:,:21,:].detach().cpu().numpy(), 
                                    labels3d[:,:21,:].detach().cpu().numpy())
        object_mpjme += compute_mpjpe(outputs3d[:,21:,:].detach().cpu().numpy(), 
                                      labels3d[:,21:,:].detach().cpu().numpy())
        mpjme += compute_mpjpe(outputs3d[:,:,:].detach().cpu().numpy(), 
                               labels3d[:,:,:].detach().cpu().numpy())
        if i == 0:
            predicted_joints = outputs3d[:,:,:].detach().cpu().numpy()
            print(predicted_joints)
        else:
            predicted_joints = np.concatenate((predicted_joints, outputs3d[:,:,:].detach().cpu().numpy()), axis=0)
        running_loss += loss.data
        # img_names.append(img_name)
    np.save("predicted_3D_joints.npy", predicted_joints)
    # np.save("img_names.npy", np.array(img_names)) 
    print('test error: %.5f' % (running_loss / (i+1)))
    print('test MPJME object: %.5f' % (object_mpjme / (i+1) * 1000) )
    print('test MPJME hand: %.5f' % (hand_mpjme / (i+1)* 1000))
    print('test MPJME: %.5f' % (mpjme / (i+1) * 1000) )
