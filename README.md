# Curriculum-Enhanced-Supervised-Attention-Network-for-Person-Re-Identification
The operation of Curriculum Enhanced Supervised Attention Network for Person Re-Identification

## Dataset pre-processing
 To Run the code, run the following order in the folder curriculum_clastering:
 
 
* 1.Extract the feature: 

````
python extracter.py
````
The output is in tests/test-data/input/feat/with_pretrain_imagenet.mat

* 2.Turn the dataset into .txt files with labels:

````
python get_file_id.py
````
The output is in tests/test-data/input/dataset.txt

* 3.Divide the dataset according to the distribution density

````
python tests/reid.py
````

The output is preserved in .txt files in tests/test-data/output-reid/*.txt

* 4.Re-store the dataset images according to step 3:
````
python tests/test-data/output-reid/put_img.py
````

* 5. Reset the dataset into the original begining(for re-training):
````
python prepare.py
````

## Training
 The training examples are as follows:
 ````
python train.py --gpu_ids 1 --batchsize 64 --lr 0.1 --conv_lr 0.01 --stage1
python train.py --gpu_ids 1 --batchsize 32 --lr 0.001 --conv_lr 0.001 --stage2 -- w1 0.3 --w2 0.7
CUDA_VISIBLE_DEVICES=0,1 python PCB.py --batchsize 64 --lr 0.1 --conv_lr 0.01 -- ABN --curri
````
In Train.py the args are as follows:
````
--gpu_ids:which GPU
--name:which model
--dataset:dataset, default market1502
--stage1:first stage
--stage2:second stage
--ABN:with Attention Mechanism
--att_w:weight of Attention Mechanism
                  
 --all:train with all dataset (for ablation experiments)
 --resume:
 --resume_path:the resume pth
--lr:lrearning rate in backbone
--conv_lr:learning rate in the network added
--stepsize:change strategy of learning rate
--adapt:learning rate adapt
--w1,w2:ratio of the two subsets
--erasing_p:erase randomly
--epoch:epochs
--which_epoch:which epoch to test
--batchsize:batchsize
````
In PCB.py the args are as follows:
````
--d:dataset, default market 1501
--ABN:with Attention Mechanism
--curri:the third stage training
--justglobal:join in global part
--evaluate:test only
--re_rank:re-rank test
--lr:learning rate in backbone
--conv_lr:learning rate in the network added
--stepsize:change strategy of learning rate
--adapt:learning rate adapt
--s2:if the step in layer4 of resnet50 is 2
--random_erasing:erase randomly
--epoch:epochs
--which_epoch:which epoch to test
--batchsize:batchsize
--height, width:trainig height and width
--features:dimension of features
--dropout:dropout ratio
````

## Testing
 The testing examples are as follows:
 
 * first stage test
 
 ````
 python test.py --gpu_ids 1 --batchsize 64 --lr 0.1 --conv_lr 0.01 -- stage1
 ````

 * second stage test
 
 ````
 python test.py --gpu_ids 1 --batchsize 32 --lr 0.001 --conv_lr 0.001 -- stage2 --w1 0.3 --w2 0.7
 ````
 
 * Third stage test
 
 ````
 CUDA_VISIBLE_DEVICES=0,1 python PCB.py --batchsize 64 --lr 0.1 --conv_lr 0.01 --ABN --curri
 ````
 
 * The manuscript(with training and testing):
 ````
 bash con.sh
 ````
 


