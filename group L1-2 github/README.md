# group L1/2 regularization for filter pruning of convolutional neural networks

In this project, we have used a group L1/2 regularization term (GL1/2)
to prune the redundant filters in CNNs.

## regularization Train
Install PyTorch(>= 1.1.0) first  
```bash
example: python main.py \
--dataset=mnist\
 --network=lenet\
 --penalty=0.5\
 --reg_param=2.e-3
```
the reg_param λ for GL1/2 and GL

|Network  |GL1/2 |GL |
| ------   |:---: | :---:    |
|LeNet    |2.e-03  |5.e-03 |
|ResNet20   |2.e-04 |5.e-04 |
|VGG16    |7.e-06  |7.e-06|
|ResNet50  |3.e-05   |4.e-05 |
