# CIFAR10
![resnet_cifar10](https://i.imgur.com/ifynl8t.png)

resnet101 for cifar10, 
the final acc is around 0.930
# CIFAR100
![resnet_cifar100](https://i.imgur.com/MhOd9Yu.png)

resnet101 for cifar10, 
the final acc is around 0.778(but I use the wide resnet,k=4)
# ImageNet
1. Download ImageNet, The we have 1000 ***.tar file
2. unzip each tar file,you can use:
   from lib.datasets.data import read_tar
   read_tar()    # unzip
3. run 
the detail about ImageNet:(https://github.com/duanyzhi/ImageNet_Label)
-------------------------------------------------------------------------------
you mabey need (DeepLearning)[https://github.com/duanyzhi/deep_learning]

something about resnet:https://duanyzhi.github.io/ResNet/
# -------------------------------
# HOW TO RUN(terminal run)
run cifar10 for train/test:
    python main.py --pattern train --data cifar10
    python main.py --pattern test --data cifar10
    
run cifar100 for train/test:
    python main.py --pattern train --data cifar100
    python main.py --pattern test --data cifar100
    
run ImageNet for train/test:
    python main.py --pattern train --data ImageNet
    python main.py --pattern test --data ImageNet　
