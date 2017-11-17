# AKBC-2017-Paper-14

Implementations of ER-MLP, HolE and ComplEx models, to reproduce results in "Revisiting Simple Neural Networks for Learning Representations of Knowledge Graphs".
Paper accepted in 6th Workshop on Automated Knowledge Base Construction at NIPS 2017.

Filenames and descriptions:

1. batch_gen.cpp : C++ code to quickly create batches (including bernoulli negative sampling) during training  
2. make.sh       : Makefile to compile batch_gen.cpp  
3. NN.py         : Python script to train the model. Uses tensorflow framework  
4. logs          : Folder that stores summary of training, and periodic embedding dumps. Point tensorboard to it  
5. validate.py   : Python script to generate scores of embeddings after training. Searches in logs/ for stored embeddings.  
6. Tester        : Contains files to calculate metrics (Hits@n, MRR, MR). Run Makefile and ./Test_NN after moving output of                        validate.py into this folder.
7. data          : Folder containing Knowledge Graph. 3 files expected in it : train.txt, valid.txt, test.txt. In all 3 files,                    each line should be a tab/space separated integer triple (eg:234 24 148) representing (headId, relationId,                          TailId).
8. Other models  : Contains equivalent of NN.py and validate.py for ComplEx and HolE.





