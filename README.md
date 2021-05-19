![model3](https://user-images.githubusercontent.com/62461020/118403487-c78fe780-b683-11eb-9d1a-4f1fb0f6aac1.png)
# Temporal-Action-Localization

In this paper, we propose a new network based on Gated Recurrent Unit (GRU) and two novel post-processing ideas for Temporal Action Localization (TAL) task. Specifically, we propose a new design for the output layer of the GRU resulting in the so called GRU-Splitted model. Moreover, linear interpolation is used to generate the action proposals with precise start and end times. Finally, to rank the generated proposals appropriately, we use a Learn to Rank (LTR) approach. We evaluated the performance of the proposed method on Thumos14 dataset. Results show the superiority of the performance of the proposed method compared to state-of-the-art. Especially in the mean Average Precision (mAP) metric at Intersection over Union (IoU) 0.7, we get 27.52% which is 5.82% better than that of state-of-the-art methods.

**Data Preparation**

We have used the prepared I3D features of the Thumos14 dataset from the [RecapNet](https://github.com/tianwangbuaa/RecapNet). Please download the features and put them in the *data* folder.

**Training & Evaluating The Probability Prediction Model**
To train the network run the Model.py and your trained model will be save in data folder. After training you can generate proposals by using the Eval_Gen.py and finally to get the AR@AN and R@100-tIoU, run the Eval_Metric.py.



In progress...
