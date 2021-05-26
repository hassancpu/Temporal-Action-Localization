![model3](https://user-images.githubusercontent.com/62461020/118403487-c78fe780-b683-11eb-9d1a-4f1fb0f6aac1.png)
# Temporal-Action-Localization

In this paper, we propose a new network based on Gated Recurrent Unit (GRU) and two novel post-processing ideas for Temporal Action Localization (TAL) task. Specifically, we propose a new design for the output layer of the GRU resulting in the so called GRU-Splitted model. Moreover, linear interpolation is used to generate the action proposals with precise start and end times. Finally, to rank the generated proposals appropriately, we use a Learn to Rank (LTR) approach. We evaluated the performance of the proposed method on Thumos14 dataset. Results show the superiority of the performance of the proposed method compared to state-of-the-art. Especially in the mean Average Precision (mAP) metric at Intersection over Union (IoU) 0.7, we get 27.52% which is 5.82% better than that of state-of-the-art methods.

**Data Preparation**

We have used the prepared I3D features of the Thumos14 dataset from the [RecapNet](https://github.com/tianwangbuaa/RecapNet). Please download the features and put them in the *data* folder.

**Training & Evaluating The Probability Prediction Model**

To train the network run the *Model* and your trained model will be saved in the *Trained_Model* folder. After training, you can generate proposals using the *Eval_Gen* and finally to get the AR@AN and R@100-tIoU, run the *Eval_Metric*. If you want to generate the proposals whose temporal boundaries are computed by Interpolation, run the *Eval_Gen_Interpolation*.  

**Ranking The Proposals By Ranking Modules**

For better ranking the generated proposals, train one of the ranking modules (*Model_prop*, *Model_prop_LTR*) and then evaluate it using the *Eval_Gen_Prop*. To train both models you need to extract features from the generated proposals and to do that, we use a similar approach to the [BSN](https://github.com/wzmsltw/BSN-boundary-sensitive-network) but with some modifications in the *prop_feat* and *prop_feat_all*. Finally, to evaluate the generated proposals with this approach, run the *Eval_Gen_Prop*.

**Qualitative Results**

In the below figures, you can see some of our generated proposals. For more information and results please refer to the paper.

![action_prop1](https://user-images.githubusercontent.com/62461020/119692934-15190b00-be61-11eb-91b6-5db092a9e1d0.jpg)
![action_prop2](https://user-images.githubusercontent.com/62461020/119692950-177b6500-be61-11eb-9b09-e74bea92f42e.jpg)


In progress...
