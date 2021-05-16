# Temporal-Action-Localization

This repository is for my master thesis which was about Temporal Action Localization (TAL) task. In this paper, we propose a new network based on Gated Recurrent Unit (GRU) and two novel post-processing ideas for TAL task. Specifically, we propose a new design for the output layer of the GRU resulting in the so called GRU-Splitted model. Moreover, linear interpolation is used to generate the action proposals with precise start and end times. Finally, to rank the generated proposals appropriately, we use a Learn to Rank (LTR) approach. We evaluated the performance of the proposed method on Thumos14 dataset. Results show the superiority of the performance of the proposed method compared to state-of-the-art. Especially in the mean Average Precision (mAP) metric at Intersection over Union (IoU) 0.7, we get 27.52% which is 5.82% better than that of state-of-the-art methods.

![image of the method]

