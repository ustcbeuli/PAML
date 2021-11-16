# PAML
PyTorch implementation of the paper: "Preference-Adaptive Meta-Learning for Cold-Start Recommendation", IJCAI, 2021.

# Introduction
In recommender systems, the cold-start problem is a critical issue. To alleviate this problem, an emerging direction adopts meta-learning frameworks and achieves success. Most existing works aim to learn globally shared prior knowledge across all users so that it can be quickly adapted to a new user with sparse interactions. However, globally shared prior knowledge may be inadequate to discern users’ complicated behaviors and causes poor generalization. Therefore, we argue that prior knowledge should be locally shared by users with similar preferences who can be recognized by social relations. To this end, in this paper, we propose a Preference-Adaptive Meta-Learning approach (PAML) to improve existing meta-learning frameworks with better generalization capacity. Specifically, to address two challenges imposed by social relations, we first identify reliable implicit friends to strengthen a user’s social relations based on our defined palindrome paths. Then, a coarse-fine preference modeling method is proposed to leverage social relations and capture the preference. Afterwards, a novel preference-specific adapter is designed to adapt the globally shared prior knowledge to the preference-specific knowledge so that users who have similar tastes share similar knowledge. We conduct extensive experiments on two publicly available datasets. Experimental results validate the power of social relations and the effectiveness of PAML.

# Requirements
python 3.6+
pytorch 1.1+

# Run example
python data_generation.py
python main.py

# Citation
@inproceedings{wang2021preference,
  title={Preference-adaptive meta-learning for cold-start recommendation},
  author={Wang, Li and Jin, Binbin and Huang, Zhenya and Zhao, Hongke and Lian, Defu and Liu, Qi and Chen, Enhong},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  pages={1607--1614},
  year={2021}
}
