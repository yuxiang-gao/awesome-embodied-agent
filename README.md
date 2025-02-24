# 🤖 Awesome-Embodied-Agent

---

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This is more of a personal collection of papers, datasets, and benchmarks related to embodied agents. The goal is to keep track of the latest research in the field and to have a quick reference to the most relevant papers. I mainly focus on works that (IMO) have the ingredients for building a generalist embodied agent (with a focus on humanoid robots).

- [🤖 Awesome-Embodied-Agent](#-awesome-embodied-agent)
  - [📃 Papers](#-papers)
    - [🌏 World Model](#-world-model)
      - [🌈 Diffusion](#-diffusion)
      - [➡️ Auto-regressive](#️-auto-regressive)
      - [🦾 Self-supervised](#-self-supervised)
      - [Others](#others)
    - [🗺️ Generation-conditioned](#️-generation-conditioned)
    - [🤖 VLA](#-vla)
    - [🐒 Imitation Learning](#-imitation-learning)
      - [🌈 Diffusion-based](#-diffusion-based)
    - [🎪 Reinforcement Learning](#-reinforcement-learning)
    - [🧙 Survey](#-survey)
  - [💽 Datasets](#-datasets)
    - [🧠 Real2sim](#-real2sim)
  - [🏋️ Benchmarks](#️-benchmarks)
  - [🧠 Thoughts](#-thoughts)

## 📃 Papers

### 🌏 World Model

#### 🌈 Diffusion

- **Towards Physical Understanding in Video Generation: A 3D Point Regularization Approach**
Yunuo Chen1,2*, Junli Cao1,2, Anil Kag2, Vidit Goel2, Sergei Korolev2, Chenfanfu Jiang1, Sergey Tulyakov2, Jian Ren2
1University of California, Los Angeles, 2Snap Inc., 2025
[[paper](https://arxiv.org/abs/2502.03639)][[project](https://snap-research.github.io/PointVidGen/)][[code](https://github.com/snap-research/PointVidGen)]

  > - a novel video generation framework that integrates 3-dimensional geometry and dynamic awareness
  > - PointVid dataset -> latent diffusion model -> track 2D objects with 3D Cartesian coordinates
  > - cross attention between video and point in corresponding channels for a better alignment between the two modalities.
  > - applying a **misalignment penalty** to the video diffusion process

> [!NOTE]
>
> This could be useful for our data since hey are naturally annotated with ee-pose and finger-tips can be calculated from joint positions.

#### ➡️ Auto-regressive

- **Learning Robotic Video Dynamics with Heterogeneous Masked Autoregression**
Lirui Wang, Kevin Zhao*, Chaoqi Liu*, Xinlei Chen, MIT, UIUC, FAIR Meta, 2025
[[paper](https://arxiv.org/pdf/2502.04296)][[project](https://liruiw.github.io/hma/)][[code](https://github.com/liruiw/HMA)]

  > HMA is a real-time robotic video simulation for high-fidelity and controllable interactions, leveraging the general masked autoregressive dynamic models and heterogeneous training.
  >
  > - an iteration nof [HPT](https://github.com/liruiw/HPT)
  > - pretrained video dynamic models from hetereogeneous data over 40 datasets and 3 million trajectories from real robot teleops, human videos, simulation
  > - token **concatenation and modulaton** for action conditioned masked autoregressive video and action generation

> [!NOTE]
>
> [Lirui's](https://liruiw.github.io/) new work, which uses a cross-attention stem and head architecture similar to HPT, bu this one focuses on action-conditioned generation.

#### 🦾 Self-supervised

- **Intuitive physics understanding emerges from self-supervised pretraining on natural videos**
Quentin Garrido, Nicolas Ballas, Mahmoud Assran, Adrien Bardes, Laurent Najman, Michael Rabbat, Emmanuel Dupoux, Yann LeCun. 2025
[[paper](https://arxiv.org/abs/2502.11831)][[code](https://github.com/facebookresearch/jepa-intuitive-physics)]

  > V-JEPA (Video Joint Embedding Predictive Architecture) is a non-auto-regressive model that takes a self-supervised learning approach. It learns by predicting missing or masked parts of a video in an abstract representation space.
  > Trained on a mixture of three popular video datasets, referred to as VideoMix2M (Bardes et al., 2024): Kinetics 710 (K710 (Kay et al.,2017)), Something-Something-v2 (SSv2 (Goyal et al., 2017b)) and HowTo100M (HowTo (Miech et al., 2019)).
  >
  > Eval Datasets:
  >
  > - IntPhys
  > - GRASP
  > - InfLevel-lab

> [!NOTE]  
>
> This paper also focuses on physical understanding, but it uses a self-supervised learning approach. It is interesting to see how the model can be used for embodied agents.

- **DINO-Foresight: Looking into the Future with DINO**
Efstathios Karypidis, Ioannis Kakogeorgiou, Spyros Gidaris, Nikos Komodakis, 2024
[[paper](https://arxiv.org/abs/2412.11673)][[code](https://github.com/Sta8is/DINO-Foresight)]

  > - masked feature transformer in a self-supervised manner to predict the evolution of VFM features over time
  > - training on masked  dinov2 features, and predict on latent space

- **DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning**
Gaoyue Zhou, Hengkai Pan, Yann LeCun and Lerrel Pinto, New York University, Meta AI, 2024
[[Paper]](https://arxiv.org/abs/2411.04983) [[Code]](https://github.com/gaoyuezhou/dino_wm) [[Data]](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28) [[Project Website]](https://dino-wm.github.io/)

#### Others

- **Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis**
Kensuke Nakamura, Lasse Peters, Andrea Bajcsy, 2025
[[paper](https://arxiv.org/abs/2502.00935)]

> - Latent Safety Filters: a latent-space generalization of HJ reachability that tractably operates directly on raw observation data (e.g., RGB images) by performing safety analysis in the latent embedding space of a generative world model.
> - Prevent unsafe states and generate actions that prevent future failures in latent space.

### 🗺️ Generation-conditioned

- **Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations**
Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, Jianyu Chen, 2024
[[paper](https://arxiv.org/abs/2412.14803)][[project](https://video-prediction-policy.github.io/)]

- **Strengthening Generative Robot Policies through Predictive World Modeling**
Han Qi, Haocheng Yin, Yilun Du, Heng Yang, School of Engineering and Applied Sciences, Harvard University,  2025
[[paper](https://arxiv.org/abs/2502.00622)][[project](https://computationalrobotics.seas.harvard.edu/GPC/)][code coming soon]

### 🤖 VLA

- **HELIX: A VISION-LANGUAGE-ACTION MODEL FOR GENERALIST HUMANOID CONTROL**
[[blog](https://www.figure.ai/news/helix)]

  > 7B VLM at 7-9HZ, 80M Transformer at 200Hz(?)
  > full upbody
  > Dataset:
  >
  > - multi-robot, multi-operator dataset of diverse teleoperated behaviors, ~500 hours in total.
  > - an auto-labeling VLM to generate hindsight instructions, prompted with: "What instruction would you have given the robot to get the action seen in this video?"

> [!NOTE]
>
> Similar Work:
>
> - [HiRT](https://arxiv.org/pdf/2410.05273)
> - [Latent code as bridge](https://arxiv.org/pdf/2405.04798)
> - [RoboDual](https://arxiv.org/pdf/2410.08001)

- **π0: A Vision-Language-Action Flow Model for General Robot Control**
  Physical Intelligence
  [[paper](https://www.physicalintelligence.company/download/pi0.pdf)][[blog](https://physicalintelligence.company/blog/pi0)]

- **UP-VLA: A Unified Understanding and Prediction Model for Embodied Agent**
  Jianke Zhang, Yanjiang Guo, Yucheng Hu, Xiaoyu Chen, Xiang Zhu, Jianyu Chen, 2025
  [[paper](https://arxiv.org/abs/2501.18867)]

> training with both multi-modal Understanding and future Prediction objectives

- **DexVLA: Vision-Language Model with Plug-In  Diffusion Expert for General Robot Control**
  Junjie Wen, Yichen Zhu, Jinming Li, Zhibin Tang, Chaomin Shen, Feifei Feng, 2025
  [[paper](https://arxiv.org/abs/2502.05855)][[project](https://dex-vla.github.io/)][[code](https://github.com/juruobenruo/DexVLA)]

  > Multi-head billion-param diffusion action expert for cross-embodiment control
  > qwen2-vl-2b+ [scale-dp](#scale-dp)-1b
  >
  > Curriculum learning:
  >
  > 1. cross-embodiment pre-training stage
  > 2. embodiment-specific alignment
  > 3. task-specific adaptation

- **Towards Synergistic, Generalized, and Efficient Dual-System for Robotic Manipulation (RoboDual)**
  Qingwen Bu, Hongyang Li, Li Chen, Jisong Cai, Jia Zeng, Heming Cui, Maoqing Yao, Yu Qiao, 2024
  [[paper](https://arxiv.org/pdf/2410.08001)]

  > Generalist & specialist
  > Generalist:
  >
  > - Prismatic-7B (similar to OpenVLA) (siglip+dinov2 [see here](https://github.com/TRI-ML/prismatic-vlms))
  >
  > Specialist:
  >
  > - DiT <- action+ViT through perceiver resampler +latent

- **HiRT: Enhancing Robotic Control with Hierarchical Robot Transformers**
  Jianke Zhang, Yanjiang Guo, Xiaoyu Chen, Yen-Jen Wang, Yucheng Hu, Chengming Shi, Jianyu Chen, 2024
  [[paper](https://arxiv.org/pdf/2410.05273)]

  > - VLMs running at low frequencies to capture temporarily invariant features while enabling real-time interaction through a high-frequency vision-based policy guided by the slowly updated features
  > - Trained on 20 tasks from Metaworld, 5 tasks from Franka-Kitchen, and 4 skills from the real
  world

- **From LLMs to Actions: Latent Codes as Bridges in Hierarchical Robot Control**
  Yide Shentu, Philipp Wu, Aravind Rajeswaran, Pieter Abbeel, 2024
  [[paper](https://arxiv.org/pdf/2405.04798)]

  > “User: can you help me $x_{txt} ? Assistant: yes, \<ACT\>.”
  >
  > Data: 400 trajectories for each reasoning task and 1200 trajectories for each long horizon task.
  >
  > staged training strategies

### 🐒 Imitation Learning

#### 🌈 Diffusion-based

- **Scaling Diffusion Policy in Transformer to 1 Billion Parameters for Robotic Manipulation ( ScaleDP )**<a name="scale-dp"></a>
  Minjie Zhu, Yichen Zhu, Jinming Li, Junjie Wen, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, Jian Tang, 2024
  [[paper](https://arxiv.org/abs/2409.14411)]

  > - DPT suffers from large gradient issues, making the optimization of Diffusion Policy unstable
  >   - factorize the
  feature embedding of observation into multiple affine layers, and integrate it into the transformer blocks
  >   - non-causal attention which allows the policy network to “see” future actions during prediction
  >
  > Obs:
  >
  > - for DP-T increasing size does not improve performance
  > - num head 4->6 improves performance, but 8 does not
  > - a consistent decline in performance with each additional layer

  > Key Modifications:
  >
  > - AdaLN block instead of cross-attention block
  > - Non-causal Attention: remove the causal mask in the self-attention layer

> [!NOTE]
>
> RDT uses DiT with:
>
> - ~~RMSNorm~~ QKNorm:  avoid numerical instability
> - ~~linear~~ MLP Decoder
> - Alternating Condition Injection: for modality imbalance

- **The Ingredients for Robotic Diffusion Transformers ( DiT policy )**
Sudeep Dasari, Oier Mees, Sebastian Zhao, Mohan Kumar Srirama, Sergey Levine, 2024
[[paper](https://arxiv.org/pdf/2410.10088)][[code](https://github.com/sudeepdasari/dit-policy)]

  > DiT policy
  >
  > - FiLM + resnet + sinusoidal fourier features
  > - adaLN-Zero attention: This simple trick improves performance by 30\%+ on long horizon, dexterous, real-world manipulation tasks containing over 1000 decisions!
  > - self-attention encoder + diffusion decoder with adaLN-Zero

- [Decoder Only Transformer Policy - simple but powerful model for behavioral cloning](https://github.com/IliaLarchenko/dot_policy)

> [!NOTE]
> very interesting idea about simplifying IL models (14M parameters with 2-3M trainable). It is worth to try it on our data.
> However, the reported results are only on simulated envs.

> [!IMPORTANT]
> Overall, I think a minimalist IL model that we can run in realtime on CPU is needed. Coupled with VLM for high-level reasoning, this could be a good stand-in for the full VLA model.

### 🎪 Reinforcement Learning

- **Learning to Manipulate Anywhere: A Visual Genralizable Framework For Visual Reinforcement Learning**
Zhecheng Yuan, Tianming Wei, Shuiqi Cheng, Gu Zhang, Yuanpei Chen, Huazhe Xu, 2024
[[paper](https://arxiv.org/abs/2407.15815)][[project](https://gemcollector.github.io/maniwhere/)][[code](https://github.com/gemcollector/maniwhere)]

  > Use multi-view representation objective to help sim-to-real transfer
  
### 🧙 Survey

- **Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI**, arXiv:2407.06886, 2024
Yang Liu, Weixing Chen, Yongjie Bai, Guanbin Li, Wen Gao, Liang Lin.
[[Paper](https://arxiv.org/pdf/2407.06886)]

## 💽 Datasets

- **Open-X Embodiment**
[[overview](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit?gid=0#gid=0)]

- **BridgeData V2: A Dataset for Robot Learning at Scale**
Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe Hansen-Estruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, Sergey Levine, 2024
[[paper](https://arxiv.org/abs/2308.12952)][[project](https://rail-berkeley.github.io/bridgedata/)]

  > - 60,096 trajectories
  > - 50,365 teleoperated demonstrations
  > - 9,731 rollouts from a scripted pick-and-place policy
  > - 24 environments grouped into 4 categories
  > - 13 skills
  >
  > The majority of the data comes from 7 distinct toy kitchens, which include some combination of sinks, stoves, and microwaves. The remaining environments come from diverse sources, including various tabletops, standalone toy sinks, a toy laundry machine, and more.
  >

- **DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset**, 2024
[[paper](https://arxiv.org/abs/2403.12945)][[project](https://droid-dataset.github.io/)]

### 🧠 Real2sim

- **Re3Sim: Generating High-Fidelity Simulation Data via 3D-Photorealistic Real-to-Sim for Robotic Manipulation**
Xiaoshen Han, Minghuan Liu, Yilun Chen, Junqiu Yu, Xiaoyang Lyu, Yang Tian, Bolun Wang, Weinan Zhang, Jiangmiao Pang, 2025
[[paper](https://arxiv.org/abs/2502.08645)][[project](https://xshenhan.github.io/Re3Sim/)]

-- **Evaluating Real-World Robot Manipulation Policies in Simulation (SimplerEnv)**
[[paper](https://arxiv.org/pdf/2405.05941)][[project](https://simpler-env.github.io/)]

## 🏋️ Benchmarks

- **All Robots in One: A New Standard and Unified Dataset for Versatile, General-Purpose Embodied Agents**, arXiv:2408.10899, 2024
Zhiqiang Wang, Hao Zheng, Yunshuang Nie, Wenjun Xu, Qingwei Wang, Hua Ye, Zhe Li, Kaidong Zhang, Xuewen Cheng, Wanxi Dong, Chang Cai, Liang Lin, Feng Zheng, Xiaodan Liang
[[Paper](https://arxiv.org/pdf/2408.10899)][[Project](https://imaei.github.io/project_pages/ario/)]

- **CALVIN - A benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks**, arXiv:2112.03227, 2022
Oier Mees, Lukas Hermann, Erick Rosete, Wolfram Burgard
[[paper](https://arxiv.org/pdf/2112.03227)][[project](https://github.com/mees/calvin)]

## 🧠 Thoughts
