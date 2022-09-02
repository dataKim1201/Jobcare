# Getting Start
---
** 본 모델은 dacon 경진대회에 TAVE라는 팀명으로 참가했을시 개발한 모델입니다. 
![잡케어](잡케어.png)

본문은 모든 부분을 포괄하고 있지 않고 단지 ipynb파일을 github에 배포용으로 모델을 개발해보기 위해 서비스화 해보는 작업을 거치는 중입니다.

대회 당시 저희는 코랩환경에서 모델개발을 진행했습니다.

* 잡케어 데이터셋과 같은 범주형 변수에 적합한 모델인 catboost 모델을 사용했습니다.
google colab에서 작성하였고, GPU를 이용해 학습을 진행했습니다.
optuna 과정과 eli5 라이브러리의 permutation feature importance 기반의 데이터 전처리를 사용함으로써 LB 점수를 많이 끌어올릴 수 있었습니다. 
public LB 38위 / 0.70425, Private LB 68위/0.70342 로 대회를 마무리하였습니다.
감사합니다:)


- 해당 파일을 ipynb로 저장해 배포했고 이를 local환경에서 바로 실행할 수 있는 서비스화 해서 진행했습니다.
(** colab에서도 실행 가능합니다.)

## 1. What about

- 저희가 진행한 작업은 크게 4가지로 보실 수 있습니다.
1. 전처리: 데이콘에서 제공받은 데이터를 학습용 데이터로 변환하는 작업입니다.
2. Eli5: 변수 중요도를 파악하기 위한 라이브러리로 id를 포함한 설명이 부족한 작업을 제거하는 과정을 거쳤습니다.
3. CatboostClassifier: 부스팅 방법을 활용한 분류 모델로 
4. Optuna: 하이퍼파라미터 튜닝

## 2. How you Setup

- Eli5 등 Optuna는 이미 작업이 완료가 되었고 해당 내용은 ipyb 파일에서 확인하시면 됩니다.

The code in this repo is an CatboostCalsifier example of the template.  
Try `python train.py` to run code.

Before you Try you must be in env like that
- python: 3.7.12 (default, Sep 10 2021, 00:21:48) 
[GCC 7.5.0]
- pandas: 1.1.5
- numpy: 1.19.5
- sklearn: 1.0.2
- catboost: 1.0.4

how to install catboost
`pip install catboost`

### 사전정의 file format
Eli5와 Optuna를 통해서 얻은 사전 하이퍼파라미터 및 제거 컬럼
```javascript
cols_drop = ["id","person_prefer_f","person_prefer_g",
            "person_prefer_d_3_attribute_d_m_contents_attribute_d_attribute_d_m"]  
best_params = {"iterations": 1422,
            "objective": "CrossEntropy",
            "bootstrap_type": "Bayesian",
            "od_wait": 666,
            "learning_rate": 0.9782109291187356,
            "reg_lambda": 70.72533306533951,
            "random_strength": 47.81900485462368,
            "depth": 3,
            "min_data_in_leaf": 20,
            "leaf_estimation_iterations": 5,
            "one_hot_max_size": 1,
            "bagging_temperature": 0.07799233624102353,
            "eval_metric":"F1"}
```

### 모듈 참조하는 법
https://etloveguitar.tistory.com/139 
해당 블로그에서 사용자 정의 함수를 어떻게 import하는 지 구체적으로 설명이 잘되어 있었다. 해당 링크를 참조하면서 파일을 구성했다.

### Folder Structure
  ```
  JobCare-template/
  │
  ├── train.py - main script to start training
  │
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   └── model.py
  │  
  └── util/ - preprocess and Load Data
      └── util.py
  ```
