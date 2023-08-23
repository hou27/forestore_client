# SW 경진대회 출품작(Forestore)

<p align="center">
 <br>
 <div width="400" style="background: none;" align="center">
  <img src='https://github.com/hou27/problem_solving/assets/65845941/590c5474-900e-42e0-be91-164e2f2e5b4b' alt="Guess me Logo" width="700" />
 </div>
</p>
<p align="center"><a href="https://www.youtube.com/watch?v=97iRtm13fLo">Demo Video</a>
</p>

## 👋 Project Overview

**Forestore**는 **재고 관리의 어려움**으로 인해

- 매진되어 소비자가 필요할 때 물품이 없거나
- 많은 물품이 남아 폐기해야하는 경우

를 효과적으로 해결하기 위해 만들어진 **재고량 예측**을 통해 **재고 관리를 효율적으로 할 수 있는 서비스**입니다.

## 📖 Table of Contents

<ol>
 <li><a href="#features">Features</a></li>
 <li><a href="#flowchart">Flow Chart</a></li>
 <li><a href="#expectation">Expectation</a></li>
 <li><a href="#competitiveness">Competitiveness</a></li>
 <li><a href="#gettingstarted">Getting Started</a></li>
</ol>

<h2 id="features"> ✨ Key Features </h2>

- 예상 재고 소진 시기 확인(그래프 또는 수치로 확인 가능)
- 원하는 품목 검색
- etc...

<h2 id="flowchart"> 🛠️  Flow Chart </h2>

<img width="500" alt="Guessme_project_architecture" src="https://github.com/hou27/mock_mart_data/assets/65845941/888f0978-2048-461c-8f6f-ff12a20cd75d">

> **Forestore**는 **재고량 예측**을 통해 **재고 관리를 효율적으로 할 수 있는 서비스**입니다.  
> 주기적으로 모델이 실시간 데이터를 기반으로 예측하며, 사용자는 모델이 업데이트 해둔 '예측된 재고량'을 확인할 수 있습니다.

<h2 id="expectation"> ✨ Expectations </h2>
 
 ### 👍 서비스 부분
- 마트 점주 매진되어 판매를 못하거나 폐기할 품목이 줄어들어 매출 및 마트 운영 만족도가 증가하며, 
소비자 역시 매진으로 인한 소비를 못하는 문제를 해결할 수 있어 만족도가 증가하게 됩니다.
- 소, 도매에 모두 도입하여 자동 발주 기능까지 확장 가능하며 마트 운영, 즉 무인 운영 및 효과적인 재고 관리가 가능합니다.
- 재고량 데이터 뿐만 아니라 다양한 시계열 데이터에 쉽게 적용할 수 있어 확장성이 매우 뛰어납니다.  
 ### 🤝 모델 부분
- LSTM에 강화학습을 더한 모델이기 때문에, LSTM이 학습한 패턴을 바탕으로 강화학습을 통해 예측하게 됩니다. 
따라서, 변동적인 현실의 데이터에 보다 높은 성능을 보일 수 있습니다.
- RL-LSTM의 구조는 복잡한 시간적 패턴에서 최적의 행동을 학습할 수 있어 다양한 실세계 시나리오에서의 활용이 가능합니다.
- 환경에 적응하는 SAC 알고리즘을 통해 다른 시계열 데이터로 학습한 모델로 다른 데이터 세트를 예측할때 높은 성능을 가져 접목성에서 우수합니다.

### 🚉 Client

- python 3.8.10
- streamlit

### 😄 ML(재고량 예측)

- python 3.8.10
- tensorflow
- torch

<h2 id="gettingstarted"> 🏃 Getting Started </h2>

아래 링크에서 서비스를 만나보실 수 있습니다.  
https://myongjimart.streamlit.app/
