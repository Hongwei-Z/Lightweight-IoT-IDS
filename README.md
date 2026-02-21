## Lightweight IoT Intrusion Detection via Feature and Sample Reduction with Multi-Client Ensemble for Zero-Day Attacks

### Abstract:
The rapid proliferation of the Internet of Things (IoT) exposes heterogeneous and resource-constrained devices to an increasingly diverse and evolving range of network attacks. Conventional centralized intrusion detection systems (IDS) and complex machine learning models often experience high computational and communication costs while lacking the capability to detect unknown threats. This paper proposes a lightweight distributed IDS framework that integrates multi-stage data compression, including feature selection, dimensionality reduction, and clustering-based sample compression, with lightweight classifiers and ensemble-based decision fusion. The framework is evaluated under two scenarios: (i) an extreme seven-client setting, where each client is trained on a single attack type, and (ii) a realistic five-client setting with heterogeneous attack distributions. Experimental results demonstrate that the proposed approach significantly reduces training cost and energy consumption while maintaining high detection performance. In the seven-client case, recall remains above 90% even with only 10% of samples retained under extreme heterogeneity, whereas in the five-client case, recall exceeds 97% under the same compression level. On the Raspberry Pi platform, total energy consumption decreases from 0.002015 kWh (without compression) to 0.000145 kWh, corresponding to a 92.8% reduction, while the prediction throughput reaches 7,117 to 173,005 samples per second across the five clients, ensuring real-time applicability. Ensemble strategies further improve robustness and enable effective detection of rare and zero-day attacks. These findings highlight the framework’s ability to balance performance, power consumption, and efficiency, making it well-suited for practical lightweight IoT deployments.


### File Structure:

1. **Dataset:**
  - [`Dataset/Preprocessing.ipynb`](Dataset/Preprocessing.ipynb): Data preprocessing workflow.

2. **Baseline:**
  - [`Baseline`](Baseline): Use [`LightGBM`](Baseline/Base_LightGBM.ipynb), [`MLP`](Baseline/Base_MLP.ipynb), and [`Random Forest`](Baseline/Base_RF.ipynb) to conduct experiments on the processed dataset as baseline models, and perform model hyperparameter tuning.
  
3. **GlobalConfig:**
  - [`GlobalConfiguration.py`](GlobalConfig/GlobalConfiguration.py): Apply feature and sample compression methods to generate the global configuration models.

4. **Demo:**
  - [`Demo.ipynb`](Demo/Demo.ipynb): A simple demonstration of model training results after feature and sample compression of the training data.
  - [`Simulation_5_clients.py`](Demo/Simulation_5_clients.py), [`Simulation_7_clients.py`](Demo/Simulation_7_clients.py), [`Simulation_Zero-day.py`](Demo/Simulation_Zero-day.py): Demonstrations of client models trained under different scenarios (five clients, seven clients, and zero-day attack scenarios with complete exclusion of a specific attack type).
  - [`Ensemble.py`](Demo/Ensemble.py): Perform ensemble learning on the client models.

5. **Log:**
  - [`Log`](Log): Code execution logs.



### How to Cite

If you use this project, please cite our paper:

IEEE Citation:      
```text

```

BibTeX: 
```text

```


Author: Hongwei Zhang

Updated on 2026-02-21

EOF