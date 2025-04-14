# Introduction

This repository presents an analytical study examining the interrelationships among ten parameters influencing the implementation of knowledge management within Bank X. The dataset, provided in `bank.csv`, serves as the foundation for this analysis. The study employs a hybrid methodological approach, integrating the Analytic Hierarchy Process (AHP) and Artificial Neural Networks (ANN) to evaluate and model the factors contributing to successful knowledge management practices.

---

# Execution Instructions

To replicate the analysis and execute the code, please follow the steps outlined below:

1. **Repository Acquisition**: Download the repository by cloning it using Git or by downloading the ZIP archive directly from the repository's webpage.

2. **Python Environment**: Ensure that Python version 3.12.1 is installed on your system, as this version was utilized during the development and testing phases. Utilizing the same Python version is recommended to maintain consistency and avoid potential compatibility issues.

3. **Dependency Installation**: Install the required Python libraries by executing the following command in your terminal or command prompt:

   ```bash
   pip install -r requirements.txt
   ```

4. **Program Execution**: Run the primary analysis script by executing:

   ```bash
   python caha.py
   ```

This will initiate the computational processes involving AHP and ANN methodologies, culminating in the analysis of the knowledge management implementation factors within Bank X.

You should get output like this

```
.....

Epoch 496/500
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.9585 - loss: 0.4813 - val_accuracy: 1.0000 - val_loss: 0.4800
Epoch 497/500
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.9835 - loss: 0.4825 - val_accuracy: 1.0000 - val_loss: 0.4800
Epoch 498/500
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.9585 - loss: 0.4810 - val_accuracy: 1.0000 - val_loss: 0.4800
Epoch 499/500
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - accuracy: 0.9585 - loss: 0.4807 - val_accuracy: 1.0000 - val_loss: 0.4800
Epoch 500/500
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - accuracy: 0.9890 - loss: 0.4820 - val_accuracy: 1.0000 - val_loss: 0.4800
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step
[0.83, 0.81, 0.72, 0.77, 0.45, 0.61, 0.19, 0.74, 0.23, 0.77]
[0.9   0.875 0.75  0.833 0.333 0.25  0.125 0.75  0.167 0.857]
```