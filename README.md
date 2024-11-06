# Quantification Over Time

This repository provides codes of experiments in our paper 

*Quantification Over Time*

Feiyu Li, Hassan H. Gharakheili, and Gustavo Batista

## Experiment

The codes are implemented in python 3.9. The whole environment of this project is listed in `requirements.txt` and `requirements.yaml`. Result tables in the paper can be reproduced by simply running the main script:

```bash
python3 run_experiment.py --run {experiment}
```

in which `--run textual` means experiments on Twitter sentiment datasets, `--run numeral` means experiments on non-textual datasets, `--run sota_qot` means comparisons with proposed approach and state-of-the-art conducted approaches.

The experimental setup details are described in the paper.
