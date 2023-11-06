# FRCSyn competition submission

This repository contains the code to create a submission for the FRCSyn competition, hosted at WACV 2024.

## Directory structure

For the script to run, you need the dataset folder laid out this way:

```
/path/to/dataset/root/
|-- comparison_files/
|   |-- sub-tasks_1.1_1.2/
|   +-- sub-tasks_2.1_2.2/
+-- Real/
|   |-- AgeDB-processed/
|   |-- BUPT-BalancedFace-processed/
|   |-- CASIA-WebFace/
|   |-- CFP-FP-processed/
|   +-- ROF-processed/
+-- Synth/
    |-- DCFace/
    +-- GANDiffFace/
```

## Dependencies

Before running the script, ensure you have the dependencies in the `requirements.txt` file installed:

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

## Running the experiments

There are three experiments: real faces only, synthetic faces only, and a mixture of the two.

To train the network each experiment, run the following command:

```
python main.py fit -c experiments/train.yml -c experiments/<name>.yml --data.datasets_root <root_to_datasets>
```

where `<name>` is one of `real`, `synth`, or `mixed`, and `<root_to_datasets>` is the path to the dataset root folder, which must be laid out as described above.

### Using multiple GPUs

To use multiple GPUs, append the following to the end of the command line:

```
--trainer.devices <number_of_gpus> --trainer.strategy ddp
```

For instance, to train the `real` experiment on 4 GPUs, run the following command:

```
python main.py fit -c experiments/train.yml -c experiments/real.yml --data.datasets_root <root_to_datasets> --trainer.devices 4 --trainer.strategy ddp
```

> [!WARNING]
> Do not evaluate the model's performance using multiple GPUs, as it can lead to incorrect results. Instead, use a single GPU for evaluation.