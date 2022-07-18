# Installation
This project uses gradient descent to find out the optimal latent space representation for 
convolutional autoencoders in image coding.

In order to start first create a dataset folder that can contain:
 1) A kodak folder containing the [kodak dataset](http://r0k.us/graphics/kodak/).
 2) A clic folder containing the [clic dataset](http://compression.cc/tasks/).

Then from the root folder run:
```bash
conda env create -f environment.yml 
conda activate FeatureOptimization
```

In order to install all the required dependencies

# Reproducing results
In order to train the networks on the kodak dataset for all QPs and for all models run:
```bash
cp configs/kodak_config.yaml src/config-defaults.yaml
cd src
wandb sweep paper_sweep.yaml
wandb agent {agent_id}
```

To run this you need to have a [weights and biases](https://wandb.ai/site) account (it is free for academic research and personal projects). the agent_id will be displayed when you run the wandb sweep command.

On top of that [VVC](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM) should be installed, and its repo should be in the same folder as the one of this project.

# Using the codec to encode images
In order to encode an image just change the config_defaults.yaml file with the parameters you 
prefer and add modify the input_path, compressed_path and output_path fields (the last two are required only if you want to save the actual result).
and run 
```bash
cd src
python codec.py
```
The compressed file size will be slightly bigger than the actual one since pickle is used to save the python object. Making an ad hoc binary file would probably reduce its size by some bytes but this was simpler and allows the user to obtain the object in the format required by compressai to decompress it. 

