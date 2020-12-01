# Genetic Programming Operators into Artificial Machine Learning Losses

*This repository holds all the necessary code to run the very-same experiments described in the paper "Genetic Programming Operators into Artificial Machine Learning Losses".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

---

## Structure

 * `core`
   * `linker`
     * `gp.py`: Provides a customized Genetic Programming implementation that can use loss functions entities as terminals;
     * `node.py`: Provides a customized node structure that can use loss functions entities as terminals;
     * `space.py`: Provides a customized tree space that can use loss functions entities as terminals;
     * `terminal.py`: Provides customizable terminals that can be used to build loss functions;
   * `model.py`: Defines the base Machine Learning architecture;
 * `models`
   * `cnn.py`: Defines a ResNet18 architecture;
   * `mlp.py`: Defines the Multi-Layer Perceptron;
 * `outputs`: Folder that holds the output files, such as `.pkl` and `.txt`;
 * `utils`
   * `loader.py`: Utility to load datasets and split them into training, validation and testing sets;
   * `object.py`: Wraps objects for usage in command line;
   * `target.py`: Implements the objective functions to be optimized;
   * `wrapper.py`: Wraps the optimization task into a single method.
   
---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```Python
pip install -r requirements.txt
```

### Data configuration

If you wish to download the medical-based datasets, please contact us. Otherwise, you can use `torchvision` to load pre-implemented datasets.

---

## Usage

### Finding Optimized Losses

The first step is to find an optimized loss function using Genetic Programming, which is guided by the validation set accuracy. To accomplish such a step, one needs to use the following script:

```Python
python find_optimized_loss.py -h
```

*Note that `-h` invokes the script helper, which assists users in employing the appropriate parameters.*

### Evaluating Optimized Losses

After conducting the optimization task, one needs to evaluate the created loss function using training and testing sets. Please, use the following script to accomplish such a procedure:

```Python
python evaluate_optimized_loss.py -h
```

### Bash Script

Instead of invoking every script to conduct the experiments, it is also possible to use the provided shell script, as follows:

```Bash
./pipeline.sh
```

Such a script will conduct every step needed to accomplish the experimentation used throughout this paper. Furthermore, one can change any input argument that is defined in the script.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---
