# Deep Learning Assignment #1
>Artur Khayaliev, BS4-DS

## 1. Docker + TF/Keras/PyTorch + MNIST = ‘A’
#### How to run:
1. Clone this repository.
    ```
    git clone https://github.com/zytfo/dl-assignment-1.git
    ```
2. Move to repository folder.
    ```
    cd dl-assignment-1
    ```
3. Build the container.
    ```
    docker build . -t mnist
    ```
4. Run `--fit` to train the model.
    ```
    docker run --mount type=bind,source="$(pwd)",target=/root/example/ mnist python app.py --fit
    ```
5. Run `--predict` to start prediction.
    ```
    docker run --mount type=bind,source="$(pwd)",target=/root/example/ mnist python app.py --predict model.pt
    ```
    
##### You can have several arguments for `--fit` option as well as for `--predict` option.
`--fit <args>`:
* `--file_name <string>` - name of the file where the model will be stored (default: model)
* `--batch-size <int>` - input batch size for training (default: 64)
* `--seed <int>` - random seed (default: 1)
* `--log-interval <int>` - how many batches to wait before logging training status
* `--momentum <float>` - SGD momentum (default: 0.5)
* `--lr <float>` - learning rate (default: 0.01)
* `--epochs <int>` - number of epochs to train (default: 1)

`--predict <path_to_model> <args>`:
* `--test-batch-size <int>` - input batch size for testing (default: 1000)
* `--epochs <int>` - number of epochs to train (default: 1)

# 2. TF/Keras/PyTorch comparison.
#### 1. Tensorflow
The most popular ML framework on github (~120k stars, ~72k forks). Google uses it in a variety of its products (Youtube, Chrome, Maps etc.). Since it is open source, many recognizable companies such as eBay, Intel, etc. uses Tensorflow.
Tensorfow originally was written in Python, but there are some "experimental" environments in Java, C++ and Go. Due to huge open-source community, Tensorflow also have wrappers for C# and others. There are various features, such as Tensorboard, which allows developers manage the training process with visualization. Another feature is serving. Serving allows to serve models at scale in a production environment. Tensorflow also supports device inference with low latency for mobile phones. 
##### Advantages:
* Open-source -> big community.
* A lot of documentation (official/unofficial) with examples.
* Relative cross-platform.
##### Disadvantages:
* It is difficult to write code without proper knowledge on how to do it.
* Due to the specificity of Tensorflow architecture, to make changes in the neural network, you have to rebuild the whole graph. It makes debug difficult.

#### 2. Keras
A small python-based library, that can work on top of Tensorflow. It supports many neural network types and makes prototyping very simple. This is the best tool for beginners. Keras is used in areas of translation, image/speech recognition and so on.

##### Advantages:
* Supports many neural network types.
* Prototyping is easy.
* Simple interface.
* Easy to learn and use.
* Supports multiple GPUs.

##### Disadvantages:
* Not as customizable as you may want.
* Less functionality comparing to another frameworks.

#### 3. PyTorch
PyTorch was created by Facebook to help its services and now its used by Salesforce and Twitter. PyTorch is defined by run mode, meaning for each iteration in an epoch a computational graph is created, so every iteration can be different and we have a dynamic graph.
##### Advantages:
* A dynamic graph which means we can change every single operation.
* Allows to use common debugging tools like PDB or PyCharm.
* Modelling process is simple.
* It has data parallelism, many pretrained models.
* Allows distributed training.
* Good for prototyping or small-scale projects.
##### Disadvantages:
* It lacks monitoring and visualisation tools.
* It lacks model serving.


# 3. Stackoverflow Answer.
[Very helpful instruction how to run your image using bind.](https://stackoverflow.com/a/54331409/10956750)
