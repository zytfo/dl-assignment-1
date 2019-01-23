# Deep Learning Assignment #1
>Artur Khayaliev, BS4-DS

## 1. Docker + TF/Keras/PyTorch + MNIST = ‘A’
#### How to run:
1. Clone this repository.
    ```
    git clone []
    ```
2. Move to repository folder.
    ```
    cd []
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