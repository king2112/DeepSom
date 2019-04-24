
# <center>DeePanc: A Python-basedFeature Construction Tool from Simple Somatic Mutational load of Protein-coding genes</center>

### Authors:  MohammadHossein RezaieKheirabadi,  Hamed Dashti, Hamid R. Rabiee, and Hamid Alinejad-Rokny
<!-- ### <center> United International University, Dhaka, Bangladesh  </center> -->

&nbsp;

## 1. Download Package
### 1.1. Direct Download
You can directly [download](https://github.com/king2112/DeePanc/archive/master.zip) by clicking the link.

**Note:** The package will download in zip format `(.zip)` named `DeePanc-master-2.zip`.

***`or,`***

### 1.2. Clone a GitHub Repository (Optional)

Cloning a repository syncs it to our local machine (Example for Linux-based OS). After clone, we can add and edit files and then push and pull updates.
- Clone over HTTPS: `user@machine:~$ git clone https://github.com/king2112/DeePanc.git `

- Clone over SSH: `user@machine:~$ git clone git@github.com:king2112/DeePanc.git `

**Note #1:** If the clone was successful, a new sub-directory appears on our local drive. This directory has the same name (PyFeat) as the `GitHub` repository that we cloned.

**Note #2:** We can run any Linux-based command from any valid location or path, but by default, a command generally runs from `/home/user/`.

**Note #2.1:** `user` is the name of our computer but your computer name can be different (Example: `/home/bioinformatics/`).

## 2. Installation Process
### 2.1. Required Python Packages
`Major (Generate Features):`
- Install: python (version >= 3.6)
- Install: numpy (version >= 1.13.0)
- Install: Tensorflow (version >= 2.2.4)
- Install: Keras (version >= 2.2.4)

`Minor (Performance Measures):`
- Install: sklearn (version >= 0.19.0)
- Install: pandas (version >= 0.21.0)
- Install: matplotlib (version >= 3.0.1)

### 2.2. How to download
`Using PIP3:`  pip3 install `<package name>`
```console
user@machine:~$ pip3 install scikit-learn
```
**`or,`**

`Using anaconda environment:` conda install `<package name>`

```console
user@machine:~$ conda install scikit-learn
```

## 3. Working Procedure

You can run commands in console or using the Jupyter Notebook instead.  

### 3.1. Importing  and Preparing Dataset
Use read_csv() for importing dataset

```console
$ pd.read_csv("Dataset directory")
```
Use np.shape() for checking the shape of the dataset.

```console
$ np.shape(data)
```
By using .as_matrix() we can change the type of dataset into matrix
```console
$ data.as_matrix()
```
#### 3.1.1. Scaling Dataset and Dividing to Training and Test Data

Dividing all of the elements by maximum number
```console
$ data / np.max(data)
```
By importing train_test_split from sklearn.model_selection we can divide our dataset into traing and test dataset.

```console
$ X_train, X_test,y_train,y_test = train_test_split(inputs,b,test_size= Test size Portion , random_state=Number of the random ceed)
```

Use np.reshape() in order to reshape your dataset into desire input shape

```console
$ data =np.reshape(data, [desire shape])
```

### 3.2. Designing Autoencoder Deep Neural Network

You first have to determine what is the shape of input

```console
$ input = Input(shape = (desire shape))
```
Using Dense layer for input layer

```console
$ Layer = Dense(Arguments)(input)
```

#### Table 1: Arguments Details for Dense Layer
&nbsp;

|   Argument     |    Type     |   Default | Help   |
|     :---       |   :---:       |  :---:    | ---:|
| units | Positive integer | --- | dimensionality of the output space. |
| activation | --- | If you don't specify anything, no activation is applied | In this work we utilized 'relu' for first 4 Dense layers |
| use_bias | Boolean | False | whether the layer uses a bias vector |
| kernel_initializer | --- |   | Initializer for the kernel weights matrix, we have used glorot_uniform |
| bias_initializer| --- |  | Initializer for the bias vector |
| kernel_regularizer | --- |   | Regularizer function applied to the kernel weights matrix |
| bias_regularizer | --- |  | Regularizer function applied to the bias vector |
| activity_regularizer | --- |  | Regularizer function applied to the output of the layer (its "activation") |
| kernel_constraint | --- |  | Constraint function applied to the kernel weights matrix |
| bias_constraint | --- |  | Constraint function applied to the bias vector  |


&nbsp;

You can use Batchnormalization() for biasing input dataset.

```console
$ layer = Batchnormalization()(layer1)
```

#### Table 2: Arguments Details for Batchnormalization Layer
&nbsp;

|   Argument     |    Type    | Help   |
|     :---       |   :---:    | ---:|
| axis | Integer |  the axis that should be normalized (typically the features axis). For instance, after a Conv2D layer with  data_format="channels_first", set axis=1 in BatchNormalization.  |
| momentum | --- |  Momentum for the moving mean and the moving variance.  |
| epsilon | --- |  Small float added to variance to avoid dividing by zero.  |
| center | False |   If True, add offset of beta to normalized tensor. If False, beta is ignored. |
| scale | False |   If True, multiply by gamma. If False, gamma is not used. When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling will be done by the next layer. |
| beta_initializer |   |   Initializer for the beta weight.|
| gamma_initializer |   |  Initializer for the gamma weight.|
| moving_mean_initializer |   |  Initializer for the moving mean.|
| moving_variance_initializer |   |  Initializer for the moving variance.|
| beta_regularizer |   |  Optional regularizer for the beta weight.|
| gamma_regularizer |   |  Optional regularizer for the gamma weight.|
| beta_constraint |   |  Optional constraint for the beta weight.|
| gamma_constraint |   |  Optional constraint for the gamma weight.|

&nbsp;



Creating object from the deep learning Network

```console
$ Model = Model(input, out)
```



### 3.3. Determining Optimizer and Compiling Method

You can use Adam optimizer by call it from Keras

```console
$ adam = keras.optimizers.Adam(lr= desired learning rate)
```
After that using Compile() for compiling the Autoencoder Model

```console
$ model.compile(optimizer= desired optimizer , loss='desired loss function'))
```
**Note #1:**  In the Liver Cancer Subtype Identification with deep learning approach, we have exploited Adam optimizer with lr=0.00005 and MSE for loss function with default parameters.

Use Summary() for seeing detail of the model

```console
$ model.summary()
```
For the aim of preventing over-fitting we have used Early Stoping monitoring.
You can use EarlyStopping by calling it from keras.callbacks

```console
$ earlyStopping =keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
```
**Note #2:**  You can modify the EarlyStopping with desire Arguments.
&nbsp;


#### Table 3: Arguments Details for the EarlyStopping
|   Argument     |    Type    | Help   |
|     :---       |   :---:    | ---:|
|    monitor      |   ---    | quantity to be monitored.|
|    min_delta      |   Positive double    | minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.|
|    patience      |   Positive Integer    |  number of epochs with no improvement after which training will be stopped.|
|    verbose      |    Integer    | verbosity mode.|
|    mode      |   auto, min, max    | In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.|
|    baseline      |   ---    | Baseline value for the monitored quantity to reach. Training will stop if the model doesn't show improvement over the baseline.|
|    restore_best_weights      |   ---    | quantity to be monitored.|
|    monitor      |   False    |  whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.|






#### 3.1.2. Evaluation Purpose


#### Table 1: Arguments Details for the Features Generation


#### Table 2: Feature Description


### 3.2. Run Machine Learning Classifiers (Optional)



&nbsp;

#### Table 3: Arguments Details for the Machine Learning Classifiers


&nbsp;
### 3.3. Training Model (Optional)



&nbsp;

#### Table 4: Arguments Details for Training Model


### 3.4. Evaluation Model (Optional)



&nbsp;

#### Table 5: Arguments Details for Evaluation Model


## References

**[1]** Bin Liu, Fule Liu, Longyun Fang, Xiaolong Wang, and Kuo-Chen Chou. repdna: a
python package to generate various modes of feature vectors for dna sequences by in-
corporating user-defined physicochemical properties and sequence-order effects. Bioin-
formatics, 31(8):1307–1309, 2014.


=======
MOHEMM
>>>>>>> c2529519364e67154520318d9ad259d25dfa2800
