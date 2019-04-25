
# <center>DeePanc: A Python-based Feature Construction Tool from Simple Somatic Mutational load of Protein-coding genes</center>

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

&nbsp;


### 3.4. Training and Fitting Model

Then use model.fit() for training the model

```console
$ autoencoder_train = model.fit(X_train,X_train,epochs=10000, batch_size=32, shuffle=True, validation_data=(X_test, X_test), callbacks=[earlyStopping])
```
**Note #2:**  You can modify the Fit() with desire Arguments.

&nbsp;


#### Table 4: Arguments Details for the EarlyStopping
|   Argument     |    Type    | Help   |
|     :---       |   :---:    | ---:|
|    x      |   Numpy array of training data (if the model has a single input)    | or list of Numpy arrays (if the model has multiple inputs). If input layers in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.  x can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).|
|    y      |    Numpy array of target (label) data (if the model has a single output)   |  or list of Numpy arrays (if the model has multiple outputs). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.  y can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).|
|    batch_size      |   Integer or None    |  Number of samples per gradient update. If unspecified, batch_size will default to 32.|
|    epochs      |     Integer, Number of epochs to train the model    |An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch,  epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.|
|    verbose      |   Integer (0, 1, or 2)  | Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
callbacks: List of keras.callbacks.Callback instances. List of callbacks to apply during training and validation (if ).|
|    validation_split      |   Float between 0 and 1    | Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling.|
|    validation_data      |   tuple (x_val, y_val) or tuple  (x_val, y_val, val_sample_weights) on which to evaluate the loss and any model metrics at the end of each epoch    | The model will not be trained on this data.  validation_data will override validation_split.|
|    shuffle      |   Boolean (whether to shuffle the training data before each epoch) or str (for 'batch')    | 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.|
|     class_weight       |   Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only)    |  This can be useful to tell the model to "pay more attention" to samples from an under-represented class.|
|     sample_weight       |   Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).    | You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape  (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().|
|     initial_epoch       |   Integer    | Epoch at which to start training (useful for resuming a previous training run).|
|     steps_per_epoch       |   Integer or None    | Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined. |
|     validation_steps       |   ---    |  Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping. |
|     validation_freq       |   Only relevant if validation data is provided. Integer or list/tuple/set    | If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a list, tuple, or set, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs. |



&nbsp;



#### 3.4.1. Plotting Losses ( Validation and training )

Use model.history() for getting the history of the training section
For Example:
``` console
$ loss = autoencoder_train.history['loss']
```
Then plot it with plt.figure()

```console
$ loss = autoencoder_train.history['loss']
$ val_loss = autoencoder_train.history['val_loss']
$ epochs = range(epochs)
$ plt.figure()
$ plt.plot(epochs, loss, 'bo', label='Training loss')
$ plt.plot(epochs, val_loss, 'b', label='Validation loss')
$ plt.title('Training and validation loss')
$ plt.legend()
$ plt.show()
```


### Extracting Features which has been Constructed

You then may use Model(inputs=model.input, outputs=model.get_layer('bottleneck').output)
**Note #3:**  You can change 'bottleneck' into your desire layer which you want to extract its values.

```console
$ m2 = Model(inputs=model.input, outputs=model.get_layer('bottleneck').output)
$ Y = m2.predict(patients)
$ Y =np.reshape(Y , (473, 3))
$ df = pd.DataFrame(Y)
$ df.to_csv("features.csv",index = False)
```





## References



**[1]** Bailey, Peter, David K Chang, Katia Nones, Amber L Johns, Ann-Marie Patch,
Marie-Claude Gingras, David K Miller, Angelika N Christ, Tim JC Bruxner, and Michael C Quinn.
2016. 'Genomic analyses identify molecular subtypes of pancreatic cancer', Nature, 531: 47.

**[2]** Cao, Dong-Sheng, Qing-Song Xu, and Yi-Zeng Liang. 2013. 'propy:
 a tool to generate various modes of Chou’s PseAAC', Bioinformatics, 29: 960-62.

**[3]** Chen, Quanjun, Xuan Song, Harutoshi Yamada, and Ryosuke Shibasaki. 2016.
 "Learning deep representation from big and heterogeneous data for traffic accident inference."
 In Thirtieth AAAI Conference on Artificial Intelligence.

**[4]** Chowdhury, Shahana Yasmin, Swakkhar Shatabda, and Abdollah Dehzangi. 2017.
 'iDNAprot-es: Identification of DNA-binding proteins using evolutionary and structural features',
  Scientific reports, 7: 14938.

**[5]** Miotto, Riccardo, Li Li, Brian A Kidd, and Joel T Dudley. 2016.
 'Deep patient: an unsupervised representation to predict the future of patients from the electronic health records',
  Scientific reports, 6: 26094.

**[6]** Tamimi, Rulla M, Graham A Colditz, Aditi Hazra, Heather J Baer, Susan E Hankinson, Bernard Rosner,
 Jonathan Marotti, James L Connolly, Stuart J Schnitt, and Laura C Collins. 2012.
  'Traditional breast cancer risk factors in relation to molecular subtypes of breast cancer',
   Breast cancer research and treatment, 131: 159-67.

**[7]** Yamashita, Taro, Marshonna Forgues, Wei Wang, Jin Woo Kim, Qinghai Ye,
 Huliang Jia, Anuradha Budhu, Krista A Zanetti, Yidong Chen, and Lun-Xiu Qin. 2008.
  'EpCAM and α-fetoprotein expression defines novel prognostic subtypes of hepatocellular carcinoma',
   Cancer research, 68: 1451-61.



=======

## This project is licensed under the MIT License - see the  [LICENSE.md](https://github.com/king2112/DeePanc/blob/master/LICENSE) file for details.
