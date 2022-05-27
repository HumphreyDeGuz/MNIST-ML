---
title: "CS559 Machine Learning Fundamentals & Applications"
author: "Humphrey De Guzman"
date: "12/9/2021"
indent: true
header-includes:
  - \usepackage{indentfirst}
  - \usepackage{setspace}\onehalfspacing
output:
  pdf_document:
    toc: true
---

\newpage

# Task 1

The handwritten number recognition problem is a very commonly studied case within the field of machine learning and specifically computer vision. The MNIST data set is the most notable data set for such a problem and contains roughly 70,000 handwritten digits split up into a 60,000 and 10,000 train test set.

![Visualization of MNIST digit]("mnist_example.png"){width=30%}

The data points themselves are typically read as a vector of length 784, which is sometimes translated into a 28x28 array, both containing values of 0-255 which represent a gray scale pixel value. Today we look to solve this problem through several models including: Logistic Regression, SVM, Random Forest, and CNN.

## Logisitc Regression

As the name suggests, logistic regression is a regressive model that utilizes the fitting of a linear relationship amongst your data and remapping the association through a function for classification. Because this model is typically used to solve the binary classification problem or two-class problem, the sigmoid function is often used as it is monotonically increasing smooth function which most importantly maps values to a range of (0,1). For our case, we will use an extension of the model called multinomial logistic regression as we have ten classes to identify. This functions similarly to classic logistic regression except many smaller models are created for each class and weighted to classify all outcomes. I choose this model as it is fairly basic and represents some sort of "*benchmark*" for the other models. The biggest fall back of logistic regression is that it is mostly focused on linearity and can fail to predict well when relationships are more complex. Because the data is large and multi-class, we use `SAGA` as our solver.

![Multinomial Logistic Regression ]("multinomial.png"){width=35%}

## SVM

SVM stands for Support Vector Machine and similar to logistic regression is used for classification. The goal of SVM is to build a hyperplane that resides in the same space as our data, tuning this hyperplane such that the distance between the classes is "maximized". This isn't necessarily always true to account for parameters such as noise, but the goal is to cut our space in an equidistant way from decision boundaries that would identify a class. It is also important to note that while this method linear in nature, the use of kernel functions allow for non-linearity via dot product substitution. Again, (*like logistic regression*), this technique is commonly used for the two class problem but can be extended to numerous classes through several the collection of several models. For our case we opt to use the radial basis function as a kernel to help the model determine multiple classes in a similar fashion to our multinomial logistic regression. I choose SVM as it is a step up from our previous model. Unlike logistic regression, we can capture non-linear relationships through the use of a kernel, though there is more computation done within this model as a trade off.

![Nonlinear Kernels for SVM]("kernel.png"){width=35%}

## Random Forest

Random forest is a type of ensemble model that utilizes the consensus of several decision trees in order to classify an object. A decision tree is a type of model whose goal is to split up the data through discrete criterion until a set of hyper-parameters are met. Splits are based off of some criterion function that estimates the amount of information gained for a split. Decision trees are useful as they can be easily translated into intuitive steps for a better understanding of the classification. Random forest is an extension of this method, building multiple independent trees that are often small. This allows for better efficiency in run time along with a reduction in overfitting compared to a single large tree. Unlike the previous two models, random forest are used naturally in multi-class problems. For our model we utilize trees that stop when all of our decisions no longer give us information (no split). Random forest is the most complicated out of all the models and is fairly robust ensemble method. Though, for its robustness we create this black box nature that is not seen in a singular decision tree, which is much easier to interpret to the real world.

![Random Forest Diagram]("rngForest.png"){width=25%}

## Results

Below is a table of the results for each algorithm.  All the models took in data with the shape (60000, 784), or 60,000 vectors of length 784. As we can see, the logistic regression performs the worst out of all the models in terms of accuracy. In comparison, we see that the SVC and Random Forest perform similarly in accuracy, though the Random Forest runs much faster in exchange for a small accuracy decrease. Overall we can see better performance as we increase in the "quality" of our models, which is to be expected, with Random Forest being the most efficient in terms of accuracy for run time.

```{r echo=FALSE, message=FALSE, warning=FALSE, results='asis'}
library(knitr)
Accuracy <- c(.9255,.9792,.9702)
Run_time <- c('3m 9s','4m 35s','32.3s')
df <- data.frame(Accuracy,Run_time)
rownames(df) <- c('Logisitc Regression','SVC','Random Forest')
knitr::kable(df)
```



# Task 2

In this section we aim to build a CNN or convolutional neural network to solve our handwritten digits classification problem. Because we are free to choose and tweak the hyper-parameters as we see fit, I chose to recreate and slightly modify an already existing architecture. After reading several of the papers presented within the given **[website](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)**. In the end I chose to follow LeNet's architecture with slight modifications.

## LeNet

LeNet also known as LeNet-5 is a deep learning model first developed in 1989 to solve the very same problem we face now. The model was a pioneer for its time and is one of the earliest CNNs created. The initial model works via a 32x32 image input, which then gets put through a convolution layer of kernel 5 and depth 6. This then gets pooled down by 2. Afterwards we repeat another convolution with kernel 5 and depth 16. We pool one last time by a factor of two again and flatten the data into a vector of 120 dimensions. This then gets densely connected to another layer of length 84 and then to 10, which we then finds the classification. After every convolution and dense layer an activation function is used, where the original paper uses the hyperbolic tangent function except with the last, which uses the softmax function.

![Diagram of LeNet-5 Architecture]("lenet.png")

## Keras Implementation

For my recreation of the model, I stay faithful to most of the parameters, with an exception to the activation functions, where we change from the hyperbolic tangent function to the infamous 'RelU' function. The ReLU function stands for rectified linear unit and is represented by the expression $f(x) = max(0,l(x))$ where $l(x)$ is a linear function. The advantage of this in its speed compared to say the sigmoid function or the original tangent function. Leaky ReLU can also be used if one is afraid of vanishing gradients, which is just a modified version where 0 is replaced by another linear function with a near flat slope. The code below is its representation using the Keras framework. Note we do not need to pad our numbers because the MNIST data set uses original dimensions of 28x28, (technically it is flattened as a vector over an array but I reshape the data using numpy).

```{python eval=FALSE, include=TRUE}
LeNet = keras.models.Sequential([
    keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu', padding='same'),
    keras.layers.AvgPool2D(pool_size=2, strides=2),
    keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu'),
    keras.layers.AvgPool2D(pool_size=2, strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(120,activation='relu'),
    keras.layers.Dense(84, activation='relu'),
    keras.layers.Dense(10,activation='softmax')])
```


## Results

Below shows the output of the console when training the model. We achieve an accuracy of ~98.86% and our loss via Spare Cross Entropy is ~3.56%. This is a sharp increase in performance. I also provide the model weights seen inside the directory `/checkpoints/my_checkpoint`.

\small
`Epoch 1/7`
`1875/1875 [==============================] - 30s 16ms/step - loss: 0.2189 - accuracy: 0.9337 - val_loss: 0.0732 - val_accuracy: 0.9772`

`Epoch 2/7`
`1875/1875 [==============================] - 30s 16ms/step - loss: 0.0704 - accuracy: 0.9780 - val_loss: 0.0570 - val_accuracy: 0.9797`

`Epoch 3/7`
`1875/1875 [==============================] - 31s 17ms/step - loss: 0.0500 - accuracy: 0.9840 - val_loss: 0.0414 - val_accuracy: 0.9860`

`Epoch 4/7`
`1875/1875 [==============================] - 31s 16ms/step - loss: 0.0402 - accuracy: 0.9880 - val_loss: 0.0390 - val_accuracy: 0.9869`

`Epoch 5/7`
`1875/1875 [==============================] - 31s 16ms/step - loss: 0.0337 - accuracy: 0.9889 - val_loss: 0.0385 - val_accuracy: 0.9886`

`Epoch 6/7`
`1875/1875 [==============================] - 30s 16ms/step - loss: 0.0268 - accuracy: 0.9919 - val_loss: 0.0294 - val_accuracy: 0.9905`

`Epoch 7/7`
`1875/1875 [==============================] - 30s 16ms/step - loss: 0.0229 - accuracy: 0.9928 - val_loss: 0.0356 - val_accuracy: 0.9886`

\normalsize

\newpage

# Task 3

Now that we've trained models, let us see how it performs on real handwritten digits! We first will generate 50 numbers by hand writing the digits 0-9 in 5 different styles. This data then gets converted into the usable MNIST format in which we can then run our models on for our predictions.

## Handwritten Digits

Shown below is an image of all the 50 unique digits I create to classify the model. Notice the difference in style choice, where we have a control style, which is my regular handwriting, a serif style, a digital style, an italics style, and a swirly style.

![Handwritten Digits]("example.png"){width=50%}

## Pre-processing

Before I can predict these numbers, they must be first made readable. I follow a similar methodology that MNIST uses with the exception of using an estimation for centering the digits rather than a center of mass calculation, along with a scale estimation. Below is the code I use for this transformation. I use the library cv2 to edit the images.

```{python eval=FALSE, include=TRUE}
def to_MNIST(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    correction = np.log(0.5*255)/np.log(np.mean(gray))
    gamma = np.power(gray, correction).clip(0,255).astype(np.uint8)
    contrast = cv2.convertScaleAbs(gamma, alpha=2.5)
    invert = cv2.bitwise_not(contrast)
    resized = cv2.resize(invert,(28,28))
    return resized
```

\newpage

Below is a representation of how an original digit looks, before it is transformed into usable data. I also include all of the post transformation data show afterwards.

![]("og0.png"){width=50%}
![]("digit0.png"){width=50%}

### Regular Style

![]("conversions/MNIST_reg0.png") ![]("conversions/MNIST_reg1.png") ![]("conversions/MNIST_reg2.png") ![]("conversions/MNIST_reg3.png") ![]("conversions/MNIST_reg4.png") ![]("conversions/MNIST_reg5.png") ![]("conversions/MNIST_reg6.png") ![]("conversions/MNIST_reg7.png") ![]("conversions/MNIST_reg8.png") ![]("conversions/MNIST_reg9.png")

### Serif Style

![]("conversions/MNIST_alt0.png") ![]("conversions/MNIST_alt1.png") ![]("conversions/MNIST_alt2.png") ![]("conversions/MNIST_alt3.png") ![]("conversions/MNIST_alt4.png") ![]("conversions/MNIST_alt5.png") ![]("conversions/MNIST_alt6.png") ![]("conversions/MNIST_alt7.png") ![]("conversions/MNIST_alt8.png") ![]("conversions/MNIST_alt9.png")

### Digital Style

![]("conversions/MNIST_sqr0.png") ![]("conversions/MNIST_sqr1.png") ![]("conversions/MNIST_sqr2.png") ![]("conversions/MNIST_sqr3.png") ![]("conversions/MNIST_sqr4.png") ![]("conversions/MNIST_sqr5.png") ![]("conversions/MNIST_sqr6.png") ![]("conversions/MNIST_sqr7.png") ![]("conversions/MNIST_sqr8.png") ![]("conversions/MNIST_sqr9.png")

### Italics Style

![]("conversions/MNIST_itl0.png") ![]("conversions/MNIST_itl1.png") ![]("conversions/MNIST_itl2.png") ![]("conversions/MNIST_itl3.png") ![]("conversions/MNIST_itl4.png") ![]("conversions/MNIST_itl5.png") ![]("conversions/MNIST_itl6.png") ![]("conversions/MNIST_itl7.png") ![]("conversions/MNIST_itl8.png") ![]("conversions/MNIST_itl9.png")

### Swirly Style

![]("conversions/MNIST_swl0.png") ![]("conversions/MNIST_swl1.png") ![]("conversions/MNIST_swl2.png") ![]("conversions/MNIST_swl3.png") ![]("conversions/MNIST_swl4.png") ![]("conversions/MNIST_swl5.png") ![]("conversions/MNIST_swl6.png") ![]("conversions/MNIST_swl7.png") ![]("conversions/MNIST_swl8.png") ![]("conversions/MNIST_swl9.png")

\newpage

## Performance

The models performed poorly as I imagined, with the following overall performance in accuracy:

  - Logistic Regression: 0.32
  - SVM Classifier: 0.40
  - Random Forest: 0.48
  - LeNet: 0.70
  
Additionally I check the accuracy of each individual style and is shown below:

```{r echo=FALSE, message=FALSE, warning=FALSE, results='asis'}
LeNet <- c(.7,.7,.7,.8,.6)
Log_Reg <- c(.4,.2,.5,.3,.2)
SVC <- c(.2,.4,.5,.4,.5)
Rng_For <- c(.3,.5,.5,.5,.6)
Total <- c(.4,.45,.55,.5,.475)
df2 <- data.frame(LeNet,Log_Reg,SVC,Rng_For,Total)
rownames(df2) <- c('Regular','Serif','Digital','Italics','Swirly')
knitr::kable(df2)
```

## Discussion

It is no surprise that the models perform worse on real data. I suspect this is mostly because of pre-processing errors rather than the actual model itself but I think highlights an important issue. Models themselves are very sensitive to the data that it is trained on, and when inputting data, ensure that the new data follows the same conjectures as the trained set. My hand written data was probably not properly normalized and centered and resulted in very skewed results, especially for each of the classes. We can see that the odd fonts actually end up getting classified better, with the digital style being best predicted over an average of all the models. That being said, LeNet predicts most of the fonts fairly similarly, with italics being best predicted and swirl being the lowest. I suspected my regular font to be best predicted but after looking at the images again, we can see that some of the numbers show fading and that may also contribute to such a lower accuracy.
