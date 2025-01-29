# ProText project - pause prediction

This repository contains the Python scripts used for predicting pauses, as part of the [Pro-TEXT project](https://pro-text.huma-num.fr), a study which seeks to deepen our understanding of textualisation by analysing the temporal dynamics of writing bursts and pauses in French.

Participants' key strokes were logged and the typed text was divided into bursts based on a 1.5 second threshold.

It is these pauses that the scripts in this repository are aiming to predict. Multiple strategies have been using , all of which are enumerated below along with the scripts used. 

More background information on the study itself, as well as data pre-processing and related work can be found in the following publications at [LIFT 2 - 2024 Scientific Days](https://lift2-2024.sciencesconf.org):
- [Analysing the Dynamics of Textualisation through Writing Bursts and Pauses in French (long abstract)](https://lift2-2024.sciencesconf.org/data/cislaru_manseri_silai_taravella_abstract.pdf)
- [Analysing the Dynamics of Textualisation through Writing Bursts and Pauses in French (poster)](https://lift2-2024.sciencesconf.org/data/cislaru_manseri_silai_taravella_poster.pdf)

In terms of results and their interpretation, you can find more information in the presentations.

# Data processing

Before launching prediction algorithms, our data was pre-processed. Here is a summary of the adjustments that were made to the initial input, all implemented in the `scripts/pre-processing/complete_data_processing.py` script. 

- All textual data (textual bursts, the parts of speech, chunks etc.) was vectorised using Word2Vec
- Outliers in terms of pause duration were eliminated using IQR or a given interval
- The columns that contained a list of values were split into multiple columns, so there would be a single value per column (necessary for certain ML algorithms) and to manage the varying length of the input.

Please note that if you want to see the full data processing steps, from the keylogger to the csv file we take as input here, the scripts are found in [this repository](https://github.com/KehinaleK/Real-Time-Writing-Analysis) .

# Continous values prediction 

## Regression with Linear Regression, Decision Tree Regressor and Random Forest Regressor

The script that was used for predicting continous values was `scripts/predicting_cont_values/prediction_regressions.py`

This script tests 3 algorithms, and uses gridsearch to find the best hyperparameters - please note that if you relaunch the script it will take a significant time for it to run. 

The results for this were poor, with a negative R2 score in all attempts. 

## Regression using TensorFlow and Keras

First we tried a simple neural network (`prediction_tf_simple.py`), where we eliminiated a lot of the features, but tried padding the input to ensure it had the same length for all values. We recommend you run this script first to ensure tensorflow is correctly installed and running on your machine. The results were poor but it was a good base to build upon.

The `prediction_tf.py` script trains and evaluates a neural network model for predicting pause duration using TensorFlow and Keras. It begins by loading and preprocessing the data, including log-transforming and clipping the target variable, handling non-numeric columns, filling NaN values, and standardizing features. The script removes outliers from the training set based on z-scores and scales the target variable. A sequential neural network model with three hidden layers and L2 regularization is defined and compiled. The model is trained on the cleaned and scaled training set, and predictions are made on the test set. The script evaluates the model's performance using Mean Absolute Error (MAE) and R² score, and visualizes the results with plots of true vs. predicted values and training vs. validation loss.

## Regression using CamemBERT embeddings

As we obtained good results with the binary classification when using CamemBERT embeddings we tested a simple sequential model using these embeddings as input, as well as the following features: totalActions, totalChars, finalChars, totalDeletions, innerDeletions, docLen, avg_shift, and num_actions.

Results are explained in detail in the corresponding presentation linked at the end of the document, but overall they were poor. We did numerous adjustments on the data to help the model recognise the variance, but due to the weak correlation between the features given as input and the target variable the results remained poor. 

## Regression using LSTM layers and RNN

The scripts implementing this are `prediction_lstm_dense_1.py` and `prediction_lstm_dense_2.py` (Please note the scripts contain multiple architectures, some of which have been commented out. Should you wish to see those results as well uncomment the specific sections.). 

As our data is sequential in nature, we also did some tests using a combination of LSTM layers and dense layers. In this case we used padding and masking to ensure uniform lengths and we handled missing values, while also using multiple input features to build a comprehensive model. 

Due to the size of our data, we've had to use global pooling average. In addition to this we've had to adjust the LSTM layers, as they tend to focus on the final state of a sequence to make a prediction. By doing that in our case, we miss potentially important information, so we used an attention mechanism to take into account different parts of the input. We have also applied a logaritmic transformation to our data due to the high variance. 

Even with all these adjustments we were unable to run tests for more than 2 texts at a time, due to technical constraints. This approach has led to R² scores that remained negative, while closer to 0 than previous approaches (-0.08152925778596498). A google colab notebook that contains a final version of this approach is saved as `colab_prediction_lstm.ipynb` and is available on this repository. The link is [here](https://colab.research.google.com/drive/1AGWeFZzXFiEwK-PdooQi-svkkIQv-KYF?usp=sharing).

A full recurrent neural network script is in `prediction_rnn.py` - please note this script has not been tested fully as it would take multiple days to get results. 

# Classification

Due to limited results obtained from attempting a prediction of continuous values, we have turned to classification. Here as well we had multiple strategies, which we implemented in a variety of ways. 

## Classification into 5 / 10 categories

In order to create a number of categories we split the pauses into 10 and then 5 "bins" so that there would be an equal number of samples in each bin. The script for this approach is in `classification/classification.py`.

In terms of algorithm we used a random forest classifier, and we attempted predictions by taking as input the text that is found after the pause on the one hand, and the text that is found before the pause on the other. We found that we obtained better results with the first approach, though the accuracy was only 0.13 for a 10 category classification.

When using neural networks (a simple sequential model) for this classification the best accuracy was 0.14.

When reducing the nmber of categories to 5, we obtained a top accuracy of 0.25 for the sequential model and taking as input the text that came after the pause we're predicting. With gradient boosting, the best accuracy was 0.21.

As the 5 category classification led to strongly skewed results towards one of the categories, we have adjusted the elimination of outliers in the processing script. With this, the top accuracy was obtained with a random forest classifier : 0.24.

## Using Camembert for embedding the input text 

In the script `classification_5cat_bert.py` we implemented the 5 category pauses and CamemBERT embeddings for the text. The text given as input to the neural network is then in this format:

> cat_2 L'intention de l cat_5 'aéroport de biard  cat_5 de  cat_4 diminuer  cat_1 la poussé des gaz sur le décollage de ses avions  cat_1 au dessus  cat_5 des zones  cat_5 habité  cat_3 à ses avantage et ses inconvéniant

The input for this script has to be a text file from the `input/fichieravecpausecat` directory.

The first results were not very satisfactory, with low accuracy and no explicability. Moreover the transformer tokenizer did not align with "traditional" tokenising, which meant that it was hard to interpret our results. 

## Binary classification

Due to poor results when using CamemBERT and 5 category classification, we decided to try binary classification.

The input texts went from:

>"L'intention de l cat_5 'aéroport de biard cat_4 diminuer la poussé des gaz cat_1 sur le décollage de ses avions."

To:

>"L 0 'intention 0 de 0 l 1 'aéroport 0 de 0 biard 1 diminuer 0 la 0 poussé 0 des 0 gaz 1 sur 0 le 0 décollage 0 de 0 ses 0 avions."

Where 0 means no siginifcant pause and 1 means significant pause. 

However this meant that the two categories were very imbalanced (Pause count: 451, No pause count: 2748). As a result we implemented oversampling / undersampling techniques as well as slightly modifying the loss function so that pause predictions would be penalised more. 

Once these techniques were in place, we attempted to tune hyper-parameters to try and obtain the best result possible. The top result obtained in this context was 0.82 accuracy, with 20 epochs for training and a learning rate of 2e-5.

## Adding part of speech information to binary classification

Initially we attempted to automatically annotate parts of speech using Spacy. However, due to tokenisation misalignment between CamemBERT and Spacy, we couldn't concatenate the embeddings (see script `bert_binary_pos_tagging.py`).

We decided therefore that instead of concatenating POS embeddings and CamemBERT tokens, we could try directly replacing words with their corresponding POS (`bert_binary_pos_replacement.py`).

We obtained relatively good results, with a 0.90 accuracy for pause and 0.92 accuracy for no pause.

## Comparison with random pause distribution

To check the validity of results obtained previously, we randomised pauses in the texts and tested the binary classification using CamemBERT embeddings (`randomise_pauses.py`). Indeed the performance of the model decreased on randomised text, which suggests that the model performs better when tested on the original structure of pauses than on a random distribution. This in turn implies that the model has found patterns in pause placement.


## Using RoBERTa

We also used another transformer (RoBERTa) in the script `roberta_binary.py`. We obtained similar results overall, though RoBERTa did perform better in some tests.

## Using LLMs

We have used T5-small and turned our task into a sequence to sequence problem, which LLMs are very good at solving. The script is `llm_binary.py`.

We therefore went from an input like "The cat sat cat_1 on the mat cat_5"" to a binary sequence like "0 1 0 0 0 1"

The results were poor, as none of the pauses were correctly identified in the tests.

# Potential areas to explore further:

- Feature selection can help with identifying the most important variables that contribute to model performance. Methods such as statistical tests, model based selection or even recursive feature elimination could help improve performance, particularly when it comes to continous value predictions.
- Trying other transformes - only CamemBERT and RoBERTa were used, but as the texts are in French it would be interesting to see if mBERT could lead to better results.
- LLM results were poor, but only one approach was tested. Debugging of existing script might be needed.

# More detailed results

This work as produced as part of a 3 month internship, during which results for each approach mentioned here were presented and discussed (in French). The presentations are saved in PDF format in the `presentations` directory. Here is a short description for each of them.

1. Initial data processing. first attempts with regression: `presentations/1_data_processing_rnn.pdf`
2. Regression results with various architectures, 10 and then 5 category classification results with various algorithms: `presentations/2_regression_classification.pdf`
3. Adjusting outlier elimination to 1-6s pause interval and first CamemBERT tests: `presentations/3_outliers_camembert.pdf` 
4. Binary classification using CamemBERT and hyper parameter tuning: `presentations/4_binary_classification.pdf`
5. Part of speech integration and randomising pause distribution: `presentations/5_pos_integration.pdf`
6. Regression using CamemBERT embeddings, statistical data adjustments, RoBERTa and LLM testing: `presentations/6_roberta_llm.pdf`

Another publications was also written during this internship and submitted to the AMLA2024 conference in Miami - however the conference got pushed to another date. This along with the poster quoted before are both in the `publications` directory on this repository.