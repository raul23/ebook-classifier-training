=====================================
Experimenting with classifying ebooks
=====================================
.. contents:: **Contents**
   :depth: 3
   :local:
   :backlinks: top

I am basing my experimentation with classifying text on the excellent scikit-learn's tutorial: `Classification of text documents using sparse features <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html>`_.

I am following along their tutorial but using my own datasets containing a bunch of ebooks (``pdf``, ``djvu``).

Introduction
============
I will be using different datasets of ebooks to test text classification. They will be of different size and categories. 

One of them is a small dataset that consists of 129 English ebooks (``pdf`` and ``djvu``) from 3 categories:

- ``computer_science`` with label 0 and 48 ebooks
- ``mathematics`` with label 1 and 50 ebooks
- ``physics`` with label 2 and 31 ebooks

It is the same dataset I `tested clustering <https://github.com/raul23/clustering-text#clustering-ebooks-pdf-djvu>`_ on.

By default, only 10% of a given ebook is `converted to text <#dataset-generation>`_ and added to the dataset. Also if an ebook is 
made of images, `OCR <#ocr>`_ is applied on 5 pages chosen randomly in the first 50% of the given ebook to extract the text.

Some stats about this small dataset:

.. code-block::

   Categories size: [48 50 31]
   129 documents - 3 categories
   
   77 documents - 5.03MB (training set)
   52 documents - 2.67MB (test set)

   vectorize training done in 0.861s at 5.837MB/s
   n_samples: 77, n_features: 5436
   vectorize testing done in 0.433s at 6.174MB/s
   n_samples: 52, n_features: 5436

|

Also as in the scikit-learn's `tutorial <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html>`_,
multiple clasification models will be used such as: ``RidgeClassifier``, ``LogisticRegression``, and ``ComplementNB``.

Results of classifying ebooks
=============================
Training a ``RidgeClassifier``
------------------------------
I put the results section at the top before explaining the `script <#script-classify-ebooks-py>`_ since it is the most important and interesting part
of this document.

Thus without further ado, here are the results from training a ``RidgeClassifier`` on the dataset of 129 documents with three categories (computer_science, mathematics, physics). 

The train and test sets are splitted as follows:

- train data: 77 ebooks (60%)
- test data: 52 ebooks (40%)

|

.. raw:: html

   <p align="center"><img src="./images/confusion_matrix_ridgeclass_small_dataset.png">
   </p>

`:information_source:` Insights from the confusion matrix for ``RidgeClassifier``

- The confusion matrix is plotted based on the predictions from the test set.
- Among the three categories, this linear model has the most "difficulties" with the physics category. It confused two physics ebooks for mathematics 
  documents which is to be expected since both domains share overlaps between words. The vice-versa situation is not found, i.e. no mathematics 
  documents were incorrectly classified as physics ones which could mean that books about physics use a more specific vocabulary than for mathematics 
  documents.
- Mathematics ebooks are well classified but one such document was classified as a computer science one. 
- The computer science category is the one that ``RidgeClassifier`` has the most success in classifying with all computer science ebooks being 
  correctly classified as such. 

|
 
.. raw:: html

   <p align="center"><img src="./images/average_feature_effect_small_dataset.png">
   </p>

.. code-block::

   top 5 keywords per class:
     computer_science mathematics   physics
   0       algorithms     riemann    energy
   1        algorithm    geometry   quantum
   2      programming        zeta  universe
   3            input       plane     light
   4          machine    theorems  particle

`:information_source:` Insights from the words with the highest average feature effects 

- This graph show words that are strongly positively correlated with one category and negatively associated 
  with the other two categories such as zeta (positive for mathematics) and universe (positive for physics).

  Those words constitute good predictive features.
- Computer science is a category that has lots of very good predictive features (e.g. programming and algorithm). No wonder that the     
  ``RidgeClassifier`` was able to correctly classify all ebooks from this category.
- When you see the word 'energy' among books from the three categories, you are almost sure that they will be about physics.
- Algorithm appears twice as good features, in the singular and plural forms. Need to do something about keeping only one
  form of a word (TODO).

Benchmarking classifiers
------------------------
+-----------------+--------------------------------------------+---------------------------------------------------+-------------------------+---------------------------+--------------------+-----------------------------------------+--------------------+---------------------------+
|                 | LogisticRegression(C=1000, max_iter=1000)  | RidgeClassifier(alpha=1e-06, solver='sparse_cg')  | KNeighborsClassifier()  | RandomForestClassifier()  | LinearSVC(C=1000)  | SGDClassifier(alpha=0.001, loss='log')  | NearestCentroid()  | ComplementNB(alpha=1000)  |
+=================+============================================+===================================================+=========================+===========================+====================+=========================================+====================+===========================+
| train time      | 0.134s                                     | 0.0447s                                           | 0.00106s                | 0.241s                    | 0.353s             | 0.00832s                                | 0.00339s           | 0.00229s                  |
+-----------------+--------------------------------------------+---------------------------------------------------+-------------------------+---------------------------+--------------------+-----------------------------------------+--------------------+---------------------------+
| test time       | 0.000615s                                  | 0.000933s                                         | 0.00966s                | 0.035s                    | 0.000555s          | 0.000608s                               | 0.000963s          | 0.000572s                 |
+-----------------+--------------------------------------------+---------------------------------------------------+-------------------------+---------------------------+--------------------+-----------------------------------------+--------------------+---------------------------+
| accuracy        | 0.942                                      | 0.962                                             | 0.962                   | 0.885                     | 0.962              | 0.942                                   | 0.923              | 0.942                     |
+-----------------+--------------------------------------------+---------------------------------------------------+-------------------------+---------------------------+--------------------+-----------------------------------------+--------------------+---------------------------+
| dimensionality  | 5436                                       | 5436                                              | -                       | -                         | 5436               | 5436                                    | -                  | 5436                      |
+-----------------+--------------------------------------------+---------------------------------------------------+-------------------------+---------------------------+--------------------+-----------------------------------------+--------------------+---------------------------+
| density         | 1.0                                        | 1.0                                               | -                       | -                         | 1.0                | 1.0                                     | -                  | 1.0                       |
+-----------------+--------------------------------------------+---------------------------------------------------+-------------------------+---------------------------+--------------------+-----------------------------------------+--------------------+---------------------------+

|

.. raw:: html

   <p align="center"><img src="./images/score_training_time_trade_off.png">
   </p>

|

.. raw:: html

   <p align="center"><img src="./images/score_test_time_trade_off.png">
   </p>

|

`:information_source:` Based on the trade-off between the test accuracy and the training/testing time, which model to choose?

- Complement naive Bayes is the model with the best trade-off between test score and training/testing time.
- KNN is the model with the best training time and test accuracy trade-off. However KNN is the second worst model in terms of testing time.

  I am kind of surprise that KNN has the best test accuracy considering that KNN is not expected to perform well with high-dimensional features
  like we find in text classification.
  
  From scikit-learn's `tutorial 
  <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#plot-accuracy-training-and-test-time-of-each-classifier>`_:
  
   Furthermore, the “curse of dimensionality” harms the ability of this model [KNN] to yield competitive accuracy in the 
   high dimensional feature space of text classification problems.
- Random Forest is the slowest model to train and make predictions and on top of that with the worst test score.

Script ``classify_ebooks.py``
=============================
Dependencies
------------
This is the environment on which the script `classify_ebooks.py <./scripts/classify_ebooks.py>`_ was tested:

* **Platform:** macOS
* **Python**: version **3.7**
* `matplotlib <https://matplotlib.org/>`_: **v3.5.2** for generating graphs
* `numpy <https://numpy.org/>`_: **v1.21.5**, for "array processing for numbers, strings, records, and objects"
* `pandas <https://pandas.pydata.org/>`_: **v1.3.5**, "High-performance, easy-to-use data structures and data analysis tool" 
* `pycld2 <https://github.com/aboSamoor/pycld2>`_: **v0.41**, for detecting the language of a given ebook in order to keep 
  books based on a chosen language
* `regex <https://pypi.org/project/regex/>`_: **v2022.7.9**, "this regex implementation is backwards-compatible with 
  the standard ``re`` module, but offers additional functionality"
* `scikit-learn <https://scikit-learn.org/>`_: **v1.0.2**, "a set of python modules for machine learning and data mining"

**Ref.:** https://docs.anaconda.com/anaconda/packages/py3.7_osx-64/

|

`:star:` **Other dependencies**

You also need recent versions of:

-  `poppler <https://poppler.freedesktop.org/>`_ (including ``pdftotext``) and `DjVuLibre <http://djvu.sourceforge.net/>`_ (including ``djvutxt``)
   can be installed for conversion of ``.pdf`` and ``.djvu`` files to ``.txt``, respectively.

Optionally:

- `diskcache <http://www.grantjenks.com/docs/diskcache/>`_: **v5.4.0** for caching persistently the converted files into ``txt``
- `Tesseract <https://github.com/tesseract-ocr/tesseract>`_ for running OCR on books - version 4 gives 
  better results. OCR is disabled by default since it is a slow resource-intensive process.

Script options
--------------
To display the script's list of options and their descriptions::

 $ python classify_ebooks.py -h
 usage: python classify_ebooks.py [OPTIONS] {input_directory}

I won't list all options (too many) but here some of the important and interesting ones:

-s, --seed SEED                        Seed for numpy's and Python's random generators. (default: 123456)
-u, --use-cache                        Highly recommended to use cache to speed up **dataset re-creation**.
-o, --ocr-enabled                      Whether to enable OCR for ``pdf``, ``djvu`` and image files. It is disabled by default. (default: false)
--ud, --update-dataset                 Update dataset with text from more new ebooks found in the directory.
--cat, --categories CATEGORY           Only include these categories in the dataset.  

|

`:information_source:` Explaining some important and interesting options/arguments

- ``input_directory`` is the path to the main directory containing the documents to classify.
- By **dataset re-creation** I mean the case when you delete the pickle dataset file and generate the dataset 
  again. If you are using cache, then the dataset generation should be quick since the text conversions were
  already computed and cached. Using the option ``-u`` is worthwhile especially if you used OCR for some of the ebooks since this procedure is very
  resource intensive and can take awhile if many pages are OCRed.
- The choices for ``-o, --ocr-enabled`` are ``{always, true, false}``
  
  - 'always': always use OCR first when doing text conversion. If the converson fails, then use the other simpler conversion tools
    (``pdftotext`` and ``djvutxt``).
  - 'true': first simpler conversion tools (``pdftotext`` and ``djvutxt``) will be used and then if a conversion method
    failed to convert an ebook to ``txt`` or resulted in an empty file, the OCR method will be used.
  - 'false': never use OCR, only use the other simpler conversion tools (``pdftotext`` and ``djvutxt``).
- The option ``--cat, --categories CATEGORY [CATEGORY ...]`` takes the following default values: 
  
  ``['computer_science', 'mathematics', 'physics']``

Caching
-------
TODO

Ebooks directory
----------------
TODO

Dataset generation
------------------
TODO

OCR
---
TODO

Filtering a dataset: select texts only in English and from valid categories
---------------------------------------------------------------------------
TODO
