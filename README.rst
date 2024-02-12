=======================================================
Part 1: Experimenting with Training an Ebook Classifier
=======================================================
.. contents:: **Contents**
   :depth: 3
   :local:
   :backlinks: top

Introduction
============
I am basing my experimentation with training a text classifier on the excellent scikit-learn's tutorial: `Classification of text documents using sparse features <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html>`_.

I am following along their tutorial but using my own `three datasets <#datasets>`_ containing a bunch of text from ebooks (``pdf`` and ``djvu``). They are of different size and categories.

.. The main motivation of experimenting with text classification is to use the best trained models in order to eventually build an 
   ebook organizer that will automatically categorize ebooks into their corresponding folders (associated with labels such as 
   artificial intelligence or calculus).

Also as in the scikit-learn's `tutorial <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html>`_,
multiple classifiers are used such as: ``RidgeClassifier``, ``LogisticRegression``, and ``ComplementNB``.

The best classifier will then be used for `part 2 <https://github.com/raul23/automated-ebook-organization>`_ 
to automate the organization of ebooks within the filesystem.

Datasets
========

- In the `first dataset (129 documents, 7MB) <#small-dataset-129-documents-with-3-categories>`_, the ebooks are simply divided into 3 large 
  categories: *computer science*, *mathematics* and *physics*. 
- The `second dataset (202 documents, 17MB) <#medium-size-dataset-202-documents-with-10-categories>`_ focuses on *computer science* ebooks but this 
  broad category is divided into 10 multiple subcategories in order to test how well the classifiers can differentiate *computer science* 
  ebooks between them. 
- The `third dataset (982 documents, 74MB) <#large-dataset-982-documents-with-43-categories>`_ is based on the second dataset but further divides 
  the other two broad categories (*mathematics* and *physics*) and includes more text from *computer science* ebooks. A total of 43 subcategories 
  are found in this third dataset.

Small dataset: 129 documents with 3 categories
----------------------------------------------
The first classifiers I am testing are those trained on a small dataset of 129 English documents (``pdf`` and ``djvu``) from 
3 categories:

- ``computer_science`` with label 0 and 48 ebooks
- ``mathematics`` with label 1 and 50 ebooks
- ``physics`` with label 2 and 31 ebooks

The train and test sets are splitted as follows:

- train data: 77 documents (60%)
- test data: 52 documents (40%)

This toy dataset can be interesting for quickly testing ideas about improving text classification since the training and 
testing times are very reasonable.

It is the same dataset I `tested clustering <https://github.com/raul23/clustering-text#clustering-ebooks-pdf-djvu>`_ on.

By default, only 10% of a given ebook is `converted to text <#dataset-creation>`_ and added to the dataset. Also if an ebook is 
made of images, `OCR <#ocr>`_ is applied on 5 pages chosen randomly in the first 50% of the given ebook to extract the text.

.. TODO: explain why 50% of ebook for OCR

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

The parameters for ``TfidfVectorizer``::

 max_df=0.5    min_df=5    ngram_range=(1, 1)   norm=l2

A sample of the kind of ebooks that serve as the basis for this small dataset:

- computer_science:
 
  - `A Discipline of Programming <https://www.amazon.ca/Discipline-Programming-Dijkstra/dp/013215871X>`_
  - `Algorithms in C <https://www.amazon.com/Algorithms-Computer-Science-Robert-Sedgewick/dp/0201514257/>`_
  - `bash Cookbook: Solutions and Examples for bash Users <https://www.amazon.ca/bash-Cookbook-Solutions-Examples-Users/dp/0596526784>`_
  - `Coding All-in-One For Dummies <https://www.amazon.ca/Coding-All-Dummies-Nikhil-Abraham/dp/1119363020/>`_
  - `Data Structures with C <https://www.amazon.com/Data-Structures-C-SIE-Lipschutz/dp/0070701989>`_
- mathematics:

  - `An Introduction to the Theory of the Riemann Zeta-Function 
    <https://www.amazon.com/Introduction-Zeta-Function-Cambridge-Advanced-Mathematics/dp/0521335353>`_
  - `Category Theory for the Sciences <https://www.amazon.com/Category-Theory-Sciences-MIT-Press/dp/0262028131>`_
  - `Introductory Non-Euclidean Geometry <https://www.amazon.com/Introductory-Non-Euclidean-Geometry-Dover-Mathematics-ebook/dp/B00A41V6Q2>`_
  - `Models of Peano Arithmetic <https://www.amazon.com/Models-Peano-Arithmetic-Oxford-Guides/dp/019853213X>`_
  - `The taming of chance <https://www.amazon.com/Taming-Chance-Ideas-Context/dp/0521388848>`_
- phyics:

  - `Gauge Theory of elementary particle physics <https://www.amazon.com/Gauge-Theory-elementary-particle-physics/dp/0198519613>`_
  - `Introduction to particle physics <https://www.amazon.com/Introduction-Particle-Physics-English-French/dp/0471653721>`_
  - `Student Friendly Quantum Field Theory <https://www.amazon.com/Student-Friendly-Quantum-Field-Theory/dp/0984513957>`_
  - `The Inflationary Universe <https://www.amazon.com/Inflationary-Universe-Alan-Guth/dp/0201328402>`_
  - `The Strongest Magnetic Fields in the Universe <https://www.amazon.com/Strongest-Magnetic-Fields-Universe-Sciences-ebook/dp/B01JAK55B4/>`_

Medium-size dataset: 202 documents with 10 categories
-----------------------------------------------------
The second dataset consists of 202 English documents (``pdf`` and ``djvu``) from 10 categories:

- ``algorithms``: with label 0 and 22 ebooks
- ``artificial intelligence``: with label 1 and 12 ebooks
- ``artificial neural networks``: with label 2 and 19 ebooks
- ``compiler``: with label 3 and 26 ebooks
- ``computer security``: with label 4 and 28 ebooks
- ``data structures``: with label 5 and 17 ebooks
- ``database``: with label 6 and 13 ebooks
- ``linux``: with label 7 and 17 ebooks
- ``machine learning``: with label 8 and 33 ebooks
- ``penetration testing``: with label 9 and 15 ebooks

`:information_source:` As you can see, these classes are actually all sub-categories from the broader *computer science* category.

The train and test sets are splitted as follows:

- train data: 121 documents (60%)
- test data: 81 documents (40%)

By default, only 10% of a given ebook is `converted to text <#dataset-creation>`_ and added to the dataset. No OCR was applied
this time.

Some stats about this medium-size dataset:

.. code-block::

   Categories size: [22 12 19 26 28 17 13 17 33 15]
   202 documents - 10 categories
   
   121 documents - 10.22MB (training set)
   81 documents - 7.24MB (test set)
   
   vectorize training done in 1.378s at 7.421MB/s
   n_samples: 121, n_features: 8549
   vectorize testing done in 0.941s at 7.686MB/s
   n_samples: 81, n_features: 8549

Large dataset: 982 documents with 43 categories
-----------------------------------------------
The third dataset consists of 982 English documents (``pdf`` and ``djvu``) from 43 categories::

   abstract algebra, algebra, algorithms, antimatter, artificial intelligence, artificial neural networks, astronomy, 
   black holes, c, calculus, category theory, chaos, compiler, complex analysis, computer security, cosmology, cpp, 
   data structures, database, general relativity, history [computer science], history [mathematics], history [physics], 
   linux, machine learning, magnetism, non-euclidean geometry, partial differential equations, particle physics, 
   penetration testing, plasma, prime numbers, probability, programming, python, quantum computing, quantum field theory, 
   quantum mechanics, real analysis, riemann hypothesis, special relativity, statistics, superconductivity

`:information_source:` These classes are all sub-categories from the three broader categories: *computer science*, *mathematics*, and
*physics*.

The train and test sets are splitted as follows:

- train data: 589 documents (60%)
- test data: 393 documents (40%)

By default, only 10% of a given ebook is `converted to text <#dataset-creation>`_ and added to the dataset. Also if an ebook is 
made of images, `OCR <#ocr>`_ is applied on 5 pages chosen randomly in the first 50% of the given ebook to extract the text.

Some stats about this large dataset:

.. code-block::

   Categories size: [15 14 22  8 12 23 20 46 15 27 18 17 26 13 29 13 24 20 13 41 13 33 42 23 33 22 12 24 27 15  
                     7 15 30 20 26  7 35 52 11 25 21 27 46]
   982 documents - 43 categories
   
   589 documents - 43.73MB (training set)
   393 documents - 30.44MB (test set)
   
   vectorize training done in 6.496s at 6.732MB/s
   n_samples: 589, n_features: 28446
   vectorize testing done in 3.902s at 7.803MB/s
   n_samples: 393, n_features: 28446

Results of classifying ebooks ⭐
================================
I put the results section at the top before explaining the `script <#script-classify-ebooks-py>`_ since it is one of the most important and interesting part of this document.

Thus without further ado, here are the results from training multiple classifiers on `three different datasets of ebook text <#datasets>`_.

Part A: classifiers trained on the small dataset
------------------------------------------------
These are the classification results from models trained on the `small dataset (129 documents) <#small-dataset-129-documents-with-3-categories>`_ with three categories (computer_science, mathematics, physics).

Classifying with ``RandomModel`` (baseline)
"""""""""""""""""""""""""""""""""""""""""""
All classifiers need to be at least much better than the baseline ``RandomModel`` which randomly generates the labels (from 0 to 2) for 
the ebooks to be classified:

.. code-block:: python

   self.labels_ = np.random.randint(0, self.n_clusters, X.shape[0])

|

Command used to generate the confusion matrix shown next::

 python train_classifier.py ~/Data/ebooks -s 12345 --clf RandomModel --cat computer_science mathematics physics
 
.. commit=dce386f074472f72684bf4efb95ea59bc23312e2

|

``RandomModel`` accuracy on small dataset::

 Score (normalized): 0.308
 Score (count): 16
 Total count: 52

|

.. raw:: html

   <p align="center"><img src="./images/confusion_matrix_RandomModel_small_dataset.png">
   </p>

`:information_source:` No feature effect plot could be generated since this random model doesn't have coefficients (no ``coef_``).

Classifying with ``RidgeClassifier``
""""""""""""""""""""""""""""""""""""
The first classifier I tried is a ``RidgeClassifier(solver='sparse_cg', tol=1e-02)`` trained on the `dataset 
of 129 documents <#small-dataset-129-documents-with-3-categories>`_ with three categories (computer_science, 
mathematics, physics). It is the same model with the same parameters as in scikit-learn's `tutorial <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#analysis-of-a-bag-of-words-document-classifier>`_.

Command used to generate the next plots::

 python train_classifier.py ~/Data/ebooks -s 12345 --cat computer_science mathematics physics
 
.. commit=dce386f074472f72684bf4efb95ea59bc23312e2
 
`:information_source:` Explaining the script's options

- When not specifying any particular classifier with the option ``--clf``, the default classifier 
  ``RidgeClassifier(solver='sparse_cg', tol=1e-02)`` is used.
- The option ``--cat`` specifies the only categories to include in the dataset.

|

``RidgeClassifier`` accuracy on small dataset::
 
 Score (normalized): 0.942
 Score (count): 49
 Total count: 52

|

.. raw:: html

   <p align="center"><img src="./images/confusion_matrix_ridgeclass_small_dataset.png">
   </p>

`:information_source:` Insights from the confusion matrix for ``RidgeClassifier``

- The confusion matrix is plotted based on the predictions from the test set.
- Among the three categories, this linear model has the most "difficulties" with the *physics* category. It confused two *physics* ebooks for 
  *mathematics* documents which is to be expected since both domains share overlaps between words. The vice-versa situation is not found, i.e. no 
  *mathematics* ebooks were incorrectly classified as *physics* ones which could mean that books about *physics* use a more specific vocabulary 
  than for *mathematics* ones.
- *Mathematics* ebooks are well classified but one such document was classified as a *computer science* ebook. 
- The *computer science* category is the one that ``RidgeClassifier`` has the most success with all *computer science* ebooks being 
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

- The average feature effects are computed based on the training set.
- This graph shows words that are strongly positively correlated with one category and negatively associated 
  with the other two categories such as zeta (positive for *mathematics*) and universe (positive for *physics*).

  Those words constitute good predictive features.
- *Computer science* is a category that has lots of very good predictive features (e.g. programming and algorithm). No wonder that the     
  ``RidgeClassifier`` was able to correctly classify all ebooks from this category.
- When you see the word 'energy' among ebooks from the three categories, you are almost sure that they will be about *physics*.
- *Algorithm* appears twice as good features, in the singular and plural forms. Need to do something about keeping only one
  form of a word (TODO).

Classifying with ``ComplementNB`` (odd results)⁉️
""""""""""""""""""""""""""""""""""""""""""""""""
Command used to generate the next plots::

 $ python train_classifier.py ~/Data/ebooks -s 12345 --clf ComplementNB alpha=1000 --cat computer_science mathematics physics

.. commit=dce386f074472f72684bf4efb95ea59bc23312e2

`:information_source:` The parameter ``alpha=1000`` comes from `tuning its hyperparameters <#benchmarking-classifiers>`_.

|

``ComplementNB`` accuracy on small dataset::

 Score (normalized): 0.942
 Score (count): 49
 Total count: 52

|

.. raw:: html

   <p align="center"><img src="./images/confusion_matrix_ComplementNB_small_dataset.png">
   </p>

`:information_source:` At first glance, the confusion matrix coming from ``ComplementNB`` looks almost as good as the one from `RidgeClassifier <#classifying-with-ridgeclassifier>`_. However, the next plot about the average feature effects tells another story about this model's performance on the training set.

|

.. raw:: html

   <p align="center"><img src="./images/average_feature_effect_ComplementNB_small_dataset.png">
   </p>

`:information_source:` What is really going on here? The average effects for each top 5 keywords seem to be almost the same for all classes.

- Average effects for each top 5 keywords per class::

   computer_science: [0.16902425, 0.16804379, 0.15740153, 0.1529318 , 0.15351916]
   mathematics: [0.16900307, 0.16802233, 0.15739999, 0.15292876, 0.15352894]
   physics: [0.16900022, 0.16801978, 0.15738953, 0.15292028, 0.15352079]
- The model's coefficients seem to be very similar between each class::

   computer_science: [8.60059669, 8.60056681, 8.60094647, ..., 8.60074224, 8.60053628, 8.60082752]
   mathematics: [8.60082058, 8.60044876, 8.60090342, ..., 8.60075364, 8.6007128, 8.6008339 ]
   physics: [8.60055778, 8.60041649, 8.60095444, ..., 8.60070866, 8.60052311, 8.60094642]

  **NOTE:** These are the coefficents upon which the average feature effects are computed.
- Here are the coefficents for `RidgeClassifier <#classifying-with-ridgeclassifier>`_ as a comparison::

   computer_science: [-0.0370117 ,  0.03214876,  0.01486401, ...,  0.02848551, -0.01713074,  0.00178766]
   mathematics: [ 0.09391498, -0.04700096, -0.01501172, ..., -0.00338542, 0.0700915 , -0.03325268]
   physics: [-0.05675082,  0.0149598 ,  0.00025892, ..., -0.02538427, -0.05347232,  0.0313287 ])

|

.. code-block::

   top 5 keywords per class:
     computer_science mathematics     physics
   0        algorithm   algorithm   algorithm
   1       algorithms  algorithms  algorithms
   2          integer     integer     integer
   3            shall       shall       shall
   4         integers    integers    integers

`:information_source:` The top 5 keywords (or any topK for that matter) are the same for all classes. It seems that even though ``ComplementNB``'s 
coefficients are almost the same values between all classes, the small differences are enough to help the model to correctly differentiate when
making its predictions!? 

Still not sure what is really happening here with ``ComplementNB``'s odd behavior even though it is giving good
predictions on the test set (as seen from its confusion matrix).

Benchmarking classifiers
""""""""""""""""""""""""
Also like in the scikit-learn's `tutorial <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#benchmarking-classifiers>`_, 
multiple models were tested by analyzing the trade-off between training/testing time and their test score.

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

- ``ComplementNB(alpha=1000)`` 👍 is the model with the best trade-off between test score and training/testing time.

  Though ``RidgeClassifier`` is also a good choice since it has the highest test score and relatively quick training/testing time (especially
  the testing time).
- KNN is the model with the best training time and test accuracy trade-off. However KNN is the second worst model in terms of testing time, i.e.
  it is very slow to make predictions.

  I am kind of surprise that KNN has one the best test accuracy considering that KNN is not expected to perform well with high-dimensional features
  like we find in text classification.
  
  From scikit-learn's `tutorial 
  <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#plot-accuracy-training-and-test-time-of-each-classifier>`_:
  
   Furthermore, the “curse of dimensionality” harms the ability of this model [KNN] to yield competitive accuracy in the 
   high dimensional feature space of text classification problems.
- ``RandomForestClassifier()`` 👎 is the slowest model to train and make predictions and on top of that with the worst test score.

  However, this is expected to happen when working with high-dimensional feature space since most problems become linearly separable and
  hence linear models (e.g. ``RidgeClassifier``) exhibit better overall performance as stated in scikit-learn's `tutorial 
  <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#plot-accuracy-training-and-test-time-of-each-classifier>`_:
  
   for high-dimensional prediction problems, linear models are often better suited as most problems become linearly 
   separable when the feature space has 10,000 dimensions or more.

Part B: classifiers trained on the medium-size dataset
------------------------------------------------------
These are the classification results from models trained on the `medium-size dataset (202 documents) <#medium-size-dataset-202-documents-with-10-categories>`_ with ten categories: algorithms, artificial intelligence, artificial neural networks, compiler, computer security, data structures, database, linux, machine learning, penetration testing.

Classifying with ``RandomModel`` (baseline)
"""""""""""""""""""""""""""""""""""""""""""
All classifiers need to be at least much better than the baseline ``RandomModel`` which randomly generates the labels (from 0 to 9) for 
the ebooks to be classified:

.. code-block:: python

   self.labels_ = np.random.randint(0, self.n_clusters, X.shape[0])

|

Command used to generate the confusion matrix shown next::

 python train_classifier.py ~/Data/organize -s 12345 --clf RandomModel

.. commit=dce386f074472f72684bf4efb95ea59bc23312e2

|

``RandomModel`` accuracy on medium-size dataset::

 Score (normalized): 0.111
 Score (count): 9
 Total count: 81

|

.. raw:: html

   <p align="center"><img src="./images/confusion_matrix_RandomModel_medium_dataset.png">
   </p>

`:information_source:` No feature effect plot could be generated since this random model doesn't have coefficients (no ``coef_``).


Classifying with ``RidgeClassifier`` [medium]
"""""""""""""""""""""""""""""""""""""""""""""
A ``RidgeClassifier(solver='sparse_cg', tol=1e-02)`` was trained on the `dataset 
of 202 documents <#medium-size-dataset-202-documents-with-10-categories>`_ with ten categories. It is the same model with the same parameters as in scikit-learn's `tutorial <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#analysis-of-a-bag-of-words-document-classifier>`_.

Command used to generate the next plots::

 python train_classifier.py ~/Data/organize -s 12345

.. commit dce386f074472f72684bf4efb95ea59bc23312e2 with i==250 for generating medium dataset

|

``RidgeClassifier`` accuracy on medium-size dataset::

 Score (normalized): 0.815
 Score (count): 66
 Total count: 81

|

.. raw:: html

   <p align="center"><img src="./images/confusion_matrix_ridgeclass_medium_dataset2.png">
   </p>

`:information_source:` ``RidgeClassifier`` is doing a very good job even with its default parameters.

- ``RidgeClassifier`` struggles a lot with classifying *data structures* ebooks, confusing three of them as *algorithms* documents. 
  On the other hand, it does perfectly in classifying ebooks about *algorithms*, getting all eight of them. The subject of 
  *data stuctures* has a more specific vocabulary than *algorithms* and this might explain why it makes this misclassification 
  with *data structures* ebooks only and not the opposite.
- *Artificial intelligence* is another category that ``RidgeClassifier`` has difficulties in classifying. It confuses ebooks
  about *AI* for documents expressly about *artificial neural networks* (no surprise), *compiler*, and *computer security*. I am curious about
  investigating why it made the latter two misclassifications.
- *Machine learning* really is a category that ``RidgeClassifier`` does a great job with getting 15 ebooks correctly over a total of 16 documents.
- Not enough ebooks about *database* but it got all three correctly.

|

.. raw:: html

   <p align="center"><img src="./images/average_feature_effect_ridgeclass_medium_dataset2.png">
   </p>

.. code-block::

   top 5 keywords per class:
     algorithms artificial intelligence artificial neural networks    compiler computer security
   0   integers            intelligence                     neural    compiler          security
   1    integer              artificial                      layer   compilers           attacks
   2    sorting                  turing                 artificial  expression            attack
   3        log                      ai                     vector      syntax            secure
   4        mod                thinking               architecture     lexical    authentication


     data structures    database     linux machine learning penetration testing
   0             int    database     linux           vector         penetration
   1           trees         sql    kernel           kernel            security
   2           array        dbms    device       regression              python
   3           items      server  hardware         training               linux
   4            void  relational    driver   classification       vulnerability

`:information_source:` The average feature effects plot is getting too crowded and very hard to read! 🔎

- Obviously, the words that are part of the category constitute the best predictive features:
  security (positively correlated with *computer security*), database, intelligence, linux, neural.
  
  If you could also rely on the filename, then the task of ebooks classification could be tried
  with regex. You might not achieve as good results as with machine learning but for some users
  it might be good enough, especially if the ebooks are well named and contain some of these good
  predictive words.
  
  However, the classifiers are very robust in that they can work well even if the ebooks have pure gibberish
  as filenames or are wrongly named since these models only care about the content of the documents. Looking at my own 
  collection of ebooks, I have some of them that were lazily named with odd titles that don't give much
  information about their content (e.g. ``2 copy.pdf``). But the classifiers should still be able to classify them
  without much problem.
- Some words can be strongly positively correlated with more than two classes such as kernel (positively
  associated with *linux*, *machine learning* and *artificial neural networks*).

Classifying with ``ComplementNB`` (again odd results)⁉️
""""""""""""""""""""""""""""""""""""""""""""""""""""""
Command used to generate the next plots::

 $ python train_classifier.py ~/Data/organize -s 12345 --clf ComplementNB

.. commit dce386f074472f72684bf4efb95ea59bc23312e2 with i==250 for generating medium dataset

`:information_source:` I used the scikit-learn's default values for ``ComplementNB``'s parameters.

|

``ComplementNB`` accuracy on medium-size dataset::

 Score (normalized): 0.679
 Score (count): 55
 Total count: 81

| 

.. raw:: html

   <p align="center"><img src="./images/confusion_matrix_ComplementNB_medium_dataset.png">
   </p>

`:information_source:` Overall, ``ComplementNB``'s predictions are not as good as those from `RidgeClassifier 
<#classifying-with-ridgeclassifier-medium>`_

- *Data structures* continues being a very difficult category to predict. However, ``ComplementNB`` is doing a worse job
  than ``RidgeClassifier`` in that respect: confusing 4 *data structures* ebooks for *algorithms* ones and being able
  to correctly categorize only one *data structures* ebook.
- *Penetration testing* is another category that ``ComplementNB`` struggles more than ``RidgeClassifier`` does:
  only one ebook was correctly classified as such vs 5 for ``RidgeClassifier`` (over a total of 6 documents from that category).
  
  ``ComplementNB`` confused 5 *penetration testing* ebooks for *computer security* ones (which technically it is the case).
- Like with ``RidgeClassifier``, ``ComplementNB`` does a perfect job in classifying all *algorithms* ebooks correctly.
- Also, *machine learning* presents an easy category to classify: 14 ebooks correctly classify as such over a total 16 documents from that category.
- Where ``ComplementNB`` is doing a relatively better job (but not that significant) than ``RidgeClassifier`` is with
  the *computer security* category: only one misclassification vs two for ``RidgeClassifier`` (over a total of 12 ebooks from that category).

|

.. raw:: html

   <p align="center"><img src="./images/average_feature_effect_ComplementNB_medium_dataset.png">
   </p>

.. code-block::

   top 5 keywords per class:
     algorithms artificial intelligence artificial neural networks  compiler computer security
   0   security                security                   security  security          security
   1    integer                compiler                   compiler  compiler          compiler
   2   compiler                 integer                     kernel   integer            kernel
   3     kernel                  kernel                    integer    string           integer
   4     string                  string                     string    kernel            server
   
   
     data structures  database     linux machine learning penetration testing
   0        security  security  security         security            security
   1        compiler  compiler    kernel           kernel            compiler
   2         integer   integer  compiler         compiler             integer
   3          kernel    kernel     linux          integer              kernel
   4          string    server   integer           string               linux

`:information_source:` Again the same odd results like when ``ComplementNB`` was trained on the `small dataset 
<#classifying-with-complementnb-odd-results>`_.

- The average feature effects look similar for all classes.
- Same top 5 keywords for all classes.
- But even though ``ComplementNB`` is acting weird with its top 5 keywords, its scores on the test set are not terrible as it can be seen
  from the previous confusion matrix.

Benchmarking classifiers [medium]
"""""""""""""""""""""""""""""""""
`:information_source:` Having problems training ``LogisticRegression`` on the medium-size dataset (202 documents)::

   STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

   Increase the number of iterations (max_iter) or scale the data as shown in:
       https://scikit-learn.org/stable/modules/preprocessing.html
   Please also refer to the documentation for alternative solver options:

I will try eventually what they suggest: increase ``max_iter`` or do some `preprocessing 
<https://scikit-learn.org/stable/modules/preprocessing.html>`_ of the dataset.

|

Command used to generate the next plots::

 $ python train_classifier.py ~/Data/organize -s 12345 -b
 
.. commit 3f2ae11

.. python train_classifier.py ~/Data/organize/ -s 12345 --ht --clfs ComplementNB LogisticRegression RidgeClassifier KNeighborsClassifier RandomForestClassifier NeaestCentroid LinearSVC SGDClassifier 

.. talk about hyper tune the other benchmarking results and add command for benchmarking

|

Here are the benchmarking results of multiple classifiers trained on the `medium-size dataset 
<#medium-size-dataset-202-documents-with-10-categories>`_:

+-----------------+---------------------------------------------------+-------------------------+---------------------------+------------------+-----------------------------------------+--------------------+----------------------------+
|                 | RidgeClassifier(alpha=0.001, solver='sparse_cg')  | KNeighborsClassifier()  | RandomForestClassifier()  | LinearSVC(C=10)  | SGDClassifier(alpha=1e-06, loss='log')  | NearestCentroid()  | ComplementNB(alpha=10000)  |
+=================+===================================================+=========================+===========================+==================+=========================================+====================+============================+
| train time      | 0.202s                                            | 0.00198s                | 0.34s                     | 0.363s           | 0.0429s                                 | 0.00817s           | 0.00663s                   |
+-----------------+---------------------------------------------------+-------------------------+---------------------------+------------------+-----------------------------------------+--------------------+----------------------------+
| test time       | 0.00166s                                          | 0.0209s                 | 0.0491s                   | 0.00163s         | 0.0021s                                 | 0.00264s           | 0.00151s                   |
+-----------------+---------------------------------------------------+-------------------------+---------------------------+------------------+-----------------------------------------+--------------------+----------------------------+
| accuracy        | 0.815                                             | 0.728                   | 0.617                     | 0.815            | 0.877                                   | 0.79               | 0.667                      |
+-----------------+---------------------------------------------------+-------------------------+---------------------------+------------------+-----------------------------------------+--------------------+----------------------------+
| dimensionality  | 8549                                              | -                       | -                         | 8549             | 8549                                    | -                  | 8549                       |
+-----------------+---------------------------------------------------+-------------------------+---------------------------+------------------+-----------------------------------------+--------------------+----------------------------+
| density         | 1.0                                               | -                       | -                         | 0.998            | 1.0                                     | -                  | 1.0                        |
+-----------------+---------------------------------------------------+-------------------------+---------------------------+------------------+-----------------------------------------+--------------------+----------------------------+

|

The next two plots about the trade-off between test score and training/test time will help us in determining the best classifier to choose:

.. raw:: html

   <p align="center"><img src="./images/score_training_time_trade_off_medium.png">
   </p>

|

.. raw:: html

   <p align="center"><img src="./images/score_test_time_trade_off_medium.png">
   </p>

`:information_source:` 

- ``SGDClassifier(loss='log')`` 👍 is the model with the best trade-off between test score and training/testing time: highest test score (0.877) and 
  relatively quick training/testing time (both under 0.05s).

  For reference, here are the top 5 keywords per class for ``SGDClassifier``::
  
     top 5 keywords per class:
        algorithms artificial intelligence artificial neural networks    compiler computer security
      0    sorting            intelligence                     neural    compiler          security
      1        mod                  turing                      layer      tokens            secure
      2    solving              artificial                  nonlinear  expression               log
      3        log                thinking               architecture   compilers             trust
      4      graph                      ai             neuralnetworks      symbol           session


        data structures    database          linux machine learning penetration testing
      0           trees    database          linux       regression         penetration
      1             int        dbms         kernel            https              python
      2         records  relational       hardware           kernel       vulnerability
      3           items      entity  configuration      statistical              import
      4          record         sql           unix              org            security
      
  ⚠️ 'https' and 'org' as top 5 key words for *machine learning*?
  
  For comparison, here are the top 5 key words for out-of-the-box `RidgeClassifier <#classifying-with-ridgeclassifier-medium>`_.
- ``RandomForestClassifier()`` 👎 `continues <#benchmarking-classifiers>`_ to underperform with text classification: worst in all respects.

  C'mon ``RandomForestClassifier``, you only had one job! 😞

Part C: classifiers trained on the large dataset
------------------------------------------------
These are the classification results from models trained on the `large dataset (982 documents) <#large-dataset-982-documents-with-43-categories>`_ with 43 categories::

   abstract algebra, algebra, algorithms, antimatter, artificial intelligence, artificial neural networks, astronomy, 
   black holes, c, calculus, category theory, chaos, compiler, complex analysis, computer security, cosmology, cpp, 
   data structures, database, general relativity, history [computer science], history [mathematics], history [physics], 
   linux, machine learning, magnetism, non-euclidean geometry, partial differential equations, particle physics, 
   penetration testing, plasma, prime numbers, probability, programming, python, quantum computing, quantum field theory, 
   quantum mechanics, real analysis, riemann hypothesis, special relativity, statistics, superconductivity

Classifying with ``RandomModel`` (baseline)
"""""""""""""""""""""""""""""""""""""""""""
All classifiers need to be at least much better than the baseline ``RandomModel`` which randomly generates the labels (from 0 to 2) for 
the ebooks to be classified:

.. code-block:: python

   self.labels_ = np.random.randint(0, self.n_clusters, X.shape[0])

|

Command used to generate the confusion matrix shown next::

 python train_classifier.py ~/Data/organize -s 12345 --clf RandomModel
 
.. commit dce386f074472f72684bf4efb95ea59bc23312e2

|

``RandomModel`` accuracy on large dataset::

 Score (normalized): 0.0229
 Score (count): 9
 Total count: 393
 
|

.. raw:: html

   <p align="center"><img src="./images/confusion_matrix_RandomModel_large_dataset.png">
   </p>

`:information_source:` No feature effect plot could be generated since this random model doesn't have coefficients (no ``coef_``).

Classifying with ``RidgeClassifier`` [large]
""""""""""""""""""""""""""""""""""""""""""""
A ``RidgeClassifier(solver='sparse_cg', tol=1e-02)`` was trained on the `dataset 
of 982 documents <#large-dataset-982-documents-with-43-categories>`_ with 43 categories. It is the same model with the same parameters as in scikit-learn's `tutorial <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#analysis-of-a-bag-of-words-document-classifier>`_.

Command used to generate the next plot::

 python train_classifier.py ~/Data/organize -s 12345

.. commit dce386f074472f72684bf4efb95ea59bc23312e2

|

``RidgeClassifier`` accuracy on large dataset::

 Score (normalized): 0.728
 Score (count): 286
 Total count: 393

|

.. raw:: html

   <p align="center"><img src="./images/confusion_matrix_ridgeclass_large_dataset.png">
   </p>

`:information_source:` Since there are so many categories to analyze, I will just focus on the most interesting cases

- *Special relativity* and *General relativity*: 

  - ``RidgeClassifier`` confuses more *special relativity* ebooks for *general relativity* ones than vice versa.
  - 6 over 23 ebooks about *general relativity* were confused for *black holes* documents which is understandable but I will have
    to think about how to help the model better differentiate ebooks from these two categories (e.g. increase the number of pages
    converted to text, add more ebooks about both classes, perform preprocessing, ...) [TODO]
  - Many of the ebooks from these two categories (*special relativity* and *general relativity*) focus on 
    both subjects. Maybe I could create another category (*Special and General Relativity*) for ebooks that treat 
    both topics extensively. [TODO]
- *C vs CPP*: programming languages

  - ``RidgeClassifier`` confuses more *c* ebooks for *cpp* ones than vice versa.
  - Again work to be done to improve the model's accuracy with these cases. [TODO]
- Some categories that ``RidgeClassifier`` achieved great accuracy:

  - *Black holes*: 20/21
  - *Compiler*: 9/9
  - *History [mathematics]*: 11/11
  - *Machine learning*: 15/15
  - *Probability*: 9/10
  - *Python*: 11/11
  - *Quantum mechanics*: 17/19
  - *Superconductivity*: 11/11

- Some categories that ``RidgeClassifier`` achieved low accuracy:

  - *Artificial neural networks*: 4/12 [8 ebooks were confused for *machine learning* ones]
  - *Complex analysis*: 4/8 [3 ebooks were confused for *Riemann hypothesis* ones]
  - *Prime numbers*: 0/5 [All ebooks were confused for *Riemann hypothesis* ones]
- Some categories where I need to add more ebooks in the test set: *antimatter* (1), *cosmology* (2), *history [computer science]* (0),
  *plasma* (1), *quantum computing* (2)
|

Top 5 keywords per class (for all 43 categories):

.. code-block::

     abstract algebra         algebra  algorithms  antimatter artificial intelligence artificial neural networks  astronomy    black holes
   0         integers         algebra   algorithm  antimatter            intelligence                    network      stars          black
   1            prove              ir  algorithms    universe                 program                   learning        sun          holes
   2          integer              ca    integers       stars                      ai                     vector        sky           hole
   3              mod  multiplication       trees      energy              artificial                     neural  astronomy      spacetime
   4          theorem         formula     sorting   particles                    test                     output       moon  gravitational
   
   
             c  calculus  category theory     chaos   compiler complex analysis computer security   cosmology       cpp data structures
   0       int     graph         category     chaos   compiler            plane          security   cosmology   classes      structures
   1  variable        2x       categories    random       code         analytic            server    universe  template       algorithm
   2      file    domain       structures     shall    machine              sin              user    galaxies  operator      algorithms
   3   program    graphs         identity   initial   language               oo            secure      cosmic  compiler             int
   4      char  calculus  transformations  behavior  languages          formula            attack  relativity     const         program


      database general relativity history [computer science] history [mathematics] history [physics]   linux machine learning  magnetism
   0  database         relativity                    machine               history           history   linux         learning   magnetic
   1  security      gravitational                    century                square            motion  kernel          machine  magnetism
   2    access           einstein                   machines               ancient          theories   shell       algorithms       axis
   3  instance             tensor                 processing               algebra        scientific    user        algorithm   electric
   4     users              frame                 historical                 greek        philosophy    code          feature     moment


     non-euclidean geometry partial differential equations particle physics penetration testing     plasma prime numbers   probability
   0               geometry                   differential         particle         penetration     plasma        primes   probability
   1                  plane                             dx        particles             testing   magnetic         prime        random
   2              euclidean                       boundary          nuclear            security   electric       theorem        events
   3                     ab                           wave             spin               tools  radiation       density  distribution
   4               triangle                        partial       scattering                 web  electrons          base        sample

   
      programming  python quantum computing quantum field theory quantum mechanics real analysis riemann hypothesis special relativity
   0  programming  python           quantum              quantum           quantum      sequence            riemann         relativity
   1     programs    code       computation              feynman         mechanics       integer               zeta            lorentz
   2         code      py                le            invariant              wave      rational         hypothesis           geometry
   3       design   press            michel                dirac          particle        metric              prime       relativistic
   4      program  module            vector                   eq            energy         limit            formula           einstein


       statistics  superconductivity
   0  probability  superconductivity
   1   statistics        temperature
   2  statistical    superconductors
   3   experiment    superconducting
   4       sample     superconductor

Benchmarking classifiers [large]
""""""""""""""""""""""""""""""""
Command used to generate the next plots::

 $ python train_classifier.py ~/Data/organize -s 12345 -b
 
.. commit 7767bf6fb8e0484926975d847a610336ad101daf

|

Here are the benchmarking results of multiple classifiers trained on the `large dataset 
<#large-dataset-202-documents-with-43-categories>`_:

+-----------------+---------------------------------------------------+---------------------------------------+---------------------------+--------------------------------+-----------------------------------------+--------------------+----------------------------+
|                 | RidgeClassifier(alpha=0.001, solver='sparse_cg')  | KNeighborsClassifier(n_neighbors=10)  | RandomForestClassifier()  | LinearSVC(C=10, max_iter=500)  | SGDClassifier(alpha=1e-06, loss='log')  | NearestCentroid()  | ComplementNB(alpha=1e-06)  |
+=================+===================================================+=======================================+===========================+================================+=========================================+====================+============================+
| train time      | 11.4s                                             | 0.00441s                              | 1.96s                     | 7.28s                          | 1.49s                                   | 0.053s             | 0.177s                     |
+-----------------+---------------------------------------------------+---------------------------------------+---------------------------+--------------------------------+-----------------------------------------+--------------------+----------------------------+
| test time       | 0.0634s                                           | 0.304s                                | 0.307s                    | 0.0546s                        | 0.0568s                                 | 0.0621s            | 0.0714s                    |
+-----------------+---------------------------------------------------+---------------------------------------+---------------------------+--------------------------------+-----------------------------------------+--------------------+----------------------------+
| accuracy        | 0.758                                             | 0.656                                 | 0.618                     | 0.766                          | 0.758                                   | 0.692              | 0.634                      |
+-----------------+---------------------------------------------------+---------------------------------------+---------------------------+--------------------------------+-----------------------------------------+--------------------+----------------------------+
| dimensionality  | 28446                                             | -                                     | -                         | 28446                          | 28446                                   | -                  | 28446                      |
+-----------------+---------------------------------------------------+---------------------------------------+---------------------------+--------------------------------+-----------------------------------------+--------------------+----------------------------+
| density         | 1.0                                               | -                                     | -                         | 0.941                          | 1.0                                     | -                  | 1.0                        |
+-----------------+---------------------------------------------------+---------------------------------------+---------------------------+--------------------------------+-----------------------------------------+--------------------+----------------------------+

|

The next two plots about the trade-off between test score and training/test time will help us in determining the best classifier to choose:

.. raw:: html

   <p align="center"><img src="./images/score_train_time_trade_off_large.png">
   </p>

|

.. raw:: html

   <p align="center"><img src="./images/score_test_time_trade_off_large.png">
   </p>

`:information_source:` 

- ``SGDClassifier(loss='log')`` 👍 is the model with the best trade-off between test score and training/testing time: second 
  highest test score (0.758) and relatively quick training/testing time (both under 1.5s).
- The training time on this large dataset is very high for some models: 

  - 11.4s for ``RidgeClassifier`` with a good test score though (0.758)
  - 7.28s for ``LinearSVC`` with the best test score though (0.766)

Conclusion
==========
In conclusion, it looks like ``SGDClassifier`` is your model of choice if you are working with a relatively large dataset of ebook text.
It was choosen as the model with the best trade-off between test score and training/testing time for both 
medium-size and large datasets: 

- `medium-size (202 documents) dataset <#benchmarking-classifiers-medium>`_::

   accuracy       = 0.877
   train time     = 0.0429s
   test time      = 0.0021s
   dimensionality = 8549
- `large (982 documents) dataset <#benchmarking-classifiers-large>`_::

   accuracy       = 0.758
   train time     = 1.49s
   test time      = 0.0568s
   dimensionality = 28446
  
On the `small dataset (129 documents) <#benchmarking-classifiers>`_, it was ``ComplementNB`` that was selected as the classifier
with the best overall performance::

  accuracy       = 0.942
  train time     = 0.00229s
  test time      = 0.000572s
  dimensionality = 5436

``SGDClassifier`` fared very well also on this small dataset (it is an easy dataset to predict since it has only
three book categories; hence many models did great as well)::

  accuracy       = 0.942
  train time     = 0.00832s
  test time      = 0.000608s
  dimensionality = 5436

Next, I will be trying to improve the test score on the large dataset which is a tough one since all the test scores achieved so far 
on that dataset are below 0.8 

Preprocessing and tuning better the hyperparameters (I am not exploring enough different sets of hyperparameters) are avenues I will be exploring.

Script ``train_classifier.py``
=============================
To know how to use the script ``train_classifier.py``, go `here <./README_script.rst>`_ to read its documentation.
