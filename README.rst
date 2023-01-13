=====================================
Experimenting with classifying ebooks
=====================================
.. contents:: **Contents**
   :depth: 3
   :local:
   :backlinks: top

Introduction
============
I am basing my experimentation with classifying text on the excellent scikit-learn's tutorial: `Classification of text documents using sparse features <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html>`_.

I am following along their tutorial but using my own `three datasets <#datasets>`_ containing a bunch of text from ebooks (``pdf`` and ``djvu``). They are of different size and categories.

The main motivation of experimenting with text classification is to use the best trained models in order to eventually build an ebooks organizer that will automatically categorize ebooks into their corresponding folders (associated with labels such as artificial intelligence or calculus).

Also as in the scikit-learn's `tutorial <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html>`_,
multiple classifiers are used such as: ``RidgeClassifier``, ``LogisticRegression``, and ``ComplementNB``.

Datasets
========

- In the `first dataset (129 documents, 7MB) <#small-dataset-129-documents-with-3-categories>`_, the ebooks are simply divided into 3 large 
  categories: *computer science*, *mathematics* and *physics*. 
- The `second dataset (202 documents, 17MB) <#medium-size-dataset-202-documents-with-10-categories>`_ focuses on *computer science* ebooks but this 
  broad category is divided into 10 multiple subcategories in order to test how well the classifiers can differentiate *computer science* 
  ebooks between them. 
- The `third dataset (982 documents, 74MB) <#large-dataset-982-documents-with-43-categories>`_ is based on the second dataset but further divides 
  the other two broad categories (*mathematics* and *physics*) and includes more ebooks from *computer science*. A total of 43 subcategories 
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
I put the results section at the top before explaining the `script <#script-classify-ebooks-py>`_ since it is one the most important and interesting part of this document.

Thus without further ado, here are the results from training multiple classifiers on `three different datasets of ebooks <#datasets>`_.

Part 1: classifiers trained on the small dataset
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

 python classify_ebooks.py ~/Data/ebooks -s 12345 --clf RandomModel --cat computer_science mathematics physics
 
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

 python classify_ebooks.py ~/Data/ebooks -s 12345 --cat computer_science mathematics physics
 
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

 $ python classify_ebooks.py ~/Data/ebooks -s 12345 --clf ComplementNB alpha=1000 --cat computer_science mathematics physics

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

Part 2: classifiers trained on the medium-size dataset
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

 python classify_ebooks.py ~/Data/organize -s 12345 --clf RandomModel

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

 python classify_ebooks.py ~/Data/organize -s 12345

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

 $ python classify_ebooks.py ~/Data/organize -s 12345 --clf ComplementNB

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

 $ python classify_ebooks.py ~/Data/organize -s 12345 -b
 
.. commit 3f2ae11

.. python classify_ebooks.py ~/Data/organize/ -s 12345 --ht --clfs ComplementNB LogisticRegression RidgeClassifier KNeighborsClassifier RandomForestClassifier NeaestCentroid LinearSVC SGDClassifier 

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

Part 3: classifiers trained on the large dataset
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

 python classify_ebooks.py ~/Data/organize -s 12345 --clf RandomModel
 
.. commit dce386f074472f72684bf4efb95ea59bc23312e2

|

``RandomModel`` accuracy on large dataset::

 Score (normalized): 0.022900763358778626
 Score (count): 9.0
 
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

 python classify_ebooks.py ~/Data/organize -s 12345

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

Benchmarking classifiers [large]
""""""""""""""""""""""""""""""""
TODO

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

I won't list all options (too many) but here are some of the important and interesting ones:

**Benchmarking options:**

-b                                     Benchmarking classifiers.

**Cache options:**

-u                                     Highly recommended to use cache to speed up **dataset re-creation**.

**Dataset options:**

--cd                                  Create dataset with text from ebooks found in the directory.
--ud                                  Update dataset with text from more new ebooks found in the directory.
--cat CATEGORY                        Only include these categories in the dataset. (default: computer_science mathematics physics)  
--vect-params PARAMS                  The parameters to be used by TfidfVectorizer for vectorizing the dataset. 
                                      (default: max_df=0.5 min_df=5 ngram_range='(1, 1)' norm=l2)

**Hyperparameter tuning options:**

--hyper-tune                           Perform hyperparameter tuning.
--clfs CLF                             The names of classifiers whose hyperparameters will be tuned with grid search.
                                       (default: RidgeClassifier ComplementNB)

**OCR options:**

-o                                     Whether to enable OCR for ``pdf``, ``djvu`` and image files. It is disabled by default. (default: false)

**Classification options:**

--clf CLF_PARAMS                       The name of the classifier along with its parameters to be used for classifying ebooks. 
                                       (default: RidgeClassifier tol=1e-2 solver=sparse_cg)

|

`:information_source:` Explaining some important and interesting options/arguments

- ``input_directory`` is the path to the main directory containing the documents to classify.

  The following options require to specify an ``input_directory``:
  
  - ``--hyper-tune``: hyperparameter tuning
  - ``-b``: benchmarking
- ``-b`` uses right now hard-coded parameter values for multiple classifiers. However, I will eventualy
  make it possible to upload a JSON file with custom parameter values for different classifiers when
  using this option (TODO).
- By **dataset re-creation** I mean the case when you delete the pickle dataset file and generate the dataset 
  again. If you are using cache, then the dataset generation should be quick since the text conversions were
  already computed and cached. Using the option ``-u`` is worthwhile especially if you used OCR for some of the ebooks since this procedure is very
  resource intensive and can take awhile if many pages are OCRed.
- ``--vect-params PARAMS [PARAMS ...]``: the parameters for ``TfidfVectorizer`` are given one after the other like this::

   --vect-params max_df=0.2 min_df=1 ngram_range='(1,1)' norm=l2
   
  `:warning:` It is important to escape any parentheses on the terminal by placing them within single quotes or after a backslash
  (e.g. ``ngram_range=\(1,1)\)``).
- ``--clfs [CLF [CLF ...]]``: the names of the classifiers are those used in scikit-learn's modules. For example::

   python classify_ebooks.py ~/Data/ebooks --hyper-tune --clfs KNeighborsClassifier NearestCentroid LogisticRegression
   
- ``--clf CLF_PARAMS``: the name of the classifier and its parameters are the ones used in scikit-learn's modules. For example::
  
   python classify_ebooks.py ~/Data/ebooks --clf KNeighborsClassifier n_neighbors=5
- The choices for ``-o`` are ``{always, true, false}``
  
  - 'always': always use OCR first when doing text conversion. If the converson fails, then use the other simpler conversion tools
    (``pdftotext`` and ``djvutxt``).
  - 'true': first simpler conversion tools (``pdftotext`` and ``djvutxt``) will be used and then if a conversion method
    failed to convert an ebook to ``txt`` or resulted in an empty file, the OCR method will be used.
  - 'false': never use OCR, only use the other simpler conversion tools (``pdftotext`` and ``djvutxt``).

Start the classification of ebooks ⭐
-------------------------------------
To **quickly** start the classification of ebooks, all you need is to provide the directory containing said ebooks::

 python classify_ebooks.py ~/Data/ebooks
 
The script will generate the dataset and then train the default classifier (``RidgeClassifier``) and 
display the confusion matrix and features effects graph.

To specify a classifier with its parameters, use the ``--clf`` option::

 python classify_ebooks.py ~/Data/ebooks --clf 

Caching
-------
`:information_source:` About the caching option (``--use-cache``) supported by the script ``classify_ebooks.py.py``

- Cache is used to save the converted ebook files into ``txt`` to
  avoid re-converting them which can be a time consuming process. 
  `DiskCache <http://www.grantjenks.com/docs/diskcache/>`_, a disk and file 
  backed cache library, is used by the ``classify_ebooks.py.py`` script.
- Default cache folder used: ``~/.classify_ebooks``
- The MD5 hashes of the ebook files are used as keys to the file-based cache.
- These hashes of ebooks (keys) are then mapped to a dictionary with the following structure:

  - key: ``convert_method+convert_only_percentage_ebook+ocr_only_random_pages``
  
    where 
    
    - ``convert_method`` is either ``djvutxt`` or ``pdftotext``
    - ``convert_only_percentage_ebook`` is the percentage of a given ebook that is converted to ``txt``
    - ``ocr_only_random_pages`` is the number of pages chosen randomly in the first 50% of a given ebook
      that will be OCRed
      
    e.g. djvutxt+15+3
    
  - value: the extracted text based on the options mentioned in the associated key
  
  Hence, you can have multiple extracted texts associated with a given ebook with each of the text
  extraction based on different values of the options mentioned in the key.

|

`:warning:` Important things to keep in mind when using the caching option

* When enabling the cache with the flag ``--use-cache``, the ``classify_ebooks.py`` 
  script has to cache the converted ebooks (``txt``) if they were
  not already saved in previous runs. Therefore, the speed up of some of the
  tasks (dataset re-creation and updating) will be seen in subsequent executions of the 
  script.
* Keep in mind that caching has its caveats. For instance if a given ebook
  is modified (e.g. a page is deleted) then the ``classify_ebooks.py`` 
  script has to run the text conversion again since the keys in the cache are the MD5 hashes of
  the ebooks.
* There is no problem in the
  cache growing without bounds since its size is set to a maximum of 1 GB by
  default (check the ``--cache-size-limit`` option) and its eviction policy
  determines what items get to be evicted to make space for more items which
  by default it is the least-recently-stored eviction policy (check the
  ``--eviction-policy`` option).

Ebooks directory
----------------
`:warning:` In order to run the script `classify_ebooks.py <./scripts/classify_ebooks.py>`_, you need first to have a main directory (e.g. ``./ebooks/``) with all the ebooks (``pdf`` and ``djvu``) you want to test classification on. Each ebook should be in a folder whose name should correspond to the category of said ebook.

For example:

- ../ebooks/**biology**/Cell theory.djvu
- ../ebooks/**philosophy**/History of Philosophy in Europe.pdf
- ../ebooks/**physics**/Electricity.pdf

Then, you need to give the path to the main directory to the script, like this::

 $ python classify_ebooks.py ~/Data/ebooks/
 
The next section explains in details the generation of a dataset containing text from these ebooks.

Dataset creation
----------------
To start creating a dataset containing texts from ebooks after you have setup your `directory of ebooks <#ebooks-directory>`_, the option
``--cd`` and the input directory are necessary::

 $ python classify_ebooks.py --cd ~/Data/ebooks/
 
`:information_source:` Explaining the text conversion procedure

- ``--cd, --create-dataset`` tells the script to start creating the dataset if it is not already found within the specified directory.
- The script will try to convert each ebook to text by using ``pdftotext`` or ``djvutxt`` depending on the type of file.
- By default, OCR is not used (``--ocr-enabled`` is set to 'false') since it is a very resource intensive procedure. The other
  simpler conversion methods (``pdftotext`` or ``djvutxt``) are used instead which are very quick and reliable in their text conversion of ebooks.
- By default, only 10% of a given ebook is converted to text. The option ``--cope, --convert-only-percentage-ebook PAGES`` controls
  this percentage.
- If the text conversion fails with the simpler tools (``pdftotext`` or ``djvutxt``) because an ebook is composed of images 
  for example, then a warning message is printed suggesting you to use OCR which should be able to fix the problem but if too many ebooks
  are images then it might not be practicable to use OCR if updating the dataset afterward.
- The hash of each ebook is computed so as to avoid adding duplicates in the dataset. Also the hashes are used as keys in the cache if
  caching is used (i.e. the option ``-u, --use-cache`` is enabled).

|

`:information_source:` The first time the script is run, the dataset of text (from ebooks) will be created. This dataset is a `Bunch <https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html>`_ object (a dictionary-like object that allows you to access its values by keys or attributes) with the following structure:

- ``data``: list of shape (n_samples,)
- ``filenames``: list of shape (n_samples,)
- ``target_names``:  list of shape (n_classes,)
- ``target``: ndarray of shape (n_samples,)
- ``DESCR``: str, the full description of the dataset

It is the same structure as the one used by scikit-learn for their `datasets <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html>`_.

The label used by ``target`` is automatically generated by assigning integers (from the range ``[0, number of classes - 1]``) to each sample. 

The dataset is saved as a pickle file under the main directory that you provided to the script.

The next times the script is run, the dataset will be loaded from disk as long as you don't delete or move the pickle file saved directly under the main directory.

|

Creating the ebooks dataset using cache (``-u`` option) without OCR support (i.e. the ``-o true`` option is not used)::

 $ python classify_ebooks.py --cd -u ~/Data/ebooks/

First time running the script with a cleared cache:

.. raw:: html

   <p align="left"><img src="https://raw.githubusercontent.com/raul23/clustering-text/main/images/dataset_generation_first_time_used_cache.png">
   </p>

|

Second time running the script with some of the text conversions already cached:


.. raw:: html

   <p align="left"><img src="https://github.com/raul23/clustering-text/blob/main/images/dataset_generation_second_time_used_cache.png">
   </p>

|

Warning message shown when a text conversion fails (e.g. the ebook is made up of images):

.. raw:: html

   <p align="left"><img src="https://github.com/raul23/clustering-text/blob/main/images/dataset_generation_conversion_failed_use_ocr.png">
   </p>
   
`:information_source:` The dataset generation can be re-run again after with the ``-o true --ud`` options which enable the use of OCR for those
problematic ebooks that couldn't be converted to ``txt`` with simpler methods (``pdftotext`` and ``djvutxt``).

|

When a duplicate is found (based on MD5 hashes), the correponding ebook is not processed further:

.. raw:: html

   <p align="left"><img src="https://github.com/raul23/clustering-text/blob/main/images/dataset_generation_found_duplicate.png">
   </p>

|

At the end of the dataset creation, some results are shown about the number of texts
added to the dataset and cache, books rejected and duplicates found

.. raw:: html

   <p align="left"><img src="https://github.com/raul23/clustering-text/blob/main/images/dataset_generation_end_results2.png">
   </p>

OCR
---
For those ebooks that couldn't be converted to ``txt`` with simpler methods (``pdftotext`` and ``djvutxt``), 
you can update the dataset using the  options ``--ud`` (update) and ``-o true`` (enable OCR)::

 $ python classify_ebooks.py -u --ud -o true ~/Data/ebooks/

`:information_source:` Explaining the options:

- ``-u`` enables the cache in order to add the converted text to the cache.
- The ``--ud`` flag refers to the action of updating the dataset pickle file that was already saved within the main ebooks directory
  (e.g. ``~/Data/ebooks/``)
- ``-o true`` enables OCR. The choices for ``-o, --ocr-enabled`` are: ``{always, true, false}``. See `Script options <#script-options>`_ for an 
  explanation of these values.
- The OCR procedure is resource intensive, thus the conversion for those problematic ebooks might take longer than usual.
- By default, OCR is applied on only 5 pages chosen randomly in the first 50% of a given ebook. This number is controlled by
  the option ``--ocr-only-random-pages PAGES``.

|

Loading a dataset and applying OCR to those ebooks that couldn't be converted to ``txt`` with simpler methods (``pdftotext`` and ``djvutxt``):

.. raw:: html

   <p align="left"><img src="https://github.com/raul23/clustering-text/blob/main/images/updating_dataset_ocr.png">
   </p>

|

Results at the end of applying OCR to all problematic ebooks (made up of images):

.. raw:: html

   <p align="left"><img src="https://github.com/raul23/clustering-text/blob/main/images/updating_dataset_ocr_end_results.png">
   </p>
   
`:information_source:` All 14 problematic ebooks (made up of images) were successfully converted to ``txt`` and added to the dataset and cache.

Updating a dataset
------------------
After a dataset is created and saved, you can update it with new texts from more ebooks by using the ``--ud`` option::

 $ python classify_ebooks.py --ud ~/Data/ebooks/

.. raw:: html

   <p align="left"><img src="https://github.com/raul23/clustering-text/blob/main/images/updating_dataset_ocr.png">
   </p>
   
`:information_source:` ``--ud`` tells the script to update the dataset pickle file saved within the main ebooks directory (e.g. ``~/Data/ebooks``).

Filtering a dataset: select texts only in English and from valid categories
---------------------------------------------------------------------------
After the dataset containing texts from ebooks is generated, you can launch the classification by providing only the input directory
containing the saved pickle file of the dataset. During the text classification, the dataset is loaded and filtered by removing 
text that is not English and not part of the specified categories (e.g. ``computer_science``, ``mathematics``, ``physics``).

Here are some samples of output from the script ``classify_ebooks.py``::

 python classify_ebooks.py ~/Data/ebooks/ --verbose
 
`:information_source:` Explaining the options:

- Since the option ``--verbose`` is used, you will see more information printed in the terminal such as
  if the text is in English or its category.
- By default, the three mentioned categories are choosen. But you can control the categories you want to include in the filtered dataset with the
  ``--cat`` option::

   python classify_ebooks.py -u ~/Data/ebooks/ --cat chemistry physics

| 
 
Showing the categories that will be kept:

.. raw:: html

   <p align="left"><img src="https://github.com/raul23/clustering-text/blob/main/images/filtering_keeping_categories.png">
   </p>

|

Texts rejected for not being in English:

.. raw:: html

   <p align="left"><img src="https://github.com/raul23/clustering-text/blob/main/images/filtering_rejected_french_spanish.png">
   </p>
   
|

Texts rejected for not being part of the specified categories (``computer_science``, ``mathematics``, ``physics``):

.. raw:: html

   <p align="left"><img src="https://github.com/raul23/clustering-text/blob/main/images/filtering_rejected_politics.png">
   </p>

|

What it looks like in the terminal if the option ``--verbose`` is not used: only the list of rejected texts is shown after the
filtering is completed

.. raw:: html

   <p align="left"><img src="https://github.com/raul23/clustering-text/blob/main/images/filtering_no_verbose.png">
   </p>

`:information_source:` You will see in my list of ebooks that the text from the ebook ``abstract algebra.pdf`` was rejected even though it
is from an English mathematics ebook. ``pycld2`` detected the text as not being in English because the text conversion (``pdftotext``) didn't 100% succeeded and introduced too many odd characters (e.g. ``0ß Å ÞBð``) mixed with english words. It seems that it is the only ebook over 153 converted documents that has this problem.
