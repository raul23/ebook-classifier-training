=====================================
Experimenting with classifying ebooks
=====================================
.. contents:: **Contents**
   :depth: 4
   :local:
   :backlinks: top

I am basing my experimentation with classifying text on the excellent scikit-learn's tutorial: `Classification of text documents using sparse features <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html>`_.

I am following along their tutorial but using my own datasets containing a bunch of ebooks (``pdf``, ``djvu``).

Introduction
============
I will be using different datasets of ebooks to test text classification. One of them is a small dataset that consists of 
129 English ebooks (``pdf`` and ``djvu``) from 3 categories:

- ``computer_science`` with label 0 and 48 ebooks
- ``mathematics`` with label 1 and 50 ebooks
- ``physics`` with label 2 and 31 ebooks

By default, only 10% of a given ebook is `converted to text <#dataset-generation>`_ and added to the dataset. Also if an ebook is 
made of images, `OCR <#ocr>`_ is applied on 5 pages chosen randomly in the first 50% of the given ebook to extract the text.

Also as in the scikit-learn's `tutorial <https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html>`_,
multiple clasification models will be used such as: ``RidgeClassifier``, ``LogisticRegression``, and ``ComplementNB``.

Script ``classify_ebooks.py``
=============================
Dependencies
------------
TODO

Script options
--------------
TODO

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

Results of classifying ebooks
=============================
Here are the results from training a ``RidgeClassifier`` on the dataset of 129 documents with three categories (computer_science,
mathematics, physics) and the train and test sets splitted as follows:

- train data: 77 ebooks (60%)
- test data: 52 ebooks (40%)

.. raw:: html

   <p align="center"><img src="./images/confusion_matrix_ridgeclass_small_dataset.png">
   </p>

`:information_source:` Insights from the confusion matrix for ``RidgeClassifier``

- The confusion matrix is plotted based on the predictions from the test set.
- Among the three categories, this linear model has the most difficulties with the physics category. It confused two physics ebooks for mathematics documents which is to be expected since both domains share overlaps between words. The vice-versa situation is not found, i.e. no mathematics documents were incorrectly classified as physics ones which is to be expected since ebooks about physics use a more specific vocabulary than for mathematics documents.
- Mathematics ebooks are well classified but one such document was classified as a computer science document. 
- The computer science category is the one that ``RidgeClassifier`` has the most success in classifying with all computer science ebooks being correctly classified as such. It means that computer science documents all share more specific vocabulary than the one for the other two domains (mathematics and physics).

|
 
.. raw:: html

   <p align="center"><img src="./images/average_feature_effect_small_dataset.png">
   </p>
 
