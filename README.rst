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
I put the results section at the top before explaining the `script <#script-classify-ebooks-py>`_ since it is the most important and interesting part
of this document.

Thus without further ado, here are the results from training a ``RidgeClassifier`` on the dataset of 129 documents with three categories (computer_science, mathematics, physics). 

The train and test sets splitted as follows:

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
