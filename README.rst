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
