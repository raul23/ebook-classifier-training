=====================================
Script ``train_classifier.py``
=====================================
.. contents:: **Contents**
   :depth: 3
   :local:
   :backlinks: top

------------
Dependencies
------------
This is the environment on which the script `train_classifier.py <./scripts/train_classifier.py>`_ was tested:

* **Platform:** macOS
* **Python**: version **3.8**
* `matplotlib <https://matplotlib.org/>`_: **v3.7.2** for generating graphs
* `numpy <https://numpy.org/>`_: **v1.24.3**, for "array processing for numbers, strings, records, and objects"
* `pandas <https://pandas.pydata.org/>`_: **v2.0.3**, "High-performance, easy-to-use data structures and data analysis tool" 
* `pycld2 <https://github.com/aboSamoor/pycld2>`_: **v0.41**, for detecting the language of a given ebook in order to keep 
  books based on a chosen language
* `regex <https://pypi.org/project/regex/>`_: **v2023.12.25**, "this regex implementation is backwards-compatible with 
  the standard ``re`` module, but offers additional functionality"
* `scikit-learn <https://scikit-learn.org/>`_: **v1.3.0**, "a set of python modules for machine learning and data mining"

|

`:star:` **Other dependencies**

You also need recent version of:

-  `poppler <https://poppler.freedesktop.org/>`_ (including ``pdftotext``) and `DjVuLibre <http://djvu.sourceforge.net/>`_ (including ``djvutxt``)
   can be installed for conversion of ``.pdf`` and ``.djvu`` files to ``.txt``, respectively.

Optionally:

- `diskcache <http://www.grantjenks.com/docs/diskcache/>`_: **v5.6.3** for caching persistently the converted files into ``txt``
- `Ghostscript <https://www.ghostscript.com/>`_ for converting ``pdf`` to ``png`` when applying OCR on a given document.
- `nltk <https://www.nltk.org/>`_: **v3.8.1** for detecting the language of a given ebook
- `Tesseract <https://github.com/tesseract-ocr/tesseract>`_ for running OCR on books - version 4 gives 
  better results. OCR is disabled by default since it is a slow resource-intensive process.

--------------
Script options
--------------
To display the script's list of options and their description::

 $ python train_classifier.py -h
 usage: python train_classifier.py [OPTIONS] {input_directory}

I won't list all options (too many) but here are some of the important and interesting ones:

**Cache options:**

-u                                     Highly recommended to use cache to speed up **dataset re-creation**.

**OCR options:**

-o                                     Whether to enable OCR for ``pdf``, ``djvu`` and image files. It is disabled by default. (default: false)

**Dataset options:**

--cd                                  Create dataset with text from ebooks found in the directory.
--ud                                  Update dataset with text from more new ebooks found in the directory.
--cat CATEGORY                        Only include these categories in the dataset. (default: None)  
--vect-params PARAMS                  The parameters to be used by TfidfVectorizer for vectorizing the dataset. 
                                      (default: max_df=0.5 min_df=5 ngram_range='(1, 1)' norm=l2)

**Benchmarking options:**

-b                                     Benchmarking classifiers.

**Hyperparameter tuning options:**

--hyper-tuning                         Perform hyperparameter tuning.
--clfs CLF                             The names of classifiers whose hyperparameters will be tuned with grid search.
                                       (default: RidgeClassifier ComplementNB)

**Classification options:**

--clf CLF_PARAMS                       The name of the classifier along with its parameters to be used for classifying ebooks. 
                                       (default: RidgeClassifier tol=1e-2 solver=sparse_cg)

|

`:information_source:` Explaining some important and interesting options/arguments

- ``input_directory`` is the path to the main directory containing the documents to classify.

  The following options require to specify an ``input_directory``:
  
  - ``--hyper-tuning``: hyperparameter tuning
  - ``-b``: benchmarking
- ``-b`` uses right now **hard-coded parameter** values for multiple classifiers. However, I will eventually
  make it possible to upload a JSON file with custom parameter values for different classifiers when
  using this option (TODO).
- By **dataset re-creation** I mean the case when you delete the pickle dataset file and generate the dataset 
  again. If you are using cache, then the dataset generation should be quick since the text conversions were
  already computed and cached. Using the option ``-u`` is worthwhile especially if you used OCR for some of the ebooks since this procedure is very
  resource intensive and can take awhile if many pages are OCRed.
- ``--vect-params PARAMS [PARAMS ...]``: the parameters for ``TfidfVectorizer`` are given one after the other like this::

   --vect-params max_df=0.2 min_df=1 ngram_range='(1,1)' norm=l2
   
  `:warning:` It is important to escape any parentheses on the terminal by placing them within single quotes or after a backslash
  (e.g. ``ngram_range=\(1,1\)``).
- ``--clfs [CLF [CLF ...]]``: the names of the classifiers are those used in scikit-learn's modules. For example::

   python train_classifier.py ~/Data/ebooks --hyper-tune --clfs KNeighborsClassifier NearestCentroid LogisticRegression
   
- ``--clf CLF_PARAMS``: the name of the classifier and its parameters are the ones used in scikit-learn's modules. For example::
  
   python train_classifier.py ~/Data/ebooks --clf KNeighborsClassifier n_neighbors=5
- The choices for ``-o`` are ``{always, true, false}``
  
  - 'always': always use OCR first when doing text conversion. If the converson fails, then use the other simpler conversion tools
    (``pdftotext`` and ``djvutxt``).
  - 'true': first simpler conversion tools (``pdftotext`` and ``djvutxt``) will be used and then if a conversion method
    failed to convert an ebook to ``txt`` or resulted in an empty file, the OCR method will be used.
  - 'false': never use OCR, only use the other simpler conversion tools (``pdftotext`` and ``djvutxt``).

Start the training of the ebook classifier ⭐
---------------------------------------------
To **quickly** start the training of the ebook classifier, all you need is to provide the directory containing said ebooks::

 python train_classifier.py ~/Data/ebooks
 
The script will generate the dataset and then train the default classifier (``RidgeClassifier``) and 
display the confusion matrix and features effects graph.

To specify a classifier with its parameters, use the ``--clf`` option::

 python train_classifier.py ~/Data/ebooks --clf 

Cache options
-------------
`:information_source:` About the caching option (``--use-cache``) supported by the script ``train_classifier.py.py``

- Cache is used to save the converted ebook files into ``txt`` to
  avoid re-converting them which can be a time consuming process. 
  `DiskCache <http://www.grantjenks.com/docs/diskcache/>`_, a disk and file 
  backed cache library, is used by the ``train_classifier.py.py`` script.
- Default cache folder used: ``~/.train_ebook_classifier``
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

* When enabling the cache with the flag ``--use-cache``, the ``train_classifier.py`` 
  script has to cache the converted ebooks (``txt``) if they were
  not already saved in previous runs. Therefore, the speed up of some of the
  tasks (dataset re-creation and updating) will be seen in subsequent executions of the 
  script.
* Keep in mind that caching has its caveats. For instance if a given ebook
  is modified (e.g. a page is deleted) then the ``train_classifier.py`` 
  script has to run the text conversion again since the keys in the cache are the MD5 hashes of
  the ebooks.
* There is no problem in the
  cache growing without bounds since its size is set to a maximum of 1 GB by
  default (check the ``--cache-size-limit`` option) and its eviction policy
  determines what items get to be evicted to make space for more items which
  by default it is the least-recently-stored eviction policy (check the
  ``--eviction-policy`` option).

Dataset options
---------------

Ebooks directory
****************
`:warning:` In order to run the script `train_classifier.py <./scripts/train_classifier.py>`_, you need first to have a main directory (e.g. ``./ebooks/``) with all the ebooks (``pdf`` and ``djvu``) you want to test classification on. Each ebook should be in a folder whose name should correspond to the category of said ebook.

For example:

- ../ebooks/**biology**/Cell theory.djvu
- ../ebooks/**philosophy**/History of Philosophy in Europe.pdf
- ../ebooks/**physics**/Electricity.pdf

Then, you need to give the path to the main directory to the script, like this::

 $ python train_classifier.py ~/Data/ebooks/
 
The next section explains in details the generation of a dataset containing text from these ebooks.

Dataset creation
****************
To start creating a dataset containing texts from ebooks after you have setup your `directory of ebooks <#ebooks-directory>`_, the option
``--cd`` and the input directory are necessary::

 $ python train_classifier.py --cd ~/Data/ebooks/
 
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

 $ python train_classifier.py --cd -u ~/Data/ebooks/

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

Updating a dataset
******************
After a dataset is created and saved, you can update it with new texts from more ebooks by using the ``--ud`` option::

 $ python train_classifier.py --ud ~/Data/ebooks/

.. raw:: html

   <p align="left"><img src="https://github.com/raul23/clustering-text/blob/main/images/updating_dataset_ocr.png">
   </p>
   
`:information_source:` ``--ud`` tells the script to update the dataset pickle file saved within the main ebooks directory (e.g. ``~/Data/ebooks``).

Filtering a dataset: select texts only in English and from valid categories
***************************************************************************
After the dataset containing texts from ebooks is generated, you can launch the classification by providing only the input directory
containing the saved pickle file of the dataset. During the text classification, the dataset is loaded and filtered by removing 
text that is not English and not part of the specified categories (e.g. ``computer_science``, ``mathematics``, ``physics``).

Here are some samples of output from the script ``train_classifier.py``::

 python train_classifier.py ~/Data/ebooks/ --verbose
 
`:information_source:` Explaining the options:

- Since the option ``--verbose`` is used, you will see more information printed in the terminal such as
  if the text is in English or its category.
- By default, the three mentioned categories are choosen. But you can control the categories you want to include in the filtered dataset with the
  ``--cat`` option::

   python train_classifier.py -u ~/Data/ebooks/ --cat chemistry physics

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

OCR options
-----------
For those ebooks that couldn't be converted to ``txt`` with simpler methods (``pdftotext`` and ``djvutxt``), 
you can update the dataset using the  options ``--ud`` (update) and ``-o true`` (enable OCR)::

 $ python train_classifier.py -u --ud -o true ~/Data/ebooks/

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
