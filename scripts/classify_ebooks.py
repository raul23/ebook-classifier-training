# This script is based on scikit-learn's tutorial: Clustering text documents using k-means
# Ref: https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
import argparse
import ast
import hashlib
import logging
import mimetypes
import os
import pickle
import random
import re
import shlex
import shutil
import subprocess
import tempfile
from argparse import Namespace
from pathlib import Path
from time import time
from types import SimpleNamespace


import matplotlib.pyplot as plt
import numpy as np
import pycld2
import regex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import Bunch

Cache = None
nltk = None
RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")

logger = logging.getLogger('clustering')
__version__ = '0.1'
_DEFAULT_MSG = ' (default: {})'

ENGLISH_VOCAB = None

# =====================
# Default config values
# =====================

# Misc options
# ============
QUIET = False
SEED = 123456

# Cache options
# =============
USE_CACHE = True
CACHE_FOLDER = os.path.expanduser('~/.classify_ebooks')
EVICTION_POLICY = 'least-recently-stored'
# In gigabytes (GiB)
CACHE_SIZE_LIMIT = 1
CLEAR_CACHE = False
REMOVE_KEYS = None
CHECK_NUMBER_ITEMS = False

# Dataset options
# ===============
UPDATE_DATASET = False
CATEGORIES = ['computer_science', 'mathematics', 'physics']

# Logging options
# ===============
LOGGING_FORMATTER = 'only_msg'
LOGGING_LEVEL = 'info'

# OCR options
# ===========
OCR_ENABLED = 'false'
OCR_ONLY_RANDOM_PAGES = 5
OCR_COMMAND = 'tesseract_wrapper'

# convert_to_txt options
# ======================
# Some of the general options affect this command's behavior a lot, especially
# the OCR ones
OUTPUT_FILE = 'output.txt'
CONVERT_ONLY_PERCENTAGE_EBOOK = 10
DJVU_CONVERT_METHOD = 'djvutxt'
EPUB_CONVERT_METHOD = 'epubtxt'
MSWORD_CONVERT_METHOD = 'textutil'
PDF_CONVERT_METHOD = 'pdftotext'


class RandomModel:

    def __init__(self, n_clusters):
        self.random_state = 12345
        self.labels_ = None
        self.n_clusters = n_clusters

    def fit(self, X):
        self.labels_ = np.random.randint(0, self.n_clusters, X.shape[0])

    def set_params(self, random_state):
        self.random_state = random_state


class ArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        print_(self.format_usage().splitlines()[0])
        self.exit(2, red(f'\nerror: {message}\n'))


class MyFormatter(argparse.HelpFormatter):
    """
    Corrected _max_action_length for the indenting of subactions
    """

    def add_argument(self, action):
        if action.help is not argparse.SUPPRESS:

            # find all invocations
            get_invocation = self._format_action_invocation
            invocations = [get_invocation(action)]
            current_indent = self._current_indent
            for subaction in self._iter_indented_subactions(action):
                # compensate for the indent that will be added
                indent_chg = self._current_indent - current_indent
                added_indent = 'x' * indent_chg
                invocations.append(added_indent + get_invocation(subaction))
            # print_('inv', invocations)

            # update the maximum item length
            invocation_length = max([len(s) for s in invocations])
            action_length = invocation_length + self._current_indent
            self._action_max_length = max(self._action_max_length,
                                          action_length)

            # add the item to the list
            self._add_item(self._format_action, [action])

    # Ref.: https://stackoverflow.com/a/23941599/14664104
    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            # change to
            #    -s, --long ARGS
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    # parts.append('%s %s' % (option_string, args_string))
                    parts.append('%s' % option_string)
                parts[-1] += ' %s'%args_string
            return ', '.join(parts)


class OptionsChecker:
    def __init__(self, add_opts, remove_opts):
        self.add_opts = init_list(add_opts)
        self.remove_opts = init_list(remove_opts)

    def check(self, opt_name):
        return not self.remove_opts.count(opt_name) or \
               self.add_opts.count(opt_name)


class Result:
    def __init__(self, stdout='', stderr='', returncode=None, args=None):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.args = args

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'stdout={self.stdout}, stderr={self.stderr}, ' \
               f'returncode={self.returncode}, args={self.args}'


# ------
# Colors
# ------
COLORS = {
    'GREEN': '\033[0;36m',  # 32
    'RED': '\033[0;31m',
    'YELLOW': '\033[0;33m',  # 32
    'BLUE': '\033[0;34m',  #
    'VIOLET': '\033[0;35m',  #
    'BOLD': '\033[1m',
    'NC': '\033[0m',
}
_COLOR_TO_CODE = {
    'g': COLORS['GREEN'],
    'r': COLORS['RED'],
    'y': COLORS['YELLOW'],
    'b': COLORS['BLUE'],
    'v': COLORS['VIOLET'],
    'bold': COLORS['BOLD']
}


def color(msg, msg_color='y', bold_msg=False):
    msg_color = msg_color.lower()
    colors = list(_COLOR_TO_CODE.keys())
    assert msg_color in colors, f'Wrong color: {msg_color}. Only these ' \
                                f'colors are supported: {msg_color}'
    msg = bold(msg) if bold_msg else msg
    msg = msg.replace(COLORS['NC'], COLORS['NC']+_COLOR_TO_CODE[msg_color])
    return f"{_COLOR_TO_CODE[msg_color]}{msg}{COLORS['NC']}"


def blue(msg):
    return color(msg, 'b')


def bold(msg):
    return color(msg, 'bold')


def green(msg):
    return color(msg, 'g')


def red(msg):
    return color(msg, 'r')


def violet(msg):
    return color(msg, 'v')


def yellow(msg):
    return color(msg)


# General options
def add_general_options(parser, add_opts=None, remove_opts=None,
                        program_version=__version__,
                        title='General options'):
    checker = OptionsChecker(add_opts, remove_opts)
    parser_general_group = parser.add_argument_group(title=title)
    if checker.check('help'):
        parser_general_group.add_argument('-h', '--help', action='help',
                                          help='Show this help message and exit.')
    if checker.check('version'):
        parser_general_group.add_argument(
            '-v', '--version', action='version',
            version=f'%(prog)s v{program_version}',
            help="Show program's version number and exit.")
    if checker.check('quiet'):
        parser_general_group.add_argument(
            '-q', '--quiet', action='store_true',
            help='Enable quiet mode, i.e. nothing will be printed.')
    if checker.check('verbose'):
        parser_general_group.add_argument(
            '--verbose', action='store_true',
            help='Print various debugging information, e.g. print traceback '
                 'when there is an exception.')
    if checker.check('log-level'):
        parser_general_group.add_argument(
            '--log-level', dest='logging_level',
            choices=['debug', 'info', 'warning', 'error'], default=LOGGING_LEVEL,
            help='Set logging level.' + get_default_message(LOGGING_LEVEL))
    if checker.check('log-format'):
        parser_general_group.add_argument(
            '--log-format', dest='logging_formatter',
            choices=['console', 'only_msg', 'simple',], default=LOGGING_FORMATTER,
            help='Set logging formatter.' + get_default_message(LOGGING_FORMATTER))
    return parser_general_group


def catdoc(input_file, output_file):
    cmd = f'catdoc "{input_file}"'
    args = shlex.split(cmd)
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Everything on the stdout must be copied to the output file
    if result.returncode == 0:
        with open(output_file, 'w') as f:
            f.write(result.stdout)
    return convert_result_from_shell_cmd(result)


# Ref.: https://stackoverflow.com/a/28909933
def command_exists(cmd):
    return shutil.which(cmd) is not None


def convert_result_from_shell_cmd(old_result):
    new_result = Result()

    for attr_name, new_val in new_result.__dict__.items():
        old_val = getattr(old_result, attr_name)
        if old_val is None:
            shell_args = getattr(old_result, 'args', None)
            logger.debug(f'result.{attr_name} is None. Shell args: {shell_args}')
        else:
            if isinstance(new_val, str):
                try:
                    new_val = old_val.decode('UTF-8')
                except (AttributeError, UnicodeDecodeError) as e:
                    if type(e) == UnicodeDecodeError:
                        # old_val = b'...'
                        new_val = old_val.decode('unicode_escape')
                    else:
                        # `old_val` already a string
                        # logger.debug('Error decoding old value: {}'.format(old_val))
                        # logger.debug(e.__repr__())
                        # logger.debug('Value already a string. No decoding necessary')
                        new_val = old_val
                try:
                    new_val = ast.literal_eval(new_val)
                except (SyntaxError, ValueError) as e:
                    # NOTE: ValueError might happen if value consists of [A-Za-z]
                    # logger.debug('Error evaluating the value: {}'.format(old_val))
                    # logger.debug(e.__repr__())
                    # logger.debug('Aborting evaluation of string. Will consider
                    # the string as it is')
                    pass
            else:
                new_val = old_val
        setattr(new_result, attr_name, new_val)
    return new_result


def convert(input_file, output_file=None,
            djvu_convert_method=DJVU_CONVERT_METHOD,
            epub_convert_method=EPUB_CONVERT_METHOD,
            msword_convert_method=MSWORD_CONVERT_METHOD,
            ocr_command=OCR_COMMAND,
            ocr_enabled=OCR_ENABLED,
            ocr_only_random_pages=OCR_ONLY_RANDOM_PAGES,
            pdf_convert_method=PDF_CONVERT_METHOD, **kwargs):
    # also setup cache outside
    func_params = locals().copy()
    statuscode = 0
    check_conversion = True
    file_hash = None
    mime_type = get_mime_type(input_file)
    if mime_type == 'text/plain':
        logger.debug('The file is already in .txt')
        with open(input_file, 'r') as f:
            text = f.read()
        return text
    return_txt = False
    if output_file is None:
        return_txt = True
        output_file = tempfile.mkstemp(suffix='.txt')[1]
    else:
        output_file = Path(output_file)
        # Check first that the output text file is valid
        if output_file.suffix != '.txt':
            logger.error(red("[ERROR] The output file needs to have a .txt extension!"))
            return 1
        # Create output file text if it doesn't exist
        if output_file.exists():
            logger.info(f"Output text file already exists: {output_file.name}")
            logger.debug(f"Full path of output text file: '{output_file.absolute()}'")
        else:
            # Create output text file
            touch(output_file)
    func_params['mime_type'] = mime_type
    func_params['output_file'] = output_file
    # check_conversion = False
    if ocr_enabled == 'always':
        logger.info("OCR=always, first try OCR then conversion")
        if ocr_file(input_file, output_file, mime_type, ocr_command, ocr_only_random_pages):
            logger.info(f"{COLORS['YELLOW']}OCR failed!{COLORS['NC']} Will try conversion...")
            result = convert_to_txt(**func_params)
            statuscode = result.returncode
        else:
            # logger.info("OCR successful!")
            statuscode = 0
    elif ocr_enabled == 'true':
        logger.info("OCR=true, first try conversion and then OCR...")
        # Check if valid converted text file
        result = convert_to_txt(**func_params)
        statuscode = result.returncode
        if statuscode == 0 and isalnum_in_file(output_file):
            logger.info("Conversion terminated, will not try OCR")
            check_conversion = False
        else:
            logger.info("{COLORS['YELLOW']}Conversion failed!{COLORS['NC']} Will try OCR...")
            if ocr_file(input_file, output_file, mime_type, ocr_command, ocr_only_random_pages):
                logger.warning(yellow("OCR failed!"))
                logger.warning(f"{COLORS['YELLOW']}File couldn't be converted to txt:{COLORS['NC']} {input_file}")
                remove_file(output_file)
                return 1
            else:
                # logger.info("OCR successful!")
                statuscode = 0
    else:
        # ocr_enabled = 'false'
        logger.info("OCR=false, try only conversion...")
        result = convert_to_txt(**func_params)
        statuscode = result.returncode
        if statuscode == 0:
            logger.info('Conversion terminated')
    # Check conversion
    logger.debug('Checking converted text...')
    if check_conversion:
        if statuscode == 0 and isalnum_in_file(output_file):
            logger.debug("Converted text is valid!")
        else:
            logger.warning(yellow("[WARNING] Conversion failed!"))
            if not isalnum_in_file(output_file):
                logger.warning(yellow(f'[WARNING] The converted txt with size {os.stat(output_file).st_size} '
                                      'bytes does not seem to contain text'))
            remove_file(output_file)
            return 1
    assert statuscode == 0
    if return_txt:
        with open(output_file, 'r', encoding="utf8", errors='ignore') as f:
            text = f.read()
        assert text
        remove_file(output_file)
        return text
    else:
        return 0


# Tries to convert the supplied ebook file into .txt. It uses calibre's
# ebook-convert tool. For optimization, if present, it will use pdftotext
# for pdfs, catdoc for word files and djvutxt for djvu files.
# Ref.: https://bit.ly/2HXdf2I
def convert_to_txt(input_file, output_file, mime_type,
                   convert_only_percentage_ebook=CONVERT_ONLY_PERCENTAGE_EBOOK,
                   djvu_convert_method=DJVU_CONVERT_METHOD,
                   epub_convert_method=EPUB_CONVERT_METHOD,
                   msword_convert_method=MSWORD_CONVERT_METHOD,
                   pdf_convert_method=PDF_CONVERT_METHOD, **kwargs):
    if mime_type.startswith('image/vnd.djvu') \
         and djvu_convert_method == 'djvutxt' and command_exists('djvutxt'):
        logger.debug('The file looks like a djvu, using djvutxt to extract the text')
        last_page = int(convert_only_percentage_ebook * get_pages_in_djvu(input_file).stdout/100)
        result = djvutxt(input_file, output_file, pages=f'1-{last_page}')
    elif mime_type.startswith('application/epub+zip') \
            and epub_convert_method == 'epubtxt' and command_exists('unzip'):
        logger.debug('The file looks like an epub, using epub2txt to extract the text')
        result = epubtxt(input_file, output_file)
    elif mime_type == 'application/msword' \
            and msword_convert_method in ['catdoc', 'textutil'] \
            and (command_exists('catdoc') or command_exists('textutil')):
        msg = 'The file looks like a doc, using {} to extract the text'
        if command_exists('catdoc'):
            logger.debug(msg.format('catdoc'))
            result = catdoc(input_file, output_file)
        else:
            logger.debug(msg.format('textutil'))
            result = textutil(input_file, output_file)
    elif mime_type == 'application/pdf' and pdf_convert_method == 'pdftotext' \
            and command_exists('pdftotext'):
        logger.debug('The file looks like a pdf, using pdftotext to extract the text')
        last_page = int(convert_only_percentage_ebook * get_pages_in_pdf(input_file).stdout/100)
        result = pdftotext(input_file, output_file, first_page_to_convert=1, last_page_to_convert=last_page)
    elif (not mime_type.startswith('image/vnd.djvu')) \
            and mime_type.startswith('image/'):
        msg = f'The file looks like a normal image ({mime_type}), skipping ' \
              'ebook-convert usage!'
        logger.debug(msg)
        return convert_result_from_shell_cmd(Result(stderr=msg, returncode=1))
    else:
        logger.debug(f"Trying to use calibre's ebook-convert to convert the {mime_type} file to .txt")
        result = ebook_convert(input_file, output_file)
    return result


def djvutxt(input_file, output_file, pages=None):
    pages = f'--page={pages}' if pages else ''
    cmd = f'djvutxt "{input_file}" "{output_file}" {pages}'
    args = shlex.split(cmd)
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return convert_result_from_shell_cmd(result)


def ebook_convert(input_file, output_file):
    cmd = f'ebook-convert "{input_file}" "{output_file}"'
    args = shlex.split(cmd)
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return convert_result_from_shell_cmd(result)


def epubtxt(input_file, output_file):
    cmd = f'unzip -c "{input_file}"'
    args = shlex.split(cmd)
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not result.stderr:
        text = str(result.stdout)
        with open(output_file, 'w') as f:
            f.write(text)
        result.stdout = text
    return convert_result_from_shell_cmd(result)


def get_default_message(default_value):
    return green(f' (default: {default_value})')


def get_file_size(file_path):
    """
    This function will return the file size in MB
    Ref.: https://stackoverflow.com/a/39988702
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return file_info.st_size/int(1e6)
    else:
        logger.error(f"'{file_path}' is not a file\nAborting get_file_size()")
        return None


# Ref.: https://stackoverflow.com/a/59056837/14664104
def get_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()


# Using Python built-in module mimetypes
def get_mime_type(file_path):
    return mimetypes.guess_type(file_path)[0]


# Return number of pages in a djvu document
def get_pages_in_djvu(file_path):
    cmd = f'djvused -e "n" "{file_path}"'
    args = shlex.split(cmd)
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return convert_result_from_shell_cmd(result)


# Return number of pages in a pdf document
def get_pages_in_pdf(file_path, cmd='mdls'):
    assert cmd in ['mdls', 'pdfinfo']
    if command_exists(cmd) and cmd == 'mdls':
        cmd = f'mdls -raw -name kMDItemNumberOfPages "{file_path}"'
        args = shlex.split(cmd)
        result = subprocess.run(args, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        if '(null)' in str(result.stdout):
            return get_pages_in_pdf(file_path, cmd='pdfinfo')
    else:
        cmd = f'pdfinfo "{file_path}"'
        args = shlex.split(cmd)
        result = subprocess.run(args, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        if result.returncode == 0:
            result = convert_result_from_shell_cmd(result)
            result.stdout = int(re.findall('^Pages:\s+([0-9]+)',
                                           result.stdout,
                                           flags=re.MULTILINE)[0])
            return result
    return convert_result_from_shell_cmd(result)


def import_modules(english_detector='pycld2', use_cache=False):
    global Cache, ENGLISH_VOCAB, nltk
    if english_detector == 'nltk':
        logger.debug('importing nltk')
        import nltk
        ENGLISH_VOCAB = set(w.lower() for w in nltk.corpus.words.words())
    if use_cache:
        logger.debug('importing diskcache')
        from diskcache import Cache


def init_list(list_):
    return [] if list_ is None else list_


def is_text_english(text, method='pycld2', threshold=25):
    assert method in ['nltk', 'pycld2']
    if method == 'pycld2':
        _, _, details = pycld2.detect(text)
        guess_lang_name = details[0][0]  # Full language name in capital, e.g. 'ENGLISH
        if guess_lang_name == 'ENGLISH':
            return True
        else:
            return False
    else:
        text = text.split()
        text_vocab = set(w.lower() for w in text if w.lower().isalpha())
        unusual = text_vocab.difference(ENGLISH_VOCAB)
        prop_unusual = len(unusual) / len(text_vocab)
        msg = f'{round(prop_unusual*100)}% of words in the text vocabulary are unusual (threshold = {threshold}%)'
        if prop_unusual * 100 > threshold:
            logger.debug(f'The text is classified as non-english: {msg}')
            return False
        else:
            logger.debug(f'The text is classified as english: {msg}')
            return True


def isalnum_in_file(file_path):
    with open(file_path, 'r', encoding="utf8", errors='ignore') as f:
        isalnum = False
        for line in f:
            for ch in line:
                if ch.isalnum():
                    isalnum = True
                    break
            if isalnum:
                break
    return isalnum


def namespace_to_dict(ns):
    namspace_classes = [Namespace, SimpleNamespace]
    if type(ns) in namspace_classes:
        adict = vars(ns)
    else:
        adict = ns
    for k, v in adict.items():
        # if isinstance(v, SimpleNamespace):
        if type(v) in namspace_classes:
            v = vars(v)
            adict[k] = v
        if isinstance(v, dict):
            namespace_to_dict(v)
    return adict


# OCR on a pdf, djvu document or image
# NOTE: If pdf or djvu document, then first needs to be converted to image and then OCR
def ocr_file(file_path, output_file, mime_type,
             ocr_command=OCR_COMMAND,
             ocr_only_random_pages=OCR_ONLY_RANDOM_PAGES, **kwargs):
    # Convert pdf to png image
    def convert_pdf_page(page, input_file, output_file):
        cmd = f'gs -dSAFER -q -r300 -dFirstPage={page} -dLastPage={page} ' \
              '-dNOPAUSE -dINTERPOLATE -sDEVICE=png16m ' \
              f'-sOutputFile="{output_file}" "{input_file}" -c quit'
        args = shlex.split(cmd)
        result = subprocess.run(args, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        return convert_result_from_shell_cmd(result)

    # Convert djvu to tif image
    def convert_djvu_page(page, input_file, output_file):
        cmd = f'ddjvu -page={page} -format=tif "{input_file}" "{output_file}"'
        args = shlex.split(cmd)
        result = subprocess.run(args, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        return convert_result_from_shell_cmd(result)

    if mime_type.startswith('application/pdf'):
        result = get_pages_in_pdf(file_path)
        num_pages = result.stdout
        logger.debug(f"Result of '{get_pages_in_pdf.__repr__()}' on '{file_path}':\n{result}")
        page_convert_cmd = convert_pdf_page
    elif mime_type.startswith('image/vnd.djvu'):
        result = get_pages_in_djvu(file_path)
        num_pages = result.stdout
        logger.debug(f"Result of '{get_pages_in_djvu.__repr__()}' on '{file_path}':\n{result}")
        page_convert_cmd = convert_djvu_page
    elif mime_type.startswith('image/'):
        logger.debug(f"Running OCR on file '{file_path}' and with mime type '{mime_type}'...")
        if ocr_command in globals():
            result = eval(f'{ocr_command}("{file_path}", "{output_file}")')
            logger.debug(f"Result of '{ocr_command.__repr__()}':\n{result}")
            return 0
        else:
            logger.debug(f"Function '{ocr_command}' doesn't exit. Ending ocr.")
            return 1
    else:
        logger.info(f"Unsupported mime type '{mime_type}'!")
        return 2

    if ocr_command not in globals():
        logger.debug(f"Function '{ocr_command}' doesn't exit. Ending ocr.")
        return 1

    logger.debug(f"Will run OCR on file '{file_path}' with {num_pages} page{'s' if num_pages > 1 else ''}...")
    logger.debug(f'mime type: {mime_type}')

    # Pre-compute the list of pages to process based on ocr_only_random_pages
    if ocr_only_random_pages:
        pages_to_process = sorted(random.sample(range(1, int(0.5*num_pages)), ocr_only_random_pages))
    else:
        # `ocr_only_random_pages` is False
        logger.debug('ocr_only_random_pages is False')
        pages_to_process = [i for i in range(1, num_pages+1)]
    logger.debug(f'Pages to process: {pages_to_process}')

    text = ''
    for i, page in enumerate(pages_to_process, start=1):
        logger.debug(f'Processing page {i} of {len(pages_to_process)}')
        # Make temporary files
        tmp_file = tempfile.mkstemp()[1]
        tmp_file_txt = tempfile.mkstemp(suffix='.txt')[1]
        logger.debug(f'Running OCR of page {page} ...')
        logger.debug(f'Using tmp files {tmp_file} and {tmp_file_txt}')
        # doc(pdf, djvu) --> image(png, tiff)
        result = page_convert_cmd(page, file_path, tmp_file)
        logger.debug(f"Result of {page_convert_cmd.__repr__()}:\n{result}")
        # image --> text
        logger.debug(f"Running the '{ocr_command}' ...")
        result = eval(f'{ocr_command}("{tmp_file}", "{tmp_file_txt}")')
        logger.debug(f"Result of '{ocr_command.__repr__()}':\n{result}")
        with open(tmp_file_txt, 'r') as f:
            data = f.read()
            # logger.debug(f"Text content of page {page}:\n{data}")
        text += data
        # Remove temporary files
        logger.debug('Cleaning up tmp files')
        remove_file(tmp_file)
        remove_file(tmp_file_txt)

    # Everything on the stdout must be copied to the output file
    logger.debug('Saving the text content')
    with open(output_file, 'w') as f:
        f.write(text)
    return 0


def pdftotext(input_file, output_file, first_page_to_convert=None, last_page_to_convert=None):
    first_page = f'-f {first_page_to_convert}' if first_page_to_convert else ''
    last_page = f'-l {last_page_to_convert}' if last_page_to_convert else ''
    pages = f'{first_page} {last_page}'.strip()
    cmd = f'pdftotext "{input_file}" "{output_file}" {pages}'
    args = shlex.split(cmd)
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return convert_result_from_shell_cmd(result)


def plot_confusion_matrix(clf, y_test, pred, target_names):
    fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
    ax.xaxis.set_ticklabels(target_names)
    ax.yaxis.set_ticklabels(target_names)
    _ = ax.set_title(
        f"Confusion Matrix for {clf.__class__.__name__}\non the original documents"
    )
    plt.show()


def print_(msg):
    global QUIET
    if not QUIET:
        print(msg)


def read_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except OSError as e:
        raise


def remove_file(file_path):
    # Ref.: https://stackoverflow.com/a/42641792
    try:
        os.remove(file_path)
        return 0
    except OSError as e:
        logger.error(red(f'[ERROR] {e.filename} - {e.strerror}.'))
        return 1


# Ref.: https://stackoverflow.com/a/4195302/14664104
def required_length(nmin, nmax):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin <= len(values) <= nmax:
                if nmin == nmax:
                    msg = 'argument "{f}" requires {nmin} arguments'.format(
                        f=self.dest, nmin=nmin, nmax=nmax)
                else:
                    msg = 'argument "{f}" requires between {nmin} and {nmax} ' \
                          'arguments'.format(f=self.dest, nmin=nmin, nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return


def setup_argparser():
    width = os.get_terminal_size().columns - 5
    name_input = 'input_directory'
    msg = f'Classify ebooks based on the specified categories. By default, categories = {CATEGORIES}'
    parser = ArgumentParser(
        description="",
        usage=f"{COLORS['BLUE']}python %(prog)s [OPTIONS] {{{name_input}}}{COLORS['NC']}\n\n{msg}",
        add_help=False,
        formatter_class=lambda prog: MyFormatter(
            prog, max_help_position=50, width=width))
    general_group = add_general_options(
        parser,
        remove_opts=[],
        program_version=__version__,
        title=yellow('General options'))
    general_group.add_argument(
        '-s', '--seed', metavar='SEED', dest='seed', type=int, default=SEED,
        help="Seed for numpy's and Python's random generators." + get_default_message(SEED))
    # =============
    # Cache options
    # =============
    cache_group = parser.add_argument_group(title=yellow('Cache options'))
    cache_group.add_argument(
        '-u', '--use-cache', dest='use_cache', action='store_true',
        help='Use cache.')
    cache_group.add_argument(
        '-e', '--eviction-policy', dest='eviction_policy', metavar='POLICY',
        choices=['least-recently-stored', 'least-recently-used',
                 'least-frequently-used', 'none'], default=EVICTION_POLICY,
        help='Eviction policy which can either be: `least-recently-stored`, '
             '`least-recently-used`,  `least-frequently-used` or '
             '`none` (never evict keys).' + get_default_message(EVICTION_POLICY))
    cache_group.add_argument(
        '--csl', '--cache-size-limit', metavar='SIZE', dest='cache_size_limit',
        default=CACHE_SIZE_LIMIT, type=int,
        help='Size limit in gibibytes (GiB).'
             + get_default_message(CACHE_SIZE_LIMIT))
    mutual_cache_group = cache_group.add_mutually_exclusive_group()
    mutual_cache_group.add_argument(
        '-c', '--clear-cache', dest='clear_cache', action='store_true',
        help='Clear the cache. Be careful before using this option since everything '
             'in cache will be deleted including the text conversions.')
    mutual_cache_group.add_argument(
        '-r', '--remove-keys', metavar='KEY', dest='remove_keys',
        nargs='+', default=[],
        help='Keys (MD5 hashes of ebooks) to be removed from the cache along '
             'with the texts associated with them. Thus be careful before '
             'deleting them.')
    mutual_cache_group.add_argument(
        '-n', '--number-items', dest='check_number_items', action='store_true',
        help='Show number of items stored in cache.')
    # =======================
    # Convert-to-text options
    # =======================
    convert_group = parser.add_argument_group(title=yellow('Convert-to-text options'))
    convert_group.add_argument(
        '--cope', '--convert-only-percentage-ebook', dest='convert_only_percentage_ebook', type=int,
        metavar='PAGES', default=CONVERT_ONLY_PERCENTAGE_EBOOK,
        help='Convert this percentage of a given ebook to text.'
             + get_default_message(CONVERT_ONLY_PERCENTAGE_EBOOK))
    # ===============
    # Dataset options
    # ===============
    dataset_group = parser.add_argument_group(title=yellow('Dataset options'))
    dataset_group.add_argument(
        '--ud', '--update-dataset', dest='update_dataset', action='store_true',
        help='Update dataset with text from more new ebooks found in the directory.')
    dataset_group.add_argument(
        '--cat', '--categories', metavar='CATEGORY', dest='categories',
        nargs='+', default=None,
        help='Only include these categories in the dataset. ' + get_default_message(CATEGORIES))
    # ===========
    # OCR options
    # ===========
    ocr_group = parser.add_argument_group(title=yellow('OCR options'))
    ocr_group.add_argument(
        "-o", "--ocr-enabled", dest='ocr_enabled', default=OCR_ENABLED,
        choices=['always', 'true', 'false'],
        help='Whether to enable OCR for .pdf, .djvu and image files. It is '
             'disabled by default.' + get_default_message(OCR_ENABLED))
    ocr_group.add_argument(
        '--oorp', '--ocr-only-random-pages', dest='ocr_only_random_pages', type=int,
        metavar='PAGES', default=OCR_ONLY_RANDOM_PAGES,
        help='OCR only these number of pages chosen randomly in the first 50%% of a given ebook.'
             + get_default_message(OCR_ONLY_RANDOM_PAGES))
    # ==============
    # Input argument
    # ==============
    input_group = parser.add_argument_group(title=yellow('Input argument'))
    input_group.add_argument(
        name_input, default=None, nargs='*', action=required_length(0, 1),
        help="Path to the main directory containing the ebooks to classify.")
    return parser


def setup_log(quiet=False, verbose=False, logging_level=LOGGING_LEVEL,
              logging_formatter=LOGGING_FORMATTER):
    if not quiet:
        if verbose:
            logger.setLevel('DEBUG')
        else:
            logging_level = logging_level.upper()
            logger.setLevel(logging_level)
        # Create console handler and set level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # Create formatter
        if logging_formatter:
            formatters = {
                # 'console': '%(name)-{auto_field_width}s | %(levelname)-8s | %(message)s',
                'console': '%(asctime)s | %(levelname)-8s | %(message)s',
                'only_msg': '%(message)s',
                'simple': '%(levelname)-8s %(message)s',
                'verbose': '%(asctime)s | %(name)-{auto_field_width}s | %(levelname)-8s | %(message)s'
            }
            formatter = logging.Formatter(formatters[logging_formatter])
            # Add formatter to ch
            ch.setFormatter(formatter)
        # Add ch to logger
        logger.addHandler(ch)
        # =============
        # Start logging
        # =============
        logger.debug("Running {} v{}".format(__file__, __version__))
        logger.debug("Verbose option {}".format("enabled" if verbose else "disabled"))


def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6


# OCR: convert image to text
def tesseract_wrapper(input_file, output_file):
    # cmd = 'tesseract INPUT_FILE stdout --psm 12 > OUTPUT_FILE || exit 1
    cmd = f'tesseract "{input_file}" stdout --psm 12'
    args = shlex.split(cmd)
    result = subprocess.run(args,
                            stdout=open(output_file, 'w'),
                            stderr=subprocess.PIPE,
                            encoding='utf-8',
                            bufsize=4096)
    return convert_result_from_shell_cmd(result)


# macOS equivalent for catdoc
# See https://stackoverflow.com/a/44003923/14664104
def textutil(input_file, output_file):
    cmd = f'textutil -convert txt "{input_file}" -output "{output_file}"'
    args = shlex.split(cmd)
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return convert_result_from_shell_cmd(result)


def touch(path, mode=0o666, exist_ok=True):
    logger.debug(f"Creating file: '{path}'")
    Path(path).touch(mode, exist_ok)
    logger.debug("File created!")


class DatasetManager:

    def __init__(self, **kwargs):
        self.verbose = False
        self.input_directory = None
        self.seed = 123456
        # Cache options
        self.use_cache = USE_CACHE
        self.cache_folder = CACHE_FOLDER
        self.eviction_policy = EVICTION_POLICY
        self.cache_size_limit = CACHE_SIZE_LIMIT
        self.clear_cache = CLEAR_CACHE
        self.remove_keys = REMOVE_KEYS
        self.check_number_items = CHECK_NUMBER_ITEMS
        # convert-to-text options
        self.convert_only_percentage_ebook = CONVERT_ONLY_PERCENTAGE_EBOOK
        self.djvu_convert_method = DJVU_CONVERT_METHOD
        self.epub_convert_method = EPUB_CONVERT_METHOD
        self.msword_convert_method = MSWORD_CONVERT_METHOD
        self.pdf_convert_method = PDF_CONVERT_METHOD
        # Dataset options
        self.update_dataset = UPDATE_DATASET
        # OCR options
        self.ocr_enabled = OCR_ENABLED
        self.ocr_only_random_pages = OCR_ONLY_RANDOM_PAGES
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        self.dataset = Bunch(data=[], filenames=[], target_names=[], target=[], target_name_to_value={}, DESCR="")
        self.dataset_tmp = None
        self.ebook_formats = ['djvu', 'pdf']
        if isinstance(self.input_directory, list):
            self.input_directory = Path(self.input_directory[0])
        self.dataset_path = self.input_directory.joinpath(f'dataset_ebooks_text.pkl')
        np.random.seed(self.seed)
        random.seed(self.seed)
        # Setup cache
        cache_size_bytes = round(self.cache_size_limit * float(1 << 30), 2)
        if self.use_cache:
            self.cache = Cache(directory=self.cache_folder,
                               eviction_policy=self.eviction_policy,
                               size_limit=cache_size_bytes)
        else:
            self.cache = None
        # Dataset generation/updating/loading
        generate_dataset = True
        if self.update_dataset and self.dataset_path.exists():
            logger.info(blue('Updating dataset ...'))
            logger.info("Loading dataset ...")
            self._load_dataset()
        elif not self.update_dataset and not self.dataset_path.exists():
            logger.info(blue('Generating dataset ...'))
        elif self.update_dataset and not self.dataset_path.exists():
            self.update_dataset = False
            logger.info(f"{COLORS['YELLOW']}Dataset not found:{COLORS['NC']} {self.dataset_path}")
            logger.info(blue('Generating dataset ...'))
        else:
            generate_dataset = False
            logger.info(blue("Loading dataset ..."))
            self._load_dataset()
        if generate_dataset:
            self._generate_dataset()
            logger.info(f"{COLORS['BLUE']}Saving dataset:{COLORS['NC']} {self.dataset_path}")
            self.save_dataset()
        del self.dataset_tmp
        self.dataset_tmp = None

    @staticmethod
    def cache_folder_exists(folder):
        if Path(folder).exists():
            return True
        else:
            logger.warning(f"{COLORS['YELLOW']}Cache folder not found:{COLORS['NC']} {folder}")
            return False

    def classify_ebooks(self, categories):
        X_train, X_test, y_train, y_test, feature_names, target_names = self._vectorize_dataset(categories)

        clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        plot_confusion_matrix(clf, y_test, pred, target_names)

        return 0

    @staticmethod
    def clear_cache(cache_folder):
        if DatasetManager.cache_folder_exists(cache_folder):
            result = Cache(cache_folder).clear()
            logger.info(f'Clearing cache: {cache_folder}')
            if result:
                logger.info(green('Cache cleared!'))
            else:
                logger.info(yellow('Cache was already empty!'))

    def filter_dataset(self, only_english=True, categories_to_keep=None):
        def remove_bad_chars(text):
            return RE_BAD_CHARS.sub("", text)

        dataset = Bunch(data=[], filenames=[], target_names=set(), target=[], target_name_to_value={}, DESCR="")
        categories = self.dataset.target_names if categories_to_keep is None else categories_to_keep
        total = len(self.dataset.data)
        logger.info(f'Keeping only the following categories: {categories_to_keep}')
        number_ebooks_rejected = 0
        for i, text in enumerate(self.dataset.data):
            logger.debug(f"{COLORS['BLUE']}Processing document {i+1} of {total}:{COLORS['NC']} "
                         f"{str(self.dataset.filenames[i])[:92]}...")
            text = remove_bad_chars(text)  # Cc category
            is_in_english = is_text_english(text, method='pycld2')
            logger.debug(f'Is the text in english? Answer: {is_in_english}')
            category = self.dataset.filenames[i].parent.name
            logger.debug(f'Text category: {category}')
            if only_english and is_in_english and category in categories:
                dataset.data.append(text)
                dataset.filenames.append(self.dataset.filenames[i])
                dataset.target.append(self.dataset.target[i])
                dataset.target_names.add(self.dataset.filenames[i].parent.name)
            else:
                logger.warning(f"{COLORS['YELLOW']}[WARNING] Document rejected:{COLORS['NC']} "
                               f"{str(self.dataset.filenames[i])[:100]}")
                number_ebooks_rejected += 1
        logger.info(f'Number of ebooks rejected: {number_ebooks_rejected}')
        self._fix_target(dataset)
        return dataset

    @staticmethod
    def number_items_in_cache(cache_folder):
        if DatasetManager.cache_folder_exists(cache_folder):
            cache = Cache(cache_folder)
            logger.info(f'Cache: {cache_folder}')
            n_items = len([k for k in cache.iterkeys()])
            ending = 's' if n_items > 1 else ''
            logger.info(f"There are {COLORS['GREEN']}{n_items} item{ending}{COLORS['NC']} in cache")

    @staticmethod
    def remove_keys_from_cache(cache_folder, keys):
        if DatasetManager.cache_folder_exists(cache_folder):
            cache = Cache(cache_folder)
            logger.info(f'Removing keys from cache: {cache_folder}')
            for key in keys:
                msg1 = f'Key={key} removed from cache!'
                msg2 = f"Key={key} was not found in cache"
                result = cache.delete(key)
                if result:
                    logger.info(green(msg1))
                else:
                    logger.info(yellow(msg2))

    def save_dataset(self):
        with open(self.dataset_path, 'wb') as f:
            pickle.dump(self.dataset, f)

    @staticmethod
    def shuffle_dataset(dataset):
        idx = np.random.permutation(range(len(dataset.target)))
        for attr in dataset.__dir__():
            v = getattr(dataset, attr)
            if len(v) != len(idx):
                continue
            if isinstance(v, list):
                setattr(dataset, attr,  list(np.array(v)[idx]))
            elif isinstance(v, np.ndarray):
                setattr(dataset, attr, v[idx])

    def _add_doc_to_dataset(self, filepath, text):
        self.dataset.data.append(text)
        self.dataset.filenames.append(filepath)
        self.dataset.target.append(self.dataset.target_name_to_value[filepath.parent.name])
        self.dataset.target_names.add(filepath.parent.name)

    def _fix_target(self, dataset):
        dataset.target_names = sorted(dataset.target_names)
        new_target_name_to_value = dict(zip(dataset.target_names, [i for i in range(len(dataset.target_names))]))
        target_value_to_name = dict(zip(self.dataset.target_name_to_value.values(),
                                        self.dataset.target_name_to_value.keys()))
        new_target = []
        for target_value in dataset.target:
            new_target.append(new_target_name_to_value[target_value_to_name[target_value]])
        assert len(dataset.data) == len(dataset.filenames) == len(dataset.target)
        dataset.target = np.asarray(new_target)
        dataset.target_names = list(dataset.target_names)
        dataset.target_name_to_value = new_target_name_to_value

    def _generate_dataset(self):
        folder_names = sorted([item.name for item in self.input_directory.iterdir() if item.is_dir()])
        self.dataset.target_name_to_value = dict(zip(folder_names, [i for i in range(len(folder_names))]))
        self.dataset.target_names = set()
        if not self.use_cache:
            logger.warning(yellow(f'use_cache={self.use_cache}'))
        self._generate_ebooks_dataset()
        self._fix_target(self.dataset)

    def _generate_ebooks_dataset(self):
        self.DESC = "Dataset containing text from ebooks"
        filepaths = [filepath for filepath in self.input_directory.rglob('*')
                     if filepath.is_file() and filepath.suffix.split('.')[-1] in self.ebook_formats]
        total = len(filepaths)
        add_text_to_cache = False
        text_added_dataset = []
        text_added_cache = []
        filepaths_rejected = []
        file_hashes = []
        duplicates = []
        for i, filepath in enumerate(filepaths, start=1):
            logger.info(f"{COLORS['BLUE']}Processing document {i} of {total}:{COLORS['NC']} {filepath.name[:92]}...")
            key_to_text = None
            cache_result = None
            file_hash = get_hash(filepath)
            logger.debug(f'File hash: {file_hash}')
            if file_hash in file_hashes:
                logger.info(f"{COLORS['GREEN']}Found duplicate:{COLORS['NC']} {filepath}")
                logger.info('It will be rejected from dataset and cache')
                duplicates.append(filepath)
                continue
            else:
                file_hashes.append(file_hash)
            if self.update_dataset:
                if filepath in self.dataset_tmp.filenames:
                    idx = self.dataset_tmp.filenames.index(filepath)
                    logger.info('Document already found in the loaded dataset')
                    text = self.dataset_tmp.data[idx]
                    self._add_doc_to_dataset(filepath, text)
                    continue
                else:
                    logger.info('Document not found in the loaded dataset. Will try to add it.')
            if self.use_cache:
                cache_result = self.cache.get(file_hash)
                convert_method = self._get_convert_method(filepath)
                key_to_text = f'{convert_method}+{self.convert_only_percentage_ebook}+{self.ocr_only_random_pages}'
                logger.debug(f'key_to_text: {key_to_text}')
                if cache_result and cache_result.get(key_to_text):
                    text = cache_result[key_to_text]
                    logger.info('Found text in cache')
                else:
                    logger.debug('Text not found in cache. Will try to convert document to text')
                    text = convert(filepath, **self.__dict__)
                    add_text_to_cache = True
            else:
                logger.debug('Converting document to text')
                text = convert(filepath, **self.__dict__)
            if isinstance(text, str):
                logger.info('Adding text to dataset')
                self._add_doc_to_dataset(filepath, text)
                text_added_dataset.append(filepath)
                if self.use_cache and add_text_to_cache:
                    logger.info('Adding text to cache')
                    dict_text = {key_to_text: text}
                    if cache_result:
                        logger.debug("Updating the dictionary in cache associated with the file hash "
                                     f"'{file_hash}' + {key_to_text}")
                        cache_result.update(dict_text)
                    else:
                        logger.debug('First time adding text in cache')
                        cache_result = dict_text
                    self.cache.set(file_hash, cache_result)
                    text_added_cache.append(filepath)
            else:
                logger.warning(yellow("[WARNING] Document couldn't be converted to text (it could be formed of "
                                      f"images, try with OCR): {filepath}"))
                logger.debug(f'Return code: {text}')
                filepaths_rejected.append(filepath)
            add_text_to_cache = False
        logger.info(violet('Results from dataset creation:'))
        logger.info(f'Number of text added to dataset: {len(text_added_dataset)}')
        logger.info(f'Number of text added to cache: {len(text_added_cache)}')
        logger.info(f'Number of filepaths rejected: {len(filepaths_rejected)}')
        logger.info(f'Number of duplicates: {len(duplicates)}')
        logger.debug(f'Filepaths rejected: {filepaths_rejected}')
        logger.debug(f'Duplicates: {duplicates}')

    def _get_convert_method(self, filepath):
        mime_type = get_mime_type(filepath)
        if mime_type.startswith('image/vnd.djvu'):
            return self.djvu_convert_method
        elif mime_type == 'application/pdf':
            return self.pdf_convert_method
        else:
            return None

    def _load_dataset(self):
        with open(self.dataset_path, 'rb') as f:
            if self.update_dataset:
                self.dataset_tmp = pickle.load(f)
            else:
                self.dataset = pickle.load(f)

    def _vectorize_dataset(self, categories):
        logger.info(blue('Filtering dataset ...'))
        dataset = self.filter_dataset(categories_to_keep=categories)

        logger.info(blue('Shuffling dataset ...'))
        self.shuffle_dataset(dataset)

        # Create train (60%) and test (40%) sets
        target_names = dataset.target_names
        dataset_size = len(dataset.data)
        labels = dataset.target
        unique_labels, category_sizes = np.unique(labels, return_counts=True)
        true_k = unique_labels.shape[0]
        logger.info(f'Target names: {dataset.target_name_to_value}')
        logger.info(f'Categories size: {category_sizes}')
        logger.info(f'{len(dataset.data)} documents - {true_k} categories')
        end_position = int(0.6*dataset_size)

        # split dataset in a training set and a test set
        train_data = dataset.data[:end_position]
        y_train = dataset.target[:end_position]
        train_filenames = dataset.filenames[:end_position]
        train_data_size = len(train_data)

        test_data = dataset.data[end_position:]
        y_test = dataset.target[end_position:]
        test_filenames = dataset.filenames[end_position:]
        test_data_size = len(test_data)

        assert dataset_size == (train_data_size + test_data_size)

        # Extracting features from the training data using a sparse vectorizer
        t0 = time()
        vectorizer = TfidfVectorizer(
            sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
        )
        X_train = vectorizer.fit_transform(train_data)
        duration_train = time() - t0

        # Extracting features from the test data using the same vectorizer
        t0 = time()
        X_test = vectorizer.transform(test_data)
        duration_test = time() - t0

        feature_names = vectorizer.get_feature_names_out()

        if self.verbose:
            # compute size of loaded data
            data_train_size_mb = size_mb(train_data)
            data_test_size_mb = size_mb(test_data)

            print(
                f"{len(train_data)} documents - "
                f"{data_train_size_mb:.2f}MB (training set)"
            )
            print(f"{len(test_data)} documents - {data_test_size_mb:.2f}MB (test set)")
            print(f"{len(target_names)} categories")
            print(
                f"vectorize training done in {duration_train:.3f}s "
                f"at {data_train_size_mb / duration_train:.3f}MB/s"
            )
            print(f"n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
            print(
                f"vectorize testing done in {duration_test:.3f}s "
                f"at {data_test_size_mb / duration_test:.3f}MB/s"
            )
            print(f"n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}")

        return X_train, X_test, y_train, y_test, feature_names, target_names


def main():
    global QUIET
    exit_code = 0
    try:
        parser = setup_argparser()
        args = parser.parse_args()
        QUIET = args.quiet
        setup_log(args.quiet, args.verbose, args.logging_level, args.logging_formatter)
        if args.use_cache or args.check_number_items or args.clear_cache or args.remove_keys:
            use_cache = True
        else:
            use_cache = False
        import_modules(english_detector='pycld2', use_cache=use_cache)
        # Actions
        if args.check_number_items:
            DatasetManager.number_items_in_cache(CACHE_FOLDER)
        elif args.clear_cache:
            DatasetManager.clear_cache(CACHE_FOLDER)
        elif args.remove_keys:
            DatasetManager.remove_keys_from_cache(CACHE_FOLDER, args.remove_keys)
        elif args.input_directory:
            data_manager = DatasetManager(**namespace_to_dict(args))
            if args.categories is None:
                categories = CATEGORIES
            else:
                categories = args.categories
            exit_code = data_manager.classify_ebooks(categories)
        else:
            logger.warning(yellow('Missing input directory'))
            exit_code = 3
    except KeyboardInterrupt:
        print_(yellow('\nProgram stopped!'))
        exit_code = 2
    except Exception as e:
        print_(yellow('Program interrupted!'))
        logger.exception(e)
        exit_code = 1
    return exit_code


if __name__ == '__main__':
    retcode = main()
    msg = f'Program exited with {retcode}'
    if retcode == 1:
        logger.error(red(f'[ERROR] {msg}'))
    else:
        logger.debug(msg)
