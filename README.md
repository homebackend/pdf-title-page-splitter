[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

# pdf-title-page-splitter
**pdf-title-page-splitter** is command line tool (with limited UI support) to splits a pdf based on identified title pages. The title pages are identified using machine learning model. The tool supports both training model and using trained model to split pdf files.

## But first, what is the need for this tool anyway ?
There are a plethora of tools that allow splitting of pdf files, so why this tool?

Consider the pdf like this: [Economic and Political Weekly - Volume 26](https://archive.org/details/dli.bengal.10689.12140/). It has 1200 pages and 210.4 MB of data. This makes these pdf files notoriously difficult to read and handle. Actually this pdf file is contains multiple issues of volume 26 of [Economic and Political Weekly](https://www.epw.in/). So ideally this pdf file should be split into multiple pdf files - each for one issue of the volume.

But this still didn't answer the original question. One can easily split pdf using any other tool. However, if you have to do this for [hundreds of such files](https://archive.org/search?query=economic+and+political+weekly) the task becomes daunting. Where this tool helps is basically to train a ML model that will identify title pages within the given pdf and split pdf into multiple pdfs, each for a single issue.

In the above specific example, the following is a title page:

<img src="https://ia800301.us.archive.org/BookReader/BookReaderImages.php?zip=/13/items/dli.bengal.10689.12140/10689.12140_jp2.zip&file=10689.12140_jp2/10689.12140_0005.jp2&id=dli.bengal.10689.12140&scale=2&rotate=0" alt="Title page" width="200"/>

Using **pdf-title-page-splitter** we train model to identify such title pages and split pdf into multiple issues.

# Installation
## Requirements

**pdf-title-page-splitter** requires *python 3* and a bunch of other dependencies mentioned in `requirements.txt` file.

## Setup
Before running **pdf-title-page-splitter** the python environment needs to be setup correctly. Here we are creating a python virtual environment and installing all the dependencies. The instructions are provided for *Linux*, but ideally these should be identical for any *UNIX* like operating system.

### Create virtual environment and install dependencies

The following Change to the folder/directory containing 

```bash
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```
### Activating virtual environment

Creating virtual environment and installing dependencies is one time process. In subsequent runs you just need to activate the virtual environment:
```bash
. venv/bin/activate
```

To deactivate the virtual environment run the command: `deactivate`.

# Usage

## Creating model file

Model is trained using **create** command. It supports the following command line options:

```bash
$ python pdf-title-page-splitter.py create -h
usage: pdf-title-page-splitter.py create [-h] [-s SAVE_PATH] [-p PARALLELISM] [--pdf pdf [title-pages ...]] files [files ...]

positional arguments:
  files                 Pdf files to be used

options:
  -h, --help            show this help message and exit
  -s, --save-path SAVE_PATH
                        Save path (default: model.pkl)
  -p, --parallelism PARALLELISM
                        Number of parallel pages to process (default: number of cores)
  --pdf pdf [title-pages ...]
                        Specify a file and comma separated title pages pair. Can be used multiple times.
```

What the above command will do, is create a model-file **model.pkl** trained using specified pdf files to predict title pages for an unseen pdf.

### Example run

Consider the case where you have identified title pages of some pdfs. You can create model like so:

```bash
python3 pdf-title-page-splitter.py \
    create \
    --save-path model.pkl \
    --pdf 'first-pdf.pdf' 5 69 143 201 239 312 \
    --pdf 'second-pdf.pdf' 2 45 100 189 234 301
```

Here page numbers 5, 69, 143, 201, 239 and 312 are title pages identified in *first-pdf.pdf*. Likewise for the other pdf.

### Another run

Instead if you have a bunch of pdf files that have title page as the first page of pdf (essentially which you have already split) you can use the following command to create a model file **model.pkl**.

```bash
python3 pdf-title-page-splitter.py \
    create \
    --save-path model.pkl \
    'first-pdf.pdf' \
    'second-pdf.pdf' \
    'third-pdf.pdf' \
    'fourth-pdf.pdf'
```

## Predicting title pages

The **predict** command can be used to predict *title pages* for a bunch of pdfs given a model file generated from **create** command. The title pages identified are saved in JSON format (by default filename is *titles.json*) for subsequent processing.

```bash
$ python pdf-title-page-splitter.py predict -h
usage: pdf-title-page-splitter.py predict [-h] [-m MODEL_PATH] [-s SAVE_PATH] [-b BEGINNING_PAGE] [-e ENDING_PAGE]
                                          [-p PARALLELISM]
                                          pdf [pdf ...]

positional arguments:
  pdf                   Pdf files to be used

options:
  -h, --help            show this help message and exit
  -m, --model-path MODEL_PATH
                        Model file path (default: model.pkl)
  -s, --save-path SAVE_PATH
                        Save path (default: titles.json)
  -b, --beginning-page BEGINNING_PAGE
                        Starting page of pdf file (default=1)
  -e, --ending-page ENDING_PAGE
                        Ending page of pdf file (default=last page)
  -p, --parallelism PARALLELISM
                        Number of parallel pages to process (default: number of cores)
```

### Example run
The following command identifies title pages in all pdf files in */tmp* directory and saves the result to *my-titles.json*.

```bash
python3 pdf-title-page-splitter.py \
    predict \
    --model-path model.pkl \
    --save-path my-titles.json \
    /tmp/*.pdf
```

A sample *titles.json* file generated could be:

```json
{
    "Economic.And.Political.Volume.xviii.No27.pdf": [],
    "Economic And Political Weekly Vol.-xxii-no.49-inernetdli2015121078.pdf": [
        2,
        58,
        114,
        170
    ],
    "Economic And Political Weekly Vol-XXVIII -- Sachin Chaudhuri -- 1995 -- Economic And Political Weekly Vol-XXVIII -- aa94fb82ae80ea6297ec3a739a400d22 -- Anna\u2019s Archive.pdf": [
        5,
        77,
        149,
        261,
        325,
        389,
        495,
        555,
        623
    ]
}
```

Essentially it contains page numbers of identified *title pages*. If no *title page* was identified it will be empty list (for example: *Economic.And.Political.Volume.xviii.No27.pdf* above).

## Show title pages

Once the *title pages* have been identified by **pdf-title-page-splitter** the next step is to visually see if the identified *title pages* are corrent and potentially correct any mistakes.

The **show** commands presents to you each *title page* and you can either accept, reject or substitute the given *title page* for each pdf file in *titles.json*. The **show** command itself is split into two sub commands, viz. **run** and **from**.

The subcommand **run** is essentially to combine **predict** and **show** commands into a single step.

The subcommand **from** is used to read *titles.json* file from **predict** command as described in prior section, and present to the user UI as described above in this section.

Note, though this is not recommended, you can skip this step and directly go on to splitting of pdfs.

Command line options for show command:

```bash
$ python pdf-title-page-splitter.py show -h
usage: pdf-title-page-splitter.py show [-h] {run,from} ...

positional arguments:
  {run,from}  Available sub commands
    run       Run predict and show pages
    from      Load saved data from file and show pages

options:
  -h, --help  show this help message and exit
```

### Show title pages from *titles.json*

The supported command line options are as follow:

```bash
$ python pdf-title-page-splitter.py show from -h
usage: pdf-title-page-splitter.py show from [-h] [-l LOAD_FROM_FILE] [-s SAVE_PATH]

options:
  -h, --help            show this help message and exit
  -l, --load-from-file LOAD_FROM_FILE
                        Load title pages and pdf from file (default: titles.json)
  -s, --save-path SAVE_PATH
                        Save path (default: splits.json)
```

Here **-l** option loads predicted *title page* data as generated using using **create** command.

**-s** option specifies the file (by default *splits.json*) where the *title pages* left after user has done accepting, rejecting and/or substituting of *title pages* are stored. Essentially this file is used to split the pdf files. Note that *splits.json* has same format as *titles.json*.

### UI and user interaction

For each *title page* user is presented with a window showing the *title page*. User can take the following actions:

- **->** (right arrow key): moves to next *title page* (current page is retained)
- **<-** (left arrow key): moves to previous *title page* (current page is retained)
- **x**: delete current *title page* (during split this will be treated as non *title page*)
- **r**: replace current *title page* (this will enter user into page replacement mode)
- **n**: moves to next pdf file (if there is no next file - you will be asked if you want to save changes)
- **s**: save and quit (all changes are saved into file specified using **-s** command line option)
- **q**: quit without saving (no changes are saved)

In replacement mode (using **r** key above) the following actions are supported:

- **->** (right arrow key): moves to next page
- **<-** (left arrow key): moves to previous page
- **s**: save current page as replacement page for the *title page*
- **q**: quit page replacement mode (the same *title page* is retained - you will be dropped to same *title page* - which can be retained, rejected or substituted again, if desired)

[Title Page](doc/title-page.png "Title page")
[Wrong Title Page](doc/wrong-title-page.png "Wrong Title Page")
[Replacement Page](doc/replacement-page.png "Replacement page")


## Split pdf files

As a final step you can proceed to split pdf files. Till now no actual pdf files were written.

The supported command line options are as follows:

```bash
$ python pdf-title-page-splitter.py split -h
usage: pdf-title-page-splitter.py split [-h] {run,from} ...

positional arguments:
  {run,from}  Available sub commands
    run       Run predict and split pages
    from      Load saved data from file and show pages

options:
  -h, --help  show this help message and exit
```

The **run** command executes **predict**, and then splits the files. There is no **show** command.

The **from** command splits the pdf files based on *splits.json* file.

### **from** command

**from** command supports the following options:

```bash
$ python pdf-title-page-splitter.py split from -h
usage: pdf-title-page-splitter.py split from [-h] [-l LOAD_FROM_FILE] [--force] [--move-original-to MOVE_ORIGINAL_TO]
                                             [--split-destination SPLIT_DESTINATION] [--noop] [--move-singles MOVE_SINGLES]

options:
  -h, --help            show this help message and exit
  -l, --load-from-file LOAD_FROM_FILE
                        Load title pages and pdf from file (default: splits.json)
  --force               Force overwriting of split files (default skips file if it exists)
  --move-original-to MOVE_ORIGINAL_TO
                        Post split move the file to specified diretory (default do not move)
  --split-destination SPLIT_DESTINATION
                        Destination directory for split files (default same as source file)
  --noop                Make no actual changes (default make changes)
  --move-singles MOVE_SINGLES
                        Move files that contain only single title page and that too as first page into the specified
                        directory (default is to not move)
```

### Example run

```bash
python3 pdf-title-page-splitter.py split \
    from \
    --move-original-to splitted \
    --split-destination splits \
    --move-singles splits \
    --load-from-file splits.json
```

After running the command the output would be like so:

```bash
$ ls -1 splits
'Economic & Political Weekly  June 16-23 1900: Vol 25 24-25-economicpoliticalweekly_june16231900_25_2425.pdf'
Economic.And.Political.Weekly.Vol-XXV.No-27_split_0000.pdf
Economic.And.Political.Weekly.Vol-XXV.No-27_split_0001.pdf
Economic.And.Political.Weekly.Vol-XXV.No-27_split_0002.pdf
Economic.And.Political.Weekly.Vol-XXV.No-27_split_0003.pdf
Economic.And.Political.Weekly.Vol-XXV.No-27_split_0004.pdf
Economic.And.Political.Weekly.Vol-XXV.No-27_split_0005.pdf
Economic.And.Political.Weekly.Vol-XXV.No-27_split_0006.pdf
Economic.And.Political.Weekly.Vol-XXV.No-27_split_0007.pdf
Economic.And.Political.Weekly.Vol-XXV.No-27_split_0008.pdf
Economic.And.Political.Weekly.Vol-XXV.No-27_split_0009.pdf
Economic.And.Political.Weekly.Vol-XXV.No-27_split_0010.pdf
Economic.And.Political.Weekly.Vol-XXV.No-27_split_0011.pdf
Economic.And.Political.Weekly.Vol-XXV.No-27_split_0012.pdf
Economic.And.Political.Weekly.Vol-XXV_split_0001.pdf
Economic.And.Political.Weekly.Vol-XXV_split_0002.pdf
Economic.And.Political.Weekly.Vol-XXV_split_0003.pdf
Economic.And.Political.Weekly.Vol-XXV_split_0004.pdf
Economic.And.Political.Weekly.Vol-XXV_split_0005.pdf
Economic.And.Political.Weekly.Vol-XXV_split_0006.pdf
Economic.And.Political.Weekly.Vol-XXV_split_0007.pdf
Economic.And.Political.Weekly.Vol-XXV_split_0008.pdf
Economic.And.Political.Weekly.Vol-XXV_split_0009.pdf
Economic.And.Political.Weekly.Vol-XXV_split_0010.pdf
Economic.And.Political.Weekly.Vol-XXV_split_0011.pdf
Economic.And.Political.Weekly.Vol-XXV_split_0012.pdf
```

Note in the above output files named **_0000.pdf** are files created from page number 1 to first *title page*. If for any file first *title page* is page number 1 there is no **_0000.pdf** file.
Also, if a file has no splits defined it will be kept as is.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/homebackend/pdf-title-page-splitter.svg?style=for-the-badge
[contributors-url]: https://github.com/homebackend/pdf-title-page-splitter/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/homebackend/pdf-title-page-splitter.svg?style=for-the-badge
[forks-url]: https://github.com/homebackend/pdf-title-page-splitter/network/members
[stars-shield]: https://img.shields.io/github/stars/homebackend/pdf-title-page-splitter.svg?style=for-the-badge
[stars-url]: https://github.com/homebackend/pdf-title-page-splitter/stargazers
[issues-shield]: https://img.shields.io/github/issues/homebackend/pdf-title-page-splitter.svg?style=for-the-badge
[issues-url]: https://github.com/homebackend/pdf-title-page-splitter/issues
[license-shield]: https://img.shields.io/github/license/homebackend/pdf-title-page-splitter.svg?style=for-the-badge
[license-url]: https://github.com/homebackend/pdf-title-page-splitter/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/neeraj-jakhar-39686212b
