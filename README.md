# Sharks classification (ongoing)

![shark](https://storage.googleapis.com/kagglesdsdata/datasets/947997/1605885/sharks/tiger/00000001.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20220318%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220318T002655Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=94c4eb15816cf1770deb88344e4cadce5b0834625e0d0070cbd7862c35c83363652a6b728ef73813c9afa01214d4e27a273b9430bfedab2238ccbc044a53dbc89e56bb9d54f58d597b29efa0252f1c0cdf2ba31743f8bc3d6c2c721a4893269d1e40139f3e3436ede103a082cf6842491297076f054890ff01e90c41a701a0995cfe71b5f603b3919784a13e65c1c50bf6fd6bcd31727349b69a88a068a2d7ef8d0505c94138f97a85b78a0bfda463846d2340b1b4fa63a8620719fd18573eab9c04f53aadbcdfc90992e3bbdd7c99c5429d7382c6c7aeb4bc7e5aeac269954e888905ed202368de46c5dc6b03ea5e43936e08b9997bead3191d936ef92095ea)

## Overview
(will be updated during the project)

The goal of the project is to deliver a deep learning model classifying an open-source dataset of [Shark species](https://www.kaggle.com/larusso94/shark-species) available on Kaggle.

The project will consist of a training and evaluation scripts wrapped with [Kedro](https://kedro.readthedocs.io/en/stable/index.html) project. We are going to prepare a fully transferable setup so that you can train and run the model either locally or using a cloud provider. We have decided to use Google Cloud Platform, so you will find detailed instructions how to start it off with GCP.

Therefore, we are going to use some state-of-the-art convolutional neural networks adjusted to the needs of the dataset. We are aware that the project is not revolutionary at, but its goal is to learn how to deliver end-to-end ML model rather than make an innovative step in research.

The papers which describe the models that we are going to use are obviously:
- A. Krizhevsky, I. Sutskever, and G. Hinton. [Imagenet classification with deep convolutional neural networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf). In NIPS, 2012.
- K. Simonyan and A. Zisserman. [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf). In ICLR, 2015.
- K. He, X. Zhang, S. Ren, and J. Sun. [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf). In CVPR, 2016.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://kedro.readthedocs.io/en/stable/12_faq/01_faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
kedro install
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, go to the `.coveragerc` file.

## Project dependencies

To generate or update the dependency requirements for your project:

```
kedro build-reqs
```

This will copy the contents of `src/requirements.txt` into a new file `src/requirements.in` which will be used as the source for `pip-compile`. You can see the output of the resolution by opening `src/requirements.txt`.

After this, if you'd like to update your project requirements, please update `src/requirements.in` and re-run `kedro build-reqs`.

[Further information about project dependencies](https://kedro.readthedocs.io/en/stable/04_kedro_project_setup/01_dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, `catalog`, and `startup_error`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `kedro install` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to convert notebook cells to nodes in a Kedro project
You can move notebook code over into a Kedro project structure using a mixture of [cell tagging](https://jupyter-notebook.readthedocs.io/en/stable/changelog.html#release-5-0-0) and Kedro CLI commands.

By adding the `node` tag to a cell and running the command below, the cell's source code will be copied over to a Python file within `src/<package_name>/nodes/`:

```
kedro jupyter convert <filepath_to_my_notebook>
```
> *Note:* The name of the Python file matches the name of the original notebook.

Alternatively, you may want to transform all your notebooks in one go. Run the following command to convert all notebook files found in the project root directory and under any of its sub-folders:

```
kedro jupyter convert --all
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://kedro.readthedocs.io/en/stable/03_tutorial/08_package_a_project.html)
