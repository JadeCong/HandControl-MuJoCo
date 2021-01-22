import glob
import os

from setuptools import find_packages, setup


def find_data_files(base, globs):
    """Find all interesting data files, for setup(data_files=)

    Arguments:
      root:  The directory to search in.
      globs: A list of glob patterns to accept files.
    """

    rv_dirs = [root for root, dirs, files in os.walk(base)]
    rv = []

    for rv_dir in rv_dirs:
        files = []

        for pat in globs:
            files += glob.glob(os.path.join(rv_dir, pat))

        if not files:
            continue
        target = os.path.join('lib', 'mypy', rv_dir)
        rv.append((target, files))

    return rv


data_files = []

data_files += find_data_files('data_importer', ['*.py', '*.pyi'])

print(data_files)
# data_files += find_data_files('funtest', ['*.xsd', '*.xslt', '*.css'])


setup(name='data_importer',
      version='0.1',
      description='data_importer',
      author='xiaoyan',
      author_email='tb_xy09@163.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
