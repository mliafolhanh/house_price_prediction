import os
from setuptools import setup, find_packages

path = os.path.abspath(os.path.dirname(__file__))
readme = open(path + "/docs/README.md")

setup(name='house_price_prediction',
      version='0.1.0',
      description='predict house price',
      url='github.',
      author='hphan',
      author_email='mliafol.phan86@gmali.com',
      license='',
      packages=find_packages(exclude=["tests", "docs", ".gitignore"]),
      classifiers = [
		  "Programming Language :: Python :: 3",
      ],
      install_requires=[
        "coloredlogs>=10.0",
        "humanfriendly>=4.18",
        "numpy>=1.17.4",
        "scipy>=1.3.2",
        "statsmodels>=0.11.1",
        "scikit-learn>=0.21.3"
      ],
      zip_safe=False,
      include_package_data=True)
