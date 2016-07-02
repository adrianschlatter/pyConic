# -*- coding: utf-8 -*-
"""
:author: Adrian Schlatter
"""

from setuptools import setup

setup(name='pyConic',
      version='0.0',
      description='Python tools to work with conic sections',
      author='Adrian Schlatter',
      author_email='schlatter@phys.ethz.ch',
      license='Revised BSD',
      packages=['pyConic'],
      install_requires=['numpy'],
      include_package_data=True,
      test_suite='tests',
      zip_safe=False)
