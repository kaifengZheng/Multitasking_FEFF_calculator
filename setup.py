#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Kaifeng Zheng",
    author_email='audreyr@example.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
    ],
    description="A parallel worflow to calculate XAS using FEFF",
    entry_points={
        'console_scripts': [
            'feff_package=feff_package.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='feff_package',
    name='feff_package',
    packages=find_packages(include=['feff_package', 'FEFF_run.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/kaifengZheng/FEFF_package',
    version='0.1.0',
    zip_safe=False,
)
