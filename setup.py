from setuptools import setup, Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='crossasr',         # How you named your package folder (MyLib)
    packages=['crossasr'],   # Chose the same as "name"
    version='0.1.2',      # Start with a small number and increase it with every change you make
    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    license='MIT',
    # Give a short description about your library
    description='CrossASR++ A Modular Differential Testing Framework for Automatic Speech Recognition',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Muhammad Hilmi Asyrofi',                   # Type in your name
    author_email='mhilmia@smu.edu.sg',      # Type in your E-Mail
    # Provide either the link to your github or to your website
    url='https://github.com/soarsmu/CrossASRplus',
    # I explain this later on
    download_url='https://github.com/soarsmu/CrossASRplus/archive/refs/tags/v0.1.2.tar.gz',
    # Keywords that define your package best
    keywords=['crossasr','differential testing', 'cross-referencing', 'ASR'],
    install_requires=[            # I get to this in a second
        'jiwer',
        'normalize'
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        # Define that your audience are developers
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
