from setuptools import setup, find_packages

setup(
    name='modeconvertor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=1.4.0',
        'scikit-learn>=0.22',
        'matplotlib>=3.1.0',
        'opencv-python>=4.2.0.32',
    ],
    author='Durgesh',
    author_email='durgesh080793@gmail.com',
    description='A package for converting modes in grayscale images and videos.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/modeconvertor',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
