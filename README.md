# Hierarchical Nested Segmentation
This code implements the Hierarchical Nested Segmentation model (HNS) from:

Hierarchical Span-Based Conditional Random Fields for Labeling and Segmenting Events in Wearable Sensor Data Streams. Roy Adams, Nazir Saleheen, Edison Thomas, Abhinav Parate, Santosh Kumar, and Benjamin Marlin. International Conference on Machine Learning, 2016.

## Setup

This code requires a number of python modules listed in requirements.txt. To install these requirements using pip, call

```
pip install -r requirements.txt
```

Additionally, portions of this code is written in cython and must be compiled. To complie this code exectute,

```
python setup.py build_ext --inplace
```
