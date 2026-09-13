## How to import NumPy
`import numpy as np`

## What is an "array"
- Elements must be homogeneous
- The size is fixed
- The shape must be rectangular, which means each row has the same number of columns.

## Array attributes
`shape`, `size`, `ndim`, `dtype`

## How to create a basic array
```python
a = np.array([1, 2, 3, 4]) # Create an array from a Python list.
b = np.zeros([2, 3]) # Create an array with a shape of 2 rows and 3 columns, filled with zeros.
c = np.ones([4, 5])  # Create an array with a shape of 4 rows and 5 columns, filled with ones.
e = np.empty() # Create an uninitialized array.
f = np.arange() # Create an array from a range of elements.
g = np.linspace() # Create an array with values that are linearly spaced within a specified interval.
``` 

## Adding, removing, and sorting elements
```python
np.concatenate((a, b), axis=0) # Concatenate arrays a and b along the rows. If axis=1, it will concatenate them along the columns.
np.sort() # Sort elements in ascending order
```

## Reshape
```python
np.reshape(a, shape=(1, 6), order='C') # a is the array you want to reshape. order='C' means row-major order, while order='F' means column-major order.
```