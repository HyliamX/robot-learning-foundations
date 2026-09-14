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

## np.newaxis
It will increase the dimensions of the array by one dimension. For example, 1D array will become 2D array, 2D array will become 3D array, and so on.

## np.expand_dims
It adds a new axis at a specified position.
```python
a = np.array([[1,2], 
              [3, 4]])
a.shape # Output: (2,2), (2[position=0], 2[position=1])
b = np.expand_dims(a, axis=1)
b.shape # Output:(2, 1, 2), it will insert a new axis at position 1 and shifts the exsiting axes to the right.
```
## Indexing and slicing
- NumPy array can be indexed and sliced in a similar way to Python lists
- Using boolean expression to extract elements.
- `np.nonzero()` returns a tuple of arrays, which is the index of the values found in the array, and can be used directly to extract all elements.

## Create an array from existing data
- Slice an array to create a new array.
- `np.vstack` Stack array vertically.
- `np.hstack` Stack array horizontally.
- Indexing and slicing will return a view, which is the shallow copy of the original array, modifying data in view also modifies the original array.
- `copy` method will make a deep copy, which means it will create an independent copy of the array.

## Basic operations
- addition, subtraction, multiplication, division
- `a.sum()` will add all elements together, also you can specify the axis.
- `a.min()` returns the minimum value in the array.
- `a.max()` returns the maximum value in the array.
- `a.mean()` returns the average value in the array.