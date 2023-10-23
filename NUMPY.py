#!/usr/bin/env python
# coding: utf-8

# # Installing Numpy and Numpy Array Creation

# In[ ]:


# Install a pip package in the current Jupyter kernel
get_ipython().system(' pip install numpy')


# In[1]:


#importing a numpy package
import numpy as np


# In[3]:


lst=[1,2,3,4,5,6]
print(type(lst))
print("List:",lst)


# In[4]:


#Creating a simple array in numpy
arr=np.array(lst)
print(type(arr))
print("NUmpy Array:",arr)  #ndarray is nothing but n dimentional array.


# In[5]:


arr1=np.array((1,2,3,4,5))
print(type(arr1))
print(arr1)


# In[6]:


arr2=np.array({1,2,3,4,5})
print(type(arr2))
print(arr2)


# In[6]:


#creating ndarray  using arange function
rr=np.arange(10)
print(arr)
print(type(arr))


# # Why Numpy?

# Most powerful numerical processing library in python. Array Oriented computing.
# 
# Provides extension package to python for multi dimensional array.
# 
# Very efficient.
# 
# Scientific computation.

# # why numpy ,when we already have lists?

# In[12]:


get_ipython().run_cell_magic('time', '', 'lst=list(range(1000000))\nfor i in  range(1000000):\n    lst[i]*=lst[i]\n')


# In[13]:


get_ipython().run_cell_magic('time', '', 'arr=np.arange(1000000)\narr=arr*arr\n')


# # Attributes/Properties

# In[2]:


import numpy as np
arr=np.array([1,2,3,4,5])
print(arr)
print("Shape:",arr.shape) #check the shape of the array. in the form of tuple
print("Data type:",arr.dtype)# data-type of array's elements.
print("Item size",arr.itemsize)#it returns the length of one arry element in bytes.
print("Dimentionalit",arr.ndim) #no.of dimentions present in the array.
print("size:",arr.size) # no.of elements present in the array. is aslo equal to np.prod(arr.shape)
print("total bytes",arr.nbytes)# returs the total bytes consumed by the elements of the array


# In[4]:


ar=np.array([1,2,3,4,5])
print(ar.flags)


# In[23]:


arr=np.array([[1,2,3,4],[5,6,7,8]]) #it will form the 2 dimentional array.
print("Array: \n",arr)
print("Shape:",arr.shape) #check the shape of the array. in the form of tuple
print("Data type:",arr.dtype)
print("Item size",arr.itemsize)
print("Dimentionalit",arr.ndim)


# In[29]:


arr=np.array([1,2.7,3,4.0,5]) # to convert the entaire elements into floate value
print(arr)
print(arr.dtype)
print(arr.itemsize)


# In[30]:


arr=np.arange(1,10)
print("Array: \n",arr)
print("Shape:",arr.shape)
print("Data type:",arr.dtype)
print("Item size",arr.itemsize)
print("Dimentionalit",arr.ndim)


# In[31]:


arr=np.arange(1,10,2)
print("Array: \n",arr)
print("Shape:",arr.shape)
print("Data type:",arr.dtype)
print("Item size",arr.itemsize)
print("Dimentionalit",arr.ndim)


# **Another way to creating an numpy array using functions**

# In[18]:


arr=np.ones((3,3))
print("Array: \n",arr)
print("Shape:",arr.shape)
print("Data type:",arr.dtype)
print("Item size",arr.itemsize)
print("Dimentionalit",arr.ndim)


# In[17]:


arr=np.ones((3,3), dtype='i')
print("Array: \n",arr)
print("Shape:",arr.shape)
print("Data type:",arr.dtype)
print("Item size",arr.itemsize)
print("Dimentionalit",arr.ndim)


# In[33]:


arr=np.zeros((3,3))
print("Array: \n",arr)
print("Shape:",arr.shape)
print("Data type:",arr.dtype)
print("Item size",arr.itemsize)
print("Dimentionalit",arr.ndim)


# In[34]:


arr=np.eye(3) # identity matrincs in mathmatics nothing but suqre matrix.
print("Array: \n",arr)
print("Shape:",arr.shape)
print("Data type:",arr.dtype)
print("Item size",arr.itemsize)
print("Dimentionalit",arr.ndim)


# **type cast**

# In[3]:


arr=np.array([1.,2.,3.4,4.])
print(arr)
print(arr.astype(int))


# In[6]:


arr=np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(arr.size)


# In[35]:


arr=np.eye(3,2) # 3 rows and 2 colums
print("Array: \n",arr)
print("Shape:",arr.shape)
print("Data type:",arr.dtype)
print("Item size",arr.itemsize)
print("Dimentionalit",arr.ndim)


# # Numpy Random Numbers

# In[2]:


import numpy as np
arr=np.random.rand(5) #5 is represents the how money numbers you want to generate
print("Numpy array:",arr)


# In[27]:


arr=np.random.rand(10,2) #10 rows and 2 colums
print(arr)


# In[4]:


arr=np.random.randn(5)
print(arr)


# In[5]:


arr=np.random.randn(5,4) #5 rows and 4 colums  
print(arr)


# In[9]:


#generate a random integer b/w 0 to 9
value = np.random.randint(10)
print(value)


# In[11]:


value = np.random.randint(10,31)
print(value)


# In[12]:


#write a programm to creat a functionlity to similate of a dice roll ?


# In[15]:


import numpy as np
def fun():
    dice_roll= np.random.randint(1,7)
    print("value:",dice_roll)
fun()


# In[ ]:


#write a programm to generate an otp(4 digit)


# In[20]:


def otp():
    return np.random.randint(1000,10000)


# In[21]:


otp()


# In[25]:


#randomly genarete a 5*4 array containing values
arr=np.random.randint(10, size=(5,4))
print(arr)


# In[27]:


arr=np.random.randint(10,40, size=(5,4))
print(arr)


# In[30]:


#generate random decimal value b/w 0 to 10
arr= np.random.uniform(10)
print(arr)


# In[33]:


arr=np.random.uniform(10, size=(5,4)) #b/t 0 to 10
print(arr)


# In[34]:


arr=np.random.uniform(10,40, size=(5,4))  # 10 is minimum and 40 is maximum.
print(arr)


# # Numpy array Indexing, slicing and updating

# In[37]:


#randomly generating 1 dimentional array
arr=np.random.randint(100, size=(5,))
print(arr)
print(arr.shape)


# In[38]:


print(arr[0])


# In[39]:


print(arr[2])


# In[41]:


print(arr[::-1])


# In[19]:


#randomly generating 2 dimentional array
arr=np.random.randint(100, size=(5,4))
print(arr)


# In[43]:


print(arr[2])


# In[44]:


print(arr[1][2])


# In[45]:


print(arr[1,2]) # [row_inde, col_inde]


# In[47]:


print(arr[[1,3,4],[2,0,3]])


# In[50]:


print(arr[1][2],arr[3][0],arr[4][3])


# In[30]:


arr=np.random.randint(100, size=(10,))
print(arr)


# In[23]:


arr


# In[24]:


print(arr[1:4])
print(arr[0:-4])
print(arr[::2])


# In[25]:


arr=np.random.randint(100, size=(10,))
print(arr)


# In[26]:


print(arr[1:4:-1])


# In[27]:


arr=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(arr)


# In[28]:


print(arr[:2,1:3])


# In[60]:


print(arr[1:2, :])


# In[61]:


print(arr[1:2, 2:3])


# In[65]:


print(arr[1:,2:])


# In[66]:


print(arr[0:2, 2:])


# In[67]:


print(arr[0:2,:3])


# **Indexing with boolean Arrays**

# In[70]:


arr=np.random.randint(100, size=(10,))
print(arr)


# In[71]:


idx=[True,False,False,False,False,True,False,False,True,True]
print(arr[idx])


# **Updating value in the array**

# In[75]:


arr=np.random.randint(100, size=(3,4))
print(arr)
arr[1,1]=99
print(arr)


# In[3]:


import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])
print(x*2)


# # Numpy Flatten and Ravel

# In[3]:


import numpy as np
arr=np.random.randint(100, size=(5,10))
print(arr)
print("shape:",arr.shape)


# In[4]:


flatten_arr=arr.flatten()
print(flatten_arr)
print(flatten_arr.shape)


# In[3]:


ravel=arr.ravel()
print(ravel)
print(ravel.shape)


# In[4]:


ravel[[0,1]]=10
print(flatten_arr)
print(ravel) #it effect to the original array.
print(arr)


# **Iterating over Numpy Array**

# In[3]:


import numpy as np
arr_1d=np.random.randint(10,40,size=(10,))
print(arr_1d)
print(arr_1d.shape)


# In[4]:


for i in arr_1d:
    print(i, end=" ")


# In[6]:


arr_2d=np.random.randint(10,40,size=(5,10))
print(arr_2d)
print(arr_2d.shape)


# In[7]:


for i in arr_2d:
    print(i)


# In[8]:


for i in arr_2d.ravel():
    print(i, end=" ")


# **ndarray.flat**

# 1. **Flat:** The ndarray.flat attribute returns an iterator object that allows you to traverse the elements of the ndarray in a flattened manner, regardless of the original shape of the array. 
#     ndarray.flat returns an iterator for traversing the elements of the ndarray without creating a new array. It provides a memory-efficient way to iterate over the elements.
#     
# 2. **ndarray.flatten()** returns a new 1-dimensional array containing the flattened elements of the ndarray. It creates a new array and consumes additional memory.
# 
# Which one to use depends on your specific use case:
# 
# If you just need to iterate through the elements and don't need a separate array, **ndarray.flat** might be more memory-efficient.
# 
# If you need a separate flattened array for further computation, then using **ndarray.flatten()** is appropriate.

# In[5]:


import numpy as np
arr=np.random.randint(10,40, size=(5,4))
flatten_arr=arr.flatten()
print(arr)
print(flatten_arr)
print(arr.flat[0]) # Flat is A 1-D iterator over the array.
print(arr[0])


# In[7]:


import numpy as np
arr_2d=np.random.randint(10,40,size=(5,4))
ravel_arr=arr_2d.ravel()
for i in range(len(ravel_arr)):
    if (ravel_arr[i]% 2 !=0):
        ravel_arr[i]=-1
print("updated_array:\n", arr_2d)


# In[53]:


arr = np.random.randint(100, size =(5,4))
print(arr)
flatten_arr = arr.ravel()
for i in range(len(flatten_arr)):
    count =0
    for j in range(2, flatten_arr[i]//2 +1):
        if flatten_arr[i]% j == 0:
            count +=1
    if count ==0:
            flatten_arr[i]=-1
print(arr)


# In[14]:


arr.flat=3
print(arr)


# In[5]:


arr.flat=3;arr


# In[6]:


arr.flat[[1,4,6]]= 1
arr


# **if we what to assigning any value to the array , we can use flat.**

# **Write a program to assign a value -1 if the element is odd number**

# In[79]:


arr = np.random.randint(10,40, size=(5,5))
print("Origanal Array \n", arr)
print(arr.flat[[0,6,12,18,24]])


# In[81]:


arr[[1,2,3],[3,2,4]] =1


# In[82]:


arr


# **write a program ,assign a value -1 at even place of an array**

# In[1]:


import numpy as np
arr=np.random.randint(10,40, size=(5,4))
print("Numpy Array: \n", arr)
for i in range(5*4):
    if(i%2==0):
        arr.flat[i]=-1
print("Modified Array: \n", arr)


# In[ ]:





# In[3]:


arr=np.random.randint(10,40, size=(5,4))
arr.flat[[2,4,6,8,10,12,14,16,18]]=-1
print(arr)


# **Numpy Reshape**

# In[9]:


arr=np.random.randint(10,40, size=(5,10))
print(arr)
print(arr.shape)


# In[10]:


arr_reshape=arr.reshape(10,5)
print(arr_reshape)


# In[11]:


arr_reshape=arr.reshape(25,2)
print(arr_reshape)


# **np.niter()**

# In[11]:


arr=np.random.randint(10,40, size=(5,10))
for i in np.nditer(arr):
    print(i,end=" ")


# In[12]:


arr=np.random.randint(10,40, size=(5,10))
for i in np.nditer(arr):
    if i>20:
        i[...] = -1
print(arr)


# In[13]:


get_ipython().run_cell_magic('time', '', 'import numpy as np\narr=np.random.randint(10,40, size=(5,4))\nprint("Original Array: \\n",arr)\nfor i in np.nditer(arr, op_flags=["readwrite"]):\n    if i>20:\n        i[...] = -1\nprint("Updated arry \\n", arr)\n')


# In[2]:


get_ipython().run_cell_magic('time', '', 'arr = np.random.randint(10,40, size=(5,4))\nprint("Origanal Array \\n", arr)\nfor i in np.arange(5*4):\n    if arr.flat[i] > 20:\n        arr.flat[i]=-1\nprint("updated array: \\n",arr)\n')


# In[3]:


get_ipython().run_cell_magic('time', '', 'arr_2d=np.random.randint(10,40, size=(5,4))\nprint("Original Array: \\n",arr_2d)\nfor i in np.nditer(arr_2d, op_flags=["readwrite"]):\n    if i%2 !=0:\n        i[...] = -1\nprint("Updated arry \\n", arr_2d)\n')


# In[31]:


import numpy as np
arr = np.random.randint(10,40, size=(5,4))
print("Numpy Array: \n", arr)
for i in range(5*4):
    if(i% 2 !=0):
        arr.flat[i] = -1
print("Modified Array: \n", arr)


# **Exercise: Given an array [1, -10, 2, 3, 0, 6], print the array in this order [0, 6, -10, 2, 1, 3]**

# In[58]:


array = np.array([1, -10, 2, 3, 0, 6])
array.flat[[0,1,2,3,4,5]] = 0,6,-10,2,1,3
print(array)


# In[29]:


arr= np.array([1, -10, 2, 3, 0, 6])
print(arr[[4,5,1,2,0,3]])


# # Python Operators on NUmpy Array

# In[35]:


arr=np.array([[1,2],[3,4]])
print(arr)


# In[36]:


print(arr+4)


# In[37]:


print(arr*2)


# In[38]:


print(arr>=2)


# In[39]:


print(arr %2==0)


# **Exercise: Write a program to generate an array with shape 5*4 at random containing positive integer. Perform an update by replacing all odd numbers with -1. (Without Using a Loop)**

# In[50]:


arr = np.random.randint(100, size=(5, 4))
print(arr)
a = arr % 2 !=0
if a.any() == True:
    arr[a] = -1
print("updated array: \n",arr)


# In[52]:


arr = np.random.randint(100, size=(5, 4))
print(arr)
a = arr % 2 !=0
arr[a] = -1
print("Updated Array: \n", arr)


# # Ellipsis(...)

# **Ellipsis(...):** expands to the number of (:)objects needed for the selection tuple to index all dimensions.
# 1. it is used for the to access the elements from the ndarray. 
# 2. it is also used for slicing consept

# In[18]:


arr = np.array([1,2,3,4,5,6,7,8,9])
arr[... , 0]


# In[46]:


arr = np.array([[1,2,3,4,5,6],[5,6,7,8,10,9],[12,13,14,15,17,18]])
print("array: \n",arr)
print(arr[... , 1])


# In[35]:


print(arr[:, 1])


# In[48]:


print(arr[..., 4])


# In[44]:


print(arr[:,4])


# In[38]:


print(arr[1, ...])


# In[39]:


print(arr[1,:])


# In[21]:


import numpy as np
arr =np.random.randint(100, size =(100, 100))
factor_num= (arr %5==0)  | ((arr%7 == 0) & (arr %2 !=0))
print(arr[factor_num])


# # Numpy Maths

# In[55]:


print("Square root: ", np.sqrt(4))
print("Exponent: " , np.exp(1))
print("Trigonometric Sin: ", np.sin(0))
print("Trigonometric Cos:", np.cos(0))


# In[22]:


arr= np.array([1,2,3,4])
print("Square root: ", np.sqrt(arr))
print("Exponent: " , np.exp(arr))
print("Trigonometric Sin: ", np.sin(arr))
print("Trigonometric Cos:", np.cos(arr))


# In[28]:


# matrix multiplication
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[6,7]])
print("Using matmul function: \n",np.matmul(x,y))
print("Using dot function: \n", np.dot(x,y))
print("Using @: \n",x @y)


# In[2]:


import numpy as np
arr =np.random.randint(10,40, size =(5,4))
np.where( arr %2 !=0, -1 , arr)


# In[25]:


heights = np.array([160, 180, 146, 162, 184, 180])

weights = np.array([50, 78, 45, 51, 80, 60])

np.cov(heights, weights)


# # np.unique()

# In[28]:


arr = np.array([[1,1],[1,1],[1,1]])
print("arr: \n", arr)
np.unique(arr)


# In[29]:


print(np.unique(arr, axis =1))


# In[30]:


print(np.unique(arr, axis=0))


# In[31]:


arr = np.array([[1,1],[1,1],[1,2]])
np.unique(arr, axis=1)


# In[32]:


arr = np.array([1, 2, 6, 4, 2, 3, 2])
unique_values , index = np.unique(arr, return_index =True)
print(unique_values)
print(index)


# In[40]:


arr = np.array([1,2,3,2,3])
u,inv = np.unique(arr, return_inverse= True)


# In[41]:


u


# In[42]:


inv


# In[44]:


u[inv]


# In[47]:


arr = np.array([1,2,3,4,5,2,3,4])
unique_val, counts = np.unique(arr, return_counts = True)


# In[48]:


unique_val


# In[49]:


counts


# In[52]:


# reconstruct the oroginal array using unique values and counts
np.repeat(unique_val, counts)


# # np.in1d()

# In[54]:


# if we want to find the arr2 of elements present in the arr1 or not
arr1 = np.array([1,2,4,5,6])
arr2 = np.array([1,2,3])
result = np.in1d(arr1, arr2)  #arr1 of elements present in the arr2.


# In[55]:


arr1[result] 


# In[56]:


result1 = np.in1d(arr1,arr2, invert= True)  #arr1 of elements not present in the arr1.


# In[57]:


arr1[result1]


# In[58]:


import numpy as np
array = np.arange(4).reshape((2,2))
print(array)
array1 = np.array([0,1])
result = np.isin(array,array1)


# In[59]:


result


# In[60]:


array[result]


# In[61]:


result1 = np.isin(array, array1, invert =True)  # reverse operation i.e the list of elements not in the array.


# In[62]:


result1


# In[63]:


array[result1]


# # numpy.setdiff1d

# In[3]:


# find the unique values in arra1 that are not present in arr2.
import numpy as np
arr1 = np.array([1,2,3,4,5,2,4])
arr2 = np.array([3,4,5,6,7])
np.setdiff1d(arr1,arr2)


# In[4]:


np.setdiff1d(arr2,arr1)


# In[8]:


arr = np.arange(4).reshape(2,2)
print("array_1: \n",arr,)
arr1 = np.array([[1,2],[4,5]])
print("array_2: \n", arr1)


# In[9]:


np.setdiff1d(arr, arr1)


# In[10]:


np.setdiff1d(arr1, arr)


# # numpy.setxor1d

# In[11]:


a = np.array([1,2,3,4,2,4])
b = np.array([2,4,5,6,5,7])
np.setxor1d(a,b)


# In[12]:


np.setxor1d(b,a)


# In[14]:


a1 = np.arange(4).reshape(2,2)
print("a1: \n",a1)
b1 = np.array([[2,3],[5,6]])
print("b1: \n",b1)
np.setxor1d(a1,b1)


# # Numpy Maths

# In[15]:


np.sqrt(4)


# In[16]:


arr = np.array([1,2,3,4])


# In[17]:


np.sqrt(arr)


# In[20]:


arr1 = np.array([[1,2],[3,4]])


# In[21]:


np.sqrt(arr1)


# In[22]:


np.exp(arr1)


# In[23]:


np.sin(arr1)


# In[24]:


np.cos(arr1)


# In[25]:


#element wise operations
x = np.array([[1,2],[3,4]])
y = np.array([[3,2],[4,6]])
print("Addition: \n", np.add(x,y))
print("Subtraction: \n", np.subtract(x,y))
print("Multiplication: \n", np.multiply(x,y))
print("Division: \n", np.divide(x,y))


# **3 ways to find the matrix multiplication**

# In[32]:


x = np.array([[1,2],[3,4]])  #[(1*3 + 2*4) (1*2 + 2*6)
y = np.array([[3,2],[4,6]])   #(3*3 + 4*4)  (3*2 + 4*6)]
print(x)
print(y)
print("way 1: \n", np.matmul(x,y))
print("way 2: \n", np.dot(x,y))
print("way 3: \n", x@y)


# In[33]:


#diagonal elements
x = np.array([[1,2,3],[4,5,6]])
np.diag(x)


# In[34]:


#Transpose
x.T


# # NUMPY STATISTICS

# In[38]:


x = np.array([1,2,3,4,5,123,45,76,89,231,56,90])
print("min_value: ",np.min(x))
print("max_value: ",np.max(x))
print("mean_value: ",np.mean(x))
print("std_value: ",np.std(x))
print("variant:", np.var(x))
print("median_value: ",np.median(x))


# In[40]:


x = np.array([[1,2],[3,4]])
print("Sum :",np.sum(x))


# In[44]:


print("Sum :",np.sum(x,axis = 0))  #colum wise


# In[46]:


print("Sum :",np.sum(x, axis = 1))


# In[47]:


y = np.array([[2,4],[5,7]])
print(np.min(y, axis =0)) 
print(np.min(y, axis =1))


# In[48]:


y = np.array([[2,4],[5,7]])
print(np.max(y, axis =0)) 
print(np.max(y, axis =1))


# # Miscellaneous Topics

# **linespace**

# In[50]:


print(np.linspace(1,5,9))


# In[51]:


print(np.linspace(1,5,7))


# **sorting**

# In[57]:


arr = np.random.randint(10,100, size = (4,4))
print(arr)


# In[58]:


np.sort(arr)


# In[59]:


np.sort(arr, axis =0)


# In[60]:


np.sort(arr, axis = 1)


# **Stacking**

# In[64]:


arr = np.arange(5,15).reshape(2,5)


# In[66]:


arr


# In[67]:


arr1 = np.arange(25,35).reshape(2,5)


# In[69]:


np.vstack([arr,arr1])


# In[72]:


np.hstack([arr,arr1])


# **concatinating**

# In[71]:


np.concatenate([arr,arr1], axis =0)


# In[73]:


np.concatenate([arr,arr1], axis =1)


# **append**

# In[76]:


np.append(arr,arr1, axis =0)


# In[78]:


np.append(arr,arr1, axis = 1)


# # Numpy.where()

# In[1]:


#syntax: np.where(condition, x,y) ,if condition is true than x return else y is return
#numpy where is appaly the broadcast application


# In[2]:


import numpy as np
arr = np.random.randint(10,40, size= (5,4))
print("Array: \n", arr)
np.where((arr >30) &(arr < 38),0,1)


# # Numpy Insert()

# In[2]:


import numpy as np
arr = np.array([[1,2],[3,4],[7,8]])
np.insert(arr, 1,6, axis = 0)    # here 5 indic1ates the index, 6 is represent the inserted element at 5th position 0 is row wise


# In[20]:


np.insert(arr, 1,6, axis = 1)  # 1 is indicates the colum wise.


# In[11]:


a = np.array([[1, 1], [2, 2], [3, 3]])
a


# In[12]:


np.insert(a,5,6)


# In[23]:


print(arr)
np.insert(arr,[1],[[1],[2],[3]],axis =1) # if we want to insert at 1st position with diff values. axis= 1 means colums wise


# In[26]:


np.insert(arr,   1, [1, 2, 3], axis=1) #if we want to insert at 1st position with diff values. axis= 1 means colums wise


# In[27]:


np.array_equal(np.insert(arr,   1, [1, 2, 3], axis=1),
               np.insert(arr, [1], [[1],[2],[3]], axis=1))


# In[31]:


np.insert(arr, slice(1,3),[0,0]) # here (1,3) are index positions and [0,0] are inserted at the 1 and 3 positions.


# In[3]:


np.insert(arr, [1,3],[3.4,False])


# Question: Randomly generate a matrix of shape (1Million, 2) and perform below mentioned operations:
# 
# a. Find the distances between each 2-Dimensional data point from the centroid (i.e. mean) of the given dataset. Append the newly calculated distances as new column with the given dataset.

# In[1]:


import numpy as np
arr = np.random.randint(100, size=(1000000,2))


# In[2]:


#find the Centroid
centroid = np.mean(arr, axis =0)


# In[3]:


#find the distence
dist = np.sum((arr - centroid)**2 , axis =1)
distance_midvalue = np.sqrt(dist)


# In[14]:


stacking = np.hstack((arr, distance_midvalue.reshape([1000000, 1])))
print(stacking)


# Create an array of shape (1Million, 100). Write a function that takes Numpy array arr and returns a new array where each element represents the count of even elements in original array's row. 
# 

# In[14]:


import numpy as np
arr = np.random.randint(100, size =(1000000, 100))
even = arr %2 ==0
print("Original array: \n ", arr)
print("Total even numbers from each row of original array:")
np.sum(even, axis =1).reshape((1000000,1))


# **How to replace items that satisfy a condition without affecting the original array?**

# In[17]:


arr_1 = np.array([[1, 2, 3], [4, 5, 6]])
arr_2 = np.array([[1], [2]])
print(arr_1 + arr_2)


# In[5]:


arr_1 = np.array([[1, 2, 3], [4, 5, 6]])

arr_2 = np.array([1])

print(arr_1 + arr_2)


# In[7]:


rr_1 = np.array([[1, 2, 3], [4, 5, 6]])

arr_2 = np.array([1, 2, 3])

print(arr_1 + arr_2)


# In[14]:


arr_1 = np.array([1, 2, 3, 4, 5])

arr_2 = np.array([1, 2, 3, 4])

print(arr_1 + arr_2)


# In[15]:


arr_1 = np.array([[1], [2], [3], [4], [5]])
arr_2 = np.array([1, 2, 3, 4])
print(arr_2 + arr_1)


# In[24]:


np.zeros((3,3))


# In[23]:


np.empty((3,3))


# In[3]:


import numpy as np


# In[5]:


arr = np.random.randint(10,50, size =(4,4))


# In[6]:


arr


# In[17]:


for i in range(len(arr)):
    for j in range(len(arr[i])):
        if(arr[i][j]>40):
            print(arr[i])
            break


# In[18]:


result = np.any(arr>40, axis =1)


# In[20]:


arr[result]


# # Numpy Basic Questions

# **Write a Numpy program to get the Numpy version and show the Numpy build configuration.**

# In[15]:


import numpy as np
print(np.__version__)
print(np.show_config())


# **2. Write a NumPy program to get help with the add function.**

# In[16]:


np.info(np.add)


# **3. Write a NumPy program to test whether none of the elements of a given array are zero.**

# In[17]:


arr = np.array([1,2,3,4,5])
np.all(arr) #print(np.all(x)): This line uses the np.all() function to test if all elements in the array 'x' are non-zero (i.e., not equal to zero). In this case, all elements are non-zero, so the function returns True.


# In[18]:


arr1 = np.array([[1,7,2,3,4],[1,2,3,4,5]])
np.all(arr1)


# In[20]:


arr1 = np.array([[1,7,0,3,4],[1,1,1,1,1]])
print(np.all(arr1) ) # it takes by default 0 for the checking 
print(np.any(arr1))


# In[21]:


np.all(arr1) ==1


# In[22]:


np.any(arr1) ==1


# **4. Write a NumPy program to test if any of the elements of a given array are non-zero.**

# In[23]:


arr1 = np.array([[1,0,0,0,0],[0,0,0,0,0]])
np.all(arr1) !=0


# In[24]:


np.any(arr1) !=0


# **5. How to create a boolean array?**

# In[25]:


np.zeros(10,dtype = bool).reshape(5,2)


# In[28]:


np.ones(10, dtype5 = bool).reshape(2,5)


# In[26]:


np.ones(10, dtype5 = bool).reshape(2,5)


# In[29]:


np.full((10), False, dtype = bool).reshape(5,2)


# **4. How to extract items that satisfy a given condition from 1D array?**

# In[30]:


arr = np.array([1,2,3,4,5,6,7,8,9])
odd = arr %2 !=0
arr[odd]


# **5. How to replace items that satisfy a condition with another value in numpy array?**

# In[31]:


# Question: Replace all odd numbers in arr with -1
# input: arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# output: array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])


# In[32]:


arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
np.where( arr %2 !=0, -1, arr)


# **6. How to replace items that satisfy a condition without affecting the original array?**

# In[33]:


arr = np.array([[1,2,3,4],[5,6,7,8]])
arr1 =arr #or arr.copy()
print("Original array: \n",arr)
np.where(arr1 %2 !=0, -1, arr1)


# **7. How to reshape an array?**

# In[34]:


arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr.reshape(5,2)


# In[35]:


arr.reshape(2,5)


# In[36]:


arr.reshape(2,-1)  # -1 represents the, it will tack  automatically how many colus are needed


# **8. How to stack two arrays vertically?**

# In[38]:


arr = np.random.randint(10,40, size =(5,3))
print("arr: \n",arr)
arr1 = np.random.randint(10,40, size =(4,3))
print("arr1: \n",arr1)


# In[39]:


np.vstack((arr,arr1))


# **8. How to stack two arrays horosontal?**

# In[40]:


arr = np.random.randint(10,40, size =(5,3))
print("arr: \n",arr)
arr1 = np.random.randint(10,40, size =(5,2))
print("arr1: \n",arr1)


# In[41]:


np.hstack((arr,arr1))


# **10. How to generate custom sequences in numpy without hardcoding?**

# In[42]:


# Input: a = np.array([1,2,3])
# Output: array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])


# In[43]:


arr = np.array([1,2,3])
np.repeat(arr,3)


# In[44]:


np.tile(arr,3)


# In[45]:


np.hstack((np.repeat(arr,3),np.tile(arr,3)))


# In[46]:


np.r_[(np.repeat(arr,3),np.tile(arr,3))]  #np.r_[] is represent he to concatenate the arrays row wise


# In[47]:


np.vstack((np.repeat(arr,3),np.tile(arr,3)))


# In[48]:


np.c_[(np.repeat(arr,3),np.tile(arr,3))]  #colum wise


# **11. How to get the common items between two python numpy arrays?**
Input: a = np.array([1,2,3,2,3,4,3,4,5,6])
       b = np.array([7,2,10,2,7,4,9,4,9,8])

# Output: array([2, 4])
# In[49]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)


# **12. How to remove from one array those items that exist in another?**
Input: a = np.array([1,2,3,4,5])
       b = np.array([5,6,7,8,9])

# Output: array([1,2,3,4])
# In[51]:


import numpy as np
x = np.array([1,2,3,4,5])
y = np.array([5,6,7,8,9])
coman_values,index_x,index_y = np.intersect1d(x, y, return_indices=True)


# In[52]:


#if we want to get the comman elements.
coman_values


# In[53]:


#if we want to get index of the common value from the array.
print("array__x:",index_x)
print("array__y",index_y)


# In[54]:


np.delete(x,index_x)


# In[55]:


np.delete(y,index_y)


# In[56]:


# we can also delete the common elements using set routes
np.setdiff1d(x,y)


# In[57]:


np.setdiff1d(y,x)


# **13. How to get the positions where elements of two arrays match?**
a = np.array([1,2,3,2,3,4,3,4,5,6])

b = np.array([7,2,10,2,7,4,9,4,9,8])

Output: (array([1, 3, 5, 7]),)
# In[58]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a == b)


# **14. How to extract all numbers between a given range from a numpy array?**
Get all items between 5 and 10 from a.

Input: a = np.array([2, 6, 1, 9, 10, 3, 27])
    
Output: (array([6, 9, 10])
# In[59]:


a = np.array([2, 6, 1, 9, 10, 3, 27])
indx = np.where((a>=5) & (a<=10))
a[indx]


# In[60]:


a[np.where((a>=5) & (a<=10))]


# In[61]:


a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])
np.where(a>b, a,b)


# **16. How to swap two columns in a 2d numpy array?**

# In[62]:


arr = np.arange(9).reshape(3,3)
print("Original array:\n", arr)


# In[63]:


#column wise
arr[:,[1,2,0]]


# In[64]:


#row wise
arr[[1,0,2],:]


# **18. How to reverse the rows of a 2D array?**

# In[65]:


arr = np.arange(9).reshape(3,3)
print("Original arra: \n", arr)
arr[2::-1,:]


# In[66]:


arr[:,2::-1]


# **20. How to create a 2D array containing random floats between 5 and 10?**

# In[68]:


arr = np.random.uniform(5,10,size=(5,3))
arr


# **21. How to print only 3 decimal places in python numpy array?**

# In[1]:


import numpy as np
arr = np.arange(9)
np.set_printoptions(threshold=6)
arr


# **23.How to print the full numpy array without truncating**

# In[2]:


np.set_printoptions(threshold=9)
arr


# **24. How to convert an array of arrays into a flat 1d array?**

# In[3]:


arr = np.arange(3)
arr1 = np.arange(3,7)
arr2 = np.arange(3,9)
arr


# In[4]:


arr1


# In[5]:


a = np.concatenate([arr,arr1,arr2])
a


# In[6]:


np.set_printoptions(threshold= 13)
a


# In[ ]:




