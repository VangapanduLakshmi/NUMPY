#!/usr/bin/env python
# coding: utf-8

# ### NumPy Exercises
# ##### Now that we've learned about NumPy let's test your knowledge. We'll start off with a few simple tasks and then you'll be asked some more complicated questions.
# ##### IMPORTANT NOTE! Make sure you don't run the cells directly above the example output shown, otherwise you will end up writing over the example output!

# ##### import numpy as np
# 

# ###### 1. Write a NumPy program to find the missing data in a given array.

# In[45]:


import numpy as np
arr = np.array([[ 3.,2.,np.nan,1.],[10.,12.,10. ,9.],[ 5.,np.nan,1.,np.nan]])
print("Origina array: \n", arr)

print("Find the missing data of the said array: \n", np.isnan(arr))


# In[ ]:


## sample output


# ##### 2. Write a NumPy program to check whether two arrays are equal (element wise) or not.

# In[46]:


arr_1 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
arr_2 = np.array([[1,2,3,4,5],[7,6,8,9,10]])
print("arra_1: \n", arr_1)
print("arr_2: \n", arr_2)
condition = arr_1 == arr_2
if condition.all() == True:
    print("Two arrays are equal")
else:
    print("Two arrays are not equal")


# ###### 3. Write a NumPy program to create a 4x4 array with random values, now create a new array from the said array swapping first and last rows.

# In[3]:


arr = np.random.randint(10, 40, size=(4, 4))
print("Original array: \n", arr)
arr[[0,3]]=arr[[3,0]]
arr[[1,2]] = arr[2, 1]
print("New array after swapping first and last rows of the said array: \n",arr)


# In[ ]:


## sample output


# ###### 4. Write a NumPy program to convert a list and tuple into arrays.

# In[10]:


lst = [1,2,3,4,5]
print("List:",lst)
print(type(lst))
tpl = (6,7,8,9,10)
print("Tuple:", tpl)
print(type(tpl))
#to convert a list and tuple into arrays
arr_1 = np.array(lst)
arr_2 = np.array(tpl)
print("arr_1: ",type(arr_1))
print("arr_2: ",type(arr_2))


# ###### 5. Write a NumPy program to find common values between two arrays.

# In[32]:


arr_1 = np.array([[1,3,4,2,13],[6,7,8,9,10]])
arr_2 = np.array([[1,3,6,7,8],[9,10,11,12,13]])
print("arr_1: \n",arr_1)
print("arr_2: \n", arr_2)
com_values = np.intersect1d(arr_1, arr_2)
print("common values between two arrays:", com_values)


# ###### 6. Write a NumPy program to create a new shape to an array without changing its data.

# In[2]:


array = np.random.randint(10, size=(3, 2))
print("Original Array: \n", array)
print("Reshape 2x3: \n", array.reshape(2,3))


# In[ ]:


### sample output


# ###### 7. Write a NumPy program to count the frequency of unique values in numpy array.

# In[29]:


arr = np.array([10,10,20,10,20,20,20,30,30,50,40,40])
print("Original array: \n", arr)
uniq_values, count = np.unique(arr, return_counts = True) #returs the unique values and counts the frequency of the uniq values
print("Frequency of unique values of the said array:")
print(np.vstack((uniq_values,count))) # np.vstack is used to stack the two arrays vertically.


# In[ ]:


## sample output


# ###### 8. Write a NumPy program to broadcast on different shapes of arrays where p(3,3) + q(3).

# In[43]:


array_1 = np.array([[0,0,0],[1,2,3],[4,5,6]])
array_2 = np.array([10,11,12])
print("Original arrays:")
print("Array-1: \n", array_1)
print("Array-2: \n", array_2)
print("New Array: \n", array_1 + array_2)


# In[ ]:


## sample output


# ###### 9. Write a NumPy program to extract all the elements of the second row from a given (4x4) array.

# In[33]:


import numpy as np
arra_data = np.arange(0,16).reshape((4, 4))
print("Original array:")
print(arra_data)
print("\nExtracted data: Second row")
print(arra_data[1])


# ###### 10. Write a NumPy program to extract third and fourth elements of the first and second rows from a given (4x4) array.

# In[51]:


arr = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
print("Original array: \n", arr)
print("Extracted data: Third and fourth elements of the first and second rows: \n",arr[:2,2:])


# In[ ]:


## sample output


# ###### 11. Write a NumPy program to get the dates of yesterday, today and tomorrow.

# In[18]:


import numpy as np
today_date = np.datetime64("today")
yest_date = today_date - np.timedelta64(1,"D")
tmw_date = today_date + np.timedelta64(1, "D")
print("Yesterday Date: ", yest_date)
print("Today Date:", today_date)
print("Tomorrow Date:", tmw_date)


# ###### 12. Write a NumPy program to find the first Monday in May 2017.

# In[42]:


import numpy as np
first_monday = np.busday_offset('2017-05', 0, roll='forward') #
print("First Monday in May 2017: \n", first_monday)


# In[ ]:


## sample output

