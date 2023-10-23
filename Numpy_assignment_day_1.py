#!/usr/bin/env python
# coding: utf-8

# ### NumPy Exercises
# ##### Now that we've learned about NumPy let's test your knowledge. We'll start off with a few simple tasks and then you'll be asked some more complicated questions.
# ##### IMPORTANT NOTE! Make sure you don't run the cells directly above the example output shown, otherwise you will end up writing over the example output!

# In[1]:


import numpy as np


# **1.  Write a NumPy program to create an array of 10 zeros,10 ones, 10 fives.**

# In[73]:


arr=np.zeros(10)
print("An array of 10 zeros: \n",arr)
arr1=np.ones(10)
print("An array of 10 ones: \n",arr1)
arr2=np.ones(10)*5
print("An array of 10 fives: \n",arr2)


# In[ ]:


## sample output


# **2.  Write a NumPy program to create an array of the integers from 30 to 70.**

# In[3]:


arr=np.arange(30,71)
print("Array of the integers from 30 to70 \n", arr)


# In[ ]:


### sample output


# **3. Write a NumPy program to create an array of all the even integers from 10 to 50.**

# In[5]:


arr=np.arange(10,50,2)
print("Array of all the even integers from 10 to 50 \n",arr)


# In[ ]:


## sample output


# **4. Write a NumPy program to create a 3x3 identity matrix.**

# In[6]:


arr=np.eye(3)
print("3x3 matrix: \n",arr)


# In[ ]:


## sample output


# ##### 5. Write a NumPy program to generate a random number between 0 and 1

# In[74]:


num= np.random.rand()
print("Random number between 0 and 1: \n", num)


# In[ ]:


## sample output


# ###### 6. Write a NumPy program to generate an array of 15 random numbers from a standard normal distribution.

# In[75]:


stand_normal= np.random.randn(15)
print("15 random numbers from a standard normal distribution: \n",stand_normal)


# In[ ]:


## sample output


# ###### 7. Write a NumPy program to create a 3X4 array using and iterate over it.

# In[10]:


arr=np.array([[10,11,12,13],[14,15,16,17],[18,19,20,21]])
print("Original array: \n",arr)
print("Each element of the array is:")
for i in arr:
    for j in i:
        print(j,end=" ")


# In[ ]:


## sample output


# ###### 8. Write a NumPy program to multiply the values ​​of two given vectors.

# In[12]:


vect_1=np.array([1,8,3,5])
vect_2=np.array([10,1,4,10])
print("Vector_1 \n",vect_1)
print("Vector_2 \n",vect_2)
print("Multiply the values of two said vectors: \n",vect_1*vect_2)


# In[ ]:


## sample output


# ###### 9. Write a NumPy program to reverse an array (first element becomes last).

# In[62]:


arr=np.array([12,13,14,15,16,17,18,19, 20, 21, 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37])
print("Original array:\n",arr)
print("Reverse array: \n")
print(arr[::-1])


# In[1]:


## sample output


# ##### 10. Write a NumPy program to create a 5x5 zero matrix with elements on the main diagonal equal to 1, 2, 3, 4, 5.

# In[27]:


arr=np.array([[1,0,0,0,0],[0,2,0,0,0],[0,0,3,0,0],[0,0,0,4,0],[0,0,0,0,5]])
print(arr)


# In[57]:


arr=np.arange(1,6)
daig_matrix=np.diag(arr)
print(daig_matrix)


# In[ ]:


## sample output


# ###### 11. Write a NumPy program to compute sum of all elements, sum of each column and sum of each row of a given array.

# In[42]:


arr=np.array([[0,1],[2,3]])
print("Original array: \n",arr)
print("Sum of all elements:\n", np.sum(arr))
colum=np.sum(arr,axis=0)
print("Sum of each column: \n", colum)
row=np.sum(arr,axis=1)
print("Sum of each row: \n", row)


# In[ ]:


## sample output


# ###### 12. Write a NumPy program to compute the inner product of two given vectors.

# In[38]:


arr=np.array([[4,5],[7,10]])
print("Original Vectors: \n",arr)
total_multi=1
for i in range(len(arr)):
    for j in range(len(arr)):
        if(i==j):
            total_multi*=arr[j]
print("Inner product of said vectors: \n",sum(total_multi))
        


# In[79]:


vect_1=[4,5]
vect_2=[7,10]
print("Original vectores: \n",vect_1,"\n",vect_2)
print("Inner product of said vectors: \n", np.dot(vect_1,vect_2))


# In[ ]:


## sample output


# In[ ]:





# In[ ]:





# In[ ]:




