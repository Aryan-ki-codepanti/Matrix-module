import numpy as np

class Matrix:

	""" This class helps to simulate some matrix operations for square matrices having order 2 or 3"""
			
	matrix_count = 0
	def __init__(self,arr,order):
		
		"""This is a constructor method that will make a matrix object.It requires 
		2 positional arguments first Matrix in form of nested list having each row list as element
		and second one being a int type of order wether its square matrix of order 2 or 3"""

		self.arr = np.array(arr)
		self.order = order
		Matrix.matrix_count += 1

		if self.order == 2:
			self.a11 = self.arr[0][0] 
			self.a12 = self.arr[0][1]
			
			self.a21 = self.arr[1][0]
			self.a22 = self.arr[1][1]

			self.c11 = self.a22
			self.c12 = (-1)*self.a21
			self.c21 = (-1)*self.a12
			self.c22 = self.a11


		elif self.order == 3:
		
			self.a11 = self.arr[0][0] 
			self.a12 = self.arr[0][1]
			self.a13 = self.arr[0][2]

			self.a21 = self.arr[1][0]
			self.a22 = self.arr[1][1]
			self.a23 = self.arr[1][2]

			self.a31 = self.arr[2][0]
			self.a32 = self.arr[2][1]
			self.a33 = self.arr[2][2]

			self.c11 = (self.a22*self.a33) - (self.a32*self.a23) 
			self.c12 = -(self.a21*self.a33) + (self.a31*self.a23) 
			self.c13 = (self.a21*self.a32) - (self.a31*self.a22) 
			
			self.c21 = -(self.a22*self.a33) + (self.a32*self.a23) 
			self.c22 = (self.a11*self.a33) - (self.a31*self.a13) 
			self.c23 = -(self.a11*self.a32) + (self.a31*self.a12) 
			
			self.c31 = (self.a12*self.a23) - (self.a22*self.a13) 
			self.c32 = -(self.a11*self.a23) + (self.a21*self.a13) 
			self.c33 = (self.a11*self.a22) - (self.a21*self.a12)
	
	def cofactor(self,key):
		
		""" corresponding to an element of matrix this returns cofactor of that element ,takes string as argument
		depicting element of matrix example 'a11' and so on."""
		
		if self.order == 2:
			if key == 'a11':
				return self.c11
			elif key == 'a12':
				return self.c12
			elif key == 'a21':
				return self.c21
			else:
				return self.c22

		elif self.order == 3:
			if key == 'a11':
				return self.c11

			elif key == 'a13':
				return self.c13

			elif key == 'a12':
				return self.c12

			elif key == 'a21':
				return self.c21

			elif key == 'a22':
				return self.c22


			elif key == 'a23':
				return self.c23


			elif key == 'a31':
				return self.c31

			elif key == 'a32':
				return self.c32

			else:
				return self.c33

	def minor(self,key):

		""" corresponding to an element of matrix this returns cofactor of that element ,takes string as argument
		depicting element of matrix example 'a11' and so on."""

		if self.order == 2:
			if key == 'a11':
				return self.c11
			elif key == 'a12':
				return -(self.c12)
			elif key == 'a21':
				return -(self.c21)
			else:
				return self.c22

		elif self.order == 3:
			if key == 'a11':
				return self.c11

			elif key == 'a13':
				return self.c13

			elif key == 'a12':
				return -(self.c12)

			elif key == 'a21':
				return -(self.c21)

			elif key == 'a22':
				return self.c22


			elif key == 'a23':
				return -(self.c23)


			elif key == 'a31':
				return self.c31

			elif key == 'a32':
				return -(self.c32)

			else:
				return self.c33

	def get_transpose(self):
		
		""" this method returns transpose of matrix """
		
		if self.order == 2:
			self.transpose = np.array([[self.a11,self.a21],[self.a12,self.a22]])
			return self.transpose

		elif self.order == 3:
			self.transpose = np.array([[self.a11,self.a21,self.a31],[self.a12,self.a22,self.a32],[self.a13,self.a23,self.a33]])
			return self.transpose

	def get_adjoint(self):
		
		""" this method returns adjoint of matrix"""
		
		if self.order == 2:
			self.adjoint = np.array([[self.c11,self.c21],[self.c12,self.c22]])
			return self.adjoint

		elif self.order == 3:
			self.adjoint = np.array([[self.c11,self.c21,self.c31],[self.c12,self.c22,self.c32],[self.c13,self.c23,self.c33]])
			return self.adjoint

	def get_matrix(self):
		
		""" returns matrix as it is"""
		
		return self.arr

	def get_order(self):
		
		""" returns order of matrix as integer"""
		
		return self.order





	def determinant(self):
		
		"""this method returns determinant of matrix in form of integer"""
		
		if self.order == 2:
			self.det = (self.a11*self.c11)+(self.a12*self.c12)
			return self.det

		elif self.order == 3:

			self.det = (self.a11*self.c11) + (self.a12*self.c12) + (self.a13*self.c13) 
			return self.det

	def is_singular(self):
		
		"""returns boolean True if matrix has 0 determinant otherwise False """
		
		if self.order == 2:
			self.det = (self.a11*self.c11)+(self.a12*self.c12)
			

		elif self.order == 3:
			self.det = (self.a11*self.c11) + (self.a12*self.c12) + (self.a13*self.c13)

		if self.det == 0:
			return True
		else:
			return False

	def get_cofactor_matrix(self):
		
		""" returns matrix of cofactors"""
		
		self.cofactor_matrix = np.array([[self.c11,self.c12,self.c13],[self.c21,self.c22,self.c23],[self.c31,self.c32,self.c33]])
		return self.cofactor_matrix
	def get_inverse(self):

		""" if inverse of matrix exist it returns inverse of it else displays meesage if inverse doesn't exist"""
		
		if self.order == 2:
			self.adjoint = np.array([[self.c11,self.c21],[self.c12,self.c22]])
			self.det = (self.a11*self.c11)+(self.a12*self.c12)

		elif self.order == 3:
			self.adjoint = np.array([[self.c11,self.c21,self.c31],[self.c12,self.c22,self.c32],[self.c13,self.c23,self.c33]])
			self.det = (self.a11*self.c11) + (self.a12*self.c12) + (self.a13*self.c13)

		if self.det == 0:
			return 'Inverse does not exist'
		else:
			self.inverse = (self.adjoint)/self.det

			return self.inverse

	def get_row(self,n):

		"""takes integer value n and returns nth row """

		if self.order == 2:
			if n == 1:
				self.row = np.array([self.a11,self.a12])
			else:
				self.row = np.array([self.a21,self.a22])

		elif self.order == 3:
			if n == 1:
				self.row = np.array([self.a11,self.a12,self.a13])
			elif n == 2:
				self.row = np.array([self.a21,self.a22,self.a23])
			else:
				self.row = np.array([self.a31,self.a32,self.a33])

		return self.row

	def get_column(self,n):

		"""takes integer value n and returns nth column """
		
		if self.order == 2:
			if n == 1:
				self.column = np.array([[self.a11],[self.a21]])
			else:
				self.column = np.array([[self.a21],[self.a22]])
		elif self.order == 3:
			if n == 1:
				self.column = np.array([[self.a11],[self.a21],[self.a31]])
			elif n == 2:
				self.column = np.array([[self.a12],[self.a22],[self.a32]])
			else:
				self.column = np.array([[self.a13],[self.a23],[self.a33]])
		return self.column

	def __repr__(self):
		""" This for matrix object representation as array of integers"""
		return f"matrix\n({self.arr},{self.order})"

	def __str__(self):
		""" This for matrix object representation in form of its determinant and order"""
		return f"Order: {self.order}\nDeterminant: {self.determinant()}"

	def __add__(self,other):
		
		"""For Matrix Addition of square matrices of same order"""
		try:
			return self.arr + other.arr

		except:
			return NotImplemented

	def __sub__(self,other):
		try:
			"""For Matrix subtraction of square matrices of same order"""
			return self.arr - other.arr
		except:
			return NotImplemented
	def __mul__(self,other):
		"""For Matrix multiplication of square matrices of same order"""
		try:
			if self.order == 2 and other.order == 2:
				new_a11 = (self.a11*other.a11) + (self.a12*other.a21)
				new_a12 = (self.a11*other.a12) + (self.a12*other.a22)

				new_a21 = (self.a21*other.a11) + (self.a22*other.a21)
				new_a22 = (self.a21*other.a12) + (self.a22*other.a22)

				return np.array([[new_a11,new_a12],[new_a21,new_a22]])
			elif self.order == 3 and other.order == 3:
				new_a11 = (self.a11*other.a11) + (self.a12*other.a21) + (self.a13*other.a31)
				new_a12 = (self.a11*other.a12) + (self.a12*other.a22) + (self.a13*other.a32)
				new_a13 = (self.a11*other.a13) + (self.a12*other.a23) + (self.a13*other.a33)

				new_a21 = (self.a21*other.a11) + (self.a22*other.a21) + (self.a23*other.a31)
				new_a22 = (self.a21*other.a12) + (self.a22*other.a22) + (self.a23*other.a32)
				new_a23 = (self.a21*other.a13) + (self.a22*other.a23) + (self.a23*other.a33)

				new_a31 = (self.a31*other.a11) + (self.a32*other.a21) + (self.a33*other.a31)
				new_a32 = (self.a31*other.a12) + (self.a32*other.a22) + (self.a33*other.a32)
				new_a33 = (self.a31*other.a13) + (self.a32*other.a23) + (self.a33*other.a33)

				return np.array([[new_a11,new_a12,new_a13],[new_a21,new_a22,new_a23],[new_a31,new_a32,new_a33]])
		except:
			return NotImplemented

	@staticmethod
	def is_square_mat(arr):
		"""To determine if entered array represents square matrix or not
			returns True if square matrix else False """
		try:
			rows = len(arr)
			cols = [len(i) for i in arr]
			if len(set(cols)) == 1 and cols[0] == rows:
				return True
			return False		

		except:
			return False
# example
my_mat = Matrix([[1,1,1],[1,2,3],[9,5,6]],3)
"""print("Orignal matrix")
print(my_mat.get_matrix())
print("-"*60)
print("Transpose")
print(my_mat.get_transpose())
print("-"*60)
print("Inverse")
print(my_mat.get_inverse())
print("-"*60)
print("Cofactor matrix")
print(my_mat.get_cofactor_matrix())
print("-"*60)
print("Adjoint")
print(my_mat.get_adjoint())
print("-"*60)
print('Row 2')
print(my_mat.get_row(2))
print("-"*60)
print("Column 3")
print(my_mat.get_column(3))
print("-"*60)
print("Cofactor of a11")
print(my_mat.cofactor('a11'))
print("-"*60)
print('Minor of a11')
print(my_mat.minor('a11'))
print("-"*60)
print(my_mat.is_singular())
print('-'*60)
print('Determinant of matrix')
print(my_mat.determinant())
help(Matrix)"""

