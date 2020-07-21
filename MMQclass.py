import numpy as np
import math
class MMQ(object):

    def __init__(self, data_file):
        
        #The constructor receives the file passed to him and remove his information
        with open(data_file) as file:
            data = file.readlines()

        list = []
        for row in data:
            if not row.startswith('#') and not row.startswith('\n'):
                list.append(row.strip())

        self.polynomial_degree, self.title, self.name_x, self.name_y, *coords = list
        self.polynomial_degree= int(self.polynomial_degree) + 1

        self.coord_x = []
        self.coord_y = []
        for coord in coords:
            x, y = coord.split(',')
            x,y = float(x), float(y)
            self.coord_x.append(x)
            self.coord_y.append(y)
        self.LeastSquare()

    def LeastSquare(self):

        self.matrix_summation_x = []
        
        #for to cycle through rows
        for i in range(self.polynomial_degree):
            
            #Create an empty row
            row = []
            
            #for to cycle through columns
            for j in range(self.polynomial_degree):
                
                #Make the summation and put them in their respective columns
                row.append(np.sum(np.power(self.coord_x, (j + i) )))
            
            #Adds a row to the summation matrix.
            self.matrix_summation_x.append(row)

        #for to create a line matrix with y-related sums.
        self.matrix_summation_x_y = []
        
        #for to cycle through rows:
        for e in range(self.polynomial_degree):
            
            #Create an empty row
            row_1 = []
            
            #It makes the sums and places them in their respective places.
            row_1.append(np.sum((self.coord_y*(np.power(self.coord_x, e)))))
            
            #Adds a row to the summation matrix.
            self.matrix_summation_x_y.append(row_1)



        try:
            
            #Try to calculate the inverse matrix of the sum of X.
            self.matrix_inverse_x = np.linalg.inv(self.matrix_summation_x)
            self.matrix_result = self.matrix_inverse_x.dot(self.matrix_summation_x_y)

        except np.linalg.LinAlgError:
            #If not, let the user know.
            self.matrix_result = np.linalg.solve(self.matrix_summation_x, self.matrix_summation_x_y)
            pass


        self.Coefficient_Determination()
        self.MakeGraph(self.name_x, self.name_y, self.title)


        return self.matrix_result

    def MakeGraph(self, name_x, name_y, title):
        import matplotlib.pyplot as plt
        
        #Ploting coords.
        plt.plot(self.coord_x, self.coord_y, "o")

        #Determining plot boundaries.
        plt.xlim(self.coord_x[0]-0.2,self.coord_x[-1]+0.5)
        plt.ylim(self.coord_y[0]-0.2, self.coord_y[-1]+0.5)

        #naming the X and Y axes
        plt.xlabel(name_x)
        plt.ylabel(name_y)
        sum1 = ""
        sign = ""
        for i in range(self.polynomial_degree):
            if self.matrix_result[i][0] > 0:
                sign = " + "
            sum1 = sum1 + (str(" ") + sign + str( round(self.matrix_result[i][0], 5)) + str("*x^") + str(i))
            sign = ""

        #Title the chart.
        plt.title(title)
        
        #Writing the line equation and R²
        poly = str("f(x) = ") + sum1
        r_square = str("R²= ") + str(self.r_square)
        poly_r_square = poly + '\n' + r_square
        vet_size = len(self.coord_x)
        x = np.linspace(0, vet_size, (vet_size*50))
        plt.plot(x ,self.PolyCoefficients(x), label =  poly_r_square)


        #Showing the graph at the time of execution
        plt.legend(loc = 'best', fontsize = '13')
        plt.show()

    def Coefficient_Determination(self):

        #Calculates the mean of the ordinate axis.
        self.average_y = ((np.sum(self.coord_y))/(len(self.coord_y)))
        
        #Calculates the sum of the square of the difference between Y_i and Y_average
        self.sum_total = np.sum(np.power((self.coord_y - self.average_y), 2))
        
        #Transform a vector into a 1xN matrix
        #N = vector size
        self.new_coord_x = np.reshape(self.coord_x, (1, len(self.coord_x)))
        
        #Transforms a Kx1 matrix into a 1xK.
        # K = number of rows.
        self.new_matrix_result = np.reshape(self.matrix_result,(1, len(self.matrix_result)))
        
        #Declares a list to store f_i.
        self.predicted_model = []
        summ = 0
       
        #for to control the x_i index and size of predicted_model
        for i in range(len(self.new_coord_x[0])):
            
            #For to control the degree of the exponent and index of the result matrix.
            for j in range(self.polynomial_degree):
                
                #Calculates the value of f_i.
                summ = summ + (self.MakeSum(i, j))



            self.predicted_model.append((summ))
            summ = 0

        #Calculates the sum of squares due to regression.
        self.sum_regression = np.sum(np.power((self.predicted_model - self.average_y), 2))

        #Calculates the coefficient of determination, R².
        self.r_square = (self.sum_regression/self.sum_total)



    def PolyCoefficients(self, x):
        #Determines the number of existing coefficients.
        y = 0
        for i in range(self.polynomial_degree):
            y += self.matrix_result[i][0]*x**i
        return y

    def MakeSum(self, i, j):
        
        #Multiplies each position of the row matrix by each position of the column matrix raising to j, 
        # the matrix index value
        return (self.new_matrix_result[0][j]) * (np.power(self.new_coord_x[0][i], j))
