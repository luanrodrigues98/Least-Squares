import numpy as np
import math
class MMQ(object):

    def __init__(self, arquivo):
        #O Construtor que recebe o arquivo passado para ele e retirar suas informações.
        with open(arquivo) as file:
            dados = file.readlines()

        list = []
        for linha in dados:
            if not linha.startswith('#') and not linha.startswith('\n'):
                list.append(linha.strip())

        self.grau_polinomio, self.titulo, self.nome_x, self.nome_y, *coords = list
        self.grau_polinomio= int(self.grau_polinomio) + 1

        self.pontos_x = []
        self.pontos_y = []
        for coord in coords:
            x, y = coord.split(',')
            x,y = float(x), float(y)
            self.pontos_x.append(x)
            self.pontos_y.append(y)
        self.LeastSquare()

    def LeastSquare(self):

        self.matrix_somatorios_x = []
        #For para percorrer as linhas:
        for i in range(self.grau_polinomio):
            #Cria uma linha vazia
            linha = []
            #For para percorrer as colunas
            for j in range(self.grau_polinomio):
                #Faz os somatorios e coloca-los em suas respectivas colunas
                linha.append(np.sum(np.power(self.pontos_x, (j + i) )))
            #Adiciona uma linha a matriz de somatorios.
            self.matrix_somatorios_x.append(linha)

        #For para criar uma matriz linha com os somatorios relacionados
        # a y.
        self.matrix_somatorios_x_y = []
        #For para percorrer as linhas:
        for e in range(self.grau_polinomio):
            #Cria uma linha vazia
            linha_1 = []
            #Faz os somatorios e os coloca em seus respectivos lugares.
            linha_1.append(np.sum((self.pontos_y*(np.power(self.pontos_x, e)))))
            #Adiciona uma linha a matriz de somatorios.
            self.matrix_somatorios_x_y.append(linha_1)



        try:
            #Tenta calcular a matriz inversa dos somatorios de X.
            self.matrix_inverse_x = np.linalg.inv(self.matrix_somatorios_x)
            self.matrix_result = self.matrix_inverse_x.dot(self.matrix_somatorios_x_y)

        except np.linalg.LinAlgError:
            #Caso não consiga, avisa o usuario.
            self.matrix_result = np.linalg.solve(self.matrix_somatorios_x, self.matrix_somatorios_x_y)
            pass


        self.Coeficiente_Determinacao()
        self.MakeGraph(self.nome_x, self.nome_y, self.titulo)


        return self.matrix_result

    def MakeGraph(self, nome_x, nome_y, titulo):
        import matplotlib.pyplot as plt
        #Plotando os pontos.
        plt.plot(self.pontos_x, self.pontos_y, "o")

        #Determinando os limites de plot.
        plt.xlim(self.pontos_x[0]-0.2,self.pontos_x[-1]+0.5)
        plt.ylim(self.pontos_y[0]-0.2, self.pontos_y[-1]+0.5)

        #Nomeando os eixos X e Y.
        plt.xlabel(nome_x)
        plt.ylabel(nome_y)
        soma = ""
        sign = ""
        for i in range(self.grau_polinomio):
            if self.matrix_result[i][0] > 0:
                sign = " + "
            soma = soma + (str(" ") + sign + str( round(self.matrix_result[i][0], 5)) + str("*x^") + str(i))
            sign = ""

        #Dando titulo ao gráfico.
        plt.title(titulo)
        #Escrevendo a equação de reta e R²
        poli = str("f(x) = ") + soma
        r_quad = str("R²= ") + str(self.r_quad)
        poli_r_quad = poli + '\n' + r_quad
        tamanho_vetor = len(self.pontos_x)
        x = np.linspace(0, tamanho_vetor, (tamanho_vetor*50))
        plt.plot(x ,self.PolyCoefficients(x), label =  poli_r_quad)


        #Mostrando o grafico no momento da execução
        plt.legend(loc = 'best', fontsize = '13')
        plt.show()

    def Coeficiente_Determinacao(self):

        #Calcula a media pontos em Y.
        self.media_y = ((np.sum(self.pontos_y))/(len(self.pontos_y)))
        # Calcula a soma do quadrado da diferença entre Y_i e Y_medio
        self.soma_total = np.sum(np.power((self.pontos_y - self.media_y), 2))
        #Transforma um vetor em uma matriz 1xN
        # N = tamanho do vetor
        self.new_pontos_x = np.reshape(self.pontos_x, (1, len(self.pontos_x)))
        #Transforma uma matriz Kx1 em uma 1xK.
        # K = número de linhas da matriz.
        self.new_matrix_result = np.reshape(self.matrix_result,(1, len(self.matrix_result)))
        #Declara o uma lista para armazenar f_i.
        self.modelo_previsto = []
        somatorio = 0
        #For para controlar o indice de x_i e o tamanho de modelo_previsto

        for i in range(len(self.new_pontos_x[0])):
            #For para controlar o grau do expoente e indice da matriz resultado.
            for j in range(self.grau_polinomio):
                #Calcula o valor de f_i.
                somatorio = somatorio + (self.MakeSum(i, j))



            self.modelo_previsto.append((somatorio))
            somatorio = 0

        #Calcula a soma dos quadrados devido a regressão.
        self.soma_regressao = np.sum(np.power((self.modelo_previsto - self.media_y), 2))

        #Calcula o coeficiente de determinação R².
        self.r_quad = (self.soma_regressao/self.soma_total)



    def PolyCoefficients(self, x):
        #Determina o número de coeficientes existentes.

        y = 0
        for i in range(self.grau_polinomio):
            y += self.matrix_result[i][0]*x**i
        return y

    def MakeSum(self, i, j):
        #Multiplica cada posição da matriz linha por cada posição da matriz coluna
        #elevando a j, valor do indice da matriz
        return (self.new_matrix_result[0][j]) * (np.power(self.new_pontos_x[0][i], j))
