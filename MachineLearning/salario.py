import numpy as np
import random

#0 male | 1 female
#0 assistant | 1 associate | 2 full
#0 masters | 1 doctorate

def lerDataSet(): #Ler a base de dados
    arq = open('DiscriminationInSalariesDataset.txt', 'r')
    bd = []
    for linha in arq:
      bd.append(linha)
    arq.close
    return bd

def gerarBDs(bd): #Gerar o bd de aprendizagem e de teste
    bd_aprendizado = []
    bd_teste = []
    for i in range(0, int(80/100*len(bd))): #80% é definido para o bd de aprendizagem
        sorteado = random.randint(0, len(bd)-1)
        bd_aprendizado.append(bd[sorteado])
        del(bd[sorteado])
    bd_teste = np.copy(bd)     
    return bd_aprendizado, bd_teste


def montarMatriz(bd): #Montar matriz x e y
    matrizx = np.zeros((len(bd),5))
    matrizy = np.zeros((len(bd),1))
    i = 0
    for linha in bd:
        temp = linha.strip('\n').split(',')
        for j in range(0,6):
            if(j != 5):
                matrizx[i][j] = temp[j]
            else:
                matrizy[i][0] = temp[5]
        i = i + 1
    return matrizx, matrizy

def aprendizado(bd): 
    x , y = montarMatriz(bd) 
    w = np.array([0,0,0,0,0])
    pseudo_inversa_x = np.linalg.pinv(x) 
    w = np.dot(pseudo_inversa_x, y)
    return w

def teste(bd,w):
    x , y = montarMatriz(bd)
    resposta = np.zeros((len(bd),1))
    erro = 0
    print("Resultado               |        Valor ideal            |    Erro absoluto ")
    for i in range(0,len(bd)):
        resposta[i,0] =  np.dot(w.transpose() , x[i,:])
        print(resposta[i,0], "          |          ", y[i,0], "            |           ", abs(resposta[i,0]-y[i,0]))
        erro = erro + pow(resposta[i,0]-y[i,0],2)
    print("\nErro dentro da amostra:", erro/len(bd))

        
base_dados = lerDataSet()
base_dados_aprendizado, base_dados_teste = gerarBDs(base_dados[:])

w = aprendizado(base_dados_aprendizado[:])
teste(base_dados_teste,w)


