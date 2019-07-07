import numpy as np
import random

def lerDataSet(): #Ler a base de dados
    arq = open('IrisDataset.txt', 'r')
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
        matrizx[i][0] = 1
        for j in range(0,5):
            if(j != 4):
                matrizx[i][j+1] = temp[j]
            elif(temp[4] == 'Iris-setosa'):
                 matrizy[i][0] = 0
            elif(temp[4] == 'Iris-versicolor'):
                matrizy[i][0] = 1
            elif(temp[4] == 'Iris-virginica'):
                matrizy[i][0] = 2
        i = i + 1
    return matrizx, matrizy

def aprendizado(bd,plantadistinta): #PLA
    x , y = montarMatriz(bd) 
    w = np.array([0,0,0,0,0])
    melhor_w = np.array([0,0,0,0,0]) #Para os casos onde as retas não convergem.
    for i in range(0,len(bd)):
        if(int(y[i]) == plantadistinta):
            y[i] = -1
        else:
            y[i] = 1
    aux = 1
    cont = 0
    erros_w = 0
    erros_Melhor_w = -1
    while(aux and cont < 100):
        aux = 0
        for i in range(0,len(bd)):
            h =  np.dot(w.transpose() , x[i,:])
            
            if( np.sign(h) != np.sign(y[i])):
                w = np.copy(w + (y[i] * x[i,:]))
                aux = 1
                erros_w = erros_w + 1
                
        if(erros_Melhor_w != -1 and erros_w < erros_Melhor_w):
            erros_Melhor_w = erros_w
            melhor_w = np.copy(w)
        elif(erros_Melhor_w == -1):
            erros_Melhor_w = erros_w
            melhor_w = np.copy(w)
        erros_w = 0
        cont = cont+1

    return w

def teste(bd,w0,w1,simulacao):#A variável simulacao é uma váriavel de controle. Para evitar que na chamada da função por parte da simulação não seja impressa as matrizes. 
    x , y = montarMatriz(bd)
    resposta = np.zeros((len(bd),1))

    acertos = 0
    erros = 0

    for i in range(0,len(bd)):
        h0 =  np.dot(w0.transpose() , x[i,:])
        h1 =  np.dot(w1.transpose() , x[i,:])

        if( np.sign(h0) == np.sign(-1) ): #É sertosa
            resposta[i] = 0
        elif(np.sign(h1) == np.sign(-1) ): #É virginica
            resposta[i] = 2
        elif(np.sign(h0) == np.sign(1) and np.sign(h1) == np.sign(1)): #É versicolor
            resposta[i] = 1

    if(simulacao):
        for i in range(0,len(bd)):
            if(int(resposta[i]) == int(y[i]) ):
                acertos = acertos + 1
            else:
                erros = erros + 1
    else:
        print("Sertosa = 0 | Versicolor = 1 | Virginica = 2")
        print("Resultado final - ( Direita:Resultado da entrada de teste. | Esquerda: Gabaito )")
        for i in range(0,len(bd)):
            if(int(resposta[i]) == int(y[i]) ):
                print(resposta[i], "-",y[i], " - Acertou" )
                acertos = acertos + 1
            else:
                print(resposta[i], "-",y[i], " - Errou" )
                erros = erros + 1

        print("Acertos = ", acertos)
        print("Erros = ", erros)

    return acertos, erros 

def simulacoes(base_dados_teste, flor_w0,flor_w1,NSimulacoes):
    media_acertos = 0
    media_erros = 0
    acertos = 0
    erros = 0
    for i in range(0,NSimulacoes):
        base_dados = lerDataSet()
        base_dados_aprendizado, base_dados_teste = gerarBDs(base_dados[:])
        w0 = aprendizado(base_dados_aprendizado[:],flor_w0) #reta que melhor isola as plantas 0 (Iris-setosa) 
        w1 = aprendizado(base_dados_aprendizado[:],flor_w1) #reta que melhor isola as plantas 2 (Iris-virginica)
    
        acertos, erros  = teste(base_dados_teste, w0, w1,1)
        media_acertos = media_acertos + acertos
        media_erros = media_erros + erros

    media_acertos = media_acertos/NSimulacoes
    media_erros = media_erros/NSimulacoes
    print("100 Simulações de aprendizado e os resultados dos testes geraram as seguintes médias:")
    print("Média de acertos = ", media_acertos)
    print("Média de erros = ", media_erros)
        
base_dados = lerDataSet()
base_dados_aprendizado, base_dados_teste = gerarBDs(base_dados[:])

w0 = aprendizado(base_dados_aprendizado[:],0) #reta que melhor isola as plantas 0 (Iris-setosa) 
w1 = aprendizado(base_dados_aprendizado[:],2) #reta que melhor isola as plantas 2 (Iris-virginica)
print("Melhor w0 = ", w0)
print("Melhor w1 = ", w1)
teste(base_dados_teste, w0,w1,0)
simulacoes(base_dados_teste, 0, 2,100) #100 simulaçoes


