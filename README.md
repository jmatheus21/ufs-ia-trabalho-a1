# TRAB. DE INTELIGÊNCIA ARTIFICIAL – A1

Alunos participantes:
- EDGAR DE SOUZA DIAS
- JOÃO EMANUEL MENDONÇA APÓSTOLO
- JOSÉ MATHEUS RIBEIRO DOS SANTOS 
- MARIA EDUARDA PIRES POSSARI DOS SANTOS 
- ULISSES DE JESUS CAVALCANTE

Na pasta *modelos* estão os melhores resultados obtidos durante o treinamento para ambos os jogos testados.

## Dependências
Todas as dependências usadas e suas respectivas versões estão disponíveis no arquivo requirements.txt.

## Instruções de uso
Para utilizar o projeto, certifique-se de ter o Python 3.x instalado em sua máquina, após isso deve-se criar um ambiente virtual e ativá-lo com os comandos:
```
python -m venv ./venv/
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
Após isso, instale as dependências com o seguinte comando:
```
pip install -r requirements.txt
```
Ótimo, com o ambiente instalado e todas as dependências configuradas, pode-se executar o arquivo de treinamento para ele gerar o modelo com o comando:
```
python treinamento.py
```
Ele gerará um modelo em um arquivo *dqn_spaceinvaders.zip*, que pode ser avaliado com o comando:
```
python avaliacao.py
```