# TRAB. DE INTELIGÊNCIA ARTIFICIAL – A1 (PORTUGUÊS)

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

# ARTIFICIAL INTELLIGENCE PROJECT – A1 (ENGLISH)

Participating students:
- EDGAR DE SOUZA DIAS
- JOÃO EMANUEL MENDONÇA APÓSTOLO
- JOSÉ MATHEUS RIBEIRO DOS SANTOS 
- MARIA EDUARDA PIRES POSSARI DOS SANTOS 
- ULISSES DE JESUS CAVALCANTE

In the *modelos* folder, you can find the best results obtained during the training for both games tested.

## Dependencies
All dependencies used and their respective versions are available in the requirements.txt file.

## Usage instructions
To use the project, make sure you have Python 3.x installed on your machine. After that, you should create and activate a virtual environment using the following commands:
```
python -m venv ./venv/
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
Then, install the dependencies with the following command:
```
pip install -r requirements.txt
```
Great! With the environment set up and all dependencies configured, you can run the training file to generate the model with the command:
```
python treinamento.py
```
It will generate a model in a file named dqn_spaceinvaders.zip, which can be evaluated using the command:
```
python avaliacao.py
```