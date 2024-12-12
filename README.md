# Reconhecimento Facial para Chamada

Este projeto tem como objetivo o reconhecimento facial para marcar a presença de alunos em sala de aula, utilizando diferentes bibliotecas e frameworks para testar e escolher a melhor solução. Todos os arquivos, exceto o `main.py`, são rascunhos utilizados para testar frameworks e modelos.

## Estrutura do Projeto

- **Mp_version.py**: Versão utilizando somente o MediaPipe. Embora tenha mostrado resultados satisfatórios para interação com dispositivos móveis, a detecção facial não é ideal para múltiplas faces. Por isso, essa versão provavelmente não será a solução final.
  
- **df_version.py**: A versão principal do projeto, que utiliza a biblioteca DeepFace. A DeepFace oferece múltiplos algoritmos de reconhecimento facial, mas a implementação ainda não está 100% correta.

- **db**: Pasta contendo a base de dados com imagens faciais. A DeepFace utiliza essas imagens para realizar o reconhecimento, embora não esteja claro se essa funcionalidade será necessária no futuro.

- **Outros arquivos**: Arquivos de testes para explorar diferentes abordagens e bibliotecas de reconhecimento facial.

## Histórico de Desenvolvimento

### 15/11

Início do projeto utilizando o MediaPipe do Google, pois, segundo minhas pesquisas, é o melhor para interação com dispositivos móveis. No entanto, a detecção facial não é muito eficiente, principalmente para múltiplas faces. Planejo testar modelos do TensorFlow Lite, que são teoricamente mais eficientes.

**Observação**: Preciso de dados melhores para continuar os testes. Se possível, tirar uma foto de uma sala de aula com fotos de cada aluno para realizar testes mais precisos.

### 16/11

Os algoritmos de detecção facial para dispositivos móveis não são muito eficazes para múltiplas faces. Eles funcionam bem quando a face está centralizada, mas para reconhecimento de várias faces, o desempenho é limitado. A partir disso, decidi focar no reconhecimento facial de faces já conhecidas, que serão salvas em um arquivo `.json`.

### 12/12

Mudanças importantes no projeto:

- O arquivo `main.py` foi renomeado para `Mp_version.py`, que utiliza apenas o MediaPipe. No entanto, não é suficiente para o uso pretendido.
- O arquivo principal agora é o `df_version.py`, que utiliza a biblioteca DeepFace. A DeepFace permite o uso de vários algoritmos de reconhecimento facial, mas a implementação ainda não está 100% correta.
- A pasta `db` contém imagens que podem ser usadas pela DeepFace para treinamento e reconhecimento, embora não saibamos se essa funcionalidade será necessária.

Os outros arquivos são testes com diferentes abordagens. Recomendo a leitura da documentação da biblioteca DeepFace. Inclusive, você sugeriu a ideia de separar as faces da imagem para processá-las individualmente, e a DeepFace possui a função `extract_faces()` que faz exatamente isso. Após extrair as faces, é possível usar a função `represent()` para gerar o embedding facial.

**Próximos Passos**:

- Adicionar fotos para a base de dados.
- Realizar mais testes para verificar se o código está funcionando corretamente.
