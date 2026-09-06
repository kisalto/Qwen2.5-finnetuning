# TCC - Fine-Tuning de Modelo para Geração de Código em Microsserviços TypeScript

## Descrição geral

Este projeto foi desenvolvido como Trabalho de Conclusão de Curso com o objetivo de criar um modelo de linguagem especializado em gerar código para microsserviços em TypeScript. A ideia central é utilizar um modelo base de código (Qwen/Qwen2.5-Coder-3B), coletar repositórios de projetos reais, transformar esse material em um dataset de treinamento e aplicar técnicas de adaptação eficiente com QLoRA para melhorar a capacidade do modelo de completar trechos de código, seguir padrões de estrutura e gerar arquivos coerentes para contextos de microsserviços.

O fluxo do projeto é:

1. Coleta de repositórios que representam microsserviços em TypeScript.
2. Download e filtragem dos arquivos relevantes.
3. Limpeza e organização do conteúdo em um dataset em JSONL.
4. Fine-tuning do modelo usando LoRA/QLoRA com Hugging Face.
5. Teste de geração e completamento de código com prompt específico.

Em resumo, o projeto busca reduzir a lacuna entre um modelo geral de código e a necessidade de um modelo mais alinhado com padrões de arquitetura de microsserviços em TypeScript, especialmente em cenários de autocompletar trechos de código e gerar estruturas iniciais de módulos, controllers, serviços e interfaces.

---

## Arquitetura de pastas

A estrutura do repositório está organizada da seguinte forma:

```text
.
├── data/
│   ├── data.csv                     # CSV original com links de repositórios utilizados como fonte
│   ├── dataset_clean.jsonl          # Dataset final limpo em formato JSONL para treinamento
│   ├── dataset_head.txt             # Amostra do dataset para validação rápida
│   ├── training_data_amostra.txt    # Trechos amostrados para testes/manutenção
│   └── training_data.txt            # Dados de treino consolidados para uso em experimentos
│
├── modelo_microsservicos_qlora/
│   ├── adapter_config.json          # Configuração do adaptador LoRA
│   ├── adapter_model.safetensors    # Pesos do adaptador treinado
│   ├── chat_template.jinja          # Template do chat do modelo
│   ├── README.md                    # Documentação do adapter/modelo
│   ├── tokenizer_config.json        # Configuração do tokenizer
│   └── tokenizer.json               # Tokenizer serializado
│
├── inferencia.py                    # Carrega o modelo base + adapter e gera código a partir de um prompt
├── process_dataset.py               # Filtra e organiza os arquivos de repositórios em dataset limpo
├── requirements.txt                 # Dependências do projeto
├── testar_checkpoint.py             # Testa um checkpoint específico do treinamento
├── training.py                      # Script principal de fine-tuning com QLoRA
├── readme.md                        # Documentação do projeto
└── .venv/                           # Ambiente virtual local (opcional)
```

### Descrição dos principais arquivos

- `process_dataset.py`
  - lê um CSV com URLs de repositórios;
  - baixa os arquivos ZIP dos projetos;
  - filtra arquivos relevantes para TypeScript e ignora diretórios como `node_modules`, `dist`, `build`, etc.;
  - salva o conteúdo em `data/dataset_clean.jsonl`.

- `training.py`
  - carrega o modelo base `Qwen/Qwen2.5-Coder-3B`;
  - configura quantização 4-bit com BitsAndBytes;
  - usa LoRA com `peft` e treino supervisionado com `trl.SFTTrainer`;
  - salva adaptador e checkpoints em `modelo_microsservicos_qlora`;
  - automaticamente retoma o último checkpoint, se existir.

- `inferencia.py`
  - carrega o modelo base e o adaptador treinado;
  - recebe um prompt de código;
  - gera continuação/complemento do trecho em linguagem TypeScript.

- `testar_checkpoint.py`
  - usado para testar um checkpoint específico, útil para validar o modelo em uma etapa intermediária do treinamento.

- `data/`
  - armazena os dados brutos e processados que alimentam o treinamento.

- `modelo_microsservicos_qlora/`
  - diretório contendo os artefatos gerados pelo treinamento do modelo adaptado.

---

## Requerimentos

### Ambiente

- Python 3.10 ou superior
- Windows 10/11 ou Linux
- NVIDIA GPU com suporte CUDA (recomendado)
- 16 GB de RAM ou mais
- 8 GB de VRAM como mínimo recomendado para execução com quantização 4-bit

### Dependências principais

O projeto utiliza as seguintes bibliotecas:

- `torch`
- `transformers`
- `datasets`
- `peft`
- `trl`
- `bitsandbytes`
- `pandas`
- `requests`
- `tqdm`
- `accelerate`

Todas as dependências estão listadas em `requirements.txt`.

---

## Como rodar

### 1. Clonar e preparar o ambiente

No PowerShell:

```powershell
cd C:\Users\User\Documents\projects\tcc
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se estiver em Linux/macOS:

```bash
cd /caminho/para/o/projeto
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Preparar o dataset

O script `process_dataset.py` coleta repositórios e cria o arquivo `data/dataset_clean.jsonl`.

```powershell
python process_dataset.py
```

> Ajuste o arquivo `data/data.csv` conforme a fonte de repositórios que você deseja usar.

### 3. Executar o treinamento

```powershell
python training.py
```

Esse script:

- carrega o modelo base;
- aplica QLoRA;
- treina sobre o dataset JSONL;
- salva checkpoints e o adaptador final em `modelo_microsservicos_qlora`.

Se houver checkpoints antigos, o treinamento tenta retomar automaticamente do último ponto salvo.

### 4. Rodar inferência

Depois do treinamento, teste a geração de código com:

```powershell
python inferencia.py
```

O script usa um prompt de exemplo em TypeScript e gera a continuação do código utilizando o modelo adaptado.

### 5. Testar um checkpoint específico

Se quiser verificar uma etapa específica do treino:

```powershell
python testar_checkpoint.py
```

Antes da execução, edite a variável `CHECKPOINT_DIR` dentro do arquivo para apontar para o diretório correto do checkpoint.

---

## Observações importantes

- O projeto foi pensado para uso com GPU NVIDIA, especialmente por causa da quantização 4-bit e do treinamento com LoRA.
- O dataset depende de repositórios reais de microsserviços em TypeScript, então a qualidade dos dados influencia diretamente na qualidade do modelo.
- A geração de código pode ser mais precisa quando os prompts são bem estruturados e próximos do estilo de arquivo que se deseja criar.
- O diretório `modelo_microsservicos_qlora` contém o resultado treinado e pode ser reutilizado para inferência sem repetir o treinamento.

---

## Resultado esperado

Com o treinamento concluído, o modelo deve ser capaz de:

- completar blocos de código TypeScript;
- continuar arquivos com estrutura de microsserviços;
- sugerir padrões de controllers, services, interfaces e imports;
- manter maior coerência sintática em trechos de código.

Esse repositório representa uma base funcional de um TCC em IA aplicada à geração de código, com foco em modelos adaptados para linguagem de programação e arquitetura de serviços.
