# Otimização de Operações de E/S em Aplicações Científicas (SciML)

Este repositório contém o código, os scripts de submissão e as ferramentas de análise de dados referentes ao artigo **"Otimização de operações de E/S em aplicações científicas de aprendizado de máquina guiadas pelo Drishti"**. 

O estudo foca na mitigação de gargalos de Entrada/Saída (I/O) no treinamento de Redes Neurais Operatórias (FNO) e U-Net, utilizando o benchmark [PDEBench](https://github.com/pdebench/PDEBench) em ambientes de Computação de Alto Desempenho (HPC).

## Aviso sobre o PDEBench e Modificações
Este repositório inclui uma versão local do **PDEBench**. 
* **Dependências Nativas:** Para informações detalhadas sobre as dependências originais da física e matemática do benchmark, consulte a [documentação oficial do PDEBench](https://github.com/pdebench/PDEBench).
* **Modificações Realizadas:** O código do PDEBench contido neste repositório **foi modificado** em relação ao original. As alterações foram feitas especificamente nos *data loaders* (`pdebench/models/fno/` e `pdebench/models/unet/`) para:
  1. Instrumentar a fase de treinamento e extrair métricas de execução usando o **Darshan DXT**.
  2. Implementar estratégias de otimização de E/S, incluindo **Alinhamento de Requisições (4KB)**, **MPI-IO Coletivo** e **Buffering**.

## Configuração do Ambiente e Ferramentas

Para reproduzir este ambiente e executar os scripts de treinamento e análise, é necessário configurar o ambiente Python e as ferramentas de HPC.

### 1. Ferramentas de Perfilamento e Análise (Darshan e Drishti)
O monitoramento de E/S e a geração de recomendações dependem do **Darshan** e do **Drishti**. Ambas as ferramentas não fazem parte do ambiente Python (Conda) e **foram instaladas e compiladas localmente na máquina**, seguindo rigorosamente as diretrizes de suas respectivas documentações oficiais. 
* [Documentação do Darshan](https://darshan.readthedocs.io/en/latest/)
* [Documentação do Drishti](https://drishti-io.readthedocs.io/en/latest/)

### 2. Ambiente Python (Miniconda/Anaconda)
É necessário ter o [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou Anaconda instalado para as dependências de Machine Learning.

1. Clone este repositório:
   ```bash
   git clone [https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git](https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git)
   cd NOME_DO_REPOSITORIO
   ```

2. Recrie o ambiente Conda a partir do arquivo de requisitos:
   ```bash
   conda env create -f environment.yml
   conda activate nome_do_ambiente
   ```

## Estrutura de Scripts e Execução

O fluxo de trabalho deste repositório é dividido em execução no cluster (Slurm) e processamento de logs. Abaixo está a descrição do que cada script principal faz:

### 1. Submissão e Treinamento (Pasta `jobs/`)
* **`jobs/submit_workflow.sh`**: Script principal de orquestração. Ele dispara as rotinas de treinamento no gerenciador de filas Slurm, iterando sobre as variações físicas (ex: coeficientes do Advection e Burgers) e as estratégias de E/S.
* **`jobs/train.sh`**: O *job script* do Slurm submetido aos nós computacionais. Ele configura as variáveis de ambiente do Darshan (ex: `DXT_ENABLE_IO_TRACE=1`, `LD_PRELOAD`) e executa o código Python modificado do PDEBench alocando recursos de GPU (ex: RTX 4090).

### 2. Extração e Processamento de Logs (Pasta `scripts/`)
* **`extract-darshan-logs.py`**: Interage com os arquivos `.darshan` gerados pela execução. Ele extrai os traços binários do DXT e os converte em um formato tabular legível para análise temporal e de vazão (throughput).
* **`process-all-logs.py`**: Consolida os dados extraídos de múltiplas rodadas do Darshan/Slurm. Útil para varrer diretórios de *outputs* e unificar os tempos de E/S e gargalos apontados.
* **`process-all-models.py`**: Lê os resultados de predição gerados pelo PDEBench (métricas numéricas como MSE, nRMSE e MaxError) para garantir a integridade convergencial da rede neural após as otimizações de E/S. Gera os arquivos CSV (ex: `Evaluation_Summary_Per_Optimization.csv`).

### 3. Geração de Resultados para o Artigo (Pasta `article/`)
* **`article/create-graph.py`**: Consome os arquivos `.csv` consolidados e gera as figuras vetoriais em formato PDF (ex: `comparacao-otimizacoes-FNO.pdf`) utilizadas na seção de Resultados do artigo.

## ✒️ Autores e Créditos

Este trabalho foi desenvolvido por pesquisadores do **Instituto de Informática** da **Universidade Federal do Rio Grande do Sul (UFRGS)**:

* **Arthur A. da Silva** (`aasilva@inf.ufrgs.br`)
* **Thiago Araújo** (`tsaraujo@inf.ufrgs.br`)
* **Cristiano A. Künas** (`cakunas@inf.ufrgs.br`)
* **Philippe O. A. Navaux** (`navaux@inf.ufrgs.br`)
