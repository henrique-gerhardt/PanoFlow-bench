# benchmark_contrato

Implementacao local do contrato de benchmark para o repositório PanoFlow, baseada nos exemplos de `panoflow_agent_pack/benchmark_contract_examples`, mas aplicada fora do `agent_pack`.

## O que esta adaptacao resolve

- preserva o codigo-fonte original do PanoFlow;
- reutiliza a inicializacao oficial do modelo e a logica oficial de inferencia/CFE;
- adapta o dataset real encontrado no workspace, cujo layout e `split/<scene_id>/{frames,fflows,bflows}`;
- gera saidas padronizadas em `benchmark_contrato/results/`.

## Limitacao real encontrada

Nenhum checkpoint `.pth` foi encontrado dentro do workspace atual. Para executar `official_reproduction`, e obrigatorio fornecer um checkpoint real do paper, por exemplo `PanoFlow(CSFlow)-wo-CFE.pth`, via volume montado e variavel `PANOFLOW_CHECKPOINT`.

## Dataset alvo

Dataset encontrado durante a inspecao:

- host atual: `/Volumes/External SSD/Mestrado/Datasets/FLOW360_train_test`
- layout: `test/<scene_id>/frames/*.png` e `test/<scene_id>/fflows/*.npy`
- resolucao observada: `1024x512`

Dentro do container, monte esse dataset em um path Linux e exporte `FLOW360_ROOT` para o caminho montado.

## Build

```bash
docker build -f benchmark_contrato/Dockerfile.benchmark -t panoflow-benchmark .
```

## official_reproduction

Exemplo de execucao em Linux com NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all \
  -e FLOW360_ROOT=/datasets/FLOW360_train_test \
  -e PANOFLOW_CHECKPOINT=/models/PanoFlow(CSFlow)-wo-CFE.pth \
  -v /caminho/linux/FLOW360_train_test:/datasets/FLOW360_train_test:ro \
  -v /caminho/linux/PanoFlow(CSFlow)-wo-CFE.pth:/models/PanoFlow(CSFlow)-wo-CFE.pth:ro \
  -v "$(pwd)/benchmark_contrato/results:/app/benchmark_contrato/results" \
  panoflow-benchmark official_reproduction
```

## standardized_efficiency

```bash
docker run --rm --gpus all \
  -e PANOFLOW_CHECKPOINT=/models/PanoFlow(CSFlow)-wo-CFE.pth \
  -v /caminho/linux/PanoFlow(CSFlow)-wo-CFE.pth:/models/PanoFlow(CSFlow)-wo-CFE.pth:ro \
  -v "$(pwd)/benchmark_contrato/results:/app/benchmark_contrato/results" \
  panoflow-benchmark standardized_efficiency
```

## Arquivos gerados

- `benchmark_contrato/results/metadata.json`
- `benchmark_contrato/results/quality_metrics.json`
- `benchmark_contrato/results/efficiency_metrics.json`
- `benchmark_contrato/results/run_config.json`
- `benchmark_contrato/results/environment.json`

## Observacoes de compatibilidade

- o host alvo pode expor driver/CUDA mais novo que o runtime do container;
- esta imagem fixa PyTorch 2.1.2 com CUDA 11.8 por ser uma base conservadora para um projeto de 2022;
- o acesso a GPU depende de `--gpus all` e do NVIDIA Container Toolkit no host.
