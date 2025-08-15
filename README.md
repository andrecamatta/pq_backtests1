# PQBacktester.jl

Framework robusto de backtest quantitativo em Julia com três estratégias implementadas: Momentum, Pairs Trading e Sazonalidade. Utiliza otimização Walk-Forward com validação estatística rigorosa.

## 📋 Visão Geral

Sistema completo de backtest para estratégias quantitativas com dados reais do mercado brasileiro. Framework modular, estatisticamente sólido e fácil de estender, com controles anti-overfitting e métricas avançadas de performance.

## 🚀 Instalação e Execução

### Pré-requisitos
- Julia 1.9 ou superior
- Conexão com internet (para download de dados)

### Execução Rápida
```bash
# Ativar o ambiente Julia
julia --project=.

# Executar o backtest
include("backtest_otimizado.jl")
```

Ou diretamente no terminal:
```bash
julia backtest_otimizado.jl
```

## 📊 Estratégias Implementadas

### 1. 📈 Momentum (IBOVESPA)
- **Lógica**: Crossover de médias móveis com lag adaptativo
- **Parâmetros**: `fast=[10,20,50]` e `slow=[50,100,200]` dias
- **Características**: Anti-look-ahead bias, rebalanceamento semestral
- **Asset**: ^BVSP (Índice Bovespa)

### 2. 🔄 Pairs Trading (ITUB4/BBDC4)
- **Lógica**: Mean reversion com z-score do spread cointegrado
- **Parâmetros**: `z_enter=[1.5,2.0,2.5]` e `z_exit=[0.5,1.0]`
- **Características**: Janela dinâmica, hedge ratio adaptativo, neutralidade de mercado
- **Assets**: ITUB4.SA / BBDC4.SA (Setor Bancário)

### 3. 📅 Sazonalidade (IBOVESPA)  
- **Lógica**: Explora padrões mensais baseados em performance histórica
- **Parâmetros**: `k=[2,4,6]` meses com melhor retorno
- **Características**: Ranking sem look-ahead bias, ciclos anuais completos
- **Asset**: ^BVSP (Índice Bovespa)

## 🔧 Estrutura do Projeto

```
pq_backtests1/
├── src/
│   ├── PQBacktester.jl      # Módulo principal
│   ├── Config.jl            # Configurações e parâmetros
│   ├── Metrics.jl           # Cálculo de métricas de performance
│   ├── Costs.jl             # Modelagem de custos de transação
│   ├── Split.jl             # Divisão temporal dos dados
│   ├── Data.jl              # Carregamento e preparação de dados
│   ├── Strategies.jl        # Implementação das estratégias
│   └── Core.jl              # Motor de backtest e otimização
├── backtest_otimizado.jl    # Script de execução principal
├── Project.toml            # Dependências do projeto
└── README.md               # Este arquivo
```

## 📊 Métricas Acadêmicas Robustas (Implementação 2025)

### 🎯 Métricas Principais Robustas
- **Weighted Sharpe Ratio ± SE**: Média ponderada por split com erro padrão e p-value
- **Compound Sharpe Ratio**: Média geométrica mais robusta a outliers
- **Newey-West Sharpe**: Corrigido para autocorrelação temporal (HAC)
- **Deflated Sharpe (DSR) ± SE**: Penaliza múltiplos testes com significância estatística
- **Adjusted Information Ratio**: Calculado por split, depois agregado
- **Deflated Information Ratio (DIR)**: IR × √DSR - penaliza baixa significância

### 🔬 Metodologia Anti-Viés
- **Média Ponderada**: Evita viés de concatenação simples dos retornos
- **Teste t**: H₀: métrica = 0 com correção para pesos desiguais (Cochran 1977)
- **Bootstrap CI**: Intervalos de confiança 95% para métricas robustas
- **HAC Standard Errors**: Correção Newey-West para autocorrelação e heterocedasticidade

### 📈 Interpretação Acadêmica
- **Sharpe ± SE**: *** p<0.001, ** p<0.01, * p<0.05 (significância estatística)
- **Compound Sharpe**: Mais estável em presença de outliers
- **Newey-West**: Essencial para estratégias com dependência temporal
- **DSR ± SE**: Robustez após múltiplos testes (Bailey & Lopez de Prado 2012)
- **DIR**: Métrica híbrida que combina alpha e significância estatística

### 📚 Fundamentação Teórica
- **Lo (2002)**: Statistics of Sharpe Ratio - correções para autocorrelação
- **Bailey & Lopez de Prado (2012)**: Deflated Sharpe Ratio methodology  
- **Harvey & Liu (2015)**: Multiple testing in finance
- **Cochran (1977)**: Weighted mean standard errors

## 🎯 Framework Walk-Forward Robusto

### Metodologia Anti-Overfitting
1. **Splits Temporais**: Janelas não-overlapping com embargo days
2. **Grid Search**: Otimização apenas em dados de treino (in-sample)
3. **Validação OOS**: Teste em períodos verdadeiramente out-of-sample
4. **Re-otimização**: Parâmetros adaptados a cada split temporal
5. **Métrica Objetivo**: Maximização do Sharpe Ratio anualizado

### Configurações por Estratégia
- **Momentum**: 24 meses treino / 6 meses teste / 3 dias embargo
- **Pairs Trading**: 36 meses treino / 12 meses teste / 5 dias embargo  
- **Sazonalidade**: 24 meses treino / 12 meses teste / 2 dias embargo

### Controles de Qualidade
- Validação de amostra mínima (60+ observações)
- Alertas para splits sem trades
- Tratamento de casos degenerados (σ=0, MDD≈0)

## 📊 Dados e Configuração

- **Período**: Janeiro 2000 - Dezembro 2024
- **Frequência**: Dados diários de fechamento
- **Fonte**: Yahoo Finance (API gratuita)
- **Assets**:
  - ^BVSP: Índice Bovespa (momentum e sazonalidade)
  - ITUB4.SA: Itaú Unibanco PN (pairs trading)  
  - BBDC4.SA: Bradesco PN (pairs trading)

## 🛠️ Personalização

Para adicionar novas estratégias:
1. Criar a função de estratégia em `Strategies.jl`
2. Adicionar parâmetros em `Config.jl`
3. Registrar na lista de estratégias em `backtest_otimizado.jl`

## 📄 Formato de Saída

O framework gera relatórios detalhados com:

- **Resultados por Split**: Performance individual de cada período de teste
- **Métricas Agregadas**: Sharpe, DSR, Information Ratio com significância estatística
- **Resumo Comparativo**: Tabela comparando todas as estratégias
- **Parâmetros Utilizados**: Configurações otimizadas para cada período

### Exemplo de Métricas Geradas
- **Sharpe Ratio** com erro padrão e p-value
- **Deflated Sharpe Ratio (DSR)** para penalizar múltiplos testes
- **Newey-West Sharpe** corrigido para autocorrelação
- **Information Ratio** vs benchmark apropriado
- **Deflated Information Ratio (DIR)** combinando alpha e significância

## 🤝 Contribuições

Sinta-se à vontade para abrir issues ou pull requests para melhorias no código ou novas funcionalidades.

## ⚠️ Aviso Legal

Este software é fornecido apenas para fins educacionais e de pesquisa. Não constitui recomendação de investimento. Sempre realize sua própria análise antes de investir.