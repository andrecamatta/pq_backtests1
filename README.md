# PQBacktester.jl

Framework robusto de backtest quantitativo em Julia com três estratégias implementadas: Momentum, Pairs Trading e Sazonalidade. Utiliza otimização Walk-Forward com validação estatística rigorosa.

## 📋 Visão Geral

Sistema completo de backtest para estratégias quantitativas com dados reais do mercado brasileiro (2015-2024). Framework modular, estatisticamente sólido e fácil de estender, com controles anti-overfitting e métricas avançadas de performance.

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
- **Asset**: ^BVSP (2015-2024)

### 2. 🔄 Pairs Trading (SUZB3/KLBN11)
- **Lógica**: Mean reversion com z-score do spread cointegrado
- **Parâmetros**: `z_enter=[1.5,2.0,2.5]` e `z_exit=[0.5,1.0]`
- **Características**: Janela dinâmica, hedge ratio adaptativo, neutralidade de mercado
- **Assets**: SUZB3.SA / KLBN11.SA

### 3. 📅 Sazonalidade (IBOVESPA)  
- **Lógica**: Explora padrões mensais baseados em performance histórica
- **Parâmetros**: `k=[4,6,8]` meses com melhor retorno
- **Características**: Ranking sem look-ahead bias, ciclos anuais completos
- **Asset**: ^BVSP (2015-2024)

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

## 📈 Métricas de Performance Avançadas

### Métricas Principais
- **MAR (Modified Annual Return)**: CAGR / Max Drawdown - relação retorno/risco
- **PSR (Probabilistic Sharpe Ratio)**: Confiança estatística do Sharpe Ratio (0-100%)
- **DSR (Deflated Sharpe Ratio)**: Sharpe ajustado para múltiplos testes

### Métricas Auxiliares
- **Sharpe Ratio**: Retorno anual ajustado ao risco (√252)
- **CAGR**: Retorno anual composto geométrico  
- **Max Drawdown**: Perda máxima de pico a vale
- **Número de Trades**: Validação de robustez estatística

### Interpretação dos Resultados
- **MAR > 1.0**: Estratégia supera drawdown máximo
- **PSR > 95%**: Alta confiança estatística (recomendado > 95%)
- **DSR > 95%**: Robustez após correção para data mining

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

## 📊 Dados e Período de Análise

- **Período**: Janeiro 2015 - Dezembro 2024 (10 anos)
- **Frequência**: Dados diários de fechamento
- **Fonte**: Yahoo Finance (API gratuita)
- **Assets**:
  - ^BVSP: Índice Bovespa (2.479 observações)
  - SUZB3.SA: Suzano Papel (2.487 observações)  
  - KLBN11.SA: Klabin Units (2.487 observações)

## 🛠️ Personalização

Para adicionar novas estratégias:
1. Criar a função de estratégia em `Strategies.jl`
2. Adicionar parâmetros em `Config.jl`
3. Registrar na lista de estratégias em `backtest_otimizado.jl`

## 📄 Exemplo de Saída Atual

```
==================================================
2️⃣  Executando Estratégia: PAIRS TRADING (Janela Dinâmica)
==================================================
📊 Resultados por Split:
   Split 1 (Período de Teste: 2018-01-02 até 2018-12-28):
      MAR: 0.423291 | PSR: 0.999765 | DSR: 0.998916 | Dias: 246
      📊 Parâmetro: Compra quando z-score > 2.5 ou z-score < -2.5, Venda quando |z-score| < 1.0
   Split 2 (Período de Teste: 2022-01-07 até 2023-01-06):
      MAR: 2.241309 | PSR: 1.0 | DSR: 1.0 | Dias: 251
      📊 Parâmetro: Compra quando z-score > 2.0 ou z-score < -2.0, Venda quando |z-score| < 0.5
   📊 Resultado Geral (média de todos os splits):
      MAR: 0.408353 | PSR: 0.999997 | DSR: 0.999981
```

### 🏆 Ranking de Performance (2015-2024)
1. **Pairs Trading**: MAR=0.408, PSR=99.99%, DSR=99.99% ⭐
2. **Sazonalidade**: MAR=0.675, PSR=100%, DSR=100% 
3. **Momentum**: MAR=-0.323, PSR=0%, DSR=0%

## 🤝 Contribuições

Sinta-se à vontade para abrir issues ou pull requests para melhorias no código ou novas funcionalidades.

## ⚠️ Aviso Legal

Este software é fornecido apenas para fins educacionais e de pesquisa. Não constitui recomendação de investimento. Sempre realize sua própria análise antes de investir.