# QuadNet Estimation Challenge

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Institution](https://img.shields.io/badge/Institution-IFMG-red.svg)](https://www.ifmg.edu.br/betim)

## Sobre o Projeto
O **QuadNet** é uma plataforma experimental e numérica dedicada ao estudo de fenômenos emergentes e desafios de estimação em redes complexas de **Osciladores de Quadratura**. 

Diferente dos modelos puramente teóricos de Kuramoto, o QuadNet utiliza uma implementação eletrônica robusta que permite validar a dinâmica de fase em ambientes com ruído real, incertezas paramétricas e limitações de hardware.

## O Desafio de Estimação (The Challenge!)
O objetivo central deste repositório é fornecer ferramentas para o desafio de **observabilidade dinâmica**:
> *“Como estimar a fase de todos os osciladores da rede utilizando o menor conjunto possível de variáveis medidas?”*

Este problema é fundamental para o controle de redes de potência, sincronização de sensores e entendimento de atividades neuronais, onde nem todos os nós podem ser monitorados simultaneamente.

## Detalhes Técnicos
### O Oscilador de Quadratura
O modelo não linear utilizado foi desenvolvido por pesquisadores do **IFMG - Campus Betim** e da **UFMG**. O circuito apresenta uma regulação robusta de amplitude, permitindo que a dinâmica se concentre na evolução da fase, mapeando-se diretamente ao modelo de Kuramoto:

$$\dot{\theta}_i = \omega_i + \sum_{j=1}^{N} \sigma_{ij} \sin(\theta_j - \theta_i)$$

### Topologias Suportadas
O código permite a análise em diversas arquiteturas de rede:
- **All-to-all** (Acoplamento Global)
- **Anel** (Ring)
- **Mundo Pequeno** (Small-World)
- **Livre de Escala** (Scale-Free)

## Estrutura do Repositório
- `/EXP_00x/`: Dados brutos, resultados e análise dos experimentos.
- `quadratureNetwork.py`: Biblioteca principal para simulação da dinâmica da rede.
- `Exp00x_analyseResults.ipynb`: Notebooks para processamento de sinais (FFT, Parâmetro de Ordem e Sincronismo).
- `m10_polyModel_freq_to_R2.npz`: Modelo polinomial para calibração das frequências naturais.

## Como Citar
Se você utilizar este código ou os dados experimentais em sua pesquisa, por favor, cite os trabalhos fundamentais do grupo:

1. **Bittencourt, V.H.S., et al. (2023)**. *"Projeto, construção e modelagem não linear do oscilador de quadratura"*. Anais do SBAI, Manaus.
2. **Dias, A.C.B., et al. (2024)**. *"Verificação Experimental de Estados de Quimera em Redes de Osciladores de Quadratura"*. Anais do CBA, Rio de Janeiro.
3. **Abreu, L.F., et al. (2025)**. *"Experimental Kuramoto Platform: Electronic Implementation of Networks"*. (Preprint disponível no repositório).

## Colaboradores e Fomento
Este projeto é desenvolvido no **Laboratório de Redes Complexas do IFMG Betim** com o apoio de:
* **FAPEMIG** (APQ-00781-21)
* **CNPq** (409487/2021-0)
* **CAPES**
