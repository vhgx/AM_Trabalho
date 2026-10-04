"""
Script de Geração do Dataset Histórico de Requisitos de QA
Gera 'data/historico_requisitos.csv' com ~400 amostras baseadas em 4 arquétipos realistas:
1. CRUD Simples
2. Fluxo de Negócio Complexo
3. Integração Pesada
4. Crítico / Segurança

Utiliza semente fixa para garantir total reprodutibilidade científica e acadêmica.
"""
import os
import numpy as np
import pandas as pd

def gerar_historico_requisitos(n_amostras=400, seed=42):
    rng = np.random.RandomState(seed)
    
    # Proporções dos 4 arquétipos latentes
    # 0: CRUD Simples (35%)
    # 1: Fluxo de Negócio Complexo (30%)
    # 2: Integração Pesada (20%)
    # 3: Crítico / Segurança (15%)
    arquetipos = rng.choice(
        ['CRUD Simples', 'Fluxo Complexo', 'Integração Pesada', 'Crítico / Segurança'],
        size=n_amostras,
        p=[0.35, 0.30, 0.20, 0.15]
    )
    
    dados = []
    
    exemplos_titulos = {
        'CRUD Simples': [
            'Cadastro de categorias de produtos',
            'Listagem com paginação de usuários',
            'Edição de dados cadastrais do cliente',
            'Exclusão lógica de fornecedores inativos',
            'Visualização de detalhes de pedidos antigos',
            'Consulta de tabela de preços vigentes',
            'Cadastro rápido de tags de conteúdo',
        ],
        'Fluxo Complexo': [
            'Cálculo de alíquotas fiscais interestaduais',
            'Aprovação multinível de adiantamento financeiro',
            'Processamento de comissões de vendas por metas',
            'Fechamento contábil mensal com conciliação',
            'Esteira de renegociação de dívidas parceladas',
            'Regras de elegibilidade para seguros corporativos',
            'Motor de precificação dinâmica para alta demanda',
        ],
        'Integração Pesada': [
            'Webhook assíncrono de notificações de pagamento',
            'Sincronização bidirecional com CRM Salesforce',
            'Emissão de NF-e via SEFAZ com retry e fallback',
            'Integração com gateway bancário PIX e Boleto',
            'Envio de mensagens transacionais via mensageria Kafka',
            'Integração com transportadoras e cálculo de frete externo',
            'Importação em lote de extratos bancários via OFX',
        ],
        'Crítico / Segurança': [
            'Autenticação multifator (MFA) via TOTP e SMS',
            'Anonimização e direito ao esquecimento LGPD',
            'Criptografia de dados de cartão de crédito no cofre',
            'Renovação de tokens OAuth2 e revogação de sessões',
            'Auditoria de transações sensíveis de administradores',
            'Recuperação de senha com link temporal assinado',
            'Proteção contra brute-force e rate limit em endpoints',
        ]
    }
    
    for i, arq in enumerate(arquetipos):
        tit_base = rng.choice(exemplos_titulos[arq])
        titulo = f"{tit_base} (Lote #{i+1:03d})"
        
        if arq == 'CRUD Simples':
            regras = rng.randint(1, 5)
            cenarios = rng.randint(1, 4)
            casos = rng.randint(2, 7)
            edge = rng.randint(1, 4)
            integracoes = rng.choice([0, 1], p=[0.85, 0.15])
            seguranca = rng.choice([0, 1], p=[0.95, 0.05])
            palavras = rng.randint(40, 140)
            
        elif arq == 'Fluxo Complexo':
            regras = rng.randint(6, 15)
            cenarios = rng.randint(5, 12)
            casos = rng.randint(10, 24)
            edge = rng.randint(3, 8)
            integracoes = rng.randint(0, 3)
            seguranca = rng.choice([0, 1], p=[0.8, 0.2])
            palavras = rng.randint(150, 380)
            
        elif arq == 'Integração Pesada':
            regras = rng.randint(3, 8)
            cenarios = rng.randint(3, 7)
            casos = rng.randint(6, 16)
            edge = rng.randint(4, 9) # Falhas de rede, timeouts
            integracoes = rng.randint(3, 8)
            seguranca = rng.choice([0, 1], p=[0.6, 0.4])
            palavras = rng.randint(120, 300)
            
        else: # Crítico / Segurança
            regras = rng.randint(4, 10)
            cenarios = rng.randint(4, 9)
            casos = rng.randint(8, 18)
            edge = rng.randint(6, 12) # Invasões, vazamentos, tokens expirados
            integracoes = rng.randint(1, 4)
            seguranca = 1
            palavras = rng.randint(160, 420)
            
        # Target 1: Regressão - Esforço de teste em horas
        ruido_esforco = rng.normal(0, 0.9)
        esforco = (
            1.2
            + 0.55 * regras
            + 0.35 * cenarios
            + 0.15 * casos
            + 0.85 * integracoes
            + 2.20 * seguranca
            + 0.005 * palavras
            + ruido_esforco
        )
        esforco = max(1.0, round(float(esforco), 1))
        
        # Target 2: Classificação - Ocorrência de Bug Crítico em Produção
        # Notar relação de proteção: mais edge_cases reduzem a chance de bug escapar!
        ruido_logit = rng.normal(0, 0.6)
        logit = (
            -3.2
            + 0.18 * regras
            + 0.45 * integracoes
            + 1.15 * seguranca
            + 0.004 * palavras
            - 0.22 * edge
            + ruido_logit
        )
        prob_bug = 1.0 / (1.0 + np.exp(-logit))
        gerou_bug = int(rng.uniform(0, 1) < prob_bug)
        
        dados.append({
            'id': i + 1,
            'titulo': titulo,
            'arquetipo': arq,
            'num_regras_negocio': regras,
            'num_cenarios_bdd': cenarios,
            'num_casos_teste': casos,
            'num_edge_cases': edge,
            'num_integracoes': integracoes,
            'envolve_seguranca': seguranca,
            'tamanho_descricao': palavras,
            'esforco_teste_horas': esforco,
            'gerou_bug_producao': gerou_bug,
        })
        
    df = pd.DataFrame(dados)
    return df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, 'data', 'historico_requisitos.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = gerar_historico_requisitos()
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Dataset gerado com sucesso em: {output_path}")
    print(f"Total de registros: {len(df)}")
    print(f"Distribuição de Arquétipos:\n{df['arquetipo'].value_counts()}")
    print(f"Taxa de Bugs em Produção: {df['gerou_bug_producao'].mean():.2%}")
    print(f"Esforço médio de teste: {df['esforco_teste_horas'].mean():.2f} horas (min: {df['esforco_teste_horas'].min()}, max: {df['esforco_teste_horas'].max()})")
