"""
Extrator de Features de Requisitos e Artefatos de QA
Extrai atributos numéricos a partir dos artefatos gerados pelo LLM:
- Interpretação (Regras de negócio)
- Cenários BDD (Gherkin)
- Casos de Teste Estruturados (Markdown table)
- Edge Cases & Vulnerabilidades
- Descrição funcional bruta
"""
import re

FEATURE_NAMES = [
    'num_regras_negocio',
    'num_cenarios_bdd',
    'num_casos_teste',
    'num_edge_cases',
    'num_integracoes',
    'envolve_seguranca',
    'tamanho_descricao',
]

PALAVRAS_INTEGRACAO = [
    r'\bapi\b', r'\bapis\b', r'\bwebhook\b', r'\bwebhooks\b', r'\bgateway\b',
    r'\bendpoint\b', r'\bmicroserviço\b', r'\bmicroservice\b', r'\bkafka\b',
    r'\brabbitmq\b', r'\brest\b', r'\bsoap\b', r'\bsincroniza\w*\b',
    r'\bextern\w*\b', r'\bterceir\w*\b', r'\bpix\b', r'\bsefaz\b',
    r'\bintegra\w*\b', r'\be-?mail\b', r'\bsms\b'
]

PALAVRAS_SEGURANCA = [
    r'\bautentica\w*\b', r'\blogin\b', r'\bsenha\b', r'\btoken\b', r'\bjwt\b',
    r'\boauth\b', r'\bcriptograf\w*\b', r'\bmfa\b', r'\b2fa\b', r'\blgpd\b',
    r'\bautoriza\w*\b', r'\bpermiss\w*\b', r'\bprivilég\w*\b', r'\bvulnerabil\w*\b',
    r'\bseguranç\w*\b', r'\bbrute[- ]?force\b', r'\bhash\b', r'\bsensíve\w*\b',
    r'\bcartão\b', r'\bcredencial\w*\b'
]

def contar_regras_negocio(interpretacao: str, descricao: str) -> int:
    """Extrai quantidade de regras de negócio identificadas na interpretação."""
    texto = interpretacao or ''
    # Busca a seção de Regras de Negócio se existir
    match = re.search(r'(?:Regras de Negócio|Restrições)(.*?)(?:###|$)', texto, re.S | re.I)
    bloco = match.group(1) if match else texto
    
    # Contar itens em lista (- item, * item, 1. item)
    itens = re.findall(r'^\s*(?:[-*]|\d+\.)\s+.+', bloco, re.M)
    count = len(itens)
    if count == 0 and descricao:
        # Estimativa por linhas ou sentenças na descrição se não houver lista
        count = max(1, min(len(descricao.split('.')), 5))
    return max(1, count)

def contar_cenarios_bdd(bdd: str) -> int:
    """Conta cenários BDD (Scenario, Scenario Outline, Cenário, Esquema do Cenário)."""
    if not bdd:
        return 1
    padrao = r'^\s*(?:Scenario Outline|Scenario|Cenário|Esquema do Cenário)\s*:'
    matches = re.findall(padrao, bdd, re.M | re.I)
    return max(1, len(matches))

def contar_casos_teste(casos_teste: str) -> int:
    """Conta linhas de casos de teste em tabelas Markdown."""
    if not casos_teste:
        return 2
    linhas = [l.strip() for l in casos_teste.splitlines() if l.strip().startswith('|')]
    # Remove header e linha separadora (ex: |---|---|)
    linhas_uteis = [l for l in linhas if not re.match(r'^\|[\s\-:|]+\|$', l)]
    if len(linhas_uteis) > 1:
        return max(1, len(linhas_uteis) - 1)
    # Se não for tabela, conta listas numeradas
    itens = re.findall(r'^\s*(?:[-*]|\d+\.)\s+.+', casos_teste, re.M)
    return max(2, len(itens))

def contar_edge_cases(edge_cases: str) -> int:
    """Conta casos extremos e de ruptura identificados."""
    if not edge_cases:
        return 1
    itens = re.findall(r'^\s*(?:[-*]|\d+\.)\s+.+', edge_cases, re.M)
    return max(1, len(itens))

def contar_integracoes(texto_completo: str) -> int:
    """Identifica número de tecnologias ou referências de integração externa."""
    if not texto_completo:
        return 0
    encontrados = set()
    for regex in PALAVRAS_INTEGRACAO:
        if re.search(regex, texto_completo, re.I):
            encontrados.add(regex)
    return len(encontrados)

def verificar_seguranca(texto_completo: str) -> int:
    """Verifica se o requisito envolve aspectos críticos de segurança/conformidade."""
    if not texto_completo:
        return 0
    for regex in PALAVRAS_SEGURANCA:
        if re.search(regex, texto_completo, re.I):
            return 1
    return 0

def extrair_features_requisito(requisito) -> dict:
    """
    Recebe um objeto Requisito (ou dict equivalente) e extrai todas as 7 features
    necessárias para os modelos de Regressão, Classificação e Clusterização.
    """
    descricao = getattr(requisito, 'descricao', '') or ''
    interpretacao = getattr(requisito, 'interpretacao', '') or ''
    bdd = getattr(requisito, 'bdd', '') or ''
    casos_teste = getattr(requisito, 'casos_teste', '') or ''
    edge_cases = getattr(requisito, 'edge_cases', '') or ''
    
    texto_geral = f"{descricao}\n{interpretacao}\n{bdd}\n{casos_teste}\n{edge_cases}"
    palavras = len(descricao.split())
    
    return {
        'num_regras_negocio': contar_regras_negocio(interpretacao, descricao),
        'num_cenarios_bdd': contar_cenarios_bdd(bdd),
        'num_casos_teste': contar_casos_teste(casos_teste),
        'num_edge_cases': contar_edge_cases(edge_cases),
        'num_integracoes': contar_integracoes(texto_geral),
        'envolve_seguranca': verificar_seguranca(texto_geral),
        'tamanho_descricao': max(20, palavras),
    }

def dict_para_vetor(features: dict, colunas=FEATURE_NAMES):
    """Converte o dicionário de features em lista ordenada para os modelos."""
    return [features[col] for col in colunas]
