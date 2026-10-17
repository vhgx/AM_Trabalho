import re

FEATURE_NAMES = [
    'num_regras_negocio',
    'num_casos_teste',
    'num_integracoes',
    'envolve_seguranca',
]

PALAVRAS_INTEGRACAO = [
    r'\bapi\b', r'\bapis\b', r'\bwebhook\b', r'\bgateway\b',
    r'\bendpoint\b', r'\bmicroserviço\b', r'\bkafka\b',
    r'\brabbitmq\b', r'\brest\b', r'\bsoap\b', r'\bsincroniza\w*\b',
    r'\bextern\w*\b', r'\bterceir\w*\b', r'\bpix\b', r'\bsefaz\b',
    r'\bintegra\w*\b', r'\be-?mail\b', r'\bsms\b'
]

PALAVRAS_SEGURANCA = [
    r'\bautentica\w*\b', r'\blogin\b', r'\bsenha\b', r'\btoken\b', r'\bjwt\b',
    r'\boauth\b', r'\bcriptograf\w*\b', r'\bmfa\b', r'\b2fa\b', r'\blgpd\b',
    r'\bautoriza\w*\b', r'\bpermiss\w*\b', r'\bseguranç\w*\b', r'\bhash\b',
    r'\bcartão\b', r'\bcredencial\w*\b'
]


def contar_regras_negocio(interpretacao: str, descricao: str) -> int:
    texto = interpretacao or ''
    match = re.search(r'(?:Regras de Negócio|Restrições)(.*?)(?:###|$)', texto, re.S | re.I)
    bloco = match.group(1) if match else texto

    itens = re.findall(r'^\s*(?:[-*]|\d+\.)\s+.+', bloco, re.M)
    count = len(itens)
    if count == 0 and descricao:
        count = max(1, min(len([s for s in descricao.split('.') if s.strip()]), 5))
    return max(1, count)


def contar_casos_teste(casos_teste: str) -> int:
    if not casos_teste:
        return 2
    linhas = [l.strip() for l in casos_teste.splitlines() if l.strip().startswith('|')]
    linhas_uteis = [l for l in linhas if not re.match(r'^\|[\s\-:|]+\|$', l)]
    if len(linhas_uteis) > 1:
        return max(1, len(linhas_uteis) - 1)
    itens = re.findall(r'^\s*(?:[-*]|\d+\.)\s+.+', casos_teste, re.M)
    return max(2, len(itens))


def contar_integracoes(texto_completo: str) -> int:
    if not texto_completo:
        return 0
    encontrados = set()
    for regex in PALAVRAS_INTEGRACAO:
        if re.search(regex, texto_completo, re.I):
            encontrados.add(regex)
    return len(encontrados)


def verificar_seguranca(texto_completo: str) -> int:
    if not texto_completo:
        return 0
    for regex in PALAVRAS_SEGURANCA:
        if re.search(regex, texto_completo, re.I):
            return 1
    return 0


def extrair_features_requisito(requisito) -> dict:
    descricao = getattr(requisito, 'descricao', '') or ''
    interpretacao = getattr(requisito, 'interpretacao', '') or ''
    casos_teste = getattr(requisito, 'casos_teste', '') or ''
    bdd = getattr(requisito, 'bdd', '') or ''
    edge_cases = getattr(requisito, 'edge_cases', '') or ''

    texto_geral = f"{descricao}\n{interpretacao}\n{bdd}\n{casos_teste}\n{edge_cases}"

    return {
        'num_regras_negocio': contar_regras_negocio(interpretacao, descricao),
        'num_casos_teste': contar_casos_teste(casos_teste),
        'num_integracoes': contar_integracoes(texto_geral),
        'envolve_seguranca': verificar_seguranca(texto_geral),
    }


def dict_para_vetor(features: dict, colunas=FEATURE_NAMES):
    return [features[col] for col in colunas]
