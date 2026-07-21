"""
Verifica no site odsbrasil.gov.br (IBGE) se ha indicadores novos, removidos ou
com status alterado em relacao ao que esta mapeado em constants.py / db/indicadores.csv.

Nao edita nenhum arquivo do projeto: apenas gera um relatorio para revisao manual,
ja que a escolha da tabela/variavel/classificacao corretas exige julgamento humano
(ver historico de sessao onde isso foi mapeado manualmente).

A secao 2 (indicadores novos) sempre verifica se ha tabela a nivel de UF, ja
que so isso interessa para a aplicacao (que exibe RO, MA, TO, MT, MS, GO e DF).

Uso:
    python check_indicadores_ibge.py                   # checagem rapida de status
    python check_indicadores_ibge.py --detail           # tambem busca tabelas SIDRA candidatas p/ secao 1
    python check_indicadores_ibge.py --objetivos 1,4,6  # restringe a objetivos especificos
    python check_indicadores_ibge.py --output relatorio_ibge.md
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from constants import LIST_INDICADORES

BASE_ODS = "https://odsbrasil.gov.br"
SIDRA_DESCRITIVO = "https://sidra.ibge.gov.br/geratabela/DescritivoTabela"

OBJETIVOS = list(range(1, 19))  # o site expoe objetivos 1 a 18


def get_json(url, **kwargs):
    r = requests.get(url, timeout=30, **kwargs)
    r.raise_for_status()
    r.encoding = 'utf-8'
    return r.json()


def carregar_indicadores_site(objetivos):
    """Retorna {numero: {'objetivo', 'meta', 'nome', 'status', 'possui_ficha'}}."""
    inventario = {}
    for n in objetivos:
        url = f"{BASE_ODS}/objetivo/MetadadosAPIMeta?n={n}"
        try:
            metas = get_json(url)
        except Exception as e:
            print(f"  aviso: falha ao consultar objetivo {n}: {e}", file=sys.stderr)
            continue
        for meta in metas:
            for ind in meta.get('indicadores', []):
                inventario[ind['numero']] = {
                    'objetivo': n,
                    'meta': meta['numero'],
                    'nome': ind['nome'],
                    'status': ind['status'],
                    'possui_ficha': ind.get('possui_ficha', False),
                }
    return inventario


def carregar_constants_indicadores():
    mapeados = set()
    for _objetivo, metas in LIST_INDICADORES.items():
        for _meta, inds in metas.items():
            mapeados.update(inds.keys())
    return mapeados


def carregar_csv_indicadores(csv_path):
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        return [row for row in reader if row.get('ID_INDICADOR')]


def numero_para_ids_csv(numero, csv_ids):
    """Casa um numero oficial do site (ex: '6.3.1') com possiveis IDs do CSV
    (ex: 'Indicador 6.3.1', 'Indicador 6.3.1a', 'Indicador 6.3.1.2')."""
    alvo = f"Indicador {numero}"
    return [cid for cid in csv_ids if cid == alvo or cid.startswith(alvo)]


def numero_base_oficial(numero, inventario_site):
    """Tenta reduzir um ID do nosso CSV (que pode ter sufixos como '1.1.1c' ou
    '4.2.1.3') ao numero oficial do site, que e o que tem pagina propria."""
    if numero in inventario_site:
        return numero
    m = re.match(r'^([\d.]+?)[a-zA-Z]+$', numero)
    if m and m.group(1) in inventario_site:
        return m.group(1)
    partes = numero.split('.')
    if len(partes) > 3 and '.'.join(partes[:3]) in inventario_site:
        return '.'.join(partes[:3])
    return None


def extrair_id_tabelas(html):
    """Le a var idTabelas do HTML da pagina do indicador, ignorando linhas
    comentadas (// var idTabelas = ...), que indicam pagina sem widget ativo."""
    for line in html.splitlines():
        stripped = line.strip()
        if 'idTabelas' not in stripped or stripped.startswith('//'):
            continue
        m = re.search(r'idTabelas\s*=\s*"([0-9,]+)"', stripped)
        if m:
            return m.group(1)
    return None


def obter_tabelas_indicador(objetivo_n, numero_oficial):
    """Busca a pagina do indicador e retorna a lista de descritivos das
    tabelas SIDRA ativas (lista vazia se a pagina nao tiver widget de dados).
    Lanca RuntimeError com mensagem legivel em caso de falha de rede/HTTP."""
    slug = "indicador" + numero_oficial.replace(".", "")
    url = f"{BASE_ODS}/objetivo{objetivo_n}/{slug}"
    r = requests.get(url, timeout=30)
    r.encoding = 'utf-8'
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} em {url} (pagina indisponivel)")
    ids_tabelas = extrair_id_tabelas(r.text)
    if not ids_tabelas:
        return []
    return get_json(SIDRA_DESCRITIVO, params={'idTabelas': ids_tabelas})


def tem_dado_estadual(descritivos):
    """So interessa para a aplicacao quem tem alguma tabela com nivel 'UF'
    (Unidade Federativa) -- e o unico nivel territorial que cobre os estados
    que a aplicacao exibe (RO, MA, TO, MT, MS, GO, DF). Indicadores que so
    tem dado nacional (BR) ou regional (GR/RH) nao servem para os graficos
    por estado."""
    return any('UF' in (d.get('SiglasNiveisTerritoriais') or []) for d in descritivos)


def investigar_tabelas(objetivo_n, numero_oficial):
    slug = "indicador" + numero_oficial.replace(".", "")
    try:
        descritivos = obter_tabelas_indicador(objetivo_n, numero_oficial)
    except Exception as e:
        return [f"    {e}"]
    if not descritivos:
        return [f"    nenhuma tabela SIDRA ativa em {BASE_ODS}/objetivo{objetivo_n}/{slug} "
                f"(indicador sem widget de dados apesar do status)"]
    linhas = []
    for d in descritivos:
        niveis = d.get('SiglasNiveisTerritoriais') or []
        marcador = 'UF ok' if 'UF' in niveis else 'sem UF'
        linhas.append(f"    [{d['Id']}] {d['Nome']}  (niveis={niveis}, {marcador})")
    return linhas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--detail', action='store_true',
                         help='Busca tabelas SIDRA candidatas tambem para a secao 1 (mais lento, requisicoes extras)')
    parser.add_argument('--objetivos', type=str, default=None,
                         help='Lista de objetivos a checar, ex: 1,4,6 (default: todos, 1-18)')
    parser.add_argument('--output', type=str, default=None,
                         help='Salva o relatorio em um arquivo markdown, alem de imprimir no console')
    args = parser.parse_args()

    objetivos = [int(x) for x in args.objetivos.split(',')] if args.objetivos else OBJETIVOS

    base_dir = Path(__file__).parent
    csv_path = base_dir / 'db' / 'indicadores.csv'

    print(f"Consultando odsbrasil.gov.br (objetivos {objetivos})...")
    inventario_site = carregar_indicadores_site(objetivos)
    print(f"  {len(inventario_site)} indicadores encontrados no site")

    mapeados_constants = carregar_constants_indicadores()
    linhas_csv = carregar_csv_indicadores(csv_path)
    csv_ids = {l['ID_INDICADOR'] for l in linhas_csv}

    relatorio = ["# Relatorio de verificacao - odsbrasil.gov.br\n"]

    # 1) RBC=1 no CSV sem URL em constants.py (checagem critica: o pipeline de
    #    coleta silenciosamente ignora esses indicadores)
    sem_url = [l for l in linhas_csv if l['RBC'] == '1' and l['ID_INDICADOR'] not in mapeados_constants]
    relatorio.append(f"## 1. Indicadores com RBC=1 sem URL em constants.py ({len(sem_url)})\n")
    if not sem_url:
        relatorio.append("Nenhum. Tudo que esta marcado como ativo (RBC=1) tem URL mapeada.\n")
    for l in sem_url:
        relatorio.append(f"- **{l['ID_INDICADOR']}** — {l['DESC_INDICADOR']}")
        if args.detail:
            numero = l['ID_INDICADOR'].replace('Indicador ', '')
            numero_oficial = numero_base_oficial(numero, inventario_site)
            if numero_oficial:
                objetivo_n = inventario_site[numero_oficial]['objetivo']
                relatorio.extend(investigar_tabelas(objetivo_n, numero_oficial))
            else:
                relatorio.append("    nao foi possivel localizar pagina oficial correspondente no site")

    # 2) Indicadores 'Produzido' no site sem nenhuma linha correspondente no CSV
    #    (possiveis indicadores novos que o IBGE passou a publicar). So interessa
    #    para a aplicacao quem tiver dado a nivel de UF (unico nivel que cobre os
    #    estados exibidos); por isso sempre verificamos o nivel territorial aqui,
    #    independente de --detail.
    candidatos = [(numero, info) for numero, info in sorted(inventario_site.items())
                  if info['status'] == 'Produzido' and not numero_para_ids_csv(numero, csv_ids)]
    novos_com_uf, novos_sem_uf, novos_com_erro = [], [], []
    for numero, info in candidatos:
        try:
            descritivos = obter_tabelas_indicador(info['objetivo'], numero)
        except Exception as e:
            novos_com_erro.append((numero, info, str(e)))
            continue
        (novos_com_uf if tem_dado_estadual(descritivos) else novos_sem_uf).append((numero, info, descritivos))

    relatorio.append(f"\n## 2. Indicadores 'Produzido' no site sem nenhuma linha em indicadores.csv ({len(candidatos)})\n")
    if not candidatos:
        relatorio.append("Nenhum. Todo indicador publicado ja tem pelo menos uma linha no CSV.\n")

    relatorio.append(f"\n### 2a. Com dado por UF — relevantes para a aplicacao ({len(novos_com_uf)})\n")
    if not novos_com_uf:
        relatorio.append("Nenhum.\n")
    for numero, info, descritivos in novos_com_uf:
        relatorio.append(f"- **{numero}** (Objetivo {info['objetivo']}, Meta {info['meta']}) — {info['nome']}")
        for d in descritivos:
            niveis = d.get('SiglasNiveisTerritoriais') or []
            relatorio.append(f"    [{d['Id']}] {d['Nome']}  (niveis={niveis})")

    relatorio.append(f"\n### 2b. Sem dado por UF — ignorados, so tem dado nacional/regional ({len(novos_sem_uf)})\n")
    if not novos_sem_uf:
        relatorio.append("Nenhum.\n")
    for numero, info, descritivos in novos_sem_uf:
        niveis_encontrados = sorted({n for d in descritivos for n in (d.get('SiglasNiveisTerritoriais') or [])})
        relatorio.append(f"- **{numero}** (Objetivo {info['objetivo']}, Meta {info['meta']}) — {info['nome']} "
                          f"(niveis disponiveis: {niveis_encontrados})")

    if novos_com_erro:
        relatorio.append(f"\n### 2c. Nao foi possivel checar o nivel territorial ({len(novos_com_erro)})\n")
        for numero, info, erro in novos_com_erro:
            relatorio.append(f"- **{numero}** (Objetivo {info['objetivo']}) — {erro}")

    # 3) Indicadores mapeados em constants.py cujo status no site nao e mais 'Produzido'
    #    (dado pode ter sido descontinuado/renomeado do lado do IBGE)
    regressoes = []
    for cid in sorted(mapeados_constants):
        numero = cid.replace('Indicador ', '')
        numero_oficial = numero_base_oficial(numero, inventario_site)
        if numero_oficial and inventario_site[numero_oficial]['status'] != 'Produzido':
            regressoes.append((cid, inventario_site[numero_oficial]))
    relatorio.append(f"\n## 3. Indicadores mapeados cujo status no site nao e mais 'Produzido' ({len(regressoes)})\n")
    if not regressoes:
        relatorio.append("Nenhum.\n")
    for cid, info in regressoes:
        relatorio.append(f"- **{cid}** -> site reporta status **{info['status']}** "
                          f"(Objetivo {info['objetivo']}, Meta {info['meta']})")

    texto = "\n".join(relatorio)
    print("\n" + texto)

    if args.output:
        Path(args.output).write_text(texto, encoding='utf-8')
        print(f"\nRelatorio salvo em {args.output}")


if __name__ == '__main__':
    main()
