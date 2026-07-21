import argparse
import json.decoder
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import time
import logging
import asyncio
from asyncio import Semaphore # Importa Semaphore
import aiohttp
from tqdm.asyncio import tqdm as async_tqdm
import logging.handlers
from typing import Dict, Any, List, Tuple, Optional, Union

import pandas as pd
import numpy as np

from constants import LIST_INDICADORES, LIST_COLUNAS
from check_indicadores_ibge import (
    carregar_constants_indicadores,
    carregar_indicadores_site,
    numero_para_ids_csv,
    obter_tabelas_indicador,
    tem_dado_estadual,
)

# Configuração de logging com FileHandler
# Nome com timestamp para diferenciar cada execução (mesma convenção usada em app.py)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / datetime.now().strftime('update_db_log_%Y-%m-%d_%H-%M-%S.log')

# Handler para o arquivo
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(log_formatter)

# Handler para o console
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Configura o logger raiz
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

# Define níveis específicos se necessário (ex: menos verbose no console)
# console_handler.setLevel(logging.INFO)
# file_handler.setLevel(logging.DEBUG) # Log mais detalhado no arquivo

# Define o tipo de retorno como uma tupla onde o segundo elemento pode ser Any (dados) ou uma Exception
AsyncSidraResult = Tuple[str, Union[Any, Exception]]

async def get_sidra_data(session: aiohttp.ClientSession, url: str, indicador_name: str) -> AsyncSidraResult:
    """Obtém dados de uma URL da API SIDRA de forma assíncrona com retries."""
    max_retries = 5
    retry_count = 0
    last_exception = None
    while retry_count < max_retries:
        try:
            logging.debug(f"[Async] Tentando obter dados de: {indicador_name} - Tentativa {retry_count + 1}/{max_retries}")
            async with session.get(url, timeout=90) as response:
                response.raise_for_status() # Levanta exceção para status 4xx/5xx
                # Log Content-Type se não for JSON (aviso)
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' not in content_type.lower():
                    logging.warning(f"[Async] Content-Type inesperado '{content_type}' para {indicador_name}. Tentando decodificar mesmo assim.")
                # Tenta decodificar JSON, tratando o content_type se necessário
                data: Any = await response.json(encoding='UTF-8', content_type=None) # content_type=None para flexibilidade
                
                # Validação da estrutura dos dados recebidos
                if not isinstance(data, list) or len(data) < 2:
                    logging.warning(f"[Async] Estrutura de dados inválida recebida para {indicador_name} (não é lista ou tem menos de 2 elementos). Dados: {str(data)[:200]}") # Loga parte dos dados para debug
                    # Considera isso uma falha e força uma nova tentativa (ou falha final se retries esgotarem)
                    raise ValueError("Estrutura de dados inválida da API")
                    
                logging.info(f"[Async] Sucesso ao obter e validar dados de: {indicador_name}")
                return indicador_name, data # Retorna nome + dados validados
        except aiohttp.ClientResponseError as e:
            logging.warning(f"[Async] Erro HTTP {e.status} para {indicador_name}: {e.message} - Tentativa {retry_count + 1}")
            last_exception = e
        except asyncio.TimeoutError:
            logging.warning(f"[Async] Timeout ao obter dados de {indicador_name} - Tentativa {retry_count + 1}")
            last_exception = asyncio.TimeoutError(f"Timeout para {indicador_name}")
        except Exception as e:
            # Captura erros de JSONDecode aqui também
            logging.warning(f"[Async] Exceção ao obter/decodificar dados de {indicador_name}: {e} - Tentativa {retry_count + 1}")
            last_exception = e

        retry_count += 1
        if retry_count < max_retries:
            await asyncio.sleep(5) # Espera assíncrona

    # Log detalhado da falha final, incluindo a última exceção
    final_error_msg = f"[Async] Falha ao obter dados para {indicador_name} após {max_retries} tentativas."
    final_exception = last_exception if last_exception else Exception(f"Falha desconhecida para {indicador_name}")
    logging.error(f"{final_error_msg} Último erro: {final_exception}", exc_info=isinstance(last_exception, Exception)) # Adiciona stack trace se for exceção real
    
    # Retorna nome + exceção em caso de falha final
    return indicador_name, final_exception


@lru_cache(maxsize=1)
def load_indicadores() -> pd.DataFrame:
    df = pd.read_csv(
        Path(__file__).parent / 'db/indicadores.csv', sep=';'
    )
    return df


def filter_indicadores(list_indicadores: Dict[str, Any], indicadores_ids: List[str]) -> Dict[str, Any]:
    return {
        objetivo: {
            meta: {
                indicador: url
                for indicador, url in metas.items()
                if indicador in indicadores_ids
            }
            for meta, metas in metas_objetivo.items()
        }
        for objetivo, metas_objetivo in list_indicadores.items()
    }


def converter_tipos_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte os tipos de dados do DataFrame para formatos mais apropriados.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados a serem convertidos
        
    Returns:
        pd.DataFrame: DataFrame com os tipos de dados convertidos
    """
    # Colunas que devem ser numéricas
    colunas_numericas = ['VLR_VAR']
    
    # Colunas que devem ser categóricas
    colunas_categoricas = [
        'ID_INDICADOR', 'CODG_UND_MED', 'CODG_UND_FED', 'CODG_VAR',
        'CODG_SEXO', 'CODG_IDADE', 'CODG_RACA', 'CODG_SIT_DOM',
        'CODG_NIV_INSTR', 'CODG_REND_MENSAL_DOM_PER_CAP',
        'COD_GRU_IDADE_NIV_ENS', 'CODG_INF_ESC', 'CODG_ETAPA_ENS',
        'CODG_SET_ATIV', 'CODG_ECO_REL_AGUA', 'CODG_TIP_DIN_ECO_REL_AGUA',
        'CODG_TIPO_DOENCA', 'CODG_DEF', 'CODG_ATV_TRAB'
    ]
    
    # Converter colunas numéricas
    for col in colunas_numericas:
        if col in df.columns:
            try:
                # Modificado: Limpeza antes da conversão numérica
                df[col] = df[col].astype(str)
                codes_to_nan = ["...", "-", "X"]
                df[col] = df[col].replace(codes_to_nan, np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Adicionado: Preenche NaN com 0 após a conversão
                df[col] = df[col].fillna(0)
            except Exception as e:
                logging.error(f"Erro ao converter coluna numérica '{col}': {e}")
                # Decide se quer parar ou continuar com a coluna como objeto
                # df[col] = pd.Series(dtype='object') # Exemplo: Mantém como objeto se falhar
    
    # Converter CODG_ANO para string
    if 'CODG_ANO' in df.columns:
        try:
            df['CODG_ANO'] = df['CODG_ANO'].astype(str)
        except Exception as e:
            logging.error(f"Erro ao converter CODG_ANO para string: {e}")
    
    # Converter colunas categóricas
    for col in colunas_categoricas:
        if col in df.columns:
            try:
                df[col] = df[col].astype('category')
            except Exception as e:
                logging.error(f"Erro ao converter coluna '{col}' para category: {e}")
    
    return df


async def process_indicadores(filtered_list_indicadores, url_base, list_colunas,
                             df_und_med_global: pd.DataFrame, 
                             df_variavel_global: pd.DataFrame, 
                             df_indicadores: pd.DataFrame) -> None:
    """Processa indicadores de forma assíncrona com barra de progresso."""
    base_dir = Path(__file__).parent
    results_dir = base_dir / 'db' / 'resultados'
    results_dir.mkdir(parents=True, exist_ok=True)

    CONCURRENCY_LIMIT = 15  # Limite de requisições simultâneas
    semaphore = Semaphore(CONCURRENCY_LIMIT)

    # Função auxiliar para controlar concorrência com Semaphore
    async def fetch_with_semaphore(session: aiohttp.ClientSession, url: str, indicador_name: str) -> AsyncSidraResult:
        async with semaphore:
            # logging.debug(f"Semaphore acquired for {indicador_name}") # Debug (opcional)
            result = await get_sidra_data(session, url, indicador_name)
            # logging.debug(f"Semaphore released for {indicador_name}") # Debug (opcional)
            return result

    tasks = []
    logging.info("Coletando URLs e criando tarefas de download...")
    failed_indicators: List[Tuple[str, str]] = [] # Lista para rastrear falhas

    # Modificado: O loop as_completed agora está DENTRO do bloco da sessão
    async with aiohttp.ClientSession() as session:
        # 1. Cria todas as tarefas primeiro, usando a função com semáforo
        for objetivo, metas_objetivo in filtered_list_indicadores.items():
            for meta, indicadores_meta in metas_objetivo.items():
                for indicador, endpoint in indicadores_meta.items():
                    link = url_base + str(endpoint)
                    # Cria a tarefa chamando a função auxiliar
                    tasks.append(asyncio.create_task(fetch_with_semaphore(session, link, indicador)))

        total_tasks = len(tasks)
        logging.info(f"{total_tasks} tarefas de download criadas (limite de concorrência: {CONCURRENCY_LIMIT}). Iniciando downloads e processamento...")

        # Listas para acumular dataframes auxiliares (inicializadas antes do loop)
        und_med_dfs = [df_und_med_global]
        variavel_dfs = [df_variavel_global]

        # 2. Processa tarefas conforme completam, DENTRO da sessão
        for future in async_tqdm(asyncio.as_completed(tasks), total=total_tasks, desc="Baixando e Processando Indicadores"):
            indicador = None
            try:
                indicador, result = await future

                filename_part = indicador.lower().replace(' ', '')
                arquivo_parquet = results_dir / f'{filename_part}.parquet'

                if isinstance(result, Exception):
                    logging.error(f"Falha no download para {indicador}: {result}")
                    failed_indicators.append((indicador, f"Download falhou: {result}"))
                    continue

                if result is None or not isinstance(result, list) or len(result) < 2:
                    logging.warning(f"Dados inválidos ou insuficientes recebidos para {indicador}. Pulando processamento.")
                    failed_indicators.append((indicador, "Dados inválidos/insuficientes recebidos"))
                    continue

                # --- Processamento principal --- (try...except interno)
                try:
                    df_temp = pd.DataFrame(result)
                    df_temp.columns = df_temp.iloc[0]
                    df_temp = df_temp[1:]
                    df_temp = df_temp.rename(columns=list_colunas)
                    df_temp.insert(0, 'ID_INDICADOR', f'Indicador {indicador.split("Indicador")[-1]}')

                    if 'CODG_UND_MED' in df_temp.columns and 'DESC_UND_MED' in df_temp.columns:
                        und_med_dfs.append(df_temp[['CODG_UND_MED', 'DESC_UND_MED']])
                    if 'CODG_VAR' in df_temp.columns and 'DESC_VAR' in df_temp.columns:
                        variavel_dfs.append(df_temp[['CODG_VAR', 'DESC_VAR']])

                    colunas_para_remover = [
                        'CODG_NIV_TER', 'DESC_NIV_TER', 'DESC_UND_MED', 'DESC_VAR', 'DESC_ANO'
                    ]
                    colunas_existentes = [col for col in colunas_para_remover if col in df_temp.columns]
                    if colunas_existentes:
                        df_temp = df_temp.drop(columns=colunas_existentes)

                    list_var = df_temp['CODG_VAR'].unique().tolist() if 'CODG_VAR' in df_temp.columns else []
                    if len(list_var) > 1:
                        df_combined = pd.DataFrame()
                        for cod_var in list_var:
                            df_temp_var = df_temp[df_temp['CODG_VAR'] == cod_var].copy()
                            df_temp_var = converter_tipos_dados(df_temp_var)
                            df_combined = pd.concat([df_combined, df_temp_var], ignore_index=True)
                        df_to_save = df_combined
                        df_indicadores.loc[df_indicadores['ID_INDICADOR'] == indicador, 'VARIAVEIS'] = 1
                    else:
                        df_temp = converter_tipos_dados(df_temp)
                        df_to_save = df_temp

                    df_to_save.to_parquet(arquivo_parquet)
                    logging.debug(f'Arquivo {filename_part}.parquet salvo.')
                except Exception as process_error:
                    logging.exception(f"Erro ao PROCESSAR dados para {indicador}: {process_error}")
                    failed_indicators.append((indicador, f"Erro no processamento: {process_error}"))
                    continue
                # --- Fim Processamento principal ---
            except Exception as e:
                log_msg = f"Erro geral no loop de processamento: {e}"
                if indicador:
                    log_msg = f"Erro geral no loop de processamento para {indicador}: {e}"
                    failed_indicators.append((indicador if indicador else "Desconhecido", f"Erro geral no loop: {e}"))
                logging.exception(log_msg)
        # FIM do loop as_completed (ainda dentro do bloco da sessão)

    # 3. Consolida e salva DataFrames auxiliares (APÓS o loop e fora da sessão)
    logging.info("Consolidando e salvando arquivos auxiliares (unidade_medida, variavel, indicadores)...")
    try:
        df_und_med_final = pd.concat(und_med_dfs, ignore_index=True)

        # Limpeza e remoção de duplicatas para unidade_medida
        if 'CODG_UND_MED' in df_und_med_final.columns:
            df_und_med_final = df_und_med_final.dropna(subset=['CODG_UND_MED'])
            df_und_med_final['CODG_UND_MED'] = df_und_med_final['CODG_UND_MED'].astype(str).str.strip()
            df_und_med_final = df_und_med_final[~df_und_med_final['CODG_UND_MED'].isin(['0', ''])]
            # Remove duplicatas baseado APENAS em CODG_UND_MED
            df_und_med_final = df_und_med_final.drop_duplicates(subset=['CODG_UND_MED'], keep='first')

            # Correção: Converte para numérico ANTES de ordenar
            df_und_med_final['CODG_UND_MED'] = pd.to_numeric(df_und_med_final['CODG_UND_MED'], errors='coerce')
            df_und_med_final = df_und_med_final.dropna(subset=['CODG_UND_MED']) # Remove se falhou na conversão
            df_und_med_final['CODG_UND_MED'] = df_und_med_final['CODG_UND_MED'].astype(int)

        # Ordenação Numérica: Atribui de volta à variável
        if not df_und_med_final.empty and 'CODG_UND_MED' in df_und_med_final.columns:
             df_und_med_final = df_und_med_final.sort_values(by='CODG_UND_MED')

        df_und_med_final.to_csv(
            base_dir / 'db' / 'unidade_medida.csv', header=True, index=False, sep=';', encoding='utf-8'
        )
        logging.info("unidade_medida.csv salvo com sucesso.")
    except Exception as e:
        logging.exception(f"Erro ao salvar unidade_medida.csv: {e}")

    try:
        df_variavel_final = pd.concat(variavel_dfs, ignore_index=True)

        # Padroniza CODG_VAR inválidos para '0' (ainda como string aqui)
        if 'CODG_VAR' in df_variavel_final.columns:
            df_variavel_final['CODG_VAR'] = df_variavel_final['CODG_VAR'].astype(str).fillna('0').str.strip().replace('', '0')
        else:
             df_variavel_final['CODG_VAR'] = '0'

        # Remove duplicatas baseado em CODG_VAR e DESC_VAR
        df_variavel_final = df_variavel_final.drop_duplicates(subset=['CODG_VAR', 'DESC_VAR'], keep='first')
        
        # Converte CODG_VAR para numérico ANTES de ordenar
        df_variavel_final = df_variavel_final[df_variavel_final['CODG_VAR'] != '0']
        if 'CODG_VAR' in df_variavel_final.columns:
            df_variavel_final['CODG_VAR'] = pd.to_numeric(df_variavel_final['CODG_VAR'], errors='coerce')
            df_variavel_final = df_variavel_final.dropna(subset=['CODG_VAR']) # Remove se falhou na conversão
            df_variavel_final['CODG_VAR'] = df_variavel_final['CODG_VAR'].astype(int)

        # Ordenação Numérica: Atribui de volta à variável
        if not df_variavel_final.empty and 'CODG_VAR' in df_variavel_final.columns:
            df_variavel_final = df_variavel_final.sort_values(by='CODG_VAR')

        # Salva CSV apenas com CODG_VAR e DESC_VAR
        # Garante que apenas as colunas desejadas estão presentes e na ordem correta
        if 'CODG_VAR' in df_variavel_final.columns and 'DESC_VAR' in df_variavel_final.columns:
             df_to_save = df_variavel_final[['CODG_VAR', 'DESC_VAR']]
        else: # Fallback para evitar erro se alguma coluna faltar
             df_to_save = df_variavel_final
            
        df_to_save.to_csv(
            base_dir / 'db' / 'variavel.csv', 
            header=True, index=False, sep=';', encoding='utf-8'
        )
        logging.info("variavel.csv salvo com sucesso.")
    except Exception as e:
        logging.exception(f"Erro ao salvar variavel.csv: {e}")

    try:
        df_indicadores.to_csv(base_dir / 'db' / 'indicadores.csv', sep=';', index=False)
        logging.info("indicadores.csv salvo com sucesso.")
    except Exception as e:
        logging.exception(f"Erro ao salvar indicadores.csv: {e}")

    # 4. Loga resumo de falhas
    if failed_indicators:
        logging.warning("--- Resumo de Indicadores com Falha ---")
        for name, reason in failed_indicators:
            logging.warning(f"- {name}: {reason}")
        logging.warning(f"Total de falhas: {len(failed_indicators)}")
    else:
        logging.info("Todos os indicadores foram processados sem falhas aparentes.")

    # 5. Remove parquets obsoletos: compara os arquivos existentes contra o
    #    mapeamento ATUAL de constants.py (nao contra o resultado desta execucao),
    #    entao um indicador que so falhou nesta rodada mantem seu parquet antigo
    #    como fallback, mas um indicador renomeado/desativado tem seu arquivo
    #    obsoleto removido (e assim deixa de ser versionado no git).
    esperados = {
        f"{indicador.lower().replace(' ', '')}.parquet"
        for metas_objetivo in filtered_list_indicadores.values()
        for indicadores_meta in metas_objetivo.values()
        for indicador in indicadores_meta.keys()
    }
    existentes = {p.name: p for p in results_dir.glob('*.parquet')}
    orfaos = set(existentes) - esperados
    if orfaos:
        logging.info(f"Removendo {len(orfaos)} arquivo(s) parquet obsoleto(s) (indicador nao mapeado atualmente)...")
        for nome in sorted(orfaos):
            try:
                existentes[nome].unlink()
                logging.info(f"  removido: {nome}")
            except Exception as e:
                logging.error(f"  falha ao remover {nome}: {e}")
    else:
        logging.info("Nenhum arquivo parquet obsoleto encontrado.")


def verificar_indicadores_sem_url(indicadores_ids: List[str]) -> List[str]:
    """Checagem local (sem rede): indicadores RBC=1 que nao tem URL em
    constants.py e que portanto seriam silenciosamente ignorados por
    filter_indicadores. Roda sempre, pois e instantanea."""
    mapeados_constants = carregar_constants_indicadores()
    return [i for i in indicadores_ids if i not in mapeados_constants]


def verificar_novidades_no_site() -> List[Tuple[str, dict]]:
    """Checagem online (18 requisicoes a odsbrasil.gov.br, mais 2 por
    candidato): indicadores com status 'Produzido' no site que ainda nao tem
    nenhuma linha em indicadores.csv E que possuem dado a nivel de Unidade
    Federativa (UF) -- unico nivel territorial que interessa para a
    aplicacao, que so exibe os estados ja listados (RO, MA, TO, MT, MS, GO,
    DF). Indicadores so-nacionais (sem UF) sao descartados aqui. So roda se
    --verificar-site for passado, pois depende de um servico de terceiro e
    nao deve travar a atualizacao de producao."""
    todos_ids = set(load_indicadores()['ID_INDICADOR'].tolist())
    inventario_site = carregar_indicadores_site(list(range(1, 19)))
    candidatos = [
        (numero, info) for numero, info in sorted(inventario_site.items())
        if info['status'] == 'Produzido' and not numero_para_ids_csv(numero, todos_ids)
    ]

    relevantes = []
    ignorados_sem_uf = 0
    for numero, info in candidatos:
        try:
            descritivos = obter_tabelas_indicador(info['objetivo'], numero)
        except Exception as e:
            logging.debug(f"Falha ao checar nivel territorial de {numero}: {e}")
            continue
        if tem_dado_estadual(descritivos):
            relevantes.append((numero, info))
        else:
            ignorados_sem_uf += 1

    if ignorados_sem_uf:
        logging.info(
            f"{ignorados_sem_uf} indicador(es) novo(s) no site ignorado(s) por nao terem dado a nivel de UF."
        )
    return relevantes


async def main(verificar_site: bool = False) -> None:
    logging.info("Iniciando atualização da base de dados...")
    url_base = 'https://apisidra.ibge.gov.br/values'
    db_path = Path(__file__).parent / 'db'

    # Carrega DF unidade_medida
    try:
        df_und_med = pd.read_csv(db_path / 'unidade_medida.csv', sep=';', dtype=str) # Ler como string para consistência
    except FileNotFoundError:
        logging.warning("Arquivo unidade_medida.csv não encontrado. Criando DataFrame vazio.")
        df_und_med = pd.DataFrame(columns=['CODG_UND_MED', 'DESC_UND_MED'])

    # Inicializa df_variavel vazio para acumular dados da API
    df_variavel = pd.DataFrame(columns=['CODG_VAR', 'DESC_VAR'])

    # Carrega e prepara df_indicadores
    try:
        df_indicadores = load_indicadores()
        df_indicadores = df_indicadores[df_indicadores['RBC'] == True].copy()
        if 'VARIAVEIS' not in df_indicadores.columns:
             df_indicadores['VARIAVEIS'] = 0
        else:
             df_indicadores['VARIAVEIS'] = df_indicadores['VARIAVEIS'].fillna(0).astype(int)
        indicadores_ids = df_indicadores['ID_INDICADOR'].tolist()
        logging.info(f"{len(indicadores_ids)} indicadores RBC encontrados para processar.")
    except Exception as e:
        logging.exception("Erro crítico ao carregar db/indicadores.csv. Abortando.")
        return

    # Checagem de consistencia entre indicadores.csv e constants.py: sem isso,
    # um indicador RBC=1 sem URL mapeada era descartado sem nenhum aviso.
    sem_url = verificar_indicadores_sem_url(indicadores_ids)
    if sem_url:
        logging.warning(
            f"{len(sem_url)} indicador(es) com RBC=1 NAO possuem URL em constants.py e "
            f"serao IGNORADOS nesta atualização: {', '.join(sem_url)}"
        )
        logging.warning(
            "Rode 'python check_indicadores_ibge.py --detail' para investigar as tabelas SIDRA candidatas."
        )

    if verificar_site:
        logging.info("Verificando novidades em odsbrasil.gov.br (checagem opcional, pode levar ~1 min)...")
        try:
            novos = verificar_novidades_no_site()
            if novos:
                logging.warning(f"{len(novos)} indicador(es) 'Produzido' no site ainda nao constam em indicadores.csv:")
                for numero, info in novos:
                    logging.warning(f"  - {numero} (Objetivo {info['objetivo']}, Meta {info['meta']}): {info['nome']}")
            else:
                logging.info("Nenhuma novidade encontrada no site além do que já está em indicadores.csv.")
        except Exception as e:
            logging.warning(f"Não foi possível verificar novidades em odsbrasil.gov.br: {e}")

    filtered_list_indicadores = filter_indicadores(LIST_INDICADORES, indicadores_ids)

    # Passa o mapeamento e o df_variavel inicial para a função de processamento
    await process_indicadores(filtered_list_indicadores, url_base, LIST_COLUNAS,
                           df_und_med, df_variavel, df_indicadores)

    if sem_url:
        logging.warning(
            f"Lembrete: {len(sem_url)} indicador(es) RBC=1 continuam sem URL em constants.py "
            f"e não foram atualizados nesta execução: {', '.join(sem_url)}"
        )

    logging.info("Atualização da base de dados concluída.")


if __name__ == '__main__':
    # Pequena melhoria: Verifica se aiohttp está instalado apenas uma vez
    try:
        import aiohttp
    except ImportError:
        logging.error("A biblioteca aiohttp não está instalada. Execute: pip install aiohttp")
        exit(1) # Sai se a dependência crucial estiver faltando

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verificar-site', action='store_true',
        help='Alem da checagem local, consulta odsbrasil.gov.br em busca de indicadores '
             'novos ainda nao presentes em indicadores.csv (mais lento, depende de terceiro).'
    )
    args = parser.parse_args()

    start_time = time.time()
    asyncio.run(main(verificar_site=args.verificar_site))
    end_time = time.time()
    logging.info(f"Tempo total de execução: {end_time - start_time:.2f} segundos")
