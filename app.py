import dash
from dash import (
    html, dcc, Input, Output, State, callback, dash_table,
    callback_context, ALL, MATCH, no_update
)
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import time
import warnings
from datetime import datetime
import numpy as np
from dash.exceptions import PreventUpdate
from config import *
import secrets
from dotenv import load_dotenv
from functools import lru_cache
from cache_manager import cache_manager, load_dados_indicador_cached, preload_related_indicators
from flask import session, redirect, send_from_directory, request, jsonify
import bcrypt
from generate_password import generate_password_hash, generate_secret_key, update_env_file, check_password
from flask_cors import CORS
import constants 
import logging
import math
import io

load_dotenv()

# Configuração do tema do Plotly
import plotly.io as pio

pio.templates.default = "plotly_white"

from config import (
    DEBUG, USE_RELOADER, PORT, HOST, DASH_CONFIG, SERVER_CONFIG,
    MAINTENANCE_PASSWORD
)
from constants import COLUMN_NAMES, UF_NAMES

# Configuração do Logging
log_level = logging.DEBUG if DEBUG else logging.INFO
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_filename = datetime.now().strftime(f'{log_dir}/app_log_%Y-%m-%d.log')

log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setFormatter(log_formatter)

# Configura o logger raiz
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
root_logger.addHandler(file_handler)

logging.info(f"Iniciando aplicação. Nível de log: {logging.getLevelName(log_level)}. Logando em: {log_filename}")

# Variável global para controle do modo de manutenção
MAINTENANCE_MODE = os.getenv('MAINTENANCE_MODE', 'false').lower() == 'true'


def maintenance_middleware():
    """Middleware para verificar se o sistema está em manutenção"""
    global MAINTENANCE_MODE
    if MAINTENANCE_MODE and request.remote_addr not in ['127.0.0.1']:
        if request.path.startswith('/assets/') or '_dash-component-suites' in request.path:
            return None
        return send_from_directory('assets', 'maintenance.html')
    return None


def capitalize_words(text):
    return ' '.join(word.capitalize() for word in text.split())


# Inicializa o aplicativo Dash com tema Bootstrap
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.MATERIA,
        "https://cdn.jsdelivr.net/npm/ag-grid-community@30.2.1/styles/ag-grid.min.css",
        "https://cdn.jsdelivr.net/npm/ag-grid-community@30.2.1/styles/ag-theme-alpine.min.css",
        "https://use.fontawesome.com/releases/v5.15.4/css/all.css",  # Adiciona Font Awesome para os ícones
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css"  # Adiciona Bootstrap Icons como alternativa
    ],
    assets_folder='assets',
    assets_url_path='/assets/',
    serve_locally=True,
    suppress_callback_exceptions=True,  # Suprimir exceções de callbacks para componentes dinâmicos
    **DASH_CONFIG
)

# Registra o middleware de manutenção
app.server.before_request(maintenance_middleware)

# Configurações de cache
for key, value in SERVER_CONFIG.items():
    app.server.config[key] = value

# Configura a chave secreta do Flask
app.server.secret_key = SERVER_CONFIG['SECRET_KEY']


@app.server.route('/assets/<path:path>')
def serve_static(path):
    return send_from_directory('assets', path)


CORS(app.server)


@app.server.route('/log', methods=['POST'])
def log_message():
    if not app.server.debug:
        return '', 204
    try:
        data = request.get_json(force=True)
        # print("\n====================== ERRO DO NAVEGADOR ======================")
        # print(f"Mensagem: {data.get('message', 'Sem mensagem')}")
        # print(f"Stack: {data.get('stack', 'Sem stack')}")
        # print("===============================================================\n")
        # Usamos logging.error para registrar o erro do navegador no backend
        # Corrigido: Removida a quebra de linha dentro da string de formatação
        logging.error("Erro do navegador recebido: Mensagem: %s Stack: %s",
                      data.get('message', 'Sem mensagem'),
                      data.get('stack', 'Sem stack'))
        return jsonify({"status": "logged", "success": True})
    except Exception as e:
        # print(f"Erro ao processar log do cliente: {str(e)}")
        # Usamos logging.exception para capturar o erro e o traceback
        logging.exception("Erro ao processar log do cliente:")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.server.route('/_dash-component-suites/<path:path>')
def serve_dash_files(path):
    return send_from_directory('_dash-component-suites', path)


# Função original para carregar dados do indicador (sem cache)
def _load_dados_indicador_original(indicador_id):
    """Função original para carregar dados do indicador (sem cache)."""
    try:
        nome_arquivo = indicador_id.lower().replace("indicador ", "")
        arquivo_parquet = f'db/resultados/indicador{nome_arquivo}.parquet'
        if not os.path.exists(arquivo_parquet):
            logging.warning("Arquivo parquet não encontrado para %s: %s", indicador_id, arquivo_parquet)
            return pd.DataFrame()
        try:
            df_load = pd.read_parquet(arquivo_parquet)
            if df_load.empty:
                logging.warning("Arquivo parquet vazio para %s: %s", indicador_id, arquivo_parquet)
                return pd.DataFrame()
        except Exception as e:
            logging.exception("Erro ao ler arquivo parquet para %s", indicador_id)
            return pd.DataFrame()
        return df_load
    except Exception as e:
        logging.exception("Erro geral em _load_dados_indicador_original para %s", indicador_id)
        return pd.DataFrame()


# Função com cache de dois níveis
def load_dados_indicador_cache(indicador_id):
    """Carrega dados do indicador usando cache de dois níveis (memória e disco)."""
    return load_dados_indicador_cached(indicador_id, _load_dados_indicador_original)


def limpar_cache_indicadores():
    """Limpa o cache de indicadores."""
    # Limpa o cache de dois níveis
    cache_manager.clear()
    # Limpa o cache LRU da função original (se ainda estiver sendo usado em algum lugar)
    if hasattr(load_dados_indicador_cache, 'cache_clear'):
        load_dados_indicador_cache.cache_clear()


@app.server.route('/limpar-cache')
def limpar_cache():
    try:
        session.clear()
        limpar_cache_indicadores()
        return redirect('/')
    except Exception as e:
        logging.exception("Erro ao limpar cache:")
        return redirect('/')


@app.server.route('/cache-stats')
def view_cache_stats():
    """Exibe estatísticas do cache."""
    stats = cache_manager.get_stats()
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Estatísticas do Cache</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            .stats {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .stat-item {{ margin-bottom: 10px; }}
            .stat-label {{ font-weight: bold; }}
            .hit-rate {{ font-size: 1.2em; color: #27ae60; }}
            .actions {{ margin-top: 20px; }}
            .btn {{ display: inline-block; padding: 10px 15px; background-color: #3498db; color: white;
                   text-decoration: none; border-radius: 4px; margin-right: 10px; }}
            .btn:hover {{ background-color: #2980b9; }}
            .btn-danger {{ background-color: #e74c3c; }}
            .btn-danger:hover {{ background-color: #c0392b; }}
            .timestamp {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Estatísticas do Cache</h1>
        <div class="timestamp">Dados atualizados em: {time.strftime('%d/%m/%Y %H:%M:%S')}</div>

        <div class="stats">
            <div class="stat-item hit-rate">
                <span class="stat-label">Taxa de acerto:</span> {stats['hit_rate']:.2%}
            </div>
            <div class="stat-item">
                <span class="stat-label">Acertos em memória:</span> {stats['memory_hits']}
            </div>
            <div class="stat-item">
                <span class="stat-label">Acertos em disco:</span> {stats['disk_hits']}
            </div>
            <div class="stat-item">
                <span class="stat-label">Erros:</span> {stats['misses']}
            </div>
            <div class="stat-item">
                <span class="stat-label">Pré-carregamentos:</span> {stats['preloads']}
            </div>
            <div class="stat-item">
                <span class="stat-label">Tamanho do cache em memória:</span> {stats['memory_cache_size']}/{stats['memory_cache_maxsize']}
            </div>
        </div>

        <div class="actions">
            <a href="/limpar-cache" class="btn btn-danger">Limpar Cache</a>
            <a href="/" class="btn">Voltar para o Painel</a>
        </div>
    </body>
    </html>
    """
    return html


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>IMB - Painel ODS</title>
        <link rel="icon" type="image/x-icon" href="/assets/favicon.ico">
        {%favicon%}
        {%css%}
        <meta http-equiv="Cache-Control" content="max-age=31536000">
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
        (function() {
            function enviarLog(mensagem, stack) {
                fetch('/log', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: typeof mensagem === 'object' ? JSON.stringify(mensagem) : String(mensagem),
                        stack: stack || new Error().stack
                    })
                })
                .then(response => response.ok ? response.json() : Promise.reject('Falha na requisição: ' + response.status))
                .then(data => console.log("Log enviado:", data))
                .catch(err => console.log("Erro ao enviar log:", err));
            }
            const originalConsoleError = console.error;
            console.error = function() {
                const args = Array.from(arguments);
                const message = args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : String(arg)).join(' ');
                enviarLog(message, new Error().stack);
                originalConsoleError.apply(console, arguments);
            };
            window.addEventListener('error', function(event) {
                enviarLog(event.message, event.error ? event.error.stack : null);
                return false;
            });
            window.addEventListener('unhandledrejection', function(event) {
                enviarLog(
                    'Promessa rejeitada não tratada: ' + (event.reason ? event.reason.toString() : 'Razão desconhecida'),
                    event.reason && event.reason.stack ? event.reason.stack : null
                );
            });
        })();
        </script>
    </body>
</html>
'''


@lru_cache(maxsize=1)
def load_objetivos():
    try:
        df_obj = pd.read_csv('db/objetivos.csv', low_memory=False, encoding='utf-8', dtype=str, sep=';',
                             on_bad_lines='skip')
        if df_obj.iloc[(0,)].name == '#':
            df_obj = df_obj.iloc[1:]
        required_columns = ['ID_OBJETIVO', 'RES_OBJETIVO', 'DESC_OBJETIVO', 'BASE64']
        if not all(col in df_obj.columns for col in required_columns):
            return pd.DataFrame(columns=required_columns)
        return df_obj
    except Exception as e:
        return pd.DataFrame(columns=['ID_OBJETIVO', 'RES_OBJETIVO', 'DESC_OBJETIVO', 'BASE64'])


@lru_cache(maxsize=1)
def load_metas():
    try:
        return pd.read_csv('db/metas.csv', low_memory=False, encoding='utf-8', dtype=str, sep=';', on_bad_lines='skip')
    except Exception as e:
        return pd.DataFrame()


@lru_cache(maxsize=1)
def load_indicadores():
    try:
        df_ind = pd.read_csv('db/indicadores.csv', low_memory=False, encoding='utf-8', dtype=str, sep=';',
                             on_bad_lines='skip')
        # Converte RBC e RANKING_ORDEM para numérico, tratando erros
        df_ind['RBC'] = pd.to_numeric(df_ind['RBC'], errors='coerce')
        if 'RANKING_ORDEM' in df_ind.columns:
            df_ind['RANKING_ORDEM'] = pd.to_numeric(df_ind['RANKING_ORDEM'], errors='coerce').fillna(0).astype(
                int)  # Converte para int, NaN vira 0
        else:
            df_ind['RANKING_ORDEM'] = 0  # Adiciona coluna com 0 se não existir
        # Filtra por RBC == 1
        return df_ind.loc[df_ind['RBC'] == 1]
    except Exception as e:
        # Retorna DataFrame vazio com as colunas esperadas em caso de erro
        return pd.DataFrame(
            columns=['ID_INDICADOR', 'ID_META', 'ID_OBJETIVO', 'DESC_INDICADOR', 'VARIAVEIS', 'GRAFICO_LINHA', 'RBC',
                     'RANKING_ORDEM'])


@lru_cache(maxsize=1)
def load_sugestoes_visualizacao():
    try:
        return pd.read_csv('db/sugestoes_visualizacao.csv', low_memory=False, encoding='utf-8', dtype=str, sep=';',
                           on_bad_lines='skip')
    except Exception as e:
        return pd.DataFrame()


@lru_cache(maxsize=1)
def load_unidade_medida():
    cols = ['CODG_UND_MED', 'DESC_UND_MED']
    try:
        try:
            df_um = pd.read_csv('db/unidade_medida.csv', low_memory=False, encoding='utf-8', dtype=str, sep=';')
        except:
            df_um = pd.read_csv('db/unidade_medida.csv', low_memory=False, encoding='utf-8', dtype=str, sep=',')
        if len(df_um.columns) == 1 and ',' in df_um.columns[0]:
            df_um = pd.DataFrame([x.split(',') for x in df_um[df_um.columns[0]]])
            if len(df_um.columns) >= 2:
                df_um = df_um.iloc[:, [0, 1]]
                df_um.columns = cols
        if not all(col in df_um.columns for col in cols):
            return pd.DataFrame(columns=cols)
        for col in cols:
            df_um[col] = df_um[col].str.strip().str.strip('"')
        return df_um
    except Exception as e:
        return pd.DataFrame(columns=cols)


@lru_cache(maxsize=1)
def load_variavel():
    cols = ['CODG_VAR', 'DESC_VAR']
    try:
        try:
            df_var = pd.read_csv('db/variavel.csv', low_memory=False, encoding='utf-8', dtype=str, sep=';')
        except:
            df_var = pd.read_csv('db/variavel.csv', low_memory=False, encoding='utf-8', dtype=str, sep=',')
        if len(df_var.columns) == 1 and ',' in df_var.columns[0]:
            df_var = pd.DataFrame([x.split(',') for x in df_var[df_var.columns[0]]])
            if len(df_var.columns) >= 3:
                df_var = df_var.iloc[:, [0, 1, 2]]
                df_var.columns = cols
            elif len(df_var.columns) >= 2:
                df_var = df_var.iloc[:, [0, 1]]
                df_var.columns = cols[:2]
        return df_var
    except Exception as e:
        return pd.DataFrame(columns=cols)


df = load_objetivos()
df_metas = load_metas()
df_indicadores = load_indicadores()
df_unidade_medida = load_unidade_medida()
df_variavel = load_variavel()

if not df.empty:
    row_objetivo_0 = df.iloc[(0,)]
    initial_header = row_objetivo_0['RES_OBJETIVO']
    # Substitui a descrição do objetivo 0 pelo texto personalizado
    initial_content = html.Div([
        html.P("Os Objetivos de Desenvolvimento Sustentável (ODS) constituem um compromisso global voltado à erradicação da pobreza, à proteção do meio ambiente e do clima, bem como à promoção da paz e da prosperidade para todas as pessoas, em todos os lugares. No Brasil, as Nações Unidas atuam de forma articulada com instituições públicas e privadas, além da sociedade civil, para viabilizar o cumprimento da Agenda 2030 e assegurar um desenvolvimento inclusivo, equitativo e sustentável."),
        html.P("Nesse contexto, o Instituto Mauro Borges desenvolveu, para os estados que compõem a Região Brasil Central — Distrito Federal, Goiás, Maranhão, Mato Grosso, Mato Grosso do Sul, Rondônia e Tocantins —, um painel de monitoramento dos ODS. A ferramenta possibilita o acompanhamento da evolução tanto dos objetivos quanto de suas respectivas metas. É importante destacar que, para alguns objetivos e metas, ainda não há disponibilidade de dados que permitam a construção dos indicadores correspondentes. Diante disso, buscou-se maximizar a utilização das bases de dados existentes, de modo a produzir o maior número possível de indicadores com qualidade e consistência.")
    ])
else:
    initial_header = "Erro ao carregar dados"
    initial_content = "Não foi possível carregar dados dos objetivos."

initial_meta_description = ""
meta_inicial = None
if not df.empty and not df_metas.empty:
    try:
        metas_filtradas_inicial = df_metas[df_metas['ID_OBJETIVO'] == df.iloc[(0,)]['ID_OBJETIVO']]
        metas_com_indicadores_inicial = [
            meta for _, meta in metas_filtradas_inicial.iterrows()
            if not df_indicadores[df_indicadores['ID_META'] == meta['ID_META']].empty
        ]
        if metas_com_indicadores_inicial:
            meta_inicial = metas_com_indicadores_inicial[0]
            initial_meta_description = meta_inicial['DESC_META']
    except Exception as e:
        pass

# Prepara os indicadores iniciais
# Cria componentes vazios para evitar erros de callback
initial_tabs_indicadores = dbc.Tabs(id='tabs-indicadores', children=[])
initial_indicadores_section = [
    html.H5("Indicadores", className="mt-4 mb-3", style={'display': 'none'}),
    dbc.Card(dbc.CardBody(initial_tabs_indicadores), className="mt-3", style={'display': 'none'}),
    # Componente vazio para lazy-load-container
    html.Div(id={'type': 'lazy-load-container', 'index': 'placeholder'}, style={'display': 'none'})
]

if meta_inicial:
    indicadores_meta_inicial = df_indicadores[df_indicadores['ID_META'] == meta_inicial['ID_META']]


# Função auxiliar para identificar colunas de filtro
def identify_filter_columns(df):
    """Identifica colunas no DataFrame que devem ser usadas como filtros dinâmicos."""
    if df is None or df.empty:
        return []
    all_cols = set(df.columns)
    non_filter_cols = {
        'CODG_ANO', 'VLR_VAR', 'CODG_UND_FED', 'CODG_UND_MED', 'CODG_VAR',
        'DESC_ANO', 'DESC_UND_FED', 'DESC_UND_MED', 'DESC_VAR',
        'ID_INDICADOR', 'ID_META', 'ID_OBJETIVO',
        'Unidade Federativa', 'Ano', 'Variável', 'Valor', 'Unidade de Medida'
    }
    non_filter_cols.update({col for col in all_cols if col.startswith('DESC_')})
    candidate_cols = all_cols - non_filter_cols
    filter_cols = [
        col for col in candidate_cols
        if col in constants.COLUMN_NAMES and df[col].nunique(dropna=True) > 1
    ]
    return sorted(filter_cols)


# Função auxiliar para formatar número no padrão brasileiro (pt-BR)
def format_br(value):
    """Formats a number to Brazilian standard (dot for thousands, comma for decimal).
       Shows integer if no significant decimal part, otherwise shows up to 2 decimals,
       removing trailing zeros.
    """
    if pd.isna(value) or value is None:
        return ""
    try:
        f_value = float(value)
        # Check if it's effectively an integer
        if f_value == int(f_value):
            # Format as integer with thousands separators
            int_str = f"{int(f_value):,}".replace(",", ".")
            return int_str
        else:
            # Format as float with 2 decimal places first for consistent rounding
            formatted_str = f"{f_value:.2f}"  # e.g., "1459.89", "15.00", "4.90"
            int_part, dec_part = formatted_str.split('.')

            # Format integer part with dots
            int_part_formatted = f"{int(int_part):,}".replace(",", ".")

            # Only add decimal part if it's not "00"
            if dec_part == "00":
                return int_part_formatted
            else:
                # Remove trailing zeros from decimal part *before* combining
                dec_part = dec_part.rstrip('0')  # "89" -> "89", "90" -> "9"
                # Handle cases like "4.0" which become "4," -> should be "4"
                if not dec_part:  # If rstrip removed everything (e.g., was "00")
                    return int_part_formatted  # Return only integer part
                return f"{int_part_formatted},{dec_part}"

    except (ValueError, TypeError):
        logging.warning(f"Could not format value '{value}' to Brazilian standard.")
        return str(value)  # Fallback


def create_visualization(df, indicador_id=None, selected_var=None, selected_filters=None):
    """Cria uma visualização (gráfico principal, ranking, mapa e tabela) com os dados do DataFrame, aplicando filtros."""
    if df is None or df.empty:
        return dbc.Alert("Nenhum dado disponível para este indicador.", color="warning", className="textCenter p-3")

    try:
        # Log para debug dos filtros recebidos
        logging.debug(f"create_visualization para {indicador_id} - Var: {selected_var}, Filtros: {selected_filters}")

        colunas_necessarias = ['CODG_ANO', 'VLR_VAR']
        if not all(col in df.columns for col in colunas_necessarias):
            missing = [col for col in colunas_necessarias if col not in df.columns]
            return dbc.Alert(f"Dados incompletos. Colunas faltando: {', '.join(missing)}", color="warning",
                             className="textCenter p-3")

        df_filtered = df.copy()

        # Aplica filtro de VARIÁVEL PRINCIPAL
        if 'CODG_VAR' in df_filtered.columns and selected_var:
            df_filtered['CODG_VAR'] = df_filtered['CODG_VAR'].astype(str).str.strip()
            selected_var_str = str(selected_var).strip()
            df_filtered = df_filtered[df_filtered['CODG_VAR'] == selected_var_str]
            logging.debug(f"Aplicando filtro de variável {selected_var_str} - Registros restantes: {len(df_filtered)}")
            if df_filtered.empty:
                var_name = selected_var_str
                df_var_desc = load_variavel()
                if not df_var_desc.empty:
                    var_info = df_var_desc[df_var_desc['CODG_VAR'] == selected_var_str]
                    if not var_info.empty:
                        var_name = var_info['DESC_VAR'].iloc[0]
                return dbc.Alert(f"Nenhum dado encontrado para a variável '{var_name}'.", color="warning")

        # Aplica FILTROS DINÂMICOS
        if selected_filters:
            for col_code, selected_value in selected_filters.items():
                if selected_value is not None and col_code in df_filtered.columns:
                    # Convert selected value to string for comparison
                    selected_value_str = str(selected_value).strip()
                    # Compare using string representation of the column, without changing its type
                    # Convert column to string *before* filling NA and stripping
                    logging.debug(f"Aplicando filtro {col_code}={selected_value_str}")
                    df_filtered = df_filtered[
                        df_filtered[col_code].astype(str).fillna('').str.strip() == selected_value_str
                        ]
                    logging.debug(f"Após filtro {col_code} - Registros restantes: {len(df_filtered)}")
                    if df_filtered.empty:
                        filter_name = constants.COLUMN_NAMES.get(col_code, col_code)
                        return dbc.Alert(
                            f"Nenhum dado encontrado para o filtro '{filter_name}' = '{selected_value_str}'.",
                            color="warning")

        if df_filtered.empty:
            return dbc.Alert("Nenhum dado encontrado após aplicar os filtros.", color="warning")

        # Adicionado: Verifica se todos os valores restantes são zero APÓS fillna(0)
        if not df_filtered.empty and (df_filtered['VLR_VAR'] == 0).all():
            # Constrói a mensagem explicando os filtros
            filter_desc = []
            if selected_var:
                df_var_desc = load_variavel()
                var_info = df_var_desc[df_var_desc['CODG_VAR'] == str(selected_var)]
                var_name = var_info['DESC_VAR'].iloc[0] if not var_info.empty else f"Variável Cód: {selected_var}"
                filter_desc.append(f"Variável: '{var_name}'")
            if selected_filters:
                for col_code, value in selected_filters.items():
                    col_name = constants.COLUMN_NAMES.get(col_code, col_code)
                    desc_col_code = 'DESC_' + col_code[5:]
                    value_desc = str(value)
                    if desc_col_code in df.columns:
                        try:
                            desc_map = df[[col_code, desc_col_code]].drop_duplicates()
                            desc_map[col_code] = desc_map[col_code].astype(str)
                            matched_desc = desc_map[desc_map[col_code] == str(value)]
                            if not matched_desc.empty:
                                value_desc = matched_desc[desc_col_code].iloc[0]
                        except Exception:
                            pass
                    filter_desc.append(f"{col_name}: '{value_desc}'")
            message = (
                    "A combinação selecionada " +
                    ("(" + ", ".join(filter_desc) + ") " if filter_desc else "") +
                    "resultou apenas em valores iguais a zero. "
                    "Não há dados a serem exibidos. Por favor, tente outra combinação de filtros."
            )
            return dbc.Alert(message, color="info", className="textCenter p-3")

        df_original_for_table = df_filtered.copy()

        # --- Adiciona/Garante Colunas de Descrição ---
        # Descrição UF
        if 'CODG_UND_FED' in df_filtered.columns:
            df_filtered['DESC_UND_FED'] = df_filtered['CODG_UND_FED'].astype(str).map(constants.UF_NAMES)
            df_filtered = df_filtered.dropna(subset=['DESC_UND_FED'])  # Garante que temos UFs válidas
        elif 'DESC_UND_FED' not in df_filtered.columns:
            df_filtered['DESC_UND_FED'] = 'N/D'

        df_variavel_loaded = load_variavel()
        if 'CODG_VAR' in df_filtered.columns and not df_variavel_loaded.empty:
            df_filtered['CODG_VAR'] = df_filtered['CODG_VAR'].astype(str)
            df_variavel_loaded['CODG_VAR'] = df_variavel_loaded['CODG_VAR'].astype(str)

            # Merge para obter as descrições das variáveis
            df_filtered = df_filtered.merge(df_variavel_loaded[['CODG_VAR', 'DESC_VAR']], on='CODG_VAR', how='left')
            df_filtered['DESC_VAR'] = df_filtered['DESC_VAR'].fillna('N/D')
        elif 'DESC_VAR' not in df_filtered.columns:
            df_filtered['DESC_VAR'] = 'N/D'

        # Descrição Unidade de Medida
        df_unidade_medida_loaded = load_unidade_medida()
        if 'CODG_UND_MED' in df_filtered.columns and not df_unidade_medida_loaded.empty:
            df_filtered['CODG_UND_MED'] = df_filtered['CODG_UND_MED'].astype(str)
            df_unidade_medida_loaded['CODG_UND_MED'] = df_unidade_medida_loaded['CODG_UND_MED'].astype(str)
            df_filtered = df_filtered.merge(df_unidade_medida_loaded[['CODG_UND_MED', 'DESC_UND_MED']],
                                            on='CODG_UND_MED', how='left')
            df_filtered['DESC_UND_MED'] = df_filtered['DESC_UND_MED'].fillna('N/D')
        elif 'DESC_UND_MED' not in df_filtered.columns:
            df_filtered['DESC_UND_MED'] = 'N/D'

        # Descrições para Filtros Dinâmicos
        dynamic_filter_cols = identify_filter_columns(df)  # Identifica filtros no DF ORIGINAL
        processed_desc_cols = set()
        for filter_col_code in dynamic_filter_cols:
            desc_col_code = 'DESC_' + filter_col_code[5:]
            if desc_col_code in processed_desc_cols:
                continue
            if desc_col_code not in df_filtered.columns:
                if desc_col_code in df.columns and filter_col_code in df_filtered.columns and filter_col_code in df.columns:
                    try:
                        merge_data = df[[filter_col_code, desc_col_code]].drop_duplicates().copy()
                        # Garante que ambas as colunas de chave são string para o merge
                        df_filtered[filter_col_code] = df_filtered[filter_col_code].astype(str)
                        merge_data[filter_col_code] = merge_data[filter_col_code].astype(str)
                        df_filtered = pd.merge(df_filtered, merge_data, on=filter_col_code, how='left')
                        df_filtered[desc_col_code] = df_filtered[desc_col_code].fillna('N/D')
                    except Exception as merge_err:
                        df_filtered[desc_col_code] = 'N/D'
                else:
                    df_filtered[desc_col_code] = 'N/D'
            else:
                df_filtered[desc_col_code] = df_filtered[desc_col_code].fillna('N/D')
            processed_desc_cols.add(desc_col_code)

        # Garante que df_original_for_table também tenha as descrições dinâmicas
        for desc_col in processed_desc_cols:
            if desc_col not in df_original_for_table.columns and desc_col in df_filtered.columns:
                # Pega o código correspondente
                filter_col_code = 'CODG' + desc_col[4:]
                if filter_col_code in df_original_for_table.columns:
                    try:
                        merge_data_orig = df_filtered[[filter_col_code, desc_col]].drop_duplicates().copy()
                        df_original_for_table[filter_col_code] = df_original_for_table[filter_col_code].astype(str)
                        merge_data_orig[filter_col_code] = merge_data_orig[filter_col_code].astype(str)
                        df_original_for_table = pd.merge(df_original_for_table, merge_data_orig, on=filter_col_code,
                                                         how='left')
                        df_original_for_table[desc_col] = df_original_for_table[desc_col].fillna('N/D')
                    except Exception:
                        df_original_for_table[desc_col] = 'N/D'

        # Adiciona colunas faltantes no df_original_for_table se necessário (UF, Var, UndMed)
        if 'DESC_UND_FED' not in df_original_for_table.columns and 'DESC_UND_FED' in df_filtered.columns:
            df_original_for_table['DESC_UND_FED'] = df_original_for_table['CODG_UND_FED'].astype(str).map(
                constants.UF_NAMES).fillna('N/D')
        if 'DESC_VAR' not in df_original_for_table.columns and 'DESC_VAR' in df_filtered.columns:
            if 'CODG_VAR' in df_original_for_table.columns and not df_variavel_loaded.empty:
                df_original_for_table['CODG_VAR'] = df_original_for_table['CODG_VAR'].astype(str)
                df_variavel_loaded['CODG_VAR'] = df_variavel_loaded['CODG_VAR'].astype(str)
                df_original_for_table = pd.merge(df_original_for_table, df_variavel_loaded[['CODG_VAR', 'DESC_VAR']],
                                                 on='CODG_VAR', how='left')
                df_original_for_table['DESC_VAR'] = df_original_for_table['DESC_VAR'].fillna('N/D')
            else:
                df_original_for_table['DESC_VAR'] = 'N/D'
        if 'DESC_UND_MED' not in df_original_for_table.columns and 'DESC_UND_MED' in df_filtered.columns:
            if 'CODG_UND_MED' in df_original_for_table.columns and not df_unidade_medida_loaded.empty:
                df_original_for_table['CODG_UND_MED'] = df_original_for_table['CODG_UND_MED'].astype(str)
                df_unidade_medida_loaded['CODG_UND_MED'] = df_unidade_medida_loaded['CODG_UND_MED'].astype(str)
                df_original_for_table = pd.merge(df_original_for_table,
                                                 df_unidade_medida_loaded[['CODG_UND_MED', 'DESC_UND_MED']],
                                                 on='CODG_UND_MED', how='left')
                df_original_for_table['DESC_UND_MED'] = df_original_for_table['DESC_UND_MED'].fillna('N/D')
            else:
                df_original_for_table['DESC_UND_MED'] = 'N/D'

        # Ordena e limpa dados numéricos
        df_filtered['CODG_ANO'] = df_filtered['CODG_ANO'].astype(str)
        df_filtered = df_filtered.sort_values('CODG_ANO')
        df_filtered['VLR_VAR'] = pd.to_numeric(df_filtered['VLR_VAR'], errors='coerce')
        df_filtered['VLR_VAR'] = df_filtered['VLR_VAR'].fillna(0)  # Preenche NA com 0 *antes* de verificar

        # Verifica novamente se está vazio após fillna
        if df_filtered.empty:
            return dbc.Alert("Não há dados disponíveis para a combinação de filtros selecionada.", color="warning",
                             className="textCenter p-3")

        # --- Definição Dinâmica das Colunas da Tabela AG Grid ---
        base_col_defs = [
            {"field": 'ID_INDICADOR', "headerName": 'ID Indicador', "hide": True},
            {"field": 'DESC_UND_FED', "headerName": 'Unidade Federativa'},
            {"field": 'CODG_UND_FED', "headerName": 'Código UF', "hide": True},
            {"field": 'CODG_ANO', "headerName": 'Ano'},
            {"field": 'DESC_VAR', "headerName": 'Variável'},
            {"field": 'CODG_VAR', "headerName": 'Código Variável', "hide": True},
        ]
        dynamic_desc_col_defs = []
        dynamic_desc_col_names = set()
        # Usa df_original_for_table para determinar colunas da tabela
        present_columns_in_table = df_original_for_table.columns
        for filter_col_code in dynamic_filter_cols:
            desc_col_code = 'DESC_' + filter_col_code[5:]
            if desc_col_code in present_columns_in_table:  # Verifica no DF da tabela
                readable_name = constants.COLUMN_NAMES.get(desc_col_code,
                                                           desc_col_code.replace('DESC_', '').replace('_', ' ').title())
                if desc_col_code not in dynamic_desc_col_names:  # Evita duplicados
                    dynamic_desc_col_defs.append({"field": desc_col_code, "headerName": readable_name})
                    dynamic_desc_col_names.add(desc_col_code)
                    # Adiciona também o código correspondente como coluna oculta, se existir
                    if filter_col_code in present_columns_in_table:
                        dynamic_desc_col_defs.append({"field": filter_col_code, "headerName": f"Código {readable_name}", "hide": True})
        
        final_col_defs = base_col_defs + dynamic_desc_col_defs + [
            {"field": 'VLR_VAR', "headerName": 'Valor'},
            {"field": 'DESC_UND_MED', "headerName": 'Unidade de Medida'},
            {"field": 'CODG_UND_MED', "headerName": 'Código Unidade Medida', "hide": True}
        ]
        columnDefs = []
        for col_def in final_col_defs:
            field_name = col_def['field']
            if field_name in present_columns_in_table:  # Verifica novamente no DF da tabela
                base_props = {"sortable": True, "filter": True, "minWidth": 100, "resizable": True, "wrapText": True,
                              "autoHeight": True, "cellStyle": {"whiteSpace": "normal"}}
                
                # Se a coluna deve ser oculta, adiciona essa propriedade
                if col_def.get('hide', False):
                    base_props["hide"] = True
                
                # Ajuste de flex baseado na coluna
                if field_name == 'DESC_VAR':
                    flex_value = 3
                elif field_name == 'DESC_UND_FED' or field_name == 'DESC_UND_MED':
                    flex_value = 2
                elif field_name in dynamic_desc_col_names:
                    flex_value = 2  # Aumenta um pouco para descrições dinâmicas
                elif field_name == 'CODG_ANO' or field_name == 'VLR_VAR':
                    flex_value = 1
                elif field_name.startswith('CODG_'):  # Colunas de código têm menos flex
                    flex_value = 1
                else:
                    flex_value = 1
                columnDefs.append(
                    {**base_props, "field": field_name, "headerName": col_def['headerName'], "flex": flex_value})
        defaultColDef = {
            "minWidth": 100, "resizable": True, "wrapText": True, "autoHeight": True,
            "cellStyle": {"whiteSpace": "normal", 'textAlign': 'left'}
        }

        # Adiciona o ID_INDICADOR ao DataFrame para exportação
        if indicador_id and 'ID_INDICADOR' not in df_original_for_table.columns:
            df_original_for_table['ID_INDICADOR'] = indicador_id

        # --- Criação das Figuras dos Gráficos ---
        main_fig = go.Figure()  # Inicializa a figura principal
        fig_map = go.Figure()  # Inicializa a figura do mapa

        # Obter anos únicos e ano padrão
        anos_unicos = sorted(df_filtered['CODG_ANO'].unique())
        ano_default = anos_unicos[-1] if anos_unicos else None

        # Lê as flags do indicador e RANKING_ORDEM
        grafico_linha_flag = 1  # Padrão
        ranking_ordem = 0  # Padrão (0 = maior para menor, 1 = menor para maior)
        if indicador_id and not df_indicadores.empty:
            indicador_info = df_indicadores[df_indicadores['ID_INDICADOR'] == indicador_id]
            if not indicador_info.empty:
                if 'GRAFICO_LINHA' in indicador_info.columns:
                    try:
                        grafico_linha_val = pd.to_numeric(indicador_info['GRAFICO_LINHA'].iloc[0], errors='coerce')
                        if not pd.isna(grafico_linha_val): grafico_linha_flag = int(grafico_linha_val)
                    except (ValueError, TypeError):
                        pass  # Mantém padrão
                # --- Adicionado: Lê RANKING_ORDEM ---
                if 'RANKING_ORDEM' in indicador_info.columns:
                    try:
                        ranking_ordem_val = pd.to_numeric(indicador_info['RANKING_ORDEM'].iloc[0], errors='coerce')
                        if not pd.isna(ranking_ordem_val): ranking_ordem = int(ranking_ordem_val)
                    except (ValueError, TypeError):
                        pass  # Mantém padrão 0

        # --- Criação do Gráfico Principal baseado na lógica existente ---
        if grafico_linha_flag == 1:
            # --- Lógica do Gráfico de Linha (Refatorado com go.Figure) ---
            main_fig = go.Figure()  # Reinicializa para garantir que está vazia
            if 'DESC_UND_FED' in df_filtered.columns:
                df_line_data = df_filtered.sort_values(['DESC_UND_FED', 'CODG_ANO'])
                if not df_line_data.empty:
                    for uf in df_line_data['DESC_UND_FED'].unique():
                        df_state = df_line_data[df_line_data['DESC_UND_FED'] == uf]
                        # Adiciona verificação se df_state não está vazio
                        if df_state.empty: continue
                        # Modificado: Adiciona valor formatado ao customdata
                        customdata_state = np.column_stack((
                            np.full(len(df_state), uf),
                            df_state['DESC_UND_MED'].values,
                            df_state['VLR_VAR'].values,  # Original value
                            df_state['VLR_VAR'].apply(format_br).values  # Formatted value
                        ))
                        # Modificado: Usa a função format_br
                        text_values = df_state['VLR_VAR'].apply(format_br)
                        trace_name = f"<b>{uf}</b>" if uf == 'Goiás' else uf
                        color_map = {
                            'Goiás': '#229846', 'Maranhão': '#D2B48C', 'Distrito Federal': '#636efa',
                            'Mato Grosso': '#ab63fa', 'Mato Grosso do Sul': '#ffa15a', 'Rondônia': '#19d3f3',
                            'Tocantins': '#ff6692', 'Brasil': '#FF0000'  # Adiciona Brasil
                        }
                        line_color = color_map.get(uf)
                        line_width = 6 if uf == 'Goiás' else 2

                        main_fig.add_trace(go.Scatter(
                            x=df_state['CODG_ANO'], y=df_state['VLR_VAR'], name=trace_name,
                            customdata=customdata_state, text=text_values, mode='lines+markers+text',
                            texttemplate='%{text}', textposition='top center', textfont=dict(size=10),
                            marker=dict(size=10, symbol='circle', line=dict(width=1, color='white')),
                            line=dict(width=line_width, color=line_color),
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>"  # UF do customdata
                                "Ano: %{x}<br>"
                                "Valor: %{customdata[3]}<br>"  # Modificado: Usa customdata[3] (pré-formatado)
                                "Unidade: %{customdata[1]}<extra></extra>"
                            )
                        ))

                    max_y_line = df_line_data['VLR_VAR'].max()
                    y_range_line = [0, max_y_line * 1.15]

                    layout_updates_line = DEFAULT_LAYOUT.copy()
                    layout_updates_line.update({
                        'xaxis': dict(showgrid=True, zeroline=False, tickfont=dict(size=12, color='black'),
                                      tickangle=45),
                        'yaxis': dict(showgrid=True, zeroline=False, tickfont=dict(size=12, color='black'), title=None,
                                      type='linear', tickformat='d', range=y_range_line)
                    })
                    unique_years_line = sorted(df_line_data['CODG_ANO'].unique())
                    layout_updates_line['xaxis']['ticktext'] = [f"<b>{x}</b>" for x in unique_years_line]
                    layout_updates_line['xaxis']['tickvals'] = unique_years_line
                    main_fig.update_layout(layout_updates_line)
                else:
                    main_fig = go.Figure().update_layout(title='Dados insuficientes para o gráfico de linha.',
                                                         xaxis={'visible': False}, yaxis={'visible': False})

            else:  # Gráfico de linha sem UF (e.g., só 'Brasil')
                df_line_data = df_filtered.sort_values('CODG_ANO')
                if not df_line_data.empty:
                    # Adapta customdata para não ter UF, adiciona valor formatado
                    customdata_line_no_uf = np.column_stack((
                        df_line_data['DESC_UND_MED'].values,
                        df_line_data['VLR_VAR'].values,  # Original value
                        df_line_data['VLR_VAR'].apply(format_br).values  # Formatted value
                    ))
                    # Modificado: Usa a função format_br
                    text_values = df_line_data['VLR_VAR'].apply(format_br)
                    main_fig.add_trace(go.Scatter(
                        x=df_line_data['CODG_ANO'], y=df_line_data['VLR_VAR'], name='Valor',
                        customdata=customdata_line_no_uf, text=text_values, mode='lines+markers+text',
                        line=dict(color='#229846', width=3),  # Cor padrão ou específica
                        hovertemplate=(
                            "Ano: %{x}<br>"
                            "Valor: %{customdata[2]}<br>"  # Modificado: Usa customdata[2] (pré-formatado)
                            "Unidade: %{customdata[0]}<extra></extra>"
                        )
                    ))

                    max_y_line_no_uf = df_line_data['VLR_VAR'].max()
                    y_range_line_no_uf = [0, max_y_line_no_uf * 1.15]

                    layout_updates_line_no_uf = DEFAULT_LAYOUT.copy()
                    layout_updates_line_no_uf.update({
                        'showlegend': False,
                        'xaxis': dict(showgrid=True, zeroline=False, tickfont=dict(size=12, color='black'),
                                      tickangle=45),
                        'yaxis': dict(showgrid=True, zeroline=False, tickfont=dict(size=12, color='black'), title=None,
                                      type='linear', tickformat='d', range=y_range_line_no_uf)
                    })
                    unique_years_line_no_uf = sorted(df_line_data['CODG_ANO'].unique())
                    layout_updates_line_no_uf['xaxis']['ticktext'] = [f"<b>{x}</b>" for x in unique_years_line_no_uf]
                    layout_updates_line_no_uf['xaxis']['tickvals'] = unique_years_line_no_uf
                    main_fig.update_layout(layout_updates_line_no_uf)
                else:
                    main_fig = go.Figure().update_layout(title='Dados insuficientes para o gráfico de linha.',
                                                         xaxis={'visible': False}, yaxis={'visible': False})

        else:  # grafico_linha_flag == 0
            # --- Lógica do Gráfico de Barras AGRUPADO POR ANO (Refatorado com go.Figure) ---
            main_fig = go.Figure()
            if 'DESC_UND_FED' in df_filtered.columns and 'CODG_ANO' in df_filtered.columns:
                df_bar_grouped_data = df_filtered.sort_values(['CODG_ANO', 'DESC_UND_FED'])
                if not df_bar_grouped_data.empty:
                    color_map = {
                        'Goiás': '#229846', 'Maranhão': '#D2B48C', 'Distrito Federal': '#636efa',
                        'Mato Grosso': '#ab63fa', 'Mato Grosso do Sul': '#ffa15a', 'Rondônia': '#19d3f3',
                        'Tocantins': '#ff6692', 'Brasil': '#FF0000'  # Adiciona Brasil
                    }
                    for uf in df_bar_grouped_data['DESC_UND_FED'].unique():
                        df_state = df_bar_grouped_data[df_bar_grouped_data['DESC_UND_FED'] == uf]
                        if df_state.empty: continue  # Pula UF sem dados
                        # Modificado: Adiciona valor formatado ao customdata
                        customdata_state = np.column_stack((
                            np.full(len(df_state), uf),
                            df_state['DESC_UND_MED'].values,
                            df_state['VLR_VAR'].values,  # Original value
                            df_state['VLR_VAR'].apply(format_br).values  # Formatted value
                        ))
                        # Modificado: Usa a função format_br
                        text_values = df_state['VLR_VAR'].apply(format_br)
                        trace_name = f"<b>{uf}</b>" if uf == 'Goiás' else uf
                        bar_color = color_map.get(uf)

                        main_fig.add_trace(go.Bar(
                            x=df_state['CODG_ANO'], y=df_state['VLR_VAR'], name=trace_name,
                            customdata=customdata_state, text=text_values, texttemplate='%{text}',
                            textposition='outside', marker_color=bar_color, marker_line_width=1.5,
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>"  # UF do customdata
                                "Ano: %{x}<br>"
                                "Valor: %{customdata[3]}<br>"  # Modificado: Usa customdata[3] (pré-formatado)
                                "Unidade: %{customdata[1]}<extra></extra>"
                            )
                        ))

                    max_y_grouped = df_bar_grouped_data['VLR_VAR'].max()
                    y_range_grouped = [0, max_y_grouped * 1.15]

                    layout_updates_bar_grouped = DEFAULT_LAYOUT.copy()
                    layout_updates_bar_grouped.update({
                        'barmode': 'group',
                        'xaxis': dict(showgrid=True, tickfont=dict(size=12, color='black'), tickangle=45, title=None),
                        'yaxis': dict(showgrid=True, tickfont=dict(size=12, color='black'), title=None, type='linear',
                                      tickformat='d', range=y_range_grouped)
                    })
                    unique_years_bar = sorted(df_bar_grouped_data['CODG_ANO'].unique())
                    layout_updates_bar_grouped['xaxis']['ticktext'] = [f"<b>{x}</b>" for x in unique_years_bar]
                    layout_updates_bar_grouped['xaxis']['tickvals'] = unique_years_bar
                    main_fig.update_layout(layout_updates_bar_grouped)
                else:
                    main_fig = go.Figure().update_layout(title='Dados insuficientes para o gráfico de barras agrupado.',
                                                         xaxis={'visible': False}, yaxis={'visible': False})
            else:
                # Caso sem UF mas com série temporal -> Barras simples por ano
                df_bar_no_uf = df_filtered.sort_values('CODG_ANO')
                if not df_bar_no_uf.empty:
                    # Modificado: Adiciona valor formatado ao customdata
                    customdata_bar_no_uf = np.column_stack((
                        df_bar_no_uf['DESC_UND_MED'].values,
                        df_bar_no_uf['VLR_VAR'].values,  # Original value
                        df_bar_no_uf['VLR_VAR'].apply(format_br).values  # Formatted value
                    ))
                    # Modificado: Usa a função format_br
                    text_values = df_bar_no_uf['VLR_VAR'].apply(format_br)
                    main_fig.add_trace(go.Bar(
                        x=df_bar_no_uf['CODG_ANO'],
                        y=df_bar_no_uf['VLR_VAR'],
                        marker_color='#229846',  # Cor padrão ou específica
                        hovertemplate=(
                            "Ano: %{x}<br>"
                            "Valor: %{customdata[2]}<br>"  # Modificado: Usa customdata[2] (pré-formatado)
                            "Unidade: %{customdata[0]}<extra></extra>"
                        )
                    ))

                    max_y_bar_no_uf = df_bar_no_uf['VLR_VAR'].max()
                    y_range_bar_no_uf = [0, max_y_bar_no_uf * 1.15]
                    layout_updates_bar_no_uf = DEFAULT_LAYOUT.copy()
                    layout_updates_bar_no_uf.update({
                        'showlegend': False,
                        'xaxis': dict(showgrid=True, tickfont=dict(size=12, color='black'), tickangle=45, title=None),
                        'yaxis': dict(showgrid=True, tickfont=dict(size=12, color='black'), title=None, type='linear',
                                      tickformat='d', range=y_range_bar_no_uf)
                    })
                    unique_years_bar_no_uf = sorted(df_bar_no_uf['CODG_ANO'].unique())
                    layout_updates_bar_no_uf['xaxis']['ticktext'] = [f"<b>{x}</b>" for x in unique_years_bar_no_uf]
                    layout_updates_bar_no_uf['xaxis']['tickvals'] = unique_years_bar_no_uf
                    main_fig.update_layout(layout_updates_bar_no_uf)
                else:
                    main_fig = go.Figure().update_layout(title='Dados insuficientes para o gráfico de barras.',
                                                         xaxis={'visible': False}, yaxis={'visible': False})

        # --- Criação do Gráfico de Ranking (se houver UF e ano) ---
        ranking_content = dbc.Alert("Ranking não disponível (requer dados por Unidade Federativa).", color="info",
                                    className="textCenter p-3")
        if 'DESC_UND_FED' in df_filtered.columns and ano_default:
            # Filtra dados para o ano padrão (será atualizado pelo dropdown)
            df_ranking_data_initial = df_filtered[df_filtered['CODG_ANO'] == ano_default].copy()

            # Verifica se há dados para o ano padrão antes de prosseguir
            if not df_ranking_data_initial.empty:
                # Verifica unicidade por UF para o ano padrão
                counts_per_uf_ranking = df_ranking_data_initial['DESC_UND_FED'].value_counts()
                if (counts_per_uf_ranking > 1).any():
                    ranking_content = dbc.Alert(
                        "Ranking não pode ser gerado: múltiplos valores por UF para o ano selecionado. "
                        "Aplique filtros adicionais se disponíveis.", color="warning", className="textCenter p-3"
                    )
                else:
                    # Ordena baseado em VLR_VAR e RANKING_ORDEM
                    ascending_rank = (ranking_ordem == 0)  # True se for maior para menor
                    df_ranking_data_initial = df_ranking_data_initial.sort_values('VLR_VAR', ascending=ascending_rank)

                    # Define cores e opacidade
                    goias_color = 'rgba(34, 152, 70, 1)'  # '#229846' opaco
                    other_color = 'rgba(34, 152, 70, 0.2)'  # Define como 0.2 para consistência

                    # Cria o gráfico de ranking com go.Figure e go.Bar
                    fig_ranking_updated = go.Figure()
                    for _, row in df_ranking_data_initial.iterrows():
                        uf = row['DESC_UND_FED']
                        valor = row['VLR_VAR']
                        und_med = row.get('DESC_UND_MED', 'N/D')  # Usa .get() para segurança
                        bar_color = goias_color if uf == 'Goiás' else other_color
                        # Modificado: Usa a função format_br
                        text_value = format_br(valor)

                        fig_ranking_updated.add_trace(go.Bar(
                            y=[uf],  # Estados no eixo Y
                            x=[valor],  # Valores no eixo X
                            name=uf,
                            orientation='h',  # Barras horizontais
                            marker_color=bar_color,
                            text=text_value,
                            textposition='outside',  # Texto fora da barra
                            hovertemplate=(
                                f"<b>{uf}</b><br>"
                                f"Valor: {text_value}<br>"  # Usa o texto formatado
                                f"Unidade: {und_med}<extra></extra>"
                            )
                        ))

                    max_x_ranking = df_ranking_data_initial['VLR_VAR'].max() if not df_ranking_data_initial.empty else 0
                    x_range_ranking = [0, max_x_ranking * 1.15]

                    # Atualiza layout para gráfico de barras horizontal
                    fig_ranking_updated.update_layout(
                        xaxis_title=None, yaxis_title=None,
                        yaxis=dict(showgrid=False, tickfont=dict(size=12, color='black'), categoryorder='array',
                                   categoryarray=df_ranking_data_initial['DESC_UND_FED'].tolist()),
                        xaxis=dict(showgrid=True, zeroline=False, tickfont=dict(size=12, color='black'),
                                   range=x_range_ranking, tickformat='d'),
                        showlegend=False, margin=dict(l=150, r=20, t=30, b=30), bargap=0.1
                    )

                    # Define o conteúdo do ranking como o dropdown e o gráfico
                    ranking_content = html.Div([
                        html.Label("Ano:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                        dcc.Dropdown(
                            id={'type': 'year-dropdown-ranking', 'index': indicador_id},
                            options=[{'label': ano, 'value': ano} for ano in anos_unicos],
                            value=ano_default,
                            clearable=False,
                            style={'width': '100%', 'marginBottom': '10px'}
                        ),
                        dcc.Graph(id={'type': 'ranking-chart', 'index': indicador_id}, figure=fig_ranking_updated)
                    ])
            else:
                ranking_content = dbc.Alert(f"Ranking não disponível para o ano {ano_default}.", color="info",
                                            className="textCenter p-3")
                # Mesmo sem dados, cria o dropdown para permitir seleção de outro ano
                ranking_content = html.Div([
                    html.Label("Ano:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Dropdown(
                        id={'type': 'year-dropdown-ranking', 'index': indicador_id},
                        options=[{'label': ano, 'value': ano} for ano in anos_unicos],
                        value=ano_default,
                        clearable=False,
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    dbc.Alert(f"Ranking não disponível para o ano {ano_default}.", color="info",
                              className="textCenter p-3")
                ])

        # --- Criação do Mapa (se houver UF e ano) ---
        map_content = dbc.Alert("Mapa não disponível (requer dados por Unidade Federativa).", color="info",
                                className="textCenter p-3")
        map_center = {'lat': -12.95984198, 'lon': -53.27299730}  # Inicializa fora do try

        if 'DESC_UND_FED' in df_filtered.columns and ano_default:
            # Filtra dados para o ano padrão (será atualizado pelo dropdown)
            df_map_data_initial = df_filtered[df_filtered['CODG_ANO'] == ano_default].copy()

            # Verifica se há dados e unicidade por UF para o ano padrão
            if not df_map_data_initial.empty:
                counts_per_uf_map = df_map_data_initial['DESC_UND_FED'].value_counts()
                if (counts_per_uf_map > 1).any():
                    map_content = dbc.Alert(
                        "Mapa não pode ser gerado: múltiplos valores por UF para o ano selecionado. "
                        "Aplique filtros adicionais se disponíveis.", color="warning", className="textCenter p-3"
                    )
                else:
                    try:
                        with open('db/br_geojson.json', 'r', encoding='utf-8') as f:
                            geojson = json.load(f)
                        # Tenta obter unidade de medida de forma segura
                        und_med_map = df_map_data_initial['DESC_UND_MED'].dropna().iloc[0] if not df_map_data_initial[
                            'DESC_UND_MED'].dropna().empty else ''
                        # Modificado: Adiciona coluna formatada para hover
                        df_map_data_initial['VLR_VAR_FORMATADO'] = df_map_data_initial['VLR_VAR'].apply(format_br)

                        fig_map = px.choropleth(
                            df_map_data_initial,
                            geojson=geojson,
                            locations='DESC_UND_FED',
                            featureidkey='properties.name',
                            color='VLR_VAR',
                            color_continuous_scale=[  # Escala baseada no Ranking
                                [0.0, 'rgba(34, 152, 70, 0.2)'],
                                [1.0, 'rgba(34, 152, 70, 1)']
                            ],
                        )

                        # --- Atualizar Geos com Centroide FIXO 
                        map_center = {'lat': -12.95984198, 'lon': -53.27299730}
                        geos_update = dict(
                            visible=False, showcoastlines=True, coastlinecolor="White",
                            showland=True, landcolor="white", showframe=False,
                            projection=dict(type='mercator', scale=15),
                            center=map_center
                        )

                        # Aplica a atualização geo
                        fig_map.update_geos(**geos_update)
                        # ----------------------------------

                        fig_map.update_traces(
                            marker_line_color='white', marker_line_width=1,
                            customdata=df_map_data_initial[['VLR_VAR_FORMATADO']],
                            hovertemplate="<b>%{location}</b><br>Valor: %{customdata[0]}" + (
                                f" {und_med_map}" if und_med_map else "") + "<extra></extra>"
                        )

                        # --- Remove o título da barra de cores ---
                        fig_map.update_layout(coloraxis_colorbar_title_text='')
                        # -----------------------------------------

                        # Define o conteúdo do mapa como o dropdown e o gráfico
                        map_content = html.Div([
                            html.Label("Ano:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                            dcc.Dropdown(
                                id={'type': 'year-dropdown-map', 'index': indicador_id},
                                options=[{'label': ano, 'value': ano} for ano in anos_unicos],
                                value=ano_default,
                                clearable=False,
                                style={'width': '100%', 'marginBottom': '10px'}
                            ),
                            dcc.Graph(id={'type': 'choropleth-map', 'index': indicador_id}, figure=fig_map)
                        ])
                    except Exception as map_err:
                        print(f"Erro ao gerar mapa inicial: {map_err}")
                        map_content = dbc.Alert("Erro ao gerar o mapa.", color="danger", className="textCenter p-3")
            else:
                map_content = dbc.Alert(f"Mapa não disponível para o ano {ano_default}.", color="info",
                                        className="textCenter p-3")
                # Mesmo sem dados, cria o dropdown para permitir seleção de outro ano
                map_content = html.Div([
                    html.Label("Ano:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Dropdown(
                        id={'type': 'year-dropdown-map', 'index': indicador_id},
                        options=[{'label': ano, 'value': ano} for ano in anos_unicos],
                        value=ano_default,
                        clearable=False,
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    dbc.Alert(f"Mapa não disponível para o ano {ano_default}.", color="info",
                              className="textCenter p-3")
                ])

        # --- Monta o Layout da Visualização com Abas ---
        graph_layout = []

        # Conteúdo do gráfico principal (sempre exibido)
        main_chart_content = dcc.Graph(id={'type': 'main-chart', 'index': indicador_id}, figure=main_fig)

        # Cria as abas para Ranking e Mapa
        tabs_content = []
        # Adiciona aba de Ranking se o conteúdo foi gerado (não é apenas a mensagem de erro inicial)
        # Ou seja, se 'DESC_UND_FED' existe
        if 'DESC_UND_FED' in df_filtered.columns:
            tabs_content.append(dbc.Tab(ranking_content, label="Ranking", tab_id=f'tab-ranking-{indicador_id}',
                                        id={'type': 'tab-ranking', 'index': indicador_id}))
        # Adiciona aba de Mapa se o conteúdo foi gerado (não é apenas a mensagem de erro inicial)
        # Ou seja, se 'DESC_UND_FED' existe
        if 'DESC_UND_FED' in df_filtered.columns:
            tabs_content.append(dbc.Tab(map_content, label="Mapa", tab_id=f'tab-map-{indicador_id}',
                                        id={'type': 'tab-map', 'index': indicador_id}))

        # Container para as abas (se existirem)
        tabs_container = html.Div()  # Vazio por padrão
        if tabs_content:
            tabs_container = dbc.Tabs(
                id={'type': 'visualization-tabs', 'index': indicador_id},
                children=tabs_content,
                # Define a primeira aba (ranking, se existir) como ativa
                active_tab=tabs_content[0].tab_id if tabs_content else None
            )

        # Monta o layout final com gráfico principal e abas lado a lado
        visualization_card_content = dbc.CardBody([
            dbc.Row([
                # Coluna para o Gráfico Principal
                dbc.Col(main_chart_content, md=7, xs=12, className="mb-4 mb-md-0"),
                # Ocupa 7 colunas em telas médias/grandes
                # Coluna para as Abas (Ranking/Mapa)
                dbc.Col(tabs_container, md=5, xs=12)  # Ocupa 5 colunas em telas médias/grandes
            ])
        ])

        graph_layout.append(dbc.Row([
            dbc.Col(dbc.Card(visualization_card_content, className="mb-4"), width=12)
        ]))

        # --- Adiciona Tabela Detalhada sempre
        graph_layout.append(dbc.Row([
            dbc.Col(dbc.Card([
                html.Div([
                    html.H5("Dados Detalhados", className="mt-4 d-inline-block", style={'marginLeft': '20px'}),
                    html.Div([
                        # Substitui os botões simples por dropdowns
                        dbc.DropdownMenu(
                            id={'type': 'dropdown-csv', 'index': indicador_id},
                            label="Baixar CSV",
                            color="success",
                            size="sm",
                            className="me-2 d-inline-block",
                            children=[
                                dbc.DropdownMenuItem("Dados filtrados", 
                                                    id={'type': 'btn-csv-filtered', 'index': indicador_id}),
                                dbc.DropdownMenuItem("Dados completos", 
                                                    id={'type': 'btn-csv-full', 'index': indicador_id}),
                            ]
                        ),
                        dbc.DropdownMenu(
                            id={'type': 'dropdown-excel', 'index': indicador_id},
                            label="Baixar Excel",
                            color="primary",
                            size="sm",
                            className="d-inline-block",
                            children=[
                                dbc.DropdownMenuItem("Dados filtrados", 
                                                    id={'type': 'btn-excel-filtered', 'index': indicador_id}),
                                dbc.DropdownMenuItem("Dados completos", 
                                                    id={'type': 'btn-excel-full', 'index': indicador_id}),
                            ]
                        ),
                        # Componentes de download
                        dcc.Download(id={'type': 'download-csv', 'index': indicador_id}),
                        dcc.Download(id={'type': 'download-excel', 'index': indicador_id})
                    ], className="float-end me-3 mt-4 d-flex")
                ], className="d-flex justify-content-between w-100"),
                dbc.CardBody([
                    dag.AgGrid(
                        id={'type': 'detail-table', 'index': indicador_id},
                        # Usa df_original_for_table para a tabela AG Grid
                        rowData=df_original_for_table.to_dict('records'),
                        columnDefs=columnDefs,
                        defaultColDef=defaultColDef,
                        dashGridOptions={
                            "pagination": True, "paginationPageSize": 10,
                            "paginationPageSizeSelector": [5, 10, 20, 50, 100],
                            "domLayout": "autoHeight", "suppressMovableColumns": True,
                            "animateRows": True, "suppressColumnVirtualisation": True
                        },
                        style={"width": "100%"}
                    ),
                ])
            ]), className="mt-4")
        ]))
        
        # Preparar dados para exportação - garantindo todos os campos CODG
        export_data = df_original_for_table.copy()
        
        # Verificar campos CODG_ no DataFrame original e garantir que são incluídos na exportação
        if df is not None and not df.empty:
            codg_cols = [col for col in df.columns if col.startswith('CODG_')]
            for col in codg_cols:
                if col not in export_data.columns and col in df.columns:
                    export_data[col] = df[col]
        
        # Adicionar ID_INDICADOR se ainda não existir
        if indicador_id and 'ID_INDICADOR' not in export_data.columns:
            export_data['ID_INDICADOR'] = indicador_id
        
        # Reordenar colunas para agrupar campos relacionados (CODG e DESC correspondentes)
        # Primeiro identificamos todas as colunas disponíveis
        all_columns = list(export_data.columns)
        
        # Definimos a ordem desejada de pares de colunas
        ordered_pairs = [
            # Primeiro o ID_INDICADOR
            ['ID_INDICADOR'],
            # Depois campos de UF
            ['CODG_UND_FED', 'DESC_UND_FED'],
            # Depois campos de ANO
            ['CODG_ANO'],
            # Depois campos de VAR
            ['CODG_VAR', 'DESC_VAR'],
            # Depois VLR_VAR 
            ['VLR_VAR'],
            # Depois campos de UND_MED
            ['CODG_UND_MED', 'DESC_UND_MED']
        ]
        
        # Campos dinâmicos (outros CODG_ e DESC_ correspondentes)
        dynamic_pairs = []
        for col in all_columns:
            if col.startswith('CODG_') and col not in [item for sublist in ordered_pairs for item in sublist]:
                # Verifica se há um DESC correspondente
                desc_col = 'DESC_' + col[5:]
                if desc_col in all_columns:
                    dynamic_pairs.append([col, desc_col])
                else:
                    dynamic_pairs.append([col])
        
        # Construir a lista final de colunas na ordem desejada
        ordered_columns = []
        for pair in ordered_pairs + dynamic_pairs:
            for col in pair:
                if col in all_columns:
                    ordered_columns.append(col)
        
        # Adicionar quaisquer colunas restantes que não foram incluídas
        for col in all_columns:
            if col not in ordered_columns:
                ordered_columns.append(col)
        
        # Reordenar o DataFrame
        export_data = export_data[ordered_columns]
        
        # Log para debug dos campos na exportação
        logging.debug(f"Campos para exportação em {indicador_id} (reorganizados): {export_data.columns.tolist()}")
        
        # Adicionar Store para dados de download
        graph_layout.append(html.Div([
            dcc.Store(
                id={'type': 'download-data', 'index': indicador_id}, 
                data=export_data.to_json(date_format='iso', orient='split')
            )
        ], style={'display': 'none'}))
        
        return graph_layout

    except Exception as e:
        print(f"Erro em create_visualization para {indicador_id}: {e}")
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"Erro ao gerar visualização para {indicador_id}.", color="danger")


# Define o layout padrão
DEFAULT_LAYOUT = {
    'showlegend': True,
    'legend': dict(
        title=None,
        orientation="h",  # Legenda horizontal
        yanchor="top",
        y=1.2,  # Posiciona abaixo do gráfico
        xanchor="center",
        x=0.5  # Centraliza horizontalmente
    ),
    'margin': dict(l=20, r=20, t=40, b=100),  # Ajusta margens para acomodar a legenda
    'xaxis': dict(showgrid=False, zeroline=False),
    'yaxis': dict(showgrid=False, zeroline=False),
    'xaxis_automargin': True,
    'yaxis_automargin': True
}

# Define o layout do aplicativo
app.layout = dbc.Container([
    # Header com imagens e título
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardBody(dbc.Row([            
            dbc.Col(html.Img(src='/assets/img/imb720.png', className="img-fluid",
                             style={'maxWidth': '150px', 'height': 'auto'}), xs=12, sm=6, md=3, className="p-2"),
            dbc.Col(html.H1('Instituto Mauro Borges - ODS - Agenda 2030', className="align-middle",
                            style={'margin': '0', 'padding': '0'}), xs=12, sm=12, md=6,
                    className="d-flex align-items-center justify-content-center justify-content-md-start p-2")
        ], className="align-items-center"))
    ], className="mb-4", style={'marginTop': '15px', 'marginLeft': '15px', 'marginRight': '15px'}))),
    
    # Botão de informações - agora entre o header e o card principal
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Button(
                    html.I(className="bi bi-info-circle", style={"fontSize": "20px"}),
                    id="info-icon",
                    color="light",
                    outline=True,
                    className="float-end me-3 mb-2 rounded-circle",
                    style={"width": "40px", "height": "40px", "padding": "6px 0", "borderColor": "#229846", "color": "#229846"}
                ),
                dbc.Tooltip(
                    "Informações sobre o painel",
                    target="info-icon",
                    placement="left"
                )
            ], className="d-flex justify-content-end")
        ])
    ]),
    
    # Card Principal
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardBody(dbc.Row([
            # Menu Lateral (Objetivos)
            dbc.Col(dbc.Card(dbc.CardBody(dbc.Row([
                dbc.Col(html.Div(
                    html.Img(src=row['BASE64'], style={'width': '100%', 'marginBottom': '10px', 'cursor': 'pointer'},
                             className="img-fluid", id=f"objetivo{idx}", n_clicks=1 if idx == 0 else 0)), width=4)
                for idx, row in df.iterrows()
            ], className="g-2"))), lg=2),
            # Conteúdo Principal (Metas e Indicadores)
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H3(id='card-header', children=initial_header)),
                dbc.CardBody([
                    html.Div(id='card-content', children=initial_content),
                    dbc.Nav(id='metas-nav', pills=True, className="nav nav-pills gap-2",
                            style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '1rem'}, children=[]),
                    html.Div(id='meta-description', children=initial_meta_description, className="textJustify mt-4"),
                    html.Div(id='loading-indicator', children=[]),
                    html.Div(id='indicadores-section', children=initial_indicadores_section),
                    # Componente oculto para acionar o carregamento do primeiro indicador
                    html.Div(id='trigger-first-tab-load', style={'display': 'none'})
                ])
            ]), lg=10)
        ]))
    ], className="border-0 shadow-none"))),
    
    # Modal de informações sobre o projeto
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Sobre o Projeto")),
        dbc.ModalBody([
            html.H5("Instituto Mauro Borges - Painel ODS"),
            html.Hr(),
            html.H6("Sobre o Projeto", className="mt-3"),
            html.P("O Painel ODS é uma iniciativa do Instituto Mauro Borges para monitoramento dos Objetivos de Desenvolvimento Sustentável na Região Brasil Central."),
            
            html.H6("Equipe de Desenvolvimento", className="mt-3"),
            html.Ul([
                html.Li("Coordenação: Instituto Mauro Borges (IMB)"),
                html.Li("Desenvolvimento: Gerência de Dados e Estatísticas do IMB"),
                html.Li("Análise de Dados: Gerência de Dados e Estatísticas do IMB")
            ]),
            
            html.H6("Contato", className="mt-3"),
            html.P([
                html.Strong("E-mail: "), "imb@goias.gov.br",
                html.Br(),
                html.Strong("Telefone: "), "(62) 3270-8674",
                html.Br(),
                html.Strong("Site: "), html.A("www.imb.go.gov.br", href="https://www.imb.go.gov.br", target="_blank")
            ]),
            
            html.H6("Tecnologias Utilizadas", className="mt-3"),
            html.Ul([
                html.Li("Python"),
                html.Li("Dash & Plotly"),
                html.Li("Bootstrap"),
                html.Li("Pandas")
            ])
        ]),
        dbc.ModalFooter(
            dbc.Button("Fechar", id="close-info-modal", className="ms-auto", n_clicks=0)
        ),
    ], id="info-modal", is_open=False, size="lg")
], fluid=True)

# Callback para abrir/fechar o modal de informações
@app.callback(
    Output("info-modal", "is_open"),
    [Input("info-icon", "n_clicks"), Input("close-info-modal", "n_clicks")],
    [State("info-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_info_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


# Callback para carregar indicadores sob demanda quando uma aba é clicada
@app.callback(
    [Output({'type': 'lazy-load-container', 'index': MATCH}, 'children'),
     Output({'type': 'spinner-indicator', 'index': MATCH}, 'style')],
    Input('tabs-indicadores', 'active_tab'),
    State({'type': 'lazy-load-container', 'index': MATCH}, 'id'),
    prevent_initial_call=True
)
def load_indicator_on_demand(active_tab, container_id):  # <--- DEFINIÇÃO DA FUNÇÃO
    # Verificações iniciais
    if not active_tab or not container_id:
        raise PreventUpdate

    # Obtém o ID do indicador
    indicador_id = container_id['index']

    # Ignora o placeholder
    if indicador_id == 'placeholder':
        raise PreventUpdate

    # Verifica EXATAMENTE se a aba ativa corresponde a este indicador
    expected_tab_id = f"tab-{indicador_id}"
    if active_tab != expected_tab_id:
        # Não gera logs - apenas previne a atualização silenciosamente
        raise PreventUpdate

    # Se chegou aqui, é porque este indicador DEVE ser carregado
    logging.debug(f"Carregando indicador: {indicador_id} (Aba ativa: {active_tab})")

    # --- INÍCIO DO BLOCO TRY...EXCEPT ---
    try:
        # Carrega os dados do indicador
        df_dados = load_dados_indicador_cache(indicador_id)

        # Busca informações do indicador (descrição, etc.)
        indicador_info = df_indicadores[df_indicadores['ID_INDICADOR'] == indicador_id]
        if indicador_info.empty:
            logging.error("Erro: Configuração não encontrada para indicador %s", indicador_id)
            # Oculta spinner, mostra erro
            return [dbc.Alert(f"Informações de configuração não encontradas para o indicador {indicador_id}.",
                              color="danger")], {'display': 'none'}

        # Obtém a descrição do indicador (será retornada junto com o conteúdo ou erro, quando necessário)
        desc_p = html.P(indicador_info.iloc[0]['DESC_INDICADOR'], className="textJustify p-3")

        # Verifica se os dados foram carregados
        if df_dados is None or df_dados.empty:
            logging.warning("Dados não disponíveis para indicador %s", indicador_id)
            # Retorna descrição + alerta de dados não disponíveis
            return [desc_p, dbc.Alert(f"Dados não disponíveis para {indicador_id}.", color="warning")], {
                'display': 'none'}  # Oculta spinner

        # Variáveis para montar o conteúdo
        dynamic_filters_div = []
        variable_dropdown_div = []
        valor_inicial_variavel = None
        df_variavel_filtrado = pd.DataFrame()  # Inicializa vazio

        # --- Identificação das colunas de filtro dinâmico ---
        filter_cols = identify_filter_columns(df_dados)

        # --- Geração do Dropdown de Variável Principal (PRIMEIRO, pois afeta os filtros) ---
        has_variable_dropdown = not indicador_info.empty and 'VARIAVEIS' in indicador_info.columns and \
                                indicador_info['VARIAVEIS'].iloc[0] == '1'
        
        # Verifica se o indicador tem variáveis e se a coluna CODG_VAR existe nos dados
        if has_variable_dropdown and 'CODG_VAR' in df_dados.columns:
            df_variavel_loaded = load_variavel()
            variaveis_indicador = df_dados['CODG_VAR'].astype(str).unique()
            if not df_variavel_loaded.empty:
                df_variavel_filtrado = df_variavel_loaded[
                    df_variavel_loaded['CODG_VAR'].astype(str).isin(variaveis_indicador)]
                if not df_variavel_filtrado.empty:
                    # Usa a função de busca de melhor variável
                    valor_inicial_variavel = find_best_initial_var(df_dados, df_variavel_filtrado)

                    variable_dropdown_div = [html.Div([
                        html.Label("Selecione uma Variável:",
                                   style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'},
                                   id={'type': 'var-label', 'index': indicador_id}),
                        dcc.Dropdown(
                            id={'type': 'var-dropdown', 'index': indicador_id},
                            options=[{'label': desc, 'value': cod} for cod, desc in
                                     zip(df_variavel_filtrado['CODG_VAR'], df_variavel_filtrado['DESC_VAR'])],
                            value=valor_inicial_variavel, style={'width': '100%'}
                        )
                    ], style={'paddingBottom': '20px', 'paddingTop': '20px'},
                        id={'type': 'var-dropdown-container', 'index': indicador_id})]
                else:
                    # Se não há variáveis válidas, renderiza um dropdown oculto
                    variable_dropdown_div = [html.Div([
                        dcc.Dropdown(
                            id={'type': 'var-dropdown', 'index': indicador_id},
                            options=[], value=None, style={'display': 'none'}, disabled=True
                        )
                    ], id={'type': 'var-dropdown-container', 'index': indicador_id}, style={'display': 'none'})]
            else:
                # Se df_variavel_loaded está vazio, renderiza um dropdown oculto
                variable_dropdown_div = [html.Div([
                    dcc.Dropdown(
                        id={'type': 'var-dropdown', 'index': indicador_id},
                        options=[], value=None, style={'display': 'none'}, disabled=True
                    )
                ], id={'type': 'var-dropdown-container', 'index': indicador_id}, style={'display': 'none'})]
        else:
            # Se o indicador não tem variáveis ou a coluna CODG_VAR não existe, renderiza um dropdown oculto
            variable_dropdown_div = [html.Div([
                dcc.Dropdown(
                    id={'type': 'var-dropdown', 'index': indicador_id},
                    options=[], value=None, style={'display': 'none'}, disabled=True
                )
            ], id={'type': 'var-dropdown-container', 'index': indicador_id}, style={'display': 'none'})]

        # --- Busca a melhor combinação de filtros (usando a variável selecionada) ---
        best_filters = find_valid_filter_combination(df_dados, filter_cols, valor_inicial_variavel)
        initial_dynamic_filters = best_filters.copy()

        # --- Geração de Filtros Dinâmicos ---
        for idx, filter_col_code in enumerate(filter_cols):
            desc_col_code = 'DESC_' + filter_col_code[5:]
            code_to_desc = {}
            if desc_col_code in df_dados.columns:
                try:
                    mapping_df = df_dados[[filter_col_code, desc_col_code]].dropna().drop_duplicates()
                    code_to_desc = pd.Series(mapping_df[desc_col_code].astype(str).values,
                                             index=mapping_df[filter_col_code].astype(str)).to_dict()
                except Exception as map_err:
                    logging.error("Erro ao mapear código/descrição para filtro %s em %s: %s", filter_col_code,
                                  indicador_id, map_err)

            unique_codes = sorted(df_dados[filter_col_code].dropna().astype(str).unique())
            col_options = [{'label': str(code_to_desc.get(code, code)), 'value': code} for code in unique_codes]
            filter_label = constants.COLUMN_NAMES.get(filter_col_code, filter_col_code)
            md_width = 7 if idx % 2 == 0 else 5

            # Usa o valor da combinação encontrada ou o melhor valor para este filtro
            initial_value = best_filters.get(filter_col_code)
            if initial_value is None and unique_codes:
                # Se não tiver na melhor combinação, usa um valor padrão inteligente
                prefs = {
                    'CODG_DOM': ['Urbana', 'Rural', 'Total'],  # Situação do domicílio
                    'CODG_SEXO': ['Total', '4'],  # Prefere "Total" ou código 4 (ambos os sexos)
                    'CODG_RACA': ['Total', '6'],  # Prefere "Total" ou código 6 (todas as raças)
                    'CODG_IDADE': ['Total', '1140'],  # Prefere "Total" ou código 1140 (todas as idades)
                    'CODG_INST': ['Total']  # Prefere "Total" para nível de instrução
                }.get(filter_col_code, ['Total', 'Todos', 'Todas'])

                initial_value = find_best_initial_value(unique_codes, prefs)
                initial_dynamic_filters[filter_col_code] = initial_value

            dynamic_filters_div.append(dbc.Col([
                html.Label(f"{filter_label}:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id={'type': 'dynamic-filter-dropdown', 'index': indicador_id, 'filter_col': filter_col_code},
                    options=col_options, value=initial_value, style={'marginBottom': '10px', 'width': '100%'}
                )
            ], md=md_width, xs=12))

        # Adiciona log para debug dos filtros iniciais
        logging.debug(f"Indicador {indicador_id}: Filtros iniciais definidos como {initial_dynamic_filters}")
        logging.debug(f"Indicador {indicador_id}: Variável inicial definida como {valor_inicial_variavel}")

        # Gera a visualização inicial com os filtros definidos
        initial_visualization = create_visualization(
            df_dados, indicador_id, valor_inicial_variavel, initial_dynamic_filters
        )

        # --- Monta o conteúdo dinâmico final ---
        dynamic_content = []
        # Nota: A descrição do indicador já está presente na tab, não precisamos adicioná-la novamente aqui
        dynamic_content.extend(variable_dropdown_div)
        if dynamic_filters_div:
            dynamic_content.append(dbc.Row(dynamic_filters_div))

        # Adiciona o container do gráfico com a visualização inicial
        dynamic_content.append(html.Div(
            id={'type': 'graph-container', 'index': indicador_id},
            children=initial_visualization
        ))

        # Retorna conteúdo dinâmico e oculta o spinner
        return dynamic_content, {'display': 'none'}

    # --- FIM DO BLOCO TRY...EXCEPT ---
    except Exception as e_load:
        logging.exception("Erro ao carregar conteúdo para %s", indicador_id)
        # Retorna apenas o alerta de erro
        return [dbc.Alert(f"Erro ao carregar dados para {indicador_id}.", color="danger")], {'display': 'none'}


# Callback para atualizar o conteúdo do card principal (metas, indicadores)
@app.callback(
    [
        Output('card-header', 'children'),
        Output('card-content', 'children'),
        Output('metas-nav', 'children'),
        Output('meta-description', 'children'),
        Output('indicadores-section', 'children')
    ],
    [Input(f"objetivo{i}", "n_clicks") for i in range(len(df))] +
    [Input({'type': 'meta-button', 'index': ALL}, 'n_clicks')],
    prevent_initial_call=True  # Impede execução inicial
)
def update_card_content(*args):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id_str = ctx.triggered[0]['prop_id']
    triggered_value = ctx.triggered[0]['value']

    # Ignora cliques iniciais ou cliques sem valor (n_clicks=0)
    if triggered_value is None or triggered_value == 0:
        raise PreventUpdate

    try:
        # --- Clique em uma META ---
        if 'meta-button' in triggered_id_str:
            # Forma mais robusta de obter o ID usando o contexto
            if isinstance(ctx.triggered_id, dict):
                meta_id = ctx.triggered_id.get('index')
            else:  # Fallback (menos provável de ser necessário agora)
                try:
                    meta_id = json.loads(triggered_id_str.split('.')[0])['index']
                except (json.JSONDecodeError, IndexError, KeyError):
                    logging.error("Erro ao parsear meta_id de: %s", triggered_id_str)
                    raise PreventUpdate

            if not meta_id:
                raise PreventUpdate  # Não conseguiu obter o meta_id
            logging.debug("Atualizando conteúdo - Clique na Meta ID: %s", meta_id)  # Log de Debug

            meta_filtrada = df_metas[df_metas['ID_META'] == meta_id]
            if meta_filtrada.empty:
                return no_update, no_update, no_update, "Meta não encontrada.", []  # Atualiza descrição

            meta_desc = meta_filtrada['DESC_META'].iloc[0]
            objetivo_id = meta_filtrada['ID_OBJETIVO'].iloc[0]

            # Recria a barra de navegação das metas, marcando a ativa
            metas_obj_filtradas = df_metas[df_metas['ID_OBJETIVO'] == objetivo_id]
            metas_com_indicadores = []
            for _, meta in metas_obj_filtradas.iterrows():
                indicadores_meta = df_indicadores[df_indicadores['ID_META'] == meta['ID_META']]
                if not indicadores_meta.empty:
                    # Verifica se pelo menos um indicador tem dados
                    for _, row_ind in indicadores_meta.iterrows():
                        indicador_id = row_ind['ID_INDICADOR']
                        nome_arquivo = indicador_id.lower().replace("indicador ", "")
                        arquivo_parquet = f'db/resultados/indicador{nome_arquivo}.parquet'
                        if os.path.exists(arquivo_parquet):
                            metas_com_indicadores.append(meta)
                            break

            if not metas_com_indicadores:
                # Retorna o alerta na seção de indicadores com estilo
                alert_message = dbc.Alert(
                    "Não existem metas com indicadores disponíveis para este objetivo.",
                    color="warning",
                    className="mt-4",
                    style={'textAlign': 'center', 'font-weight': 'bold'}  # Adiciona estilo aqui
                )
                return header, content, [], "", [alert_message]  # Limpa descrição da meta, mostra alerta

            # Seleciona a primeira meta e gera a navegação
            meta_selecionada = metas_com_indicadores[0]
            metas_nav_children = [
                dbc.NavLink(
                    meta['ID_META'],
                    id={'type': 'meta-button', 'index': meta['ID_META']},
                    href="#",
                    active=(meta['ID_META'] == meta_id),
                    className="nav-link",
                    n_clicks=0  # Reset n_clicks
                ) for meta in metas_com_indicadores
            ]

            # Gera a seção de indicadores para a meta clicada
            indicadores_meta_selecionada = df_indicadores[df_indicadores['ID_META'] == meta_id]
            tabs_indicadores = []

            # Comentado para desabilitar pré-carregamento
            # preload_related_indicators(meta_id, df_indicadores, _load_dados_indicador_original)

            if not indicadores_meta_selecionada.empty:
                valor_inicial_variavel_primeira_aba = None

                # Filtra apenas indicadores que realmente possuem dados disponíveis
                indicadores_com_dados = []
                for _, row_ind in indicadores_meta_selecionada.iterrows():
                    indicador_id_atual = row_ind['ID_INDICADOR']
                    # Verifica se o arquivo do indicador existe
                    nome_arquivo = indicador_id_atual.lower().replace("indicador ", "")
                    arquivo_parquet = f'db/resultados/indicador{nome_arquivo}.parquet'
                    if os.path.exists(arquivo_parquet):
                        indicadores_com_dados.append(row_ind)

                # Se não houver indicadores com dados disponíveis, exibe mensagem
                if not indicadores_com_dados:
                    return header, content, metas_nav_children, meta_desc, [
                        html.H5("Indicadores", className="mt-4 mb-3"),
                        dbc.Alert("Não há dados disponíveis para os indicadores desta meta.", color="warning",
                                  className="textCenter p-3 mt-3")
                    ]

                for i, row_ind in enumerate(indicadores_com_dados):
                    indicador_id_atual = row_ind['ID_INDICADOR']
                    is_first_indicator = (i == 0)

                    if is_first_indicator:
                        # Carrega dados e cria conteúdo COMPLETO apenas para o primeiro indicador
                        df_dados = load_dados_indicador_cache(indicador_id_atual)
                        tab_content = []
                        dynamic_filters_div = []
                        valor_inicial_variavel = None

                        if df_dados is not None and not df_dados.empty:
                            try:
                                # Identifica filtros dinâmicos
                                filter_cols = identify_filter_columns(df_dados)
                                initial_dynamic_filters = {}  # Dicionário para guardar filtros iniciais

                                # Prepara os filtros dinâmicos
                                for idx, filter_col_code in enumerate(filter_cols):
                                    desc_col_code = 'DESC_' + filter_col_code[5:]
                                    code_to_desc = {}
                                    if desc_col_code in df_dados.columns:
                                        try:
                                            mapping_df = df_dados[
                                                [filter_col_code, desc_col_code]].dropna().drop_duplicates()
                                            code_to_desc = pd.Series(mapping_df[desc_col_code].astype(str).values,
                                                                     index=mapping_df[filter_col_code].astype(
                                                                         str)).to_dict()
                                        except Exception as map_err:
                                            logging.error("Erro ao mapear código/descrição para filtro %s: %s",
                                                          filter_col_code, map_err)
                                            # Adiciona log de erro para mapeamento
                                            logging.error("Erro ao mapear código/descrição para filtro %s: %s",
                                                          filter_col_code, map_err)
                                    unique_codes = sorted(df_dados[filter_col_code].dropna().astype(str).unique())
                                    col_options = [{'label': str(code_to_desc.get(code, code)), 'value': code} for code
                                                   in unique_codes]
                                    filter_label = constants.COLUMN_NAMES.get(filter_col_code, filter_col_code)
                                    md_width = 7 if idx % 2 == 0 else 5
                                    # Define o valor inicial e armazena
                                    initial_value = unique_codes[0] if unique_codes else None
                                    if initial_value is not None:
                                        initial_dynamic_filters[filter_col_code] = initial_value

                                    dynamic_filters_div.append(dbc.Col([
                                        html.Label(f"{filter_label}:", style={'fontWeight': 'bold', 'display': 'block',
                                                                              'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id={'type': 'dynamic-filter-dropdown', 'index': indicador_id_atual,
                                                'filter_col': filter_col_code},
                                            options=col_options,
                                            value=initial_value,  # Usa o valor inicial definido
                                            style={'marginBottom': '10px', 'width': '100%'}
                                        )
                                    ], md=md_width, xs=12))

                                # Dropdown de variável principal
                                indicador_info = df_indicadores[df_indicadores['ID_INDICADOR'] == indicador_id_atual]
                                has_variable_dropdown = not indicador_info.empty and 'VARIAVEIS' in indicador_info.columns and \
                                                        indicador_info['VARIAVEIS'].iloc[0] == '1'
                                variable_dropdown_div = []
                                if has_variable_dropdown:
                                    df_variavel_loaded = load_variavel()
                                    if 'CODG_VAR' in df_dados.columns:
                                        variaveis_indicador = df_dados['CODG_VAR'].astype(str).unique()
                                        if not df_variavel_loaded.empty:
                                            df_variavel_filtrado = df_variavel_loaded[
                                                df_variavel_loaded['CODG_VAR'].astype(str).isin(variaveis_indicador)]
                                            if not df_variavel_filtrado.empty:
                                                # Usar o primeiro valor disponível no dropdown
                                                valor_inicial_variavel = df_variavel_filtrado['CODG_VAR'].iloc[0]
                                                valor_inicial_variavel_primeira_aba = valor_inicial_variavel  # Salva para o Store
                                                variable_dropdown_div = [html.Div([
                                                    html.Label("Selecione uma Variável:",
                                                               style={'fontWeight': 'bold', 'display': 'block',
                                                                      'marginBottom': '5px'},
                                                               id={'type': 'var-label', 'index': indicador_id_atual}),
                                                    dcc.Dropdown(
                                                        id={'type': 'var-dropdown', 'index': indicador_id_atual},
                                                        options=[{'label': row['DESC_VAR'], 'value': row['CODG_VAR']}
                                                                 for _, row in df_variavel_filtrado.iterrows()],
                                                        value=valor_inicial_variavel,
                                                        style={'width': '100%', 'marginBottom': '15px'}
                                                    )
                                                ])]
                                            else: # df_variavel_filtrado está vazio
                                                variable_dropdown_div = [html.Div([dcc.Dropdown(id={'type': 'var-dropdown', 'index': indicador_id_atual}, options=[], value=None, style={'display': 'none'}, disabled=True)], style={'display': 'none'})]
                                        else: # df_variavel_loaded está vazio
                                            variable_dropdown_div = [html.Div([dcc.Dropdown(id={'type': 'var-dropdown', 'index': indicador_id_atual}, options=[], value=None, style={'display': 'none'}, disabled=True)], style={'display': 'none'})]
                                    else: # 'CODG_VAR' não está em df_dados
                                        variable_dropdown_div = [html.Div([dcc.Dropdown(id={'type': 'var-dropdown', 'index': indicador_id_atual}, options=[], value=None, style={'display': 'none'}, disabled=True)], style={'display': 'none'})]
                                else: # has_variable_dropdown é False
                                    variable_dropdown_div = [html.Div([dcc.Dropdown(id={'type': 'var-dropdown', 'index': indicador_id_atual}, options=[], value=None, style={'display': 'none'}, disabled=True)], style={'display': 'none'})]

                                # Cria a visualização inicial PASSANDO OS FILTROS INICIAIS
                                initial_visualization = create_visualization(
                                    df_dados, indicador_id_atual, valor_inicial_variavel, initial_dynamic_filters
                                )
                                tab_content = [html.P(row_ind['DESC_INDICADOR'], className="textJustify p-3",
                                                      style={'marginBottom': '10px'})]
                                tab_content.extend(variable_dropdown_div)
                                if dynamic_filters_div:
                                    tab_content.append(dbc.Row(dynamic_filters_div))
                                tab_content.append(html.Div(id={'type': 'graph-container', 'index': indicador_id_atual},
                                                            children=initial_visualization))
                            except Exception as e_inner:
                                logging.exception("Erro interno ao gerar conteúdo da aba %s", indicador_id_atual)
                                tab_content = [
                                    dbc.Alert(f"Erro ao gerar conteúdo para {indicador_id_atual}.", color="danger")]
                                # Usamos logging.exception aqui
                                logging.exception("Erro interno ao gerar conteúdo da aba %s", indicador_id_atual)
                        else:
                            tab_content = [
                                dbc.Alert(f"Dados não disponíveis para {indicador_id_atual}.", color="warning")]
                    else:
                        # Para os demais indicadores, cria apenas placeholder (lazy loading)
                        tab_content = [
                            html.Div([
                                html.P(row_ind['DESC_INDICADOR'], className="textJustify",
                                       style={'display': 'inline-block', 'marginRight': '10px'}),
                                dbc.Spinner(color="primary", size="sm", type="grow",
                                            spinner_style={'display': 'inline-block'},
                                            id={'type': 'spinner-indicator', 'index': indicador_id_atual})
                            ], className="p-3"),
                            html.Div(id={'type': 'lazy-load-container', 'index': indicador_id_atual},
                                     style={'minHeight': '50px'})
                        ]

                    # Adiciona Store para CADA aba (carregada ou não)
                    # Para a primeira aba, usa o valor_inicial_variavel encontrado
                    # Para as outras, o valor inicial da variável será None até serem carregadas
                    # --- MODIFICADO: Inclui initial_dynamic_filters no Store ---
                    initial_filters_for_store = {}
                    if is_first_indicator:
                        initial_filters_for_store = initial_dynamic_filters  # Usa os filtros coletados
                    
                    store_data = {
                        'selected_var': valor_inicial_variavel_primeira_aba if is_first_indicator else None,
                        'selected_filters': initial_filters_for_store 
                    }
                    tab_content.append(dcc.Store(id={'type': 'visualization-state-store', 'index': indicador_id_atual},
                                                 data=store_data))

                    # Adiciona a aba
                    tabs_indicadores.append(
                        dbc.Tab(tab_content, label=indicador_id_atual, tab_id=f"tab-{indicador_id_atual}",
                                id={'type': 'tab-indicador', 'index': indicador_id_atual}))

            # Define a primeira aba como ativa
            first_tab_id = tabs_indicadores[0].tab_id if tabs_indicadores else None

            # Monta a seção de indicadores
            indicadores_section = [
                html.H5("Indicadores", className="mt-4 mb-3"),
                dbc.Card(dbc.CardBody(
                    dbc.Tabs(id='tabs-indicadores', children=tabs_indicadores, active_tab=first_tab_id)
                ), className="mt-3")
            ] if tabs_indicadores else []  # Só mostra seção se houver indicadores

            # Retorna SEM no_update para header/content para permitir voltar ao desc do objetivo se necessário
            objetivo_row = df[df['ID_OBJETIVO'] == objetivo_id].iloc[0]
            header_obj = f"{objetivo_row['ID_OBJETIVO']} - {objetivo_row['RES_OBJETIVO']}"
            content_obj = objetivo_row['DESC_OBJETIVO']
            return header_obj, content_obj, metas_nav_children, meta_desc, indicadores_section

        # --- Clique em um OBJETIVO ---
        elif 'objetivo' in triggered_id_str:
            index = int(triggered_id_str.replace('objetivo', '').split('.')[0])
            if index >= len(df):
                return "Erro", "Objetivo não encontrado.", [], "", []
            row_obj = df.iloc[index]
            logging.debug("Atualizando conteúdo - Clique no Objetivo ID: %s (Index: %d)", row_obj['ID_OBJETIVO'],
                          index)  # Log de Debug
            header = f"{row_obj['ID_OBJETIVO']} - {row_obj['RES_OBJETIVO']}" if index > 0 else row_obj['RES_OBJETIVO']
            
            # Usa o texto personalizado para o objetivo 0, caso contrário usa o texto do CSV
            if index == 0:
                content = html.Div([
                    html.P("Os Objetivos de Desenvolvimento Sustentável (ODS) constituem um compromisso global voltado à erradicação da pobreza, à proteção do meio ambiente e do clima, bem como à promoção da paz e da prosperidade para todas as pessoas, em todos os lugares. No Brasil, as Nações Unidas atuam de forma articulada com instituições públicas e privadas, além da sociedade civil, para viabilizar o cumprimento da Agenda 2030 e assegurar um desenvolvimento inclusivo, equitativo e sustentável."),
                    html.P("Nesse contexto, o Instituto Mauro Borges desenvolveu, para os estados que compõem a Região Brasil Central — Distrito Federal, Goiás, Maranhão, Mato Grosso, Mato Grosso do Sul, Rondônia e Tocantins —, um painel de monitoramento dos ODS. A ferramenta possibilita o acompanhamento da evolução tanto dos objetivos quanto de suas respectivas metas. É importante destacar que, para alguns objetivos e metas, ainda não há disponibilidade de dados que permitam a construção dos indicadores correspondentes. Diante disso, buscou-se maximizar a utilização das bases de dados existentes, de modo a produzir o maior número possível de indicadores com qualidade e consistência.")
                ])
            else:
                content = row_obj['DESC_OBJETIVO']

            # Se for objetivo 0, limpa metas e indicadores
            if index == 0:
                return header, content, [], "", []

            # Encontra metas com indicadores para este objetivo
            metas_obj_filtradas = df_metas[df_metas['ID_OBJETIVO'] == row_obj['ID_OBJETIVO']]
            metas_com_indicadores = []
            for _, meta in metas_obj_filtradas.iterrows():
                indicadores_meta = df_indicadores[df_indicadores['ID_META'] == meta['ID_META']]
                if not indicadores_meta.empty:
                    # Verifica se pelo menos um indicador tem dados
                    for _, row_ind in indicadores_meta.iterrows():
                        indicador_id = row_ind['ID_INDICADOR']
                        nome_arquivo = indicador_id.lower().replace("indicador ", "")
                        arquivo_parquet = f'db/resultados/indicador{nome_arquivo}.parquet'
                        if os.path.exists(arquivo_parquet):
                            metas_com_indicadores.append(meta)
                            break

            if not metas_com_indicadores:
                # Retorna o alerta na seção de indicadores com estilo
                alert_message = dbc.Alert(
                    "Não existem metas com indicadores disponíveis para este objetivo.",
                    color="warning",
                    className="mt-4",
                    style={'textAlign': 'center', 'font-weight': 'bold'}  # Adiciona estilo aqui
                )
                return header, content, [], "", [alert_message]  # Limpa descrição da meta, mostra alerta

            # Seleciona a primeira meta e gera a navegação
            meta_selecionada = metas_com_indicadores[0]
            metas_nav_children = [
                dbc.NavLink(
                    meta['ID_META'],
                    id={'type': 'meta-button', 'index': meta['ID_META']},
                    href="#",
                    active=(meta['ID_META'] == meta_selecionada['ID_META']),
                    className="nav-link",
                    n_clicks=0  # Reset n_clicks
                ) for meta in metas_com_indicadores
            ]
            meta_description = meta_selecionada['DESC_META']

            # Gera a seção de indicadores para a primeira meta
            meta_id = meta_selecionada['ID_META']
            indicadores_primeira_meta = df_indicadores[df_indicadores['ID_META'] == meta_id]
            tabs_indicadores = []

            # Comentado para desabilitar pré-carregamento
            # preload_related_indicators(meta_id, df_indicadores, _load_dados_indicador_original)

            if not indicadores_primeira_meta.empty:
                # Variável para armazenar o valor inicial da variável (usado apenas para o primeiro indicador)
                valor_inicial_variavel = None
                initial_dynamic_filters = {}  # Dicionário para guardar filtros iniciais para clique em objetivo

                # Filtra apenas indicadores que realmente possuem dados disponíveis
                indicadores_com_dados = []
                for _, row_ind in indicadores_primeira_meta.iterrows():
                    indicador_id_atual = row_ind['ID_INDICADOR']
                    # Verifica se o arquivo do indicador existe
                    nome_arquivo = indicador_id_atual.lower().replace("indicador ", "")
                    arquivo_parquet = f'db/resultados/indicador{nome_arquivo}.parquet'
                    if os.path.exists(arquivo_parquet):
                        indicadores_com_dados.append(row_ind)

                # Se não houver indicadores com dados disponíveis, exibe mensagem
                if not indicadores_com_dados:
                    return header, content, metas_nav_children, meta_description, [
                        html.H5("Indicadores", className="mt-4 mb-3"),
                        dbc.Alert("Não há dados disponíveis para os indicadores desta meta.", color="warning",
                                  className="textCenter p-3 mt-3")
                    ]

                # Cria abas para todos os indicadores da primeira meta
                for i, row_ind in enumerate(indicadores_com_dados):
                    # Apenas o primeiro indicador é carregado completamente
                    is_first_indicator = (i == 0)

                    if is_first_indicator:
                        # Carrega dados apenas para o primeiro indicador
                        df_dados = load_dados_indicador_cache(row_ind['ID_INDICADOR'])
                        tab_content = []
                        dynamic_filters_div = []
                        valor_inicial_variavel = None
                        initial_dynamic_filters = {}  # Reseta para este indicador

                        if df_dados is not None and not df_dados.empty:
                            try:
                                # Identifica filtros dinâmicos
                                filter_cols = identify_filter_columns(df_dados)
                                initial_dynamic_filters = {}  # Dicionário para filtros iniciais

                                # Prepara os filtros dinâmicos
                                for idx, filter_col_code in enumerate(filter_cols):
                                    desc_col_code = 'DESC_' + filter_col_code[5:]
                                    code_to_desc = {}
                                    if desc_col_code in df_dados.columns:
                                        try:
                                            mapping_df = df_dados[
                                                [filter_col_code, desc_col_code]].dropna().drop_duplicates()
                                            code_to_desc = pd.Series(mapping_df[desc_col_code].astype(str).values,
                                                                     index=mapping_df[filter_col_code].astype(
                                                                         str)).to_dict()
                                        except Exception as map_err:
                                            logging.error("Erro ao mapear código/descrição para filtro %s: %s",
                                                          filter_col_code, map_err)
                                            # Adiciona log de erro para mapeamento
                                            logging.error("Erro ao mapear código/descrição para filtro %s: %s",
                                                          filter_col_code, map_err)
                                    unique_codes = sorted(df_dados[filter_col_code].dropna().astype(str).unique())
                                    col_options = [{'label': str(code_to_desc.get(code, code)), 'value': code} for code
                                                   in unique_codes]
                                    filter_label = constants.COLUMN_NAMES.get(filter_col_code, filter_col_code)

                                    # Define larguras alternadas para os filtros
                                    md_width = 7 if idx % 2 == 0 else 5
                                    # Define o valor inicial e armazena
                                    initial_value = unique_codes[0] if unique_codes else None
                                    if initial_value is not None:
                                        initial_dynamic_filters[filter_col_code] = initial_value

                                    dynamic_filters_div.append(
                                        dbc.Col([
                                            html.Label(f"{filter_label}:",
                                                       style={'fontWeight': 'bold', 'display': 'block',
                                                              'marginBottom': '5px'}),
                                            dcc.Dropdown(
                                                id={'type': 'dynamic-filter-dropdown', 'index': row_ind['ID_INDICADOR'],
                                                    'filter_col': filter_col_code},
                                                options=col_options,
                                                value=initial_value,  # Usa o valor inicial definido
                                                style={'marginBottom': '10px', 'width': '100%'}
                                            )
                                        ], md=md_width, xs=12)
                                    )

                                # Dropdown de variável principal
                                indicador_info = df_indicadores[
                                    df_indicadores['ID_INDICADOR'] == row_ind['ID_INDICADOR']]
                                has_variable_dropdown = not indicador_info.empty and 'VARIAVEIS' in indicador_info.columns and \
                                                        indicador_info['VARIAVEIS'].iloc[0] == '1'
                                variable_dropdown_div = []
                                if has_variable_dropdown:
                                    df_variavel_loaded = load_variavel()
                                    if 'CODG_VAR' in df_dados.columns:
                                        variaveis_indicador = df_dados['CODG_VAR'].astype(str).unique()
                                        if not df_variavel_loaded.empty:
                                            df_variavel_filtrado = df_variavel_loaded[
                                                df_variavel_loaded['CODG_VAR'].astype(str).isin(variaveis_indicador)]
                                            if not df_variavel_filtrado.empty:
                                                # Usar o primeiro valor disponível no dropdown
                                                valor_inicial_variavel = df_variavel_filtrado['CODG_VAR'].iloc[0]
                                                variable_dropdown_div = [html.Div([
                                                    html.Label("Selecione uma Variável:",
                                                               style={'fontWeight': 'bold', 'display': 'block',
                                                                      'marginBottom': '5px'},
                                                               id={'type': 'var-label',
                                                                   'index': row_ind['ID_INDICADOR']}),
                                                    dcc.Dropdown(
                                                        id={'type': 'var-dropdown', 'index': row_ind['ID_INDICADOR']},
                                                        options=[{'label': row['DESC_VAR'], 'value': row['CODG_VAR']}
                                                                 for _, row in df_variavel_filtrado.iterrows()],
                                                        value=valor_inicial_variavel,
                                                        style={'width': '100%', 'marginBottom': '15px'}
                                                    )
                                                ])]
                                            else: # df_variavel_filtrado está vazio
                                                variable_dropdown_div = [html.Div([dcc.Dropdown(id={'type': 'var-dropdown', 'index': row_ind['ID_INDICADOR']}, options=[], value=None, style={'display': 'none'}, disabled=True)], style={'display': 'none'})]
                                        else: # df_variavel_loaded está vazio
                                            variable_dropdown_div = [html.Div([dcc.Dropdown(id={'type': 'var-dropdown', 'index': row_ind['ID_INDICADOR']}, options=[], value=None, style={'display': 'none'}, disabled=True)], style={'display': 'none'})]
                                    else: # 'CODG_VAR' não está em df_dados
                                        variable_dropdown_div = [html.Div([dcc.Dropdown(id={'type': 'var-dropdown', 'index': row_ind['ID_INDICADOR']}, options=[], value=None, style={'display': 'none'}, disabled=True)], style={'display': 'none'})]
                                else: # has_variable_dropdown é False
                                    variable_dropdown_div = [html.Div([dcc.Dropdown(id={'type': 'var-dropdown', 'index': row_ind['ID_INDICADOR']}, options=[], value=None, style={'display': 'none'}, disabled=True)], style={'display': 'none'})]

                                # Cria a visualização inicial PASSANDO OS FILTROS INICIAIS
                                initial_visualization = create_visualization(
                                    df_dados, row_ind['ID_INDICADOR'], valor_inicial_variavel, initial_dynamic_filters
                                )
                                tab_content = [html.P(row_ind['DESC_INDICADOR'], className="textJustify p-3",
                                                      style={'marginBottom': '10px'})]
                                tab_content.extend(variable_dropdown_div)
                                if dynamic_filters_div:
                                    tab_content.append(dbc.Row(dynamic_filters_div))
                                tab_content.append(
                                    html.Div(id={'type': 'graph-container', 'index': row_ind['ID_INDICADOR']},
                                             children=initial_visualization))

                            except Exception as e_inner:
                                logging.exception("Erro interno ao gerar conteúdo da aba %s (clique objetivo)",
                                                  row_ind['ID_INDICADOR'])
                                tab_content = [dbc.Alert(f"Erro ao gerar conteúdo para {row_ind['ID_INDICADOR']}.",
                                                         color="danger")]
                                logging.exception("Erro interno ao gerar conteúdo da aba %s (clique objetivo)",
                                                  row_ind['ID_INDICADOR'])
                        else:
                            tab_content = [
                                dbc.Alert(f"Dados não disponíveis para {row_ind['ID_INDICADOR']}.", color="warning")]
                    else:
                        # Para os demais indicadores, cria apenas um placeholder que será carregado sob demanda
                        tab_content = [
                            # Coloca o spinner ao lado do título para economizar espaço
                            html.Div([
                                html.P(row_ind['DESC_INDICADOR'], className="textJustify",
                                       style={'display': 'inline-block', 'marginRight': '10px'}),
                                dbc.Spinner(color="primary", size="sm", type="grow",
                                            spinner_style={'display': 'inline-block'},
                                            id={'type': 'spinner-indicator', 'index': row_ind['ID_INDICADOR']})
                            ], className="p-3"),
                            # Div oculta que será substituída pelo conteúdo quando carregado
                            html.Div(id={'type': 'lazy-load-container', 'index': row_ind['ID_INDICADOR']},
                                     style={'minHeight': '50px'})
                        ]

                    # Adiciona Store para esta aba
                    # Para o primeiro indicador (is_first_indicator = True), valor_inicial_variavel e 
                    # initial_dynamic_filters já foram calculados.
                    # Para os outros (lazy-loaded), o store é inicializado com None/vazio,
                    # e a callback load_indicator_on_demand preencherá os valores corretos.
                    current_indicador_store_data = {
                        'selected_var': valor_inicial_variavel if is_first_indicator else None,
                        'selected_filters': initial_dynamic_filters if is_first_indicator else {} # Garante um dict vazio
                    }
                    tab_content.append(
                        dcc.Store(id={'type': 'visualization-state-store', 'index': row_ind['ID_INDICADOR']},
                                  data=current_indicador_store_data)
                    )

                    # Adiciona a aba ao conjunto de abas
                    tabs_indicadores.append(dbc.Tab(tab_content,
                                                    label=row_ind['ID_INDICADOR'],
                                                    tab_id=f"tab-{row_ind['ID_INDICADOR']}",
                                                    id={'type': 'tab-indicador', 'index': row_ind['ID_INDICADOR']}))

            # Adiciona um trigger para carregar o primeiro indicador automaticamente
            first_tab_id = tabs_indicadores[0].tab_id if tabs_indicadores else None

            indicadores_section = [
                html.H5("Indicadores", className="mt-4 mb-3"),
                dbc.Card(dbc.CardBody(
                    dbc.Tabs(id='tabs-indicadores', children=tabs_indicadores, active_tab=first_tab_id)
                ), className="mt-3")
            ] if tabs_indicadores else []

            return header, content, metas_nav_children, meta_description, indicadores_section
        else:
            # Caso ID não seja nem meta nem objetivo (não deve acontecer)
            raise PreventUpdate

    except Exception as e:
        logging.exception("Erro geral em update_card_content:")
        # Retorna um estado seguro em caso de erro inesperado
        return initial_header, initial_content, [], "Ocorreu um erro.", []


# --- Novas Callbacks para Atualizar o Store ---
# Callback para atualizar o store quando a VARIÁVEL PRINCIPAL muda
@app.callback(
    Output({'type': 'visualization-state-store', 'index': MATCH}, 'data', allow_duplicate=True),
    Input({'type': 'var-dropdown', 'index': MATCH}, 'value'),
    State({'type': 'dynamic-filter-dropdown', 'index': MATCH, 'filter_col': ALL}, 'value'), # <-- ADICIONADO STATE DOS FILTROS
    State({'type': 'dynamic-filter-dropdown', 'index': MATCH, 'filter_col': ALL}, 'id'),    # <-- ADICIONADO STATE DOS IDs DOS FILTROS
    State({'type': 'visualization-state-store', 'index': MATCH}, 'data'),
    prevent_initial_call=True
)
def update_store_from_variable(selected_var, current_filter_values, current_filter_ids, current_store_data):
    """Atualiza o store com a variável selecionada e os filtros atuais dos dropdowns."""
    if not callback_context.triggered:
        raise PreventUpdate

    # Constrói o dicionário de filtros atuais a partir dos valores e IDs dos dropdowns de filtro
    actual_current_filters = {}
    if current_filter_ids and current_filter_values:
        for i, filter_id_dict in enumerate(current_filter_ids):
            if filter_id_dict and i < len(current_filter_values) and current_filter_values[i] is not None: 
                filter_col = filter_id_dict.get('filter_col')
                if filter_col:
                    actual_current_filters[filter_col] = current_filter_values[i]

    new_store_data = {
        'selected_var': selected_var,
        'selected_filters': actual_current_filters  # Usa os filtros lidos diretamente dos dropdowns
    }

    # Só atualiza se o novo estado combinado for diferente do armazenado
    if new_store_data != (current_store_data or {}):
        logging.debug(
            f"Store Update (Var): Indicador {callback_context.inputs_list[0]['id']['index']} - Novo Store: {new_store_data}")
        return new_store_data
    else:
        raise PreventUpdate


# Callback para atualizar o store quando os FILTROS DINÂMICOS mudam
@app.callback(
    Output({'type': 'visualization-state-store', 'index': MATCH}, 'data', allow_duplicate=True),
    Input({'type': 'dynamic-filter-dropdown', 'index': MATCH, 'filter_col': ALL}, 'value'),
    State({'type': 'dynamic-filter-dropdown', 'index': MATCH, 'filter_col': ALL}, 'id'),
    State({'type': 'var-dropdown', 'index': MATCH}, 'value'),  # READICIONADO este state para preservar seleção de variável
    State({'type': 'visualization-state-store', 'index': MATCH}, 'data'),
    prevent_initial_call=True
)
def update_store_from_filters(filter_values, filter_ids, var_value, current_store_data):
    """Atualiza o store com os filtros selecionados e a variável atual do dropdown."""
    if not callback_context.triggered:
        raise PreventUpdate

    # Remonta o dicionário de filtros a partir dos inputs atuais
    actual_current_filters = {}
    if filter_ids and filter_values:
        for i, filter_id_dict in enumerate(filter_ids):
            if filter_id_dict and i < len(filter_values) and filter_values[i] is not None: 
                filter_col = filter_id_dict.get('filter_col')
                if filter_col:
                    actual_current_filters[filter_col] = filter_values[i]
    
    # Prioriza variável do dropdown, mas mantém valor do store se dropdown retornar None
    selected_var_from_store = current_store_data.get('selected_var') if current_store_data else None
    final_var_value = var_value if var_value is not None else selected_var_from_store
    
    # Preserva filtros existentes no store para quaisquer filtros não atualizados
    previous_filters = current_store_data.get('selected_filters', {}) if current_store_data else {}
    
    # Combina os filtros atuais com os anteriores (atuais têm precedência)
    combined_filters = {**previous_filters, **actual_current_filters}

    new_store_data = {
        'selected_var': final_var_value,
        'selected_filters': combined_filters  # Usa a combinação de filtros
    }

    # Só atualiza se o novo estado combinado for diferente do armazenado
    if new_store_data != (current_store_data or {}):
        try:
            indicador_id_str = callback_context.inputs_list[0][0]['id']['index'] 
        except (IndexError, KeyError, TypeError):
            indicador_id_str = "Desconhecido"
        logging.debug(f"Store Update (Filters): Indicador {indicador_id_str} - Novo Store: {new_store_data}")
        return new_store_data
    else:
        raise PreventUpdate


# Callback ATUALIZADA para gerar visualização QUANDO O STORE MUDA
@app.callback(
    Output({'type': 'graph-container', 'index': MATCH}, 'children'),
    [
        Input({'type': 'visualization-state-store', 'index': MATCH}, 'data')  # <-- INPUT AGORA É O STORE
    ],
    [
        State({'type': 'graph-container', 'index': MATCH}, 'id'),  # Mantém ID do container para logs
    ],
    prevent_initial_call=True  # Mantém prevent_initial_call
)
def update_visualization_from_store(store_data, container_id):  # <-- Argumentos modificados
    """Atualiza diretamente a visualização lendo o estado (var/filtros) do store."""
    ctx = callback_context
    # Não precisa checar ctx.triggered explicitamente aqui, pois o Input do store garante trigger
    if not store_data or not container_id:
        logging.debug("Vis Update from Store: Preventido (sem store_data ou container_id)")
        raise PreventUpdate

    indicador_id = container_id['index']

    # ---- Obter filtros e variável do STORE (Input) ----
    var_value = store_data.get('selected_var')
    selected_filters = store_data.get('selected_filters', {})  # Pega dict vazio se chave não existir
    if selected_filters is None:  # Garante que seja um dict
        selected_filters = {}
    # -------------------------------------------------

    logging.debug(
        f"Atualizando visualização from store para {indicador_id}. Var: {var_value}, Filtros: {selected_filters}")

    try:
        # Carrega dados do indicador
        df_dados = load_dados_indicador_cache(indicador_id)
        if df_dados is None or df_dados.empty:
            return dbc.Alert(f"Dados não disponíveis para {indicador_id}.",
                             color="warning")  # Não precisa retornar store

        # Gera nova visualização
        visualization = create_visualization(
            df_dados, indicador_id, var_value, selected_filters
        )

        return visualization  # <-- RETORNA APENAS VISUALIZAÇÃO

    except Exception as e:
        logging.exception(f"Erro ao atualizar visualização from store para {indicador_id}: {str(e)}")
        return dbc.Alert(f"Erro ao atualizar visualização: {str(e)}", color="danger")  # Não precisa retornar store


# Callback para atualizar o ranking quando o ano é alterado
@app.callback(
    Output({'type': 'ranking-chart', 'index': MATCH}, 'figure'),
    [Input({'type': 'year-dropdown-ranking', 'index': MATCH}, 'value')],
    [
        State({'type': 'ranking-chart', 'index': MATCH}, 'id'),
        State({'type': 'visualization-state-store', 'index': MATCH}, 'data')  # <-- ADICIONADO ESTADO DO STORE
    ],
    prevent_initial_call=True
)
def update_ranking_chart(selected_year, chart_id, store_data):  # <-- Argumentos modificados
    """Atualiza o gráfico de ranking quando o ano é alterado, lendo filtros do store"""
    ctx = callback_context
    if not ctx.triggered or not selected_year or not store_data:
        # Não atualiza se não houver ano ou dados no store
        logging.debug("Ranking: Update preventido (sem ano ou store_data)")
        raise PreventUpdate

    # Identificar o indicador a partir do ID do gráfico
    indicador_id = chart_id['index']

    # ---- Obter filtros e variável do STORE ----
    selected_var_value = store_data.get('selected_var')
    selected_filters = store_data.get('selected_filters', {})  # Pega dict vazio se chave não existir
    if selected_filters is None:  # Garante que seja um dict
        selected_filters = {}
    # -----------------------------------------

    # Log para debugging
    logging.debug(
        f"Atualizando ranking para {indicador_id}, Ano: {selected_year}, Var Store: {selected_var_value}, Filtros Store: {selected_filters}")

    # Carrega os dados do indicador
    df_ranking_base = load_dados_indicador_cache(indicador_id)
    if df_ranking_base is None or df_ranking_base.empty:
        logging.warning(f"Dados não disponíveis para o ranking de {indicador_id}")
        # Retorna figura vazia com aviso se não houver dados
        return go.Figure().update_layout(title='Dados não disponíveis para ranking.', xaxis={'visible': False},
                                         yaxis={'visible': False})

    logging.debug(
        f"Ranking - df_ranking_base inicial - Colunas: {df_ranking_base.columns.tolist()}, Registros: {len(df_ranking_base)}")

    # --- INÍCIO: Aplicar filtro de VARIÁVEL PRINCIPAL ---
    df_filtered_ranking = df_ranking_base.copy()
    if selected_var_value and 'CODG_VAR' in df_filtered_ranking.columns:
        selected_var_str = str(selected_var_value).strip()
        df_filtered_ranking['CODG_VAR'] = df_filtered_ranking['CODG_VAR'].astype(str).str.strip()
        df_filtered_ranking = df_filtered_ranking[df_filtered_ranking['CODG_VAR'] == selected_var_str]
        logging.debug(
            f"Ranking - Após filtro de variável principal ({selected_var_str}) - Registros: {len(df_filtered_ranking)}")
        if df_filtered_ranking.empty:
            var_name = selected_var_str
            df_var_desc = load_variavel()
            if not df_var_desc.empty:
                var_info = df_var_desc[df_var_desc['CODG_VAR'] == selected_var_str]
                if not var_info.empty:
                    var_name = var_info['DESC_VAR'].iloc[0]
            return go.Figure().update_layout(title=f'Ranking: Nenhum dado para variável \'{var_name}\'.',
                                             xaxis={'visible': False}, yaxis={'visible': False})
    # --- FIM: Aplicar filtro de VARIÁVEL PRINCIPAL ---

    # Garante que a coluna DESC_UND_MED exista desde o início (AGORA EM df_filtered_ranking)
    if 'DESC_UND_MED' not in df_filtered_ranking.columns:
        logging.debug(f"Ranking - Adicionando coluna DESC_UND_MED ao df_filtered_ranking (não existia)")
        if 'CODG_UND_MED' in df_filtered_ranking.columns:
            df_unidade_medida_loaded = load_unidade_medida()
            if not df_unidade_medida_loaded.empty:
                df_filtered_ranking['CODG_UND_MED'] = df_filtered_ranking['CODG_UND_MED'].astype(str)
                df_unidade_medida_loaded['CODG_UND_MED'] = df_unidade_medida_loaded['CODG_UND_MED'].astype(str)
                df_filtered_ranking = pd.merge(df_filtered_ranking,
                                               df_unidade_medida_loaded[['CODG_UND_MED', 'DESC_UND_MED']],
                                               on='CODG_UND_MED', how='left')
                df_filtered_ranking['DESC_UND_MED'] = df_filtered_ranking['DESC_UND_MED'].fillna('N/D')
            else:
                df_filtered_ranking['DESC_UND_MED'] = 'N/D'
        else:
            df_filtered_ranking['DESC_UND_MED'] = 'N/D'
        logging.debug(
            f"Ranking - df_filtered_ranking após DESC_UND_MED - Colunas: {df_filtered_ranking.columns.tolist()}")
    elif 'DESC_UND_MED' in df_filtered_ranking.columns:  # Garante fillna se já existir
        df_filtered_ranking['DESC_UND_MED'] = df_filtered_ranking['DESC_UND_MED'].fillna('N/D')

    # Verificar se temos dados de UF (AGORA EM df_filtered_ranking)
    if 'DESC_UND_FED' not in df_filtered_ranking.columns and 'CODG_UND_FED' not in df_filtered_ranking.columns:
        logging.warning(
            f"Ranking - Colunas de UF (DESC_UND_FED ou CODG_UND_FED) não encontradas em df_filtered_ranking para {indicador_id}")
        return go.Figure().update_layout(title='Dados não incluem informações por UF para ranking.',
                                         xaxis={'visible': False}, yaxis={'visible': False})

    # Filtros Dinâmicos (AGORA EM df_filtered_ranking)
    if selected_filters:
        for col_code, selected_value in selected_filters.items():
            if selected_value is not None and col_code in df_filtered_ranking.columns:
                df_filtered_ranking[col_code] = df_filtered_ranking[col_code].astype(str).fillna('').str.strip()
                selected_value_str = str(selected_value).strip()
                df_filtered_ranking = df_filtered_ranking[df_filtered_ranking[col_code] == selected_value_str]
        logging.debug(
            f"Ranking - df_filtered_ranking após filtros dinâmicos - Colunas: {df_filtered_ranking.columns.tolist()}, Registros: {len(df_filtered_ranking)}")

    # IMPORTANTE: Primeiro filtra pelo ANO selecionado, depois verifica unicidade
    if 'CODG_ANO' not in df_filtered_ranking.columns:
        logging.error(
            f"Ranking - Coluna CODG_ANO não encontrada em df_filtered_ranking para {indicador_id}. Colunas: {df_filtered_ranking.columns.tolist()}")
        return go.Figure().update_layout(title='Erro interno: Coluna de Ano ausente.', xaxis={'visible': False},
                                         yaxis={'visible': False})

    df_filtered_ranking['CODG_ANO'] = df_filtered_ranking['CODG_ANO'].astype(str).str.strip()
    df_ranking_ano = df_filtered_ranking[df_filtered_ranking['CODG_ANO'] == str(selected_year).strip()].copy()
    logging.debug(
        f"Ranking - df_ranking_ano após filtro de ano ({selected_year}) - Colunas: {df_ranking_ano.columns.tolist()}, Registros: {len(df_ranking_ano)}")

    if df_ranking_ano.empty:
        logging.warning(f"Ranking - Sem dados para o ano {selected_year} com os filtros aplicados em {indicador_id}")
        return go.Figure().update_layout(
            title=f'Sem dados para o ano {selected_year} com os filtros aplicados.',
            xaxis={'visible': False}, yaxis={'visible': False}
        )

    # Adiciona DESC_UND_FED se necessário
    if 'DESC_UND_FED' not in df_ranking_ano.columns and 'CODG_UND_FED' in df_ranking_ano.columns:
        logging.debug(f"Ranking - Adicionando DESC_UND_FED a df_ranking_ano para {indicador_id}")
        df_ranking_ano['DESC_UND_FED'] = df_ranking_ano['CODG_UND_FED'].astype(str).map(constants.UF_NAMES)
        # Log antes do dropna
        logging.debug(
            f"Ranking - df_ranking_ano ANTES de dropna DESC_UND_FED - Registros: {len(df_ranking_ano)}, NaNs em DESC_UND_FED: {df_ranking_ano['DESC_UND_FED'].isna().sum()}")
        df_ranking_ano = df_ranking_ano.dropna(subset=['DESC_UND_FED'])
        logging.debug(
            f"Ranking - df_ranking_ano APÓS dropna DESC_UND_FED - Colunas: {df_ranking_ano.columns.tolist()}, Registros: {len(df_ranking_ano)}")

    # Agora verifica unicidade por UF para o ano selecionado
    if 'DESC_UND_FED' not in df_ranking_ano.columns or df_ranking_ano.empty:
        logging.warning(
            f"Ranking - DESC_UND_FED não encontrada ou df_ranking_ano vazio após processamento para {indicador_id}, ano {selected_year}")
        return go.Figure().update_layout(
            title=f'Ranking não disponível para {selected_year} (dados de UF ausentes/inválidos).',
            xaxis={'visible': False}, yaxis={'visible': False}
        )

    # Verifica unicidade por UF para o ano selecionado
    counts_per_uf_ranking = df_ranking_ano['DESC_UND_FED'].value_counts()
    # ---- INÍCIO DA VERIFICAÇÃO DE UNICIDADE ----
    if (counts_per_uf_ranking > 1).any():
        logging.warning(
            f"Ranking - Múltiplos valores por UF para o ano {selected_year} e filtros aplicados. Indicador: {indicador_id}. Contagens: {counts_per_uf_ranking[counts_per_uf_ranking > 1]}")
        alert_fig = go.Figure()
        alert_fig.update_layout(
            annotations=[
                go.layout.Annotation(
                    text="Ranking não pode ser gerado: múltiplos valores por UF para o ano e filtros selecionados.<br>Verifique os filtros ou a configuração do indicador.",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    align="center",
                    font=dict(size=12)
                )
            ],
            xaxis={'visible': False}, yaxis={'visible': False},
            margin=dict(t=20, b=20, l=20, r=20),
            height=200  # Altura menor para o alerta
        )
        return alert_fig
    # ---- FIM DA VERIFICAÇÃO DE UNICIDADE ----

    # Adiciona DESC_UND_MED se necessário (agora em df_ranking_ano)
    if 'DESC_UND_MED' not in df_ranking_ano.columns:
        logging.debug(f"Ranking - Adicionando coluna DESC_UND_MED ao df_ranking_ano (não existia)")
        if 'CODG_UND_MED' in df_ranking_ano.columns:
            df_unidade_medida_loaded = load_unidade_medida()
            if not df_unidade_medida_loaded.empty:
                df_ranking_ano['CODG_UND_MED'] = df_ranking_ano['CODG_UND_MED'].astype(str)
                df_unidade_medida_loaded['CODG_UND_MED'] = df_unidade_medida_loaded['CODG_UND_MED'].astype(str)
                df_ranking_ano = pd.merge(df_ranking_ano, df_unidade_medida_loaded[['CODG_UND_MED', 'DESC_UND_MED']],
                                          on='CODG_UND_MED', how='left')
                df_ranking_ano['DESC_UND_MED'] = df_ranking_ano['DESC_UND_MED'].fillna('N/D')
            else:
                df_ranking_ano['DESC_UND_MED'] = 'N/D'
        else:
            df_ranking_ano['DESC_UND_MED'] = 'N/D'
        logging.debug(f"Ranking - df_ranking_ano após DESC_UND_MED - Colunas: {df_ranking_ano.columns.tolist()}")
    elif 'DESC_UND_MED' in df_ranking_ano.columns:  # Garante que não haja NaNs se a coluna já existir
        df_ranking_ano['DESC_UND_MED'] = df_ranking_ano['DESC_UND_MED'].fillna('N/D')

    # Lê a ordem do ranking do indicador
    ranking_ordem = 0  # Padrão
    indicador_info = df_indicadores[df_indicadores['ID_INDICADOR'] == indicador_id]
    if not indicador_info.empty and 'RANKING_ORDEM' in indicador_info.columns:
        try:
            ranking_ordem_val = pd.to_numeric(indicador_info['RANKING_ORDEM'].iloc[0], errors='coerce')
            if not pd.isna(ranking_ordem_val): ranking_ordem = int(ranking_ordem_val)
        except (ValueError, TypeError):
            pass

    # Ordena baseado em VLR_VAR e RANKING_ORDEM
    ascending = (ranking_ordem == 1)  # True se for menor para maior (1)
    df_ranking_ano = df_ranking_ano.sort_values('VLR_VAR', ascending=ascending)

    # Define cores e opacidade
    goias_color = 'rgba(34, 152, 70, 1)'
    other_color = 'rgba(34, 152, 70, 0.2)'

    # Cria o gráfico de ranking com go.Figure e go.Bar
    fig_ranking_updated = go.Figure()
    for _, row in df_ranking_ano.iterrows():
        uf = row['DESC_UND_FED']
        valor = row['VLR_VAR']
        und_med = row.get('DESC_UND_MED', 'N/D')  # Usa .get() para segurança
        bar_color = goias_color if uf == 'Goiás' else other_color
        # Modificado: Usa a função format_br
        text_value = format_br(valor)

        fig_ranking_updated.add_trace(go.Bar(
            y=[uf],  # Estados no eixo Y
            x=[valor],  # Valores no eixo X
            name=uf,
            orientation='h',  # Barras horizontais
            marker_color=bar_color,
            text=text_value,
            textposition='outside',  # Texto fora da barra
            hovertemplate=(
                f"<b>{uf}</b><br>"
                f"Valor: {text_value}<br>"  # Usa o texto formatado
                f"Unidade: {und_med}<extra></extra>"
            )
        ))

    max_x_ranking = df_ranking_ano['VLR_VAR'].max() if not df_ranking_ano.empty else 0
    x_range_ranking = [0, max_x_ranking * 1.15]

    # Atualiza layout para gráfico de barras horizontal
    fig_ranking_updated.update_layout(
        xaxis_title=None, yaxis_title=None,
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color='black'), categoryorder='array',
                   categoryarray=df_ranking_ano['DESC_UND_FED'].tolist()),
        xaxis=dict(showgrid=True, zeroline=False, tickfont=dict(size=12, color='black'), range=x_range_ranking,
                   tickformat='d'),
        showlegend=False, margin=dict(l=150, r=20, t=30, b=30), bargap=0.1
    )

    return fig_ranking_updated


@app.callback(
    Output({'type': 'choropleth-map', 'index': MATCH}, 'figure'),
    [Input({'type': 'year-dropdown-map', 'index': MATCH}, 'value')],
    [
        State({'type': 'choropleth-map', 'index': MATCH}, 'id'),
        State({'type': 'visualization-state-store', 'index': MATCH}, 'data')  # <-- ADICIONADO ESTADO DO STORE
    ],
    prevent_initial_call=True
)
def update_map_on_year_change(selected_year, chart_id, store_data):  # <-- Argumentos modificados
    """Atualiza o mapa coroplético quando o ano é alterado, lendo filtros do store"""
    import plotly.express as px  # Import local para clareza
    ctx = callback_context
    if not ctx.triggered or not selected_year or not store_data:
        logging.debug("Mapa: Update preventido (sem ano ou store_data)")
        raise PreventUpdate

    indicador_id = chart_id['index']

    # ---- Obter filtros e variável do STORE ----
    selected_var_value = store_data.get('selected_var')
    selected_filters = store_data.get('selected_filters', {})  # Pega dict vazio se chave não existir
    if selected_filters is None:  # Garante que seja um dict
        selected_filters = {}
    # -----------------------------------------

    logging.debug(
        f"Atualizando mapa para {indicador_id}, Ano: {selected_year}, Var Store: {selected_var_value}, Filtros Store: {selected_filters}")

    df_map_base = load_dados_indicador_cache(indicador_id)
    if df_map_base is None or df_map_base.empty:
        logging.warning(f"Dados não disponíveis para o mapa de {indicador_id}")
        return go.Figure().update_layout(title='Dados não disponíveis para mapa.', xaxis={'visible': False},
                                         yaxis={'visible': False})

    logging.debug(
        f"Mapa - df_map_base inicial - Colunas: {df_map_base.columns.tolist()}, Registros: {len(df_map_base)}")

    # --- INÍCIO: Aplicar filtro de VARIÁVEL PRINCIPAL ---
    df_filtered_map = df_map_base.copy()
    if selected_var_value and 'CODG_VAR' in df_filtered_map.columns:
        selected_var_str = str(selected_var_value).strip()
        df_filtered_map['CODG_VAR'] = df_filtered_map['CODG_VAR'].astype(str).str.strip()
        df_filtered_map = df_filtered_map[df_filtered_map['CODG_VAR'] == selected_var_str]
        logging.debug(
            f"Mapa - Após filtro de variável principal ({selected_var_str}) - Registros: {len(df_filtered_map)}")
        if df_filtered_map.empty:
            var_name = selected_var_str
            df_var_desc = load_variavel()
            if not df_var_desc.empty:
                var_info = df_var_desc[df_var_desc['CODG_VAR'] == selected_var_str]
                if not var_info.empty:
                    var_name = var_info['DESC_VAR'].iloc[0]
            return go.Figure().update_layout(title=f'Mapa: Nenhum dado para variável \'{var_name}\'.',
                                             xaxis={'visible': False}, yaxis={'visible': False})
    # --- FIM: Aplicar filtro de VARIÁVEL PRINCIPAL ---

    # NOVO: Garante que a coluna DESC_UND_MED exista desde o início (AGORA EM df_filtered_map)
    if 'DESC_UND_MED' not in df_filtered_map.columns:
        logging.debug(f"Mapa - Adicionando coluna DESC_UND_MED ao df_filtered_map (não existia)")
        if 'CODG_UND_MED' in df_filtered_map.columns:
            df_unidade_medida_loaded = load_unidade_medida()
            if not df_unidade_medida_loaded.empty:
                df_filtered_map['CODG_UND_MED'] = df_filtered_map['CODG_UND_MED'].astype(str)
                df_unidade_medida_loaded['CODG_UND_MED'] = df_unidade_medida_loaded['CODG_UND_MED'].astype(str)
                df_filtered_map = pd.merge(df_filtered_map, df_unidade_medida_loaded[['CODG_UND_MED', 'DESC_UND_MED']],
                                           on='CODG_UND_MED', how='left')
                df_filtered_map['DESC_UND_MED'] = df_filtered_map['DESC_UND_MED'].fillna('N/D')
            else:
                df_filtered_map['DESC_UND_MED'] = 'N/D'
        else:
            df_filtered_map['DESC_UND_MED'] = 'N/D'
        logging.debug(f"Mapa - df_filtered_map após DESC_UND_MED - Colunas: {df_filtered_map.columns.tolist()}")
    elif 'DESC_UND_MED' in df_filtered_map.columns:  # Garante fillna
        df_filtered_map['DESC_UND_MED'] = df_filtered_map['DESC_UND_MED'].fillna('N/D')

    logging.debug(
        f"Mapa - df_filtered_map antes de filtros dinâmicos - Colunas: {df_filtered_map.columns.tolist()}, Registros: {len(df_filtered_map)}")
    logging.debug(
        f"Mapa - 'DESC_UND_MED' está disponível antes de filtros dinâmicos? {'DESC_UND_MED' in df_filtered_map.columns}")

    if 'DESC_UND_FED' not in df_filtered_map.columns and 'CODG_UND_FED' not in df_filtered_map.columns:
        logging.warning(
            f"Mapa - Colunas de UF (DESC_UND_FED ou CODG_UND_FED) não encontradas em df_filtered_map para {indicador_id}")
        return go.Figure().update_layout(title='Dados não incluem informações por UF para mapa.',
                                         xaxis={'visible': False}, yaxis={'visible': False})

    # Aplica filtros dinâmicos (AGORA EM df_filtered_map)
    if selected_filters:
        for col_code, selected_value in selected_filters.items():
            if selected_value is not None and col_code in df_filtered_map.columns:
                df_filtered_map[col_code] = df_filtered_map[col_code].astype(str).fillna('').str.strip()
                selected_value_str = str(selected_value).strip()
                df_filtered_map = df_filtered_map[df_filtered_map[col_code] == selected_value_str]
                # Removido log repetido aqui
    logging.debug(
        f"Mapa - df_filtered_map após filtros dinâmicos - Colunas: {df_filtered_map.columns.tolist()}, Registros: {len(df_filtered_map)}")
    logging.debug(
        f"Mapa - 'DESC_UND_MED' está disponível após filtros dinâmicos? {'DESC_UND_MED' in df_filtered_map.columns}")

    # Garante CODG_ANO existe antes de filtrar
    if 'CODG_ANO' not in df_filtered_map.columns:
        logging.error(
            f"Mapa - Coluna CODG_ANO não encontrada em df_filtered_map para {indicador_id}. Colunas: {df_filtered_map.columns.tolist()}")
        return go.Figure().update_layout(title='Erro interno: Coluna de Ano ausente.', xaxis={'visible': False},
                                         yaxis={'visible': False})

    df_filtered_map['CODG_ANO'] = df_filtered_map['CODG_ANO'].astype(str).str.strip()
    df_map_ano = df_filtered_map[df_filtered_map['CODG_ANO'] == str(selected_year).strip()].copy()

    logging.debug(
        f"Mapa - df_map_ano após filtro de ano ({selected_year}) - Colunas: {df_map_ano.columns.tolist()}, Registros: {len(df_map_ano)}")
    logging.debug(f"Mapa - 'DESC_UND_MED' está disponível após filtro de ano? {'DESC_UND_MED' in df_map_ano.columns}")

    if df_map_ano.empty:
        logging.warning(f"Sem dados para o ano {selected_year} com os filtros aplicados")
        return go.Figure().update_layout(
            title=f'Sem dados para o ano {selected_year} com os filtros aplicados.',
            xaxis={'visible': False}, yaxis={'visible': False}
        )

    if 'DESC_UND_FED' not in df_map_ano.columns and 'CODG_UND_FED' in df_map_ano.columns:
        df_map_ano['DESC_UND_FED'] = df_map_ano['CODG_UND_FED'].astype(str).map(constants.UF_NAMES)
        df_map_ano = df_map_ano.dropna(subset=['DESC_UND_FED'])

    if 'DESC_UND_FED' not in df_map_ano.columns or df_map_ano.empty:
        logging.warning(f"Dados de UF não encontrados para o ano {selected_year}")
        return go.Figure().update_layout(
            title=f'Mapa não disponível para {selected_year}.',
            xaxis={'visible': False}, yaxis={'visible': False}
        )

    counts_per_uf_map = df_map_ano['DESC_UND_FED'].value_counts()

    if 'DESC_UND_MED' not in df_map_ano.columns and 'CODG_UND_MED' in df_map_ano.columns:
        logging.debug("Tentando obter DESC_UND_MED via CODG_UND_MED")
        df_unidade_medida_loaded = load_unidade_medida()
        if not df_unidade_medida_loaded.empty:
            df_map_ano['CODG_UND_MED'] = df_map_ano['CODG_UND_MED'].astype(str)
            df_unidade_medida_loaded['CODG_UND_MED'] = df_unidade_medida_loaded['CODG_UND_MED'].astype(str)
            df_map_ano = pd.merge(df_map_ano, df_unidade_medida_loaded[['CODG_UND_MED', 'DESC_UND_MED']],
                                  on='CODG_UND_MED', how='left')
            df_map_ano['DESC_UND_MED'] = df_map_ano['DESC_UND_MED'].fillna('N/D')
        else:
            df_map_ano['DESC_UND_MED'] = 'N/D'
    elif 'DESC_UND_MED' not in df_map_ano.columns:
        logging.debug("DESC_UND_MED e CODG_UND_MED não estão disponíveis, criando com valor padrão")
        df_map_ano['DESC_UND_MED'] = 'N/D'

    # Formata os valores para o hover
    df_map_ano['VLR_VAR_FORMATADO'] = df_map_ano['VLR_VAR'].apply(format_br)

    # Tenta obter unidade de medida de forma segura
    und_med_map = df_map_ano['DESC_UND_MED'].dropna().iloc[0] if not df_map_ano['DESC_UND_MED'].dropna().empty else ''

    # Carrega o GeoJSON do arquivo
    try:
        with open('db/br_geojson.json', 'r', encoding='utf-8') as f:
            geojson = json.load(f)
    except Exception as e:
        logging.error(f"Erro ao carregar GeoJSON: {e}")
        return go.Figure().update_layout(
            title='Erro ao carregar dados do mapa.',
            xaxis={'visible': False}, yaxis={'visible': False}
        )

    # Cria o mapa usando px.choropleth exatamente como na função create_visualization
    fig_map = px.choropleth(
        df_map_ano,
        geojson=geojson,
        locations='DESC_UND_FED',
        featureidkey='properties.name',
        color='VLR_VAR',
        color_continuous_scale=[
            [0.0, 'rgba(34, 152, 70, 0.2)'],
            [1.0, 'rgba(34, 152, 70, 1)']
        ]
    )

    # Atualiza as configurações de GEO (igual à função create_visualization)
    map_center = {'lat': -12.95984198, 'lon': -53.27299730}
    geos_update = dict(
        visible=False,
        showcoastlines=True,
        coastlinecolor="White",
        showland=True,
        landcolor="white",
        showframe=False,
        projection=dict(type='mercator', scale=15),
        center=map_center
    )
    fig_map.update_geos(**geos_update)

    # Atualiza os traces para usar linhas brancas (igual à função create_visualization)
    fig_map.update_traces(
        marker_line_color='white',
        marker_line_width=1,
        customdata=df_map_ano[['VLR_VAR_FORMATADO']],
        hovertemplate="<b>%{location}</b><br>Valor: %{customdata[0]}" + (
            f" {und_med_map}" if und_med_map else "") + "<extra></extra>"
    )

    # Remove o título da barra de cores (igual à função create_visualization)
    fig_map.update_layout(coloraxis_colorbar_title_text='')

    return fig_map


def find_best_initial_value(filter_values, preference_list=None):
    """
    Encontra o melhor valor inicial para um filtro com base em uma lista de preferências.
    
    Args:
        filter_values: Lista de valores disponíveis para o filtro
        preference_list: Lista de termos preferenciais ordenados por prioridade
        
    Returns:
        O melhor valor encontrado ou o primeiro valor se nenhuma preferência corresponder
    """
    if not filter_values:
        return None

    # Lista padrão de termos preferenciais se nenhuma for fornecida
    if preference_list is None:
        preference_list = ['Total', 'Todos', 'Todas', 'Geral', 'Ambos', 'Ambas']

    # Converte tudo para string para comparação
    filter_values_str = [str(val).strip().lower() for val in filter_values]

    # Primeiro tenta encontrar correspondências exatas
    for pref in preference_list:
        pref_lower = pref.lower()
        if pref_lower in filter_values_str:
            idx = filter_values_str.index(pref_lower)
            return filter_values[idx]

    # Depois tenta encontrar valores que contenham os termos preferenciais
    for pref in preference_list:
        pref_lower = pref.lower()
        for i, val in enumerate(filter_values_str):
            if pref_lower in val:
                return filter_values[i]

    # Se não encontrar nada, retorna o primeiro valor
    return filter_values[0]


# Função para testar diferentes combinações de filtros até encontrar uma que retorne dados
def find_valid_filter_combination(df_dados, filter_cols, var_value=None):
    """
    Tenta diferentes combinações de filtros até encontrar uma que retorne dados válidos.
    
    Args:
        df_dados: DataFrame com os dados do indicador
        filter_cols: Lista de colunas que são filtros dinâmicos
        var_value: Valor da variável principal, se aplicável
        
    Returns:
        Dicionário com a melhor combinação de filtros encontrada
    """
    logging.debug(f"Buscando combinação válida de filtros para {len(filter_cols)} filtros")

    # Verificar se há variável principal
    if var_value is not None and 'CODG_VAR' in df_dados.columns:
        df_test = df_dados[df_dados['CODG_VAR'].astype(str).str.strip() == str(var_value).strip()].copy()
        if df_test.empty:
            # Se a variável selecionada não retornar dados, não adianta testar filtros
            logging.debug(f"Variável {var_value} não retorna dados, não testando filtros")
            return {}
    else:
        df_test = df_dados.copy()

    # Se não houver filtros, não há o que testar
    if not filter_cols:
        return {}

    # Verificar se há coluna VLR_VAR antes de continuar
    if 'VLR_VAR' not in df_test.columns:
        logging.debug("Coluna VLR_VAR não encontrada no DataFrame, não é possível testar filtros")
        return {}

    # Preferências de valores por tipo de filtro
    preference_map = {
        'CODG_DOM': ['Urbana', 'Rural', 'Total'],  # Situação do domicílio
        'CODG_SEXO': ['Total', '4'],  # Prefere "Total" ou código 4 (ambos os sexos)
        'CODG_RACA': ['Total', '6'],  # Prefere "Total" ou código 6 (todas as raças)
        'CODG_IDADE': ['Total', '1140'],  # Prefere "Total" ou código 1140 (todas as idades)
        'CODG_INST': ['Total']  # Prefere "Total" para nível de instrução
    }

    # Primeiro tenta valores preferenciais para cada filtro
    best_filters = {}
    for col in filter_cols:
        unique_values = sorted(df_test[col].dropna().astype(str).unique())
        if not unique_values:
            continue

        # Usa preferências específicas para o filtro, se disponíveis
        prefs = preference_map.get(col, None)
        best_value = find_best_initial_value(unique_values, prefs)
        best_filters[col] = best_value

    # Verifica se a combinação de filtros retorna dados não-zeros
    df_filtered = df_test.copy()
    for col, val in best_filters.items():
        df_filtered = df_filtered[df_filtered[col].astype(str).str.strip() == str(val).strip()]

    # Se não restar nenhum registro, tenta filtros um a um
    if df_filtered.empty or (df_filtered['VLR_VAR'] == 0).all():
        logging.debug("Combinação inicial resultou em dados vazios, tentando filtros individuais")
        best_filters = {}

        # Testa cada filtro isoladamente (um por vez)
        for col in filter_cols:
            unique_values = sorted(df_test[col].dropna().astype(str).unique())
            for val in unique_values:
                df_test_single = df_test[df_test[col].astype(str).str.strip() == str(val).strip()]
                if not df_test_single.empty and not (df_test_single['VLR_VAR'] == 0).all():
                    best_filters[col] = val
                    break

    logging.debug(f"Melhor combinação de filtros encontrada: {best_filters}")
    return best_filters


# Adicione a seguinte função para selecionar a melhor variável inicial
def find_best_initial_var(df_dados, df_variavel_filtrado):
    """
    Encontra a melhor variável inicial baseado nos dados disponíveis
    """
    if df_variavel_filtrado.empty or 'CODG_VAR' not in df_dados.columns:
        return None

    # Tenta encontrar uma variável que tenha dados não-zeros
    for _, row in df_variavel_filtrado.iterrows():
        var_cod = row['CODG_VAR']
        df_test = df_dados[df_dados['CODG_VAR'].astype(str).str.strip() == str(var_cod).strip()]

        if not df_test.empty and not (df_test['VLR_VAR'] == 0).all():
            logging.debug(f"Encontrada variável com dados válidos: {var_cod}")
            return var_cod

    # Se não encontrar, usa a primeira variável
    return df_variavel_filtrado['CODG_VAR'].iloc[0]


# Callback para download de CSV (DADOS FILTRADOS)
@app.callback(
    Output({'type': 'download-csv', 'index': MATCH}, 'data'),
    Input({'type': 'btn-csv-filtered', 'index': MATCH}, 'n_clicks'),
    State({'type': 'download-data', 'index': MATCH}, 'data'),
    State({'type': 'btn-csv-filtered', 'index': MATCH}, 'id'),  # Para obter o ID do indicador
    prevent_initial_call=True
)
def download_csv_filtered(n_clicks, data_json, btn_id):
    """Gera o arquivo CSV para download com os dados filtrados da tabela."""
    if n_clicks is None or data_json is None:
        raise PreventUpdate
    
    try:
        # Obtém o ID do indicador do botão
        indicador_id = btn_id.get('index', '')
        
        # Formata o nome do arquivo: indicador sem espaços, pontos substituídos por underscores
        indicador_formatado = str(indicador_id).replace(' ', '').replace('.', '_')
        
        # Converte JSON para DataFrame
        from io import StringIO
        df = pd.read_json(StringIO(data_json), orient='split')
        
        # Log para verificar os campos presentes na exportação
        logging.info(f"Campos disponíveis na exportação CSV: {df.columns.tolist()}")
        
        # Retorna conteúdo CSV com nome de arquivo incluindo o indicador
        return dict(
            content=df.to_csv(index=False, encoding='utf-8-sig'),
            filename=f'{indicador_formatado}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        logging.exception("Erro ao gerar CSV filtrado: %s", str(e))
        return no_update

# Callback para download de CSV (DADOS COMPLETOS)
@app.callback(
    Output({'type': 'download-csv', 'index': MATCH}, 'data', allow_duplicate=True),
    Input({'type': 'btn-csv-full', 'index': MATCH}, 'n_clicks'),
    State({'type': 'btn-csv-full', 'index': MATCH}, 'id'),  # Para obter o ID do indicador
    prevent_initial_call=True
)
def download_csv_full(n_clicks, btn_id):
    """Gera o arquivo CSV para download com os dados completos do indicador."""
    if n_clicks is None:
        raise PreventUpdate
    
    try:
        # Obtém o ID do indicador do botão
        indicador_id = btn_id.get('index', '')
        
        # Formata o nome do arquivo: indicador sem espaços, pontos substituídos por underscores
        indicador_formatado = str(indicador_id).replace(' ', '').replace('.', '_')
        
        # Carrega os dados completos do indicador sem filtros
        df_full = load_dados_indicador_cache(indicador_id)
        
        if df_full is None or df_full.empty:
            logging.warning(f"Dados completos não disponíveis para o indicador {indicador_id}")
            return no_update
        
        # Prepara os dados para exportação, garantindo todas as colunas descritivas
        # Adiciona descrições da unidade federativa
        if 'CODG_UND_FED' in df_full.columns:
            df_full['DESC_UND_FED'] = df_full['CODG_UND_FED'].astype(str).map(constants.UF_NAMES)
        
        # Adiciona descrições de variáveis
        if 'CODG_VAR' in df_full.columns:
            df_variavel_loaded = load_variavel()
            if not df_variavel_loaded.empty:
                df_full['CODG_VAR'] = df_full['CODG_VAR'].astype(str)
                df_variavel_loaded['CODG_VAR'] = df_variavel_loaded['CODG_VAR'].astype(str)
                df_full = df_full.merge(df_variavel_loaded[['CODG_VAR', 'DESC_VAR']], 
                                             on='CODG_VAR', how='left')
        
        # Adiciona descrições de unidade de medida
        if 'CODG_UND_MED' in df_full.columns:
            df_unidade_medida_loaded = load_unidade_medida()
            if not df_unidade_medida_loaded.empty:
                df_full['CODG_UND_MED'] = df_full['CODG_UND_MED'].astype(str)
                df_unidade_medida_loaded['CODG_UND_MED'] = df_unidade_medida_loaded['CODG_UND_MED'].astype(str)
                df_full = df_full.merge(df_unidade_medida_loaded[['CODG_UND_MED', 'DESC_UND_MED']], 
                                             on='CODG_UND_MED', how='left')
        
        # Adiciona ID_INDICADOR
        if 'ID_INDICADOR' not in df_full.columns:
            df_full['ID_INDICADOR'] = indicador_id
        
        # Reordena colunas para agrupá-las logicamente
        all_columns = list(df_full.columns)
        ordered_pairs = [
            ['ID_INDICADOR'],
            ['CODG_UND_FED', 'DESC_UND_FED'],
            ['CODG_ANO'],
            ['CODG_VAR', 'DESC_VAR'],
            ['VLR_VAR'],
            ['CODG_UND_MED', 'DESC_UND_MED']
        ]
        
        # Campos dinâmicos
        dynamic_pairs = []
        for col in all_columns:
            if col.startswith('CODG_') and col not in [item for sublist in ordered_pairs for item in sublist]:
                desc_col = 'DESC_' + col[5:]
                if desc_col in all_columns:
                    dynamic_pairs.append([col, desc_col])
                else:
                    dynamic_pairs.append([col])
        
        # Constrói lista ordenada de colunas
        ordered_columns = []
        for pair in ordered_pairs + dynamic_pairs:
            for col in pair:
                if col in all_columns:
                    ordered_columns.append(col)
        
        # Adiciona colunas restantes
        for col in all_columns:
            if col not in ordered_columns:
                ordered_columns.append(col)
        
        # Reordena DataFrame
        try:
            df_full = df_full[ordered_columns]
        except Exception as e:
            logging.exception(f"Erro ao reordenar colunas para CSV completo: {e}")
            # Continua com a ordem original se houver erro
        
        # Log para verificar os campos presentes na exportação
        logging.info(f"Campos disponíveis na exportação CSV completa: {df_full.columns.tolist()}")
        
        # Retorna conteúdo CSV com nome de arquivo incluindo o indicador e sufixo 'full'
        return dict(
            content=df_full.to_csv(index=False, encoding='utf-8-sig'),
            filename=f'{indicador_formatado}_full_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        logging.exception("Erro ao gerar CSV completo: %s", str(e))
        return no_update

# Callback para download de Excel (DADOS FILTRADOS)
@app.callback(
    Output({'type': 'download-excel', 'index': MATCH}, 'data'),
    Input({'type': 'btn-excel-filtered', 'index': MATCH}, 'n_clicks'),
    State({'type': 'download-data', 'index': MATCH}, 'data'),
    State({'type': 'btn-excel-filtered', 'index': MATCH}, 'id'),  # Para obter o ID do indicador
    prevent_initial_call=True
)
def download_excel_filtered(n_clicks, data_json, btn_id):
    """Gera o arquivo Excel para download com os dados filtrados da tabela."""
    if n_clicks is None or data_json is None:
        raise PreventUpdate
    
    try:
        # Obtém o ID do indicador do botão
        indicador_id = btn_id.get('index', '')
        
        # Formata o nome do arquivo: indicador sem espaços, pontos substituídos por underscores
        indicador_formatado = str(indicador_id).replace(' ', '').replace('.', '_')
        
        # Converte JSON para DataFrame
        from io import StringIO
        df = pd.read_json(StringIO(data_json), orient='split')
        
        # Log para verificar os campos presentes na exportação
        logging.info(f"Campos disponíveis na exportação Excel filtrada: {df.columns.tolist()}")
        
        # Cria buffer de memória para o Excel
        output = io.BytesIO()
        
        # Tenta usar xlsxwriter, com fallback para openpyxl
        try:
            excel_engine = 'xlsxwriter'
            with pd.ExcelWriter(output, engine=excel_engine) as writer:
                df.to_excel(writer, sheet_name='Dados', index=False)
                
                # Auto-ajusta largura das colunas (apenas com xlsxwriter)
                worksheet = writer.sheets['Dados']
                for i, col in enumerate(df.columns):
                    # Encontra a largura máxima da coluna
                    column_len = max(
                        df[col].astype(str).map(len).max(),
                        len(str(col))
                    ) + 2  # adiciona um espaço extra
                    worksheet.set_column(i, i, column_len)
        except ImportError:
            # Fallback para openpyxl se xlsxwriter não estiver disponível
            excel_engine = 'openpyxl'
            with pd.ExcelWriter(output, engine=excel_engine) as writer:
                df.to_excel(writer, sheet_name='Dados', index=False)
                logging.info("Usando engine openpyxl para Excel (sem auto-ajuste de colunas)")
        
        # Coloca o ponteiro do buffer no início
        output.seek(0)
        
        # Retorna conteúdo Excel com nome de arquivo incluindo o indicador
        return dcc.send_bytes(output.read(), f'{indicador_formatado}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
    except Exception as e:
        logging.exception("Erro ao gerar Excel filtrado: %s", str(e))
        return no_update

# Callback para download de Excel (DADOS COMPLETOS)
@app.callback(
    Output({'type': 'download-excel', 'index': MATCH}, 'data', allow_duplicate=True),
    Input({'type': 'btn-excel-full', 'index': MATCH}, 'n_clicks'),
    State({'type': 'btn-excel-full', 'index': MATCH}, 'id'),  # Para obter o ID do indicador
    prevent_initial_call=True
)
def download_excel_full(n_clicks, btn_id):
    """Gera o arquivo Excel para download com os dados completos do indicador."""
    if n_clicks is None:
        raise PreventUpdate
    
    try:
        # Obtém o ID do indicador do botão
        indicador_id = btn_id.get('index', '')
        
        # Formata o nome do arquivo: indicador sem espaços, pontos substituídos por underscores
        indicador_formatado = str(indicador_id).replace(' ', '').replace('.', '_')
        
        # Carrega os dados completos do indicador sem filtros
        df_full = load_dados_indicador_cache(indicador_id)
        
        if df_full is None or df_full.empty:
            logging.warning(f"Dados completos não disponíveis para o indicador {indicador_id}")
            return no_update
        
        # Prepara os dados para exportação, garantindo todas as colunas descritivas
        # Adiciona descrições da unidade federativa
        if 'CODG_UND_FED' in df_full.columns:
            df_full['DESC_UND_FED'] = df_full['CODG_UND_FED'].astype(str).map(constants.UF_NAMES)
        
        # Adiciona descrições de variáveis
        if 'CODG_VAR' in df_full.columns:
            df_variavel_loaded = load_variavel()
            if not df_variavel_loaded.empty:
                df_full['CODG_VAR'] = df_full['CODG_VAR'].astype(str)
                df_variavel_loaded['CODG_VAR'] = df_variavel_loaded['CODG_VAR'].astype(str)
                df_full = df_full.merge(df_variavel_loaded[['CODG_VAR', 'DESC_VAR']], 
                                        on='CODG_VAR', how='left')
        
        # Adiciona descrições de unidade de medida
        if 'CODG_UND_MED' in df_full.columns:
            df_unidade_medida_loaded = load_unidade_medida()
            if not df_unidade_medida_loaded.empty:
                df_full['CODG_UND_MED'] = df_full['CODG_UND_MED'].astype(str)
                df_unidade_medida_loaded['CODG_UND_MED'] = df_unidade_medida_loaded['CODG_UND_MED'].astype(str)
                df_full = df_full.merge(df_unidade_medida_loaded[['CODG_UND_MED', 'DESC_UND_MED']], 
                                        on='CODG_UND_MED', how='left')
        
        # Adiciona ID_INDICADOR
        if 'ID_INDICADOR' not in df_full.columns:
            df_full['ID_INDICADOR'] = indicador_id
        
        # Reordena colunas para agrupá-las logicamente
        all_columns = list(df_full.columns)
        ordered_pairs = [
            ['ID_INDICADOR'],
            ['CODG_UND_FED', 'DESC_UND_FED'],
            ['CODG_ANO'],
            ['CODG_VAR', 'DESC_VAR'],
            ['VLR_VAR'],
            ['CODG_UND_MED', 'DESC_UND_MED']
        ]
        
        # Campos dinâmicos
        dynamic_pairs = []
        for col in all_columns:
            if col.startswith('CODG_') and col not in [item for sublist in ordered_pairs for item in sublist]:
                desc_col = 'DESC_' + col[5:]
                if desc_col in all_columns:
                    dynamic_pairs.append([col, desc_col])
                else:
                    dynamic_pairs.append([col])
        
        # Constrói lista ordenada de colunas
        ordered_columns = []
        for pair in ordered_pairs + dynamic_pairs:
            for col in pair:
                if col in all_columns:
                    ordered_columns.append(col)
        
        # Adiciona colunas restantes
        for col in all_columns:
            if col not in ordered_columns:
                ordered_columns.append(col)
        
        # Reordena DataFrame se possível
        try:
            df_full = df_full[ordered_columns]
        except Exception as e:
            logging.exception(f"Erro ao reordenar colunas para Excel completo: {e}")
            # Continua com a ordem original se houver erro
        
        # Log para verificar os campos presentes na exportação
        logging.info(f"Campos disponíveis na exportação Excel completa: {df_full.columns.tolist()}")
        
        # Cria buffer de memória para o Excel
        output = io.BytesIO()
        
        # Tenta usar xlsxwriter, com fallback para openpyxl
        try:
            excel_engine = 'xlsxwriter'
            with pd.ExcelWriter(output, engine=excel_engine) as writer:
                df_full.to_excel(writer, sheet_name='Dados', index=False)
                
                # Auto-ajusta largura das colunas (apenas com xlsxwriter)
                worksheet = writer.sheets['Dados']
                for i, col in enumerate(df_full.columns):
                    # Encontra a largura máxima da coluna
                    column_len = max(
                        df_full[col].astype(str).map(len).max(),
                        len(str(col))
                    ) + 2  # adiciona um espaço extra
                    worksheet.set_column(i, i, column_len)
        except ImportError:
            # Fallback para openpyxl se xlsxwriter não estiver disponível
            excel_engine = 'openpyxl'
            with pd.ExcelWriter(output, engine=excel_engine) as writer:
                df_full.to_excel(writer, sheet_name='Dados', index=False)
                logging.info("Usando engine openpyxl para Excel (sem auto-ajuste de colunas)")
        
        # Coloca o ponteiro do buffer no início
        output.seek(0)
        
        # Retorna conteúdo Excel com nome de arquivo incluindo o indicador e sufixo 'full'
        return dcc.send_bytes(output.read(), f'{indicador_formatado}_full_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
    except Exception as e:
        logging.exception("Erro ao gerar Excel completo: %s", str(e))
        return no_update


server = app.server

if __name__ == '__main__':
    # Verifica se o arquivo .env existe
    if not os.path.exists('.env'):
        logging.warning("Arquivo .env não encontrado. Criando com configurações padrão...")

        # Gera uma nova SECRET_KEY e atualiza o arquivo .env
        new_secret_key = generate_secret_key()
        update_env_file(generate_password_hash(MAINTENANCE_PASSWORD))

        logging.info("Arquivo .env criado com sucesso!")
        # Recarrega as variáveis de ambiente
        load_dotenv()

    if DEBUG:
        app.run_server(
            debug=DEBUG,
            use_reloader=USE_RELOADER,
            port=PORT,
            host=HOST
        )
    else:
        # Em produção, o uWSGI irá usar a variável 'server'
        app.run_server(
            debug=False,
            use_reloader=False,
            port=PORT,
            host=HOST
        )
