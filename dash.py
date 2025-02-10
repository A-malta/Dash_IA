import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import requests
from google import genai
import logging 
from google.genai import types

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="Dashboard de Desempenho")
st.markdown("""<h1 style='margin-top: -60px;'>Dashboard de Desempenho</h1>""", unsafe_allow_html=True)

df = pd.read_csv("dataset.csv")

nomes = ["Todos"] + list(df['Nome'].unique())
anos = ["Todos"] + list(df['Ano'].unique())
meses = ["Todos"] + list(df['Mês'].unique())

st.sidebar.header("Filtros")
nome_escolhido = st.sidebar.selectbox("Selecione o Nome", nomes)
ano_escolhido = st.sidebar.selectbox("Selecione o Ano", anos)
mes_escolhido = st.sidebar.selectbox("Selecione o Mês", meses)

def filtrar_dados(df, nome=None, ano=None, mes=None):
    if nome:
        df = df[df['Nome'] == nome]
    if ano:
        df = df[df['Ano'] == ano]
    if mes:
        df = df[df['Mês'] == mes]
    return df

df_resultado = filtrar_dados(
    df,
    nome=None if nome_escolhido == "Todos" else nome_escolhido,
    ano=None if ano_escolhido == "Todos" else ano_escolhido,
    mes=None if mes_escolhido == "Todos" else mes_escolhido
)

# Integração com IA
st.sidebar.header("Integração com IA")

if "ia_resposta" not in st.session_state:
    st.session_state.ia_resposta = ""

if "pergunta" not in st.session_state:
    st.session_state.pergunta = ""

pergunta = st.sidebar.text_input(
    "Digite sua pergunta ou solicitação:",
    placeholder="Escreva seu prompt para a IA...",
    key="pergunta"
)
enviar = st.sidebar.button("Enviar")

if (pergunta and pergunta != st.session_state.get("ultima_pergunta", "")) or enviar:
    st.session_state.ultima_pergunta = pergunta  

    client = genai.Client(api_key="GEMINI_API_KEY")
    dados_contexto = "\n".join([str(item) for item in df_resultado.to_dict(orient="records")])

    prompt = f"contexto:{dados_contexto}\n com base nos dados que você recebeu de todas as features filtradas faça uma sugestão com base na pergunta:{pergunta}"
    system_prompt = "responda de forma sucinta"

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                system_instruction=system_prompt
            )
        )
        logger.info(response)
        
        st.session_state.ia_resposta = response.text  

    except Exception as e:
        st.sidebar.error(f"Erro: {e}")

if st.session_state.ia_resposta:
    st.sidebar.markdown("### Resposta da IA")
    st.sidebar.write(st.session_state.ia_resposta)


# Gráficos
def grafico_01(df_resultado):
    features_mapping = {
        "Satisfacao_Tarefas": "Satisfação com Tarefas",
        "Satisfacao_Lideranca": "Satisfação com Liderança",
        "Clima_Organizacional": "Clima Organizacional",
        "Apoio_Colegas": "Apoio dos Colegas",
        "Reconhecimento_Profissional": "Reconhecimento Profissional",
        "Perspectiva_Crescimento": "Perspectiva de Crescimento",
        "Bem_Estar_Psicologico": "Bem-Estar Psicológico",
        "Saude_Fisica": "Saúde Física",
        "Liberdade_Autonomia": "Liberdade e Autonomia"
    }
    
    features = list(features_mapping.keys())
    cores = sns.color_palette("cool", n_colors=len(features))  

    if not df_resultado.empty:
        pontuacao_por_grupo = [df_resultado[feature].mean() for feature in features]
        pontuacao_por_grupo += pontuacao_por_grupo[:1] 
        
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]  

        rotation_offset = np.pi / 12  
        rotated_angles = [(angle + rotation_offset) % (2 * np.pi) for angle in angles]

        cor_fundo = "#0e1117"

        plt.rcParams["font.family"] = "DejaVu Sans"

        fig, ax = plt.subplots(figsize=(6, 6), dpi=150, subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor(cor_fundo)

        for i, (angle, valor, cor) in enumerate(zip(rotated_angles[:-1], pontuacao_por_grupo[:-1], cores)):
            ax.bar(angle, valor, width=2 * np.pi / len(features), color=cor,  
                   edgecolor=cor_fundo, align='edge')

        for i, (key, angle) in enumerate(zip(features_mapping.values(), angles[:-1])):
            ax.text(angle, 5.5, key, fontsize=8, color='white',
                    horizontalalignment='center', verticalalignment='center', rotation=0)

        ax.set_facecolor(cor_fundo)
        ax.tick_params(colors='white')
        ax.spines['polar'].set_color('white')
        ax.grid(color='white', linewidth=0.1, linestyle='solid')
        ax.set_yticks(range(0, 6, 1))
        ax.set_yticklabels(range(0, 6, 1), color='white', fontsize=7)
        ax.set_xticks(rotated_angles[:-1])
        ax.set_xticklabels([''] * len(features))
        ax.set_ylim(0, 5)

        return fig
    else:
        return None

        

def grafico_02(df_resultado):
    if not df_resultado.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        
        sns.barplot(x=df_resultado["Horas_Trabalhadas"], 
                    y=df_resultado["Vendas_Realizadas"], 
                    palette="cool", ax=ax, ci=None)  
        
        ax.set_xlabel("Horas Trabalhadas", color="white")
        ax.set_ylabel("Vendas Realizadas", color="white")
        
        max_vendas = df_resultado["Vendas_Realizadas"].max()
        ax.set_yticks(np.arange(0, max_vendas + 1, 1))  
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))  
        
        ax.tick_params(axis='both', colors='white')  
        for spine in ax.spines.values():
            spine.set_visible(False)  

        return fig
    else:
        return None



def grafico_03(df_resultado, fontsize=18):  
    if not df_resultado.empty:
        taxa_media = df_resultado["Taxa_Conversao"].mean()
        restante = 1 - taxa_media  
        
        valores = [taxa_media, restante]
        labels = ["", ""]
        
        fig, ax = plt.subplots(figsize=(8, 8)) 
        
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        
        wedges, texts, autotexts = ax.pie(
            valores,
            labels=labels,
            autopct="%.1f%%",  
            startangle=90, 
            colors=["#43bccd", "#182128"],
            textprops={'fontsize': fontsize}  
        )
        
        centro = plt.Circle((0, 0), 0.70, color="#0e1117", fc="#0e1117")
        fig.gca().add_artist(centro)
        
        for text in texts:
            text.set_color("white") 
            text.set_fontsize(fontsize)  
        for autotext in autotexts:
            autotext.set_color("white")  
            autotext.set_fontsize(fontsize)  
                
        return fig
    else:
        return None


def grafico_04(df_resultado):
    if not df_resultado.empty:
        avaliacao_media = df_resultado["Avaliacao_Cliente"].mean()
        min_val, max_val = 0, 5

        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        angle = np.pi * (1 - (avaliacao_media - min_val) / (max_val - min_val))

        theta = np.linspace(np.pi, 0, 100)
        r = np.ones_like(theta) * 0.8
        ax.plot(theta, r, color='#48cae4', linewidth=8, solid_capstyle='butt')  
        ax.plot(theta, r * 0.8, color='#182128', linewidth=8, solid_capstyle='butt')  

        pointer_length = 0.7
        base_width = 0.1
        triangle = np.array([
            [angle, pointer_length],  
            [angle - base_width / 2, 0.1],  
            [angle + base_width / 2, 0.1],  
        ])
        ax.fill(triangle[:, 0], triangle[:, 1], color='white', zorder=5)

        ax.scatter([0], [0], color='white', s=300, zorder=6)  

        for i in range(min_val, max_val + 1):
            tick_angle = np.pi * (1 - (i - min_val) / (max_val - min_val))
            ax.text(tick_angle, 0.92, f"{i}", ha='center', va='center', fontsize=10, color='white')

        ax.text(0, -0.4, "Avaliação do Cliente", ha='center', fontsize=12, color='white')
        ax.set_ylim(0, 1)
        ax.axis('off')  

        plt.figtext(0.52, 0.3, f"{avaliacao_media:.1f}", ha='center', va='center', fontsize=24, color='white', fontweight='bold')

        return fig
    else:
        return None


def grafico_05(df_resultado):
    if not df_resultado.empty:
        ligacoes_por_dia = df_resultado.groupby("Dia")["Ligacoes_Realizadas"].sum().reset_index()

        fig, ax = plt.subplots(figsize=(12, 8))

        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        ax.fill_between(
            ligacoes_por_dia["Dia"],
            ligacoes_por_dia["Ligacoes_Realizadas"],
            color="violet",
            alpha=0.1,  
            label="Ligações Realizadas"
        )

        ax.plot(
            ligacoes_por_dia["Dia"],
            ligacoes_por_dia["Ligacoes_Realizadas"],
            color="violet",
            linewidth=2
        )

        ax.set_xlabel("Dia", color="white", fontsize=14, labelpad=20)
        ax.set_ylabel("Ligações Realizadas", color="white", fontsize=14, labelpad=15)

        ax.tick_params(axis='both', colors='white', labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)

        return fig
    else:
        return None



# Layout do Dashboard
colx, col1, cola, col4, coly = st.columns([0.15, 1, 0.5, 1, 0.3])

with col1:
    st.markdown("<div class='custom-container'><h3 class='centered' style='font-size:14px;'>Ligações Realizadas</h3></div>", unsafe_allow_html=True)
    grafico5 = grafico_05(df_resultado)
    if grafico5:
        st.pyplot(grafico5, use_container_width=True)
    else:
        st.write("Nenhum dado disponível para os filtros selecionados.")

with col4:
    st.markdown("<div class='custom-container'><h3 class='centered' style='font-size:14px;'>Dimensões de um Trabalho Saudável</h3></div>", unsafe_allow_html=True)
    grafico1 = grafico_01(df_resultado)
    if grafico1:
        st.pyplot(grafico1, use_container_width=True)
    else:
        st.write("Nenhum dado disponível para os filtros selecionados.")

colx, col3, col4, colz, col2, coly = st.columns([0.15, 0.5, 0.5, 0.5, 1.1, 0.3])

with col3:
    st.markdown("<div class='custom-container'><h3 class='centered' style='font-size:12px;'>Taxa de Conversão</h3></div>", unsafe_allow_html=True)
    grafico3 = grafico_03(df_resultado)
    if grafico3:
        st.pyplot(grafico3, use_container_width=True)
    else:
        st.write("Nenhum dado disponível para os filtros selecionados.")

with col4:
    st.markdown("<div class='custom-container'><h3 class='centered' style='font-size:12px;'>Avaliação do Atendimento</h3></div>", unsafe_allow_html=True)
    grafico4 = grafico_04(df_resultado)
    if grafico4:
        st.pyplot(grafico4, use_container_width=True)
    else:
        st.write("Nenhum dado disponível para os filtros selecionados.")

with col2:
    st.markdown("<div class='custom-container'><h3 class='centered' style='font-size:14px;'>Vendas dadas horas trabalhadas diárias</h3></div>", unsafe_allow_html=True)
    grafico2 = grafico_02(df_resultado)
    if grafico2:
        st.pyplot(grafico2, use_container_width=True)
    else:
        st.write("Nenhum dado disponível para os filtros selecionados.")