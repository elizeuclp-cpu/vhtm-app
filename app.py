import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="VHTM - Verificador de Hipóteses para Torres Metálicas",
    page_icon="🗼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🗼 VHTM - Verificador de Hipóteses para Torres Metálicas")
st.caption("Versão inicial - Hipótese 1: verificação da capacidade da mísula")
st.markdown("---")


# ==========================================================
# BANCO DE DADOS DOS CABOS CONDUTORES
#
# Unidades:
# peso  = kgf/m
# área  = mm²
# E     = kgf/mm²
# alpha = 1/°C
# CR    = kgf
# D     = m
# ==========================================================

CABOS_CONDUTORES = {
    "Grosbeak": {
        "peso": 1.3028,
        "area": 374.80,
        "E": 7593.0,
        "alpha": 189e-7,
        "CR": 11427.0,
        "D": 0.02515
    },

    "Linnet": {
        "peso": 0.6883,
        "area": 198.00,
        "E": 7593.0,
        "alpha": 189e-7,
        "CR": 6393.0,
        "D": 0.01830
    },

    "Penguin": {
        "peso": 0.4330,
        "area": 125.06,
        "E": 8120.0,
        "alpha": 186e-7,
        "CR": 3790.0,
        "D": 0.01431
    },

    "Cairo AAAC 6201": {
        "peso": 0.6510,
        "area": 236.38,
        "E": 8120.0,
        "alpha": 186e-7,
        "CR": 7106.3,
        "D": 0.01990
    },

    "Raven": {
        "peso": 0.2170,
        "area": 125.09,
        "E": 7593.0,
        "alpha": 189e-7,
        "CR": 1985.0,
        "D": 0.01011
    },

    "Flint": {
        "peso": 1.0350,
        "area": 374.52,
        "E": 8120.0,
        "alpha": 186e-7,
        "CR": 11012.8,
        "D": 0.02510
    },

    "Drake": {
        "peso": 1.6290,
        "area": 468.00,
        "E": 7593.0,
        "alpha": 189e-7,
        "CR": 14245.0,
        "D": 0.02813
    }
}


# ==========================================================
# BANCO DE DADOS DOS CABOS PARA-RAIOS
# ==========================================================

CABOS_PR = {
    'EHS 5/16"': {
        "peso": 0.305,
        "area": 38.32,
        "E": 18500.0,
        "alpha": 11.5e-6,
        "CR": 4900.0,
        "D": 0.00794
    },

    'EHS 3/8"': {
        "peso": 0.407,
        "area": 51.14,
        "E": 18500.0,
        "alpha": 11.5e-6,
        "CR": 6990.0,
        "D": 0.00952
    }
}


# ==========================================================
# FUNÇÕES DE CÁLCULO
# ==========================================================

def mudanca_estado(
    T_inicial,
    temp_inicial,
    temp_final,
    peso_inicial,
    peso_final,
    vao_medio,
    E,
    area,
    alpha
):
    """
    Calcula a tração final pela equação de mudança de estado.

    T_inicial: kgf
    peso_inicial: kgf/m
    peso_final: kgf/m
    vão médio: m
    E: kgf/mm²
    área: mm²
    """

    valores_positivos = [
        T_inicial,
        peso_inicial,
        peso_final,
        vao_medio,
        E,
        area
    ]

    if any(valor <= 0 for valor in valores_positivos):
        raise ValueError(
            "Tração inicial, pesos, vão médio, módulo de elasticidade "
            "e área devem ser maiores que zero."
        )

    B = (
        (
            E
            * area
            * peso_inicial ** 2
            * vao_medio ** 2
        )
        / (
            24.0
            * T_inicial ** 2
        )
        + E
        * area
        * alpha
        * (
            temp_final
            - temp_inicial
        )
        - T_inicial
    )

    C = (
        E
        * area
        * peso_final ** 2
        * vao_medio ** 2
    ) / 24.0

    raizes = np.roots(
        [
            1.0,
            B,
            0.0,
            -C
        ]
    )

    raizes_positivas = [
        raiz.real
        for raiz in raizes
        if abs(raiz.imag) < 1e-7
        and raiz.real > 0
    ]

    if not raizes_positivas:
        raise ValueError(
            "Não foi encontrada uma raiz física positiva "
            "na equação de mudança de estado."
        )

    return max(raizes_positivas)


def tracao_na_condicao_de_vento(
    cabo,
    percentual_cr,
    temp_eds,
    temp_vento,
    pressao_vento,
    vao_medio
):
    """
    Transforma a tração EDS para a condição de:

    - temperatura coincidente com vento máximo;
    - pressão de vento meteorológica do projeto.
    """

    T_eds = (
        cabo["CR"]
        * percentual_cr
        / 100.0
    )

    carga_vento_linear = (
        pressao_vento
        * cabo["D"]
    )

    peso_composto = np.sqrt(
        cabo["peso"] ** 2
        + carga_vento_linear ** 2
    )

    T_vento = mudanca_estado(
        T_inicial=T_eds,
        temp_inicial=temp_eds,
        temp_final=temp_vento,
        peso_inicial=cabo["peso"],
        peso_final=peso_composto,
        vao_medio=vao_medio,
        E=cabo["E"],
        area=cabo["area"],
        alpha=cabo["alpha"]
    )

    return {
        "T_eds": T_eds,
        "T_vento": T_vento,
        "carga_vento_linear": carga_vento_linear,
        "peso_composto": peso_composto
    }


def calcular_alfa_maximo(
    capacidade_transversal_util,
    soma_tracoes
):
    """
    Calcula o alfa máximo quando o vão de vento é igual a zero.

    Capacidade transversal =
        2 * soma das trações * sen(alfa / 2)
    """

    if soma_tracoes <= 0:
        raise ValueError(
            "A soma das trações deve ser maior que zero."
        )

    razao = (
        capacidade_transversal_util
        / (
            2.0
            * soma_tracoes
        )
    )

    if razao >= 1.0:
        return 180.0

    if razao <= 0.0:
        return 0.0

    alfa_maximo = np.degrees(
        2.0
        * np.arcsin(razao)
    )

    return float(alfa_maximo)


def montar_curva_vao_vento(
    capacidade_transversal_util,
    soma_tracoes,
    coeficiente_vento,
    alfa_maximo,
    passo_angular
):
    """
    Calcula o vão de vento máximo para cada deflexão.

    Fcabo =
        2 * soma(T) * sen(alfa/2)

    Fvento =
        P * soma(D) * VV

    VV máximo =
        (Capacidade transversal - Fcabo)
        / (P * soma(D))
    """

    if coeficiente_vento <= 0:
        raise ValueError(
            "O coeficiente de vento deve ser maior que zero."
        )

    alfa_limite = min(
        alfa_maximo,
        60.0
    )

    angulos = list(
        np.arange(
            0.0,
            alfa_limite + 1e-9,
            passo_angular
        )
    )

    # Inclui o ponto final exato quando o alfa máximo
    # não é múltiplo do passo angular.
    if (
        not angulos
        or not np.isclose(
            angulos[-1],
            alfa_limite
        )
    ):
        angulos.append(
            alfa_limite
        )

    dados = []

    for alfa in angulos:

        forca_cabo = (
            2.0
            * soma_tracoes
            * np.sin(
                np.radians(
                    alfa / 2.0
                )
            )
        )

        capacidade_disponivel_vento = (
            capacidade_transversal_util
            - forca_cabo
        )

        if capacidade_disponivel_vento < 0:
            capacidade_disponivel_vento = 0.0

        vao_vento_maximo = (
            capacidade_disponivel_vento
            / coeficiente_vento
        )

        dados.append(
            {
                "Deflexão (°)": alfa,
                "Força dos cabos (kgf)": forca_cabo,
                "Capacidade disponível para vento (kgf)":
                    capacidade_disponivel_vento,
                "Vão de vento máximo (m)":
                    vao_vento_maximo
            }
        )

    df = pd.DataFrame(
        dados
    )

    return df, alfa_limite


def formatar_numero(
    valor,
    casas=1
):
    """
    Formatação numérica no padrão brasileiro.
    """

    texto = f"{valor:,.{casas}f}"

    texto = (
        texto
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    return texto


# ==========================================================
# INTERFACE
# ==========================================================

coluna1, coluna2 = st.columns(
    2,
    gap="large"
)


# ==========================================================
# COLUNA 1
# ==========================================================

with coluna1:

    st.subheader(
        "1. Condutor"
    )

    nome_condutor = st.selectbox(
        "Tipo de condutor",
        list(
            CABOS_CONDUTORES.keys()
        ),
        index=1
    )

    cabo_condutor = (
        CABOS_CONDUTORES[
            nome_condutor
        ]
    )

    quantidade_fases = st.number_input(
        "Quantidade de fases sustentadas pela mísula analisada",
        min_value=1,
        max_value=12,
        value=1,
        step=1,
        help=(
            "Informe somente as fases efetivamente sustentadas "
            "pela mísula que está sendo verificada."
        )
    )

    cabos_por_fase = st.selectbox(
        "Cabos por fase",
        [
            1,
            2,
            3,
            4
        ],
        index=0
    )

    percentual_eds_condutor = st.number_input(
        "Tração EDS do condutor (%CR)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.1
    )

    st.markdown("---")

    st.subheader(
        "2. Cabo para-raios"
    )

    possui_pr = st.checkbox(
        "A mísula analisada sustenta cabo para-raios?",
        value=False
    )

    if possui_pr:

        nome_pr = st.selectbox(
            "Tipo de cabo para-raios",
            list(
                CABOS_PR.keys()
            )
        )

        cabo_pr = (
            CABOS_PR[
                nome_pr
            ]
        )

        quantidade_pr = st.number_input(
            "Quantidade de cabos para-raios sustentados pela mísula",
            min_value=1,
            max_value=4,
            value=1,
            step=1
        )

        percentual_eds_pr = st.number_input(
            "Tração EDS do para-raios (%CR)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.1
        )

    else:

        nome_pr = "Não aplicável"
        cabo_pr = None
        quantidade_pr = 0
        percentual_eds_pr = 0.0

    st.markdown("---")

    st.subheader(
        "3. Condição mecânica"
    )

    vao_medio = st.number_input(
        "Vão médio para mudança de estado (m)",
        min_value=1.0,
        max_value=2000.0,
        value=300.0,
        step=1.0
    )

    temp_eds = st.number_input(
        "Temperatura EDS (°C)",
        value=20.0,
        step=0.5
    )

    temp_vento = st.number_input(
        "Temperatura coincidente com vento máximo (°C)",
        value=15.0,
        step=0.5
    )


# ==========================================================
# COLUNA 2
# ==========================================================

with coluna2:

    st.subheader(
        "4. Parâmetros meteorológicos"
    )

    pressao_vento_projeto = st.number_input(
        "Pressão de vento prevista no projeto (kgf/m²)",
        min_value=0.1,
        max_value=300.0,
        value=30.0,
        step=1.0,
        help=(
            "Esta pressão será utilizada na mudança de estado "
            "e no cálculo do vão de vento."
        )
    )

    st.markdown("---")

    st.subheader(
        "5. Dados da hipótese da torre/mísula"
    )

    capacidade_vertical = st.number_input(
        "Capacidade vertical da mísula (kgf)",
        min_value=0.1,
        value=2000.0,
        step=50.0
    )

    capacidade_transversal = st.number_input(
        "Capacidade transversal máxima da mísula (kgf)",
        min_value=0.1,
        value=3000.0,
        step=50.0
    )

    pressao_vento_hipotese = st.number_input(
        "Pressão de vento da hipótese da torre (kgf/m²)",
        min_value=0.1,
        max_value=300.0,
        value=30.0,
        step=1.0,
        help=(
            "Utilizada somente para verificar a aplicabilidade "
            "da torre. Os cálculos sempre utilizam a pressão "
            "meteorológica prevista no projeto."
        )
    )

    st.markdown("---")

    st.subheader(
        "6. Fator de segurança"
    )

    fator_seguranca = st.number_input(
        "Fator de segurança",
        min_value=1.0,
        max_value=5.0,
        value=1.40,
        step=0.05,
        help=(
            "As capacidades vertical e transversal serão "
            "divididas por este fator."
        )
    )

    passo_angular = st.selectbox(
        "Passo da tabela de deflexão (°)",
        [
            0.5,
            1.0,
            2.0,
            5.0
        ],
        index=1
    )


# ==========================================================
# BOTÃO DE CÁLCULO
# ==========================================================

st.markdown("---")

calcular = st.button(
    "🔍 Calcular Hipótese 1",
    type="primary",
    use_container_width=True
)


# ==========================================================
# CÁLCULOS
# ==========================================================

if calcular:

    try:

        # ==================================================
        # VERIFICAÇÃO DA PRESSÃO DE VENTO
        # ==================================================

        if (
            pressao_vento_hipotese
            < pressao_vento_projeto
        ):

            st.error(
                "⚠️ A torre não pode ser aplicada para a "
                "pressão de vento prevista no projeto.\n\n"
                f"Pressão da hipótese da torre: "
                f"{pressao_vento_hipotese:.1f} kgf/m²\n\n"
                f"Pressão prevista no projeto: "
                f"{pressao_vento_projeto:.1f} kgf/m²\n\n"
                "Os cálculos abaixo continuam utilizando "
                "a pressão meteorológica prevista no projeto."
            )

        else:

            st.success(
                "✅ A pressão de vento da hipótese da torre "
                "é igual ou superior à pressão prevista no projeto."
            )

        # ==================================================
        # TRAÇÃO TRANSFORMADA DO CONDUTOR
        # ==================================================

        resultado_condutor = (
            tracao_na_condicao_de_vento(
                cabo=cabo_condutor,
                percentual_cr=percentual_eds_condutor,
                temp_eds=temp_eds,
                temp_vento=temp_vento,
                pressao_vento=pressao_vento_projeto,
                vao_medio=vao_medio
            )
        )

        T_eds_condutor = (
            resultado_condutor[
                "T_eds"
            ]
        )

        T_vento_condutor = (
            resultado_condutor[
                "T_vento"
            ]
        )

        carga_vento_linear_condutor = (
            resultado_condutor[
                "carga_vento_linear"
            ]
        )

        peso_composto_condutor = (
            resultado_condutor[
                "peso_composto"
            ]
        )

        quantidade_condutores = (
            int(
                quantidade_fases
            )
            * int(
                cabos_por_fase
            )
        )

        # ==================================================
        # TRAÇÃO TRANSFORMADA DO PARA-RAIOS
        # ==================================================

        if possui_pr:

            resultado_pr = (
                tracao_na_condicao_de_vento(
                    cabo=cabo_pr,
                    percentual_cr=percentual_eds_pr,
                    temp_eds=temp_eds,
                    temp_vento=temp_vento,
                    pressao_vento=pressao_vento_projeto,
                    vao_medio=vao_medio
                )
            )

            T_eds_pr = (
                resultado_pr[
                    "T_eds"
                ]
            )

            T_vento_pr = (
                resultado_pr[
                    "T_vento"
                ]
            )

            carga_vento_linear_pr = (
                resultado_pr[
                    "carga_vento_linear"
                ]
            )

            peso_composto_pr = (
                resultado_pr[
                    "peso_composto"
                ]
            )

        else:

            T_eds_pr = 0.0
            T_vento_pr = 0.0
            carga_vento_linear_pr = 0.0
            peso_composto_pr = 0.0

        # ==================================================
        # CAPACIDADES APÓS FATOR DE SEGURANÇA
        # ==================================================

        capacidade_vertical_util = (
            capacidade_vertical
            / fator_seguranca
        )

        capacidade_transversal_util = (
            capacidade_transversal
            / fator_seguranca
        )

        # ==================================================
        # VÃO GRAVANTE MÁXIMO
        # ==================================================

        peso_linear_condutores = (
            quantidade_condutores
            * cabo_condutor["peso"]
        )

        if possui_pr:

            peso_linear_para_raios = (
                int(
                    quantidade_pr
                )
                * cabo_pr["peso"]
            )

        else:

            peso_linear_para_raios = 0.0

        peso_linear_total = (
            peso_linear_condutores
            + peso_linear_para_raios
        )

        if peso_linear_total <= 0:

            raise ValueError(
                "O peso linear total deve ser maior que zero."
            )

        vao_gravante_maximo = (
            capacidade_vertical_util
            / peso_linear_total
        )

        # ==================================================
        # SOMA DAS TRAÇÕES TRANSFORMADAS
        # ==================================================

        soma_tracoes_condutores = (
            quantidade_condutores
            * T_vento_condutor
        )

        if possui_pr:

            soma_tracoes_para_raios = (
                int(
                    quantidade_pr
                )
                * T_vento_pr
            )

        else:

            soma_tracoes_para_raios = 0.0

        soma_tracoes_vento = (
            soma_tracoes_condutores
            + soma_tracoes_para_raios
        )

        # ==================================================
        # SOMA DOS DIÂMETROS
        # ==================================================

        soma_diametros_condutores = (
            quantidade_condutores
            * cabo_condutor["D"]
        )

        if possui_pr:

            soma_diametros_para_raios = (
                int(
                    quantidade_pr
                )
                * cabo_pr["D"]
            )

        else:

            soma_diametros_para_raios = 0.0

        soma_diametros = (
            soma_diametros_condutores
            + soma_diametros_para_raios
        )

        # ==================================================
        # COEFICIENTE DE VENTO
        # ==================================================

        coeficiente_vento = (
            pressao_vento_projeto
            * soma_diametros
        )

        # ==================================================
        # ALFA MÁXIMO
        # ==================================================

        alfa_maximo = (
            calcular_alfa_maximo(
                capacidade_transversal_util=
                    capacidade_transversal_util,
                soma_tracoes=
                    soma_tracoes_vento
            )
        )

        # ==================================================
        # CURVA DO VÃO DE VENTO
        # ==================================================

        df_curva, alfa_limite_grafico = (
            montar_curva_vao_vento(
                capacidade_transversal_util=
                    capacidade_transversal_util,
                soma_tracoes=
                    soma_tracoes_vento,
                coeficiente_vento=
                    coeficiente_vento,
                alfa_maximo=
                    alfa_maximo,
                passo_angular=
                    passo_angular
            )
        )

        # ==================================================
        # RESULTADOS PRINCIPAIS
        # ==================================================

        st.markdown("---")

        st.subheader(
            "📊 Resultados da Hipótese 1"
        )

        resultado1, resultado2, resultado3, resultado4 = (
            st.columns(4)
        )

        with resultado1:

            st.metric(
                "Tração transformada do condutor",
                (
                    f"{formatar_numero(T_vento_condutor)} kgf"
                ),
                (
                    f"EDS: "
                    f"{formatar_numero(T_eds_condutor)} kgf"
                )
            )

        with resultado2:

            st.metric(
                "Vão gravante máximo",
                (
                    f"{formatar_numero(vao_gravante_maximo)} m"
                )
            )

        with resultado3:

            st.metric(
                "Alfa máximo teórico",
                (
                    f"{formatar_numero(alfa_maximo, 2)}°"
                ),
                "Vão de vento = 0 m"
            )

        with resultado4:

            st.metric(
                "Limite do gráfico",
                (
                    f"{formatar_numero(alfa_limite_grafico, 2)}°"
                ),
                "Menor entre alfa máximo e 60°"
            )

        if possui_pr:

            st.info(
                f"Tração transformada do para-raios "
                f"{nome_pr}: "
                f"{formatar_numero(T_vento_pr)} kgf por cabo."
            )

        if alfa_maximo > 60.0:

            st.info(
                "ℹ️ O alfa máximo teórico é superior a 60°. "
                "Conforme o critério definido, a tabela e o "
                "gráfico foram limitados a 60°."
            )

        # ==================================================
        # TABELA DAS TRAÇÕES
        # ==================================================

        st.markdown("---")

        st.subheader(
            "📋 Trações transformadas"
        )

        dados_tracoes = [
            {
                "Cabo": nome_condutor,
                "Tipo": "Condutor",
                "Quantidade": quantidade_condutores,
                "Tração EDS por cabo (kgf)":
                    T_eds_condutor,
                "Tração com vento por cabo (kgf)":
                    T_vento_condutor,
                "Peso próprio (kgf/m)":
                    cabo_condutor["peso"],
                "Carga de vento linear (kgf/m)":
                    carga_vento_linear_condutor,
                "Peso composto (kgf/m)":
                    peso_composto_condutor
            }
        ]

        if possui_pr:

            dados_tracoes.append(
                {
                    "Cabo": nome_pr,
                    "Tipo": "Para-raios",
                    "Quantidade": int(
                        quantidade_pr
                    ),
                    "Tração EDS por cabo (kgf)":
                        T_eds_pr,
                    "Tração com vento por cabo (kgf)":
                        T_vento_pr,
                    "Peso próprio (kgf/m)":
                        cabo_pr["peso"],
                    "Carga de vento linear (kgf/m)":
                        carga_vento_linear_pr,
                    "Peso composto (kgf/m)":
                        peso_composto_pr
                }
            )

        df_tracoes = pd.DataFrame(
            dados_tracoes
        )

        st.dataframe(
            df_tracoes,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Tração EDS por cabo (kgf)":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    ),

                "Tração com vento por cabo (kgf)":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    ),

                "Peso próprio (kgf/m)":
                    st.column_config.NumberColumn(
                        format="%.4f"
                    ),

                "Carga de vento linear (kgf/m)":
                    st.column_config.NumberColumn(
                        format="%.4f"
                    ),

                "Peso composto (kgf/m)":
                    st.column_config.NumberColumn(
                        format="%.4f"
                    )
            }
        )

        # ==================================================
        # MEMÓRIA DE CÁLCULO
        # ==================================================

        st.markdown("---")

        with st.expander(
            "📐 Memória resumida de cálculo",
            expanded=False
        ):

            parametros = [
                {
                    "Grandeza":
                        "Quantidade total de condutores",
                    "Valor":
                        quantidade_condutores,
                    "Unidade":
                        "un"
                },

                {
                    "Grandeza":
                        "Quantidade de para-raios",
                    "Valor":
                        int(quantidade_pr),
                    "Unidade":
                        "un"
                },

                {
                    "Grandeza":
                        "Peso linear dos condutores",
                    "Valor":
                        peso_linear_condutores,
                    "Unidade":
                        "kgf/m"
                },

                {
                    "Grandeza":
                        "Peso linear dos para-raios",
                    "Valor":
                        peso_linear_para_raios,
                    "Unidade":
                        "kgf/m"
                },

                {
                    "Grandeza":
                        "Peso linear total",
                    "Valor":
                        peso_linear_total,
                    "Unidade":
                        "kgf/m"
                },

                {
                    "Grandeza":
                        "Soma dos diâmetros expostos",
                    "Valor":
                        soma_diametros,
                    "Unidade":
                        "m"
                },

                {
                    "Grandeza":
                        "Coeficiente de vento P × ΣD",
                    "Valor":
                        coeficiente_vento,
                    "Unidade":
                        "kgf/m"
                },

                {
                    "Grandeza":
                        "Soma das trações transformadas",
                    "Valor":
                        soma_tracoes_vento,
                    "Unidade":
                        "kgf"
                },

                {
                    "Grandeza":
                        "Capacidade vertical informada",
                    "Valor":
                        capacidade_vertical,
                    "Unidade":
                        "kgf"
                },

                {
                    "Grandeza":
                        "Capacidade vertical após FS",
                    "Valor":
                        capacidade_vertical_util,
                    "Unidade":
                        "kgf"
                },

                {
                    "Grandeza":
                        "Capacidade transversal informada",
                    "Valor":
                        capacidade_transversal,
                    "Unidade":
                        "kgf"
                },

                {
                    "Grandeza":
                        "Capacidade transversal após FS",
                    "Valor":
                        capacidade_transversal_util,
                    "Unidade":
                        "kgf"
                },

                {
                    "Grandeza":
                        "Pressão utilizada nos cálculos",
                    "Valor":
                        pressao_vento_projeto,
                    "Unidade":
                        "kgf/m²"
                },

                {
                    "Grandeza":
                        "Fator de segurança",
                    "Valor":
                        fator_seguranca,
                    "Unidade":
                        "-"
                }
            ]

            df_parametros = pd.DataFrame(
                parametros
            )

            st.dataframe(
                df_parametros,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Valor":
                        st.column_config.NumberColumn(
                            format="%.4f"
                        )
                }
            )

            st.markdown(
                "**Força transversal dos cabos:**"
            )

            st.latex(
                r"""
                F_{cabo}
                =
                2
                \cdot
                \sum T_{vento}
                \cdot
                \sin
                \left(
                \frac{\alpha}{2}
                \right)
                """
            )

            st.markdown(
                "**Força de vento nos cabos:**"
            )

            st.latex(
                r"""
                F_{vento}
                =
                P
                \cdot
                \sum D
                \cdot
                VV
                """
            )

            st.markdown(
                "**Vão de vento máximo:**"
            )

            st.latex(
                r"""
                VV_{max}
                =
                \frac{
                T_{transversal,\ util}
                -
                F_{cabo}
                }{
                P
                \cdot
                \sum D
                }
                """
            )

            st.markdown(
                "**Vão gravante máximo:**"
            )

            st.latex(
                r"""
                VG_{max}
                =
                \frac{
                V_{vertical,\ util}
                }{
                \sum peso_{linear}
                }
                """
            )

            st.markdown(
                "**Alfa máximo, considerando VV = 0:**"
            )

            st.latex(
                r"""
                \alpha_{max}
                =
                2
                \cdot
                \arcsin
                \left(
                \frac{
                T_{transversal,\ util}
                }{
                2
                \cdot
                \sum T_{vento}
                }
                \right)
                """
            )

        # ==================================================
        # GRÁFICO
        # ==================================================

        st.markdown("---")

        st.subheader(
            "📈 Curva do vão de vento máximo"
        )

        fig, ax = plt.subplots(
            figsize=(
                10,
                5.5
            )
        )

        mostrar_marcadores = (
            len(
                df_curva
            )
            <= 31
        )

        ax.plot(
            df_curva[
                "Deflexão (°)"
            ],
            df_curva[
                "Vão de vento máximo (m)"
            ],
            marker=(
                "o"
                if mostrar_marcadores
                else None
            ),
            linewidth=2,
            color="blue"
        )

        ax.fill_between(
            df_curva[
                "Deflexão (°)"
            ],
            df_curva[
                "Vão de vento máximo (m)"
            ],
            0,
            alpha=0.12,
            color="blue"
        )

        ax.set_title(
            "Vão de vento máximo × deflexão",
            fontsize=13,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Deflexão externa (°)"
        )

        ax.set_ylabel(
            "Vão de vento máximo (m)"
        )

        ax.set_xlim(
            left=0,
            right=max(
                alfa_limite_grafico,
                0.1
            )
        )

        ax.set_ylim(
            bottom=0
        )

        ax.grid(
            True,
            alpha=0.3
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(
            fig
        )

        # ==================================================
        # TABELA POR DEFLEXÃO
        # ==================================================

        st.markdown("---")

        st.subheader(
            "📊 Tabela do vão de vento por deflexão"
        )

        st.dataframe(
            df_curva,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Deflexão (°)":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    ),

                "Força dos cabos (kgf)":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    ),

                "Capacidade disponível para vento (kgf)":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    ),

                "Vão de vento máximo (m)":
                    st.column_config.NumberColumn(
                        format="%.2f"
                    )
            }
        )

        # ==================================================
        # DOWNLOAD DA TABELA
        # ==================================================

        csv = (
            df_curva
            .to_csv(
                index=False,
                sep=";",
                decimal=","
            )
            .encode(
                "utf-8-sig"
            )
        )

        st.download_button(
            "📥 Baixar tabela em CSV",
            data=csv,
            file_name=(
                "VHTM_hipotese_1_"
                "vao_vento_deflexao.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

        # ==================================================
        # CONCLUSÃO
        # ==================================================

        st.markdown("---")

        st.subheader(
            "🎯 Conclusão"
        )

        coluna_conclusao1, coluna_conclusao2, coluna_conclusao3 = (
            st.columns(3)
        )

        with coluna_conclusao1:

            st.metric(
                "Vão gravante máximo",
                (
                    f"{formatar_numero(vao_gravante_maximo)} m"
                )
            )

        with coluna_conclusao2:

            st.metric(
                "Deflexão máxima teórica",
                (
                    f"{formatar_numero(alfa_maximo, 2)}°"
                )
            )

        with coluna_conclusao3:

            if (
                pressao_vento_hipotese
                >= pressao_vento_projeto
            ):

                st.metric(
                    "Aplicação por pressão de vento",
                    "ATENDE"
                )

            else:

                st.metric(
                    "Aplicação por pressão de vento",
                    "NÃO ATENDE"
                )

    except Exception as erro:

        st.error(
            f"❌ Erro no cálculo: {erro}"
        )

        st.exception(
            erro
        )


# ==========================================================
# RODAPÉ
# ==========================================================

st.markdown("---")

st.caption(
    "Escopo desta versão: Hipótese 1 e verificação da capacidade "
    "da mísula. As capacidades informadas devem ser compatíveis "
    "com as unidades e com o critério estrutural adotado."
)
