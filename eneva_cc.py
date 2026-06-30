import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import CoolProp.CoolProp as CP

# **PAGE CONFIG**

st.session_state.update(st.session_state)
for k, v in st.session_state.items():
    st.session_state[k] = v

from PIL import Image
import os
path = os.path.dirname(__file__)
my_file = path + '/e_logo.png'
img = Image.open(my_file)

st.set_page_config(
    page_title='Eneva Ciclo Combinado',
    layout="wide",
    page_icon=img
                   )

hide_menu = '''
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        '''
st.markdown(hide_menu, unsafe_allow_html=True)


# **FUNÇÕES**

## Função para calcular propriedades termodinâmicas pós combustão

# Calcula o estado termodinâmico completo de uma mistura de gases
# pós-combustão de CH4 com ar, dado T e P.
#
# Propriedades calculadas (via CoolProp, mistura de gases ideais):
#     cp, cv, k, h, s, ρ, μ, λ_cond, Pr
#
# Dependências: CoolProp, scipy, numpy
#     pip install CoolProp scipy numpy
#
# Referência de H e S: estado de cada componente a T=298.15 K, P=101325 Pa.
# Δh e Δs entre dois estados são termoconsistentes pois a referência cancela.

# ── Constantes globais ────────────────────────────────────────────────────────
R     = 8.314    # J/mol·K — constante dos gases ideais
T_REF = 298.15   # K  — temperatura de referência para h e s (USAR DO COOLPROP)
P_REF = 101325   # Pa — pressão de referência (USAR DO COOLPROP)

# Massas molares dos componentes (g/mol) — cacheadas para evitar chamadas repetidas
_FLUIDS = ('CarbonDioxide', 'Water', 'Oxygen', 'Nitrogen')
_M = {f: CP.PropsSI('M', 'T', 300, 'P', 101325, f) * 1000 for f in _FLUIDS}


# 1. ESTEQUIOMETRIA

def combustao_ch4(AFR: float) -> tuple[dict, float]:

    # Estequiometria da combustão completa de CH4 com ar seco.
    #
    # Reação:
    #     CH4 + 2λ·O2 + 2λ·(79/21)·N2 → CO2 + 2H2O + 2(λ-1)·O2 + 2λ·(79/21)·N2
    #
    # Parâmetros
    # ----------
    # AFR : float
    #     Relação ar/combustível mássica [kg ar / kg CH4].
    #     Estequiométrico ≈ 17.2. Turbinas a gás: tipicamente 50–70.
    #
    # Retorna
    # -------
    # fracs : dict {fluid_name: mole_fraction}
    # lam   : float — fator de excesso de ar (λ = 1 → estequiométrico)

    M_CH4, M_ar = 16.043, 28.966
    n_ar     = (AFR * M_CH4) / M_ar     # mol ar / mol CH4
    n_O2_in  = 0.21 * n_ar
    n_N2_in  = 0.79 * n_ar
    lam      = n_O2_in / 2.0

    if lam < 1.0:
        raise ValueError(
            f"λ={lam:.3f} < 1: combustão incompleta. Aumente o AFR (mínimo ≈ 17.2)."
        )

    n_CO2 = 1.0
    n_H2O = 2.0
    n_O2  = n_O2_in - 2.0   # O2 excedente
    n_N2  = n_N2_in
    n_tot = n_CO2 + n_H2O + n_O2 + n_N2

    fracs = {
        'CarbonDioxide': n_CO2 / n_tot,
        'Water':         n_H2O / n_tot,
        'Oxygen':        n_O2  / n_tot,
        'Nitrogen':      n_N2  / n_tot,
    }
    return fracs, round(lam, 4)


# 2. ESTADO TERMODINÂMICO COMPLETO

def estado_mistura(
    T: float,
    P: float,
    fracs: dict,
) -> dict:

    # Estado termodinâmico completo de uma mistura de gases ideais.
    #
    # Parâmetros
    # ----------
    # T     : float — temperatura [K]
    # P     : float — pressão [Pa]
    # fracs : dict  — {fluid_name: mole_fraction}, deve somar 1.
    #                 Nomes aceitos: 'CarbonDioxide', 'Water', 'Oxygen', 'Nitrogen'
    #
    # Retorna
    # -------
    # dict com as seguintes propriedades (todas na base mássica):
    #
    #     T_K          : temperatura de entrada [K]
    #     P_Pa         : pressão de entrada [Pa]
    #     M_mix_g/mol  : massa molar da mistura [g/mol]
    #
    #     cp_kJ/kgK    : calor específico a pressão constante [kJ/kg·K]
    #     cv_kJ/kgK    : calor específico a volume constante  [kJ/kg·K]
    #     k_cp/cv      : razão de calores específicos (−) — "gamma"
    #
    #     h_kJ/kg      : entalpia específica relativa à referência (T=298.15K, P=1atm) [kJ/kg]
    #     s_kJ/kgK     : entropia específica relativa à referência (inclui term. de mistura) [kJ/kg·K]
    #
    #     rho_kg/m3    : massa específica [kg/m³]
    #     mu_Pa.s      : viscosidade dinâmica [Pa·s]  — Wilke simplificado
    #     lambda_W/mK  : condutividade térmica [W/m·K] — Wilke simplificado
    #     Pr           : número de Prandtl [−]
    #
    # Notas
    # -----
    # • h e s são RELATIVOS à referência — use apenas diferenças (Δh, Δs) para
    #   cálculos de trabalho e calor. Diferenças são termoconsistentes.
    # • Pressão parcial de cada componente: Pᵢ = yᵢ · P (lei de Dalton).
    # • Viscosidade e condutividade via método de Wilke simplificado
    #   (boa aproximação para misturas de gases de propriedades similares).

    # Massa molar da mistura [g/mol]
    M_mix = sum(y * _M[f] for f, y in fracs.items())
    Mm    = M_mix / 1000    # kg/mol

    # Acumuladores na base molar
    cp_mol  = 0.0
    cv_mol  = 0.0
    h_mol   = 0.0
    s_mol   = 0.0
    V_mol   = 0.0   # volume molar da mistura [m³/mol] via volumes parciais
    visc_n  = 0.0   # numerador Wilke
    cond_n  = 0.0
    wilke_d = 0.0   # denominador Wilke = Σ yᵢ·√Mᵢ

    for fluid, y in fracs.items():
        Mi = _M[fluid]          # g/mol
        Pi = y * P              # pressão parcial [Pa] (Lei de Dalton)

        # ── Propriedades do componente puro via CoolProp ──────────────────
        cp_i    = CP.PropsSI('Cpmolar',      'T', T,     'P', Pi,    fluid)  # J/mol·K
        cv_i    = CP.PropsSI('Cvmolar',      'T', T,     'P', Pi,    fluid)  # J/mol·K
        h_i     = CP.PropsSI('Hmolar',       'T', T,     'P', Pi,    fluid)  # J/mol
        h_ref_i = CP.PropsSI('Hmolar',       'T', T_REF, 'P', P_REF, fluid)  # J/mol
        s_i     = CP.PropsSI('Smolar',       'T', T,     'P', Pi,    fluid)  # J/mol·K
        s_ref_i = CP.PropsSI('Smolar',       'T', T_REF, 'P', P_REF, fluid)  # J/mol·K
        rho_i   = CP.PropsSI('D',            'T', T,     'P', Pi,    fluid)  # kg/m³
        mu_i    = CP.PropsSI('viscosity',    'T', T,     'P', Pi,    fluid)  # Pa·s
        lam_i   = CP.PropsSI('conductivity', 'T', T,     'P', Pi,    fluid)  # W/m·K

        # ── Contribuição molar (gases ideais: propriedades aditivas) ──────
        cp_mol += y * cp_i
        cv_mol += y * cv_i

        # Δh relativo à referência [J/mol]
        h_mol  += y * (h_i - h_ref_i)

        # Δs relativo à ref + entropia de mistura (−R·ln yᵢ) [J/mol·K]
        s_mol  += y * ((s_i - s_ref_i) - R * np.log(y))

        # Volume parcial [m³/mol] → lei dos volumes aditivos
        V_mol  += y * (Mi / 1000) / rho_i

        # Wilke: numerador e denominador
        sq_Mi   = Mi ** 0.5
        visc_n += y * mu_i  * sq_Mi
        cond_n += y * lam_i * sq_Mi
        wilke_d += y * sq_Mi

    # ── Propriedades finais ───────────────────────────────────────────────
    rho    = Mm / V_mol              # kg/m³
    mu     = visc_n / wilke_d        # Pa·s
    lam_th = cond_n / wilke_d        # W/m·K
    cp_kg  = cp_mol / M_mix          # kJ/kg·K
    cv_kg  = cv_mol / M_mix          # kJ/kg·K
    k_gam  = cp_mol / cv_mol         # k = cp/cv

    return {
        'T_K':           round(T,                     2),
        'P_Pa':          round(P,                     1),
        'M_mix_g/mol':   round(M_mix,                 4),

        'cp_kJ/kgK':     round(cp_kg,                 5),
        'cv_kJ/kgK':     round(cv_kg,                 5),
        'k_cp/cv':       round(k_gam,                 5),

        'h_kJ/kg':       round(h_mol / Mm / 1000,     3),
        's_kJ/kgK':      round(s_mol / Mm / 1000,     5),

        'rho_kg/m3':     round(rho,                   5),
        'mu_Pa.s':       round(mu,                    9),
        'lambda_W/mK':   round(lam_th,                6),
        'Pr':            round(cp_kg * 1000 * mu / lam_th, 4),
    }


# 3. FUNÇÃO DE CONVENIÊNCIA — estado + composição de uma vez

def estado_pos_combustao(T: float, P: float, AFR: float = 60.0) -> dict:

    # Atalho: calcula estequiometria + estado completo em uma chamada.
    #
    # Parâmetros
    # ----------
    # T   : float — temperatura [K]
    # P   : float — pressão [Pa]
    # AFR : float — relação ar/combustível mássica [kg ar / kg CH4], default 60
    #
    # Retorna
    # -------
    # dict com tudo de estado_mistura() mais:
    #     'AFR'     : AFR de entrada
    #     'lambda'  : fator de excesso de ar
    #     'fracs'   : frações molares dos produtos

    fracs, lam = combustao_ch4(AFR)
    estado = estado_mistura(T, P, fracs)
    return {'AFR': AFR, 'lambda': lam, 'fracs': fracs, **estado}

# EXEMPLO

# Entrada da turbina (estado 3)
e3 = estado_pos_combustao(T=1500, P=1200000, AFR=60)

# Saída da turbina (estado 4) — você já calculou T4=829K
e4 = estado_pos_combustao(T=829, P=101325, AFR=60)

# Trabalho real da turbina
w_turbina = e3['h_kJ/kg'] - e4['h_kJ/kg']   # kJ/kg

# st.write(e3)  # → 824.6 kJ/kg
# st.write(e4)  # → 824.6 kJ/kg
#
# st.write(w_turbina)  # → 824.6 kJ/kg

# **FUNC PLOTS**
R = 8.314
T_REF = 298.15
P_REF = 101325
_FLUIDS = ('CarbonDioxide', 'Water', 'Oxygen', 'Nitrogen')
_M = {f: CP.PropsSI('M', 'T', 300, 'P', 101325, f) * 1000 for f in _FLUIDS}


def _combustao_ch4(AFR: float) -> dict:
    M_CH4, M_ar = 16.043, 28.966
    n_ar = (AFR * M_CH4) / M_ar
    n_O2_in = 0.21 * n_ar
    n_N2_in = 0.79 * n_ar
    n_CO2 = 1.0;
    n_H2O = 2.0
    n_O2 = n_O2_in - 2.0
    n_N2 = n_N2_in
    n_tot = n_CO2 + n_H2O + n_O2 + n_N2
    return {
        'CarbonDioxide': n_CO2 / n_tot,
        'Water': n_H2O / n_tot,
        'Oxygen': n_O2 / n_tot,
        'Nitrogen': n_N2 / n_tot,
    }


def _estado_mistura(T: float, P: float, fracs: dict) -> dict:
    M_mix = sum(y * _M[f] for f, y in fracs.items())
    Mm = M_mix / 1000
    cp_mol = cv_mol = h_mol = s_mol = V_mol = 0.0
    visc_n = cond_n = wilke_d = 0.0
    for fluid, y in fracs.items():
        Mi = _M[fluid]
        Pi = y * P
        cp_i = CP.PropsSI('Cpmolar', 'T', T, 'P', Pi, fluid)
        cv_i = CP.PropsSI('Cvmolar', 'T', T, 'P', Pi, fluid)
        h_i = CP.PropsSI('Hmolar', 'T', T, 'P', Pi, fluid)
        h_ref = CP.PropsSI('Hmolar', 'T', T_REF, 'P', P_REF, fluid)
        s_i = CP.PropsSI('Smolar', 'T', T, 'P', Pi, fluid)
        s_ref = CP.PropsSI('Smolar', 'T', T_REF, 'P', P_REF, fluid)
        rho_i = CP.PropsSI('D', 'T', T, 'P', Pi, fluid)
        mu_i = CP.PropsSI('viscosity', 'T', T, 'P', Pi, fluid)
        lam_i = CP.PropsSI('conductivity', 'T', T, 'P', Pi, fluid)
        cp_mol += y * cp_i
        cv_mol += y * cv_i
        h_mol += y * (h_i - h_ref)
        s_mol += y * ((s_i - s_ref) - R * np.log(y))
        V_mol += y * (Mi / 1000) / rho_i
        sq_Mi = Mi ** 0.5
        visc_n += y * mu_i * sq_Mi
        cond_n += y * lam_i * sq_Mi
        wilke_d += y * sq_Mi
    rho = Mm / V_mol
    mu = visc_n / wilke_d
    lam_th = cond_n / wilke_d
    cp_kg = cp_mol / M_mix
    k_gam = cp_mol / cv_mol
    return {
        'cp_kJ/kgK': round(cp_kg, 5),
        'k_cp/cv': round(k_gam, 5),
        'h_kJ/kg': round(h_mol / Mm / 1000, 3),
        's_kJ/kgK': round(s_mol / Mm / 1000, 5),
    }


## PLOT BRAYTON FUNÇÃO

def plot_brayton(
        states: dict,
        e3: dict,
        e4: dict,
        T2s: float,
        T4s: float,
        AFR: float,
        T1: float,
        T2: float,
        T3: float,
        T4: float,
        P1: float,
        P2: float,
        n_c: float,
        n_t: float,
) -> tuple:

    # Gera os diagramas T-s e P-h do ciclo Brayton em Plotly.
    #
    # Parâmetros
    # ----------
    # states : dict com chaves 1,2,3,4 → {'T','P','h','s','label'}
    #          h em kJ/kg, s em kJ/kg·K
    # e3, e4 : dicts retornados por estado_pos_combustao()
    # T2s    : temperatura na saída do compressor isentrópico [K]
    # T4s    : temperatura na saída da turbina isentrópica [K]
    # AFR    : relação ar/combustível convergida [kg ar / kg CH4]
    # T1..T4 : temperaturas dos 4 estados [K]
    # P1, P2 : pressão baixa e alta [Pa]
    # n_c    : eficiência isentrópica do compressor [0-1]
    # n_t    : eficiência isentrópica da turbina a gás [0-1]
    #
    # Retorna
    # -------
    # fig_ts : go.Figure — diagrama T-s
    # fig_ph : go.Figure — diagrama P-h


    NAVY = "#7EB8D4"
    TEAL = "#4DD0E1"
    ORANGE = "#F26B3A"
    GREEN = "#66BB6A"

    N = 300  # pontos nas isobaras

    # ── Isobaras (ar puro — boa aproximação visual) ──────────────────────
    T_low = np.linspace(T1, T4, N)
    T_high = np.linspace(T2, T3, N)

    s_low = np.array([CP.PropsSI('S', 'T', T, 'P', P1, "Air") for T in T_low]) / 1000
    s_high = np.array([CP.PropsSI('S', 'T', T, 'P', P2, "Air") for T in T_high]) / 1000
    h_low = np.array([CP.PropsSI('H', 'T', T, 'P', P1, "Air") for T in T_low]) / 1000
    h_high = np.array([CP.PropsSI('H', 'T', T, 'P', P2, "Air") for T in T_high]) / 1000

    # ── Coordenadas dos estados ──────────────────────────────────────────
    s1, s2, s3, s4 = (states[i]['s'] for i in [1, 2, 3, 4])
    h1, h2, h3, h4 = (states[i]['h'] for i in [1, 2, 3, 4])
    p1_bar = P1 / 1e5
    p2_bar = P2 / 1e5

    # Pontos isentrópicos ideais
    s2s = CP.PropsSI('S', 'T', T2s, 'P', P2, "Air") / 1000
    h2s = CP.PropsSI('H', 'T', T2s, 'P', P2, "Air") / 1000
    fracs_4s = _combustao_ch4(AFR)
    e4s = _estado_mistura(T4s, P1, fracs_4s)
    s4s = e4s['s_kJ/kgK']

    # Cores e rótulos dos processos
    colors_seg = [NAVY, ORANGE, GREEN, TEAL]
    labels_seg = ['1→2 Compressão', '2→3 Combustão',
                  '3→4 Expansão', '4→1 Rejeição de calor']

    # DIAGRAMA T-s
    fig_ts = go.Figure()

    # Isobaras de referência
    fig_ts.add_trace(go.Scatter(
        x=s_low, y=T_low, mode='lines',
        line=dict(color=TEAL, width=1.5, dash='dot'),
        name=f'Isobara P₁ = {p1_bar:.2f} bar',
    ))
    fig_ts.add_trace(go.Scatter(
        x=s_high, y=T_high, mode='lines',
        line=dict(color=ORANGE, width=1.5, dash='dot'),
        name=f'Isobara P₂ = {p2_bar:.2f} bar',
    ))

    # Processos reais
    T_pts = [T1, T2, T3, T4, T1]
    s_pts = [s1, s2, s3, s4, s1]
    for i in range(4):
        fig_ts.add_trace(go.Scatter(
            x=[s_pts[i], s_pts[i + 1]],
            y=[T_pts[i], T_pts[i + 1]],
            mode='lines',
            line=dict(color=colors_seg[i], width=2.5),
            name=labels_seg[i],
        ))

    # Isentrópicos ideais (tracejado)
    fig_ts.add_trace(go.Scatter(
        x=[s1, s2s], y=[T1, T2s], mode='lines',
        line=dict(color=NAVY, width=1.5, dash='dash'),
        name='1→2s (ideal)',
    ))
    fig_ts.add_trace(go.Scatter(
        x=[s3, s4s], y=[T3, T4s], mode='lines',
        line=dict(color=GREEN, width=1.5, dash='dash'),
        name='3→4s (ideal)',
    ))

    # Pontos (reais + ideais)
    pts_s = [s1, s2s, s2, s3, s4s, s4]
    pts_T = [T1, T2s, T2, T3, T4s, T4]
    pts_lbl = ['1', '2s', '2', '3', '4s', '4']
    pts_clr = [TEAL, NAVY, NAVY, ORANGE, GREEN, GREEN]
    pts_pos = ['bottom left', 'top left', 'top right',
               'top right', 'bottom right', 'bottom right']
    pts_hov = [
        f"<b>Estado 1</b><br>T = {T1:.0f} K<br>s = {s1:.4f} kJ/kg·K",
        f"<b>Estado 2s</b> (ideal)<br>T = {T2s:.1f} K<br>s = {s2s:.4f} kJ/kg·K",
        f"<b>Estado 2</b> (real)<br>T = {T2:.1f} K<br>s = {s2:.4f} kJ/kg·K",
        f"<b>Estado 3</b><br>T = {T3:.0f} K<br>s = {s3:.4f} kJ/kg·K",
        f"<b>Estado 4s</b> (ideal)<br>T = {T4s:.1f} K<br>s = {s4s:.4f} kJ/kg·K",
        f"<b>Estado 4</b> (real)<br>T = {T4:.1f} K<br>s = {s4:.4f} kJ/kg·K",
    ]
    fig_ts.add_trace(go.Scatter(
        x=pts_s, y=pts_T,
        mode='markers+text',
        marker=dict(size=10, color=pts_clr, line=dict(width=2, color='white')),
        text=pts_lbl,
        textposition=pts_pos,
        textfont=dict(size=13, color=NAVY),
        hovertext=pts_hov,
        hoverinfo='text',
        name='Estados',
        showlegend=False,
    ))

    # Anotações de irreversibilidade
    fig_ts.add_annotation(
        x=(s2 + s2s) / 2, y=(T2 + T2s) / 2 + 40,
        text=f"Δs = {s2 - s1:.4f} kJ/kg·K<br>η_c = {n_c * 100:.0f}%",
        showarrow=False, font=dict(size=14, color=NAVY),
        bgcolor='rgba(0,0,0,0.35)', bordercolor=NAVY, borderwidth=1,
    )
    fig_ts.add_annotation(
        x=(s4 + s4s) / 2 + 0.05, y=(T4 + T4s) / 2 - 40,
        text=f"Δs = {s4 - s4s:.4f} kJ/kg·K<br>η_t = {n_t * 100:.0f}%",
        showarrow=False, font=dict(size=14, color=GREEN),
        bgcolor='rgba(0,0,0,0.35)', bordercolor=GREEN, borderwidth=1,
    )

    fig_ts.update_layout(
        title=dict(
            text='Diagrama T-s: Ciclo Brayton (com irreversibilidades)',
            font=dict(size=16, color="white"),
        ),
        xaxis=dict(title='Entropia específica  s  (kJ/kg·K)', gridcolor='rgba(200,200,200,0.4)'),
        yaxis=dict(title='Temperatura  T  (K)', gridcolor='rgba(200,200,200,0.4)'),
        legend=dict(orientation='v', x=1.01, y=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=520,
    )

    # DIAGRAMA P-h
    fig_ph = go.Figure()

    # Isobaras de referência
    fig_ph.add_trace(go.Scatter(
        x=h_low, y=[p1_bar] * N, mode='lines',
        line=dict(color=TEAL, width=1.5, dash='dot'),
        name=f'Isobara P₁ = {p1_bar:.2f} bar',
    ))
    fig_ph.add_trace(go.Scatter(
        x=h_high, y=[p2_bar] * N, mode='lines',
        line=dict(color=ORANGE, width=1.5, dash='dot'),
        name=f'Isobara P₂ = {p2_bar:.2f} bar',
    ))

    # Processos reais
    h_pts = [h1, h2, h3, h4, h1]
    p_pts = [p1_bar, p2_bar, p2_bar, p1_bar, p1_bar]
    for i in range(4):
        fig_ph.add_trace(go.Scatter(
            x=[h_pts[i], h_pts[i + 1]],
            y=[p_pts[i], p_pts[i + 1]],
            mode='lines',
            line=dict(color=colors_seg[i], width=2.5),
            name=labels_seg[i],
        ))

    # Pontos
    ph_s = [h1, h2s, h2, h3, h4]
    ph_p = [p1_bar, p2_bar, p2_bar, p2_bar, p1_bar]
    ph_lbl = ['1', '2s', '2', '3', '4']
    ph_clr = [TEAL, NAVY, NAVY, ORANGE, GREEN]
    ph_pos = ['bottom left', 'top left', 'top right', 'top right', 'bottom right']
    ph_hov = [
        f"<b>Estado 1</b><br>h = {h1:.1f} kJ/kg<br>P = {p1_bar:.2f} bar",
        f"<b>Estado 2s</b> (ideal)<br>h = {h2s:.1f} kJ/kg<br>P = {p2_bar:.2f} bar",
        f"<b>Estado 2</b> (real)<br>h = {h2:.1f} kJ/kg<br>P = {p2_bar:.2f} bar",
        f"<b>Estado 3</b><br>h = {h3:.1f} kJ/kg<br>P = {p2_bar:.2f} bar",
        f"<b>Estado 4</b> (real)<br>h = {h4:.1f} kJ/kg<br>P = {p1_bar:.2f} bar",
    ]
    fig_ph.add_trace(go.Scatter(
        x=ph_s, y=ph_p,
        mode='markers+text',
        marker=dict(size=10, color=ph_clr, line=dict(width=2, color='white')),
        text=ph_lbl,
        textposition=ph_pos,
        textfont=dict(size=13, color=NAVY),
        hovertext=ph_hov,
        hoverinfo='text',
        name='Estados',
        showlegend=False,
    ))

    # Anotações de trabalho e calor
    w_c = h2 - h1
    w_t = h3 - h4
    w_n = w_t - w_c
    q_i = h3 - h2
    q_o = h4 - h1

    fig_ph.add_annotation(
        x=(h1 + h2) / 2, y=p2_bar * 1.1,
        text=f"w_comp = {w_c:.0f} kJ/kg",
        showarrow=False, font=dict(size=14, color=NAVY),
        bgcolor='rgba(0,0,0,0.35)',
    )
    fig_ph.add_annotation(
        x=(h2 + h3) / 2, y=p2_bar * 0.88,
        text=f"q_in = {q_i:.0f} kJ/kg",
        showarrow=False, font=dict(size=14, color=ORANGE),
        bgcolor='rgba(0,0,0,0.35)',
    )
    fig_ph.add_annotation(
        x=(h3 + h4) / 2, y=p1_bar * 1.8,
        text=f"w_turb = {w_t:.0f} kJ/kg",
        showarrow=False, font=dict(size=14, color=GREEN),
        bgcolor='rgba(0,0,0,0.35)',
    )
    fig_ph.add_annotation(
        x=(h4 + h1) / 2, y=p1_bar * 0.55,
        text=f"q_out = {q_o:.0f} kJ/kg",
        showarrow=False, font=dict(size=14, color=TEAL),
        bgcolor='rgba(0,0,0,0.35)',
    )

    fig_ph.update_layout(
        title=dict(
            text='Diagrama P-h: Ciclo Brayton (com irreversibilidades)',
            font=dict(size=16, color="white"),
        ),
        xaxis=dict(title='Entalpia específica  h  (kJ/kg)', gridcolor='rgba(200,200,200,0.4)'),
        yaxis=dict(title='Pressão  P  (bar)', gridcolor='rgba(200,200,200,0.4)'),
        legend=dict(orientation='v', x=1.01, y=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=520,
    )

    return fig_ts, fig_ph


## PLOT RANKINE FUNÇÃO

def plot_rankine(
    states: dict,
    n_tr: float,
    n_b: float,
) -> tuple:
    # Gera os diagramas T-s e P-h do ciclo Rankine em Plotly.
    #
    # Parâmetros
    # ----------
    # states : dict com chaves 'A','B','C','D','Bs' → {'T','P','h','s','label'}
    #          h em kJ/kg, s em kJ/kg·K
    #          A = entrada turbina vapor (superaquecido)
    #          B = saída turbina vapor (real)
    #          Bs= saída turbina vapor (isentrópico ideal)
    #          C = saída condensador (líquido saturado)
    #          D = saída bomba (líquido comprimido)
    # n_tr   : eficiência isentrópica da turbina a vapor [0-1]
    # n_b    : eficiência isentrópica da bomba [0-1]
    #
    # Retorna
    # -------
    # fig_ts : go.Figure — diagrama T-s (com domo de saturação)
    # fig_ph : go.Figure — diagrama P-h (com domo de saturação)
    #
    # Nota
    # ----
    # Para os estados B e D, calcule s via ('S','H',h,'P',P) e não via ('S','T',T,'P',P)
    # para evitar erros do CoolProp na região bifásica:
    #     sB = CP.PropsSI('S', 'H', hB, 'P', P_condensador, 'Water')
    #     sD = CP.PropsSI('S', 'H', hD, 'P', P_caldeira,    'Water')

    NAVY = "#7EB8D4"
    TEAL = "#4DD0E1"
    ORANGE = "#F26B3A"
    GREEN = "#66BB6A"
    PURPLE = "#CE93D8"
    WHITE = "white"

    N = 400  # pontos nas curvas

    # ── Extrair estados ──────────────────────────────────────────────────
    hA = states['A']['h'];
    sA = states['A']['s'];
    TA = states['A']['T']
    hB = states['B']['h'];
    sB = states['B']['s'];
    TB = states['B']['T']
    hC = states['C']['h'];
    sC = states['C']['s'];
    TC = states['C']['T']
    hD = states['D']['h'];
    sD = states['D']['s'];
    TD = states['D']['T']
    hBs = states['Bs']['h'];
    sBs = states['Bs']['s'];
    TBs = states['Bs']['T']

    PA = states['A']['P']  # Pa — pressão da caldeira
    PC = states['C']['P']  # Pa — pressão do condensador
    PA_bar = PA / 1e5
    PC_bar = PC / 1e5

    # ── Domo de saturação da água ────────────────────────────────────────
    T_crit = CP.PropsSI('Tcrit', 'Water')  # 647.1 K
    P_crit = CP.PropsSI('Pcrit', 'Water')  # 220.6 bar

    T_dome = np.linspace(273.16, T_crit - 0.1, N)
    s_liq = np.array([CP.PropsSI('S', 'T', T, 'Q', 0, 'Water') for T in T_dome]) / 1000
    s_vap = np.array([CP.PropsSI('S', 'T', T, 'Q', 1, 'Water') for T in T_dome]) / 1000
    h_liq = np.array([CP.PropsSI('H', 'T', T, 'Q', 0, 'Water') for T in T_dome]) / 1000
    h_vap = np.array([CP.PropsSI('H', 'T', T, 'Q', 1, 'Water') for T in T_dome]) / 1000
    p_dome = np.array([CP.PropsSI('P', 'T', T, 'Q', 0, 'Water') for T in T_dome]) / 1e5

    # ── Isobaras (para referência visual) ───────────────────────────────
    # Temperaturas de saturação em cada pressão
    T_sat_A = CP.PropsSI('T', 'Q', 0, 'P', PA, 'Water')  # T saturação em PA
    T_sat_C = CP.PropsSI('T', 'Q', 0, 'P', PC, 'Water')  # T saturação em PC

    # Isobara ALTA (caldeira) — 3 segmentos separados para evitar cortar o domo
    # Segmento 1: líquido comprimido (TD → T_sat_A)
    T_liq = np.linspace(TD, T_sat_A - 0.01, 80)
    s_liq_A = [CP.PropsSI('S', 'H',
                          CP.PropsSI('H', 'T', T, 'P', PA, 'Water'),
                          'P', PA, 'Water') / 1000 for T in T_liq]
    h_liq_A = [CP.PropsSI('H', 'T', T, 'P', PA, 'Water') / 1000 for T in T_liq]

    # Segmento 2: vaporização (T constante = T_sat_A, Q varia 0→1)
    Q_bif = np.linspace(0, 1, 80)
    s_bif_A = [CP.PropsSI('S', 'Q', q, 'P', PA, 'Water') / 1000 for q in Q_bif]
    h_bif_A = [CP.PropsSI('H', 'Q', q, 'P', PA, 'Water') / 1000 for q in Q_bif]
    T_bif_A = [T_sat_A] * 80

    # Segmento 3: vapor superaquecido (T_sat_A → TA)
    T_sup = np.linspace(T_sat_A + 0.01, TA, 80)
    s_sup_A = [CP.PropsSI('S', 'H',
                          CP.PropsSI('H', 'T', T, 'P', PA, 'Water'),
                          'P', PA, 'Water') / 1000 for T in T_sup]
    h_sup_A = [CP.PropsSI('H', 'T', T, 'P', PA, 'Water') / 1000 for T in T_sup]

    # Concatena os 3 segmentos
    s_isob_A = s_liq_A + s_bif_A + s_sup_A
    h_isob_A = h_liq_A + h_bif_A + h_sup_A
    T_isob_A = list(T_liq) + T_bif_A + list(T_sup)

    # Isobara BAIXA (condensador) — idem, 3 segmentos
    # Segmento 1: líquido saturado (ponto C)
    # Segmento 2: vaporização a T_sat_C
    s_bif_C = [CP.PropsSI('S', 'Q', q, 'P', PC, 'Water') / 1000 for q in Q_bif]
    h_bif_C = [CP.PropsSI('H', 'Q', q, 'P', PC, 'Water') / 1000 for q in Q_bif]
    T_bif_C = [T_sat_C] * 80

    # Segmento 3: vapor superaquecido (T_sat_C → TB+5) se TB > T_sat_C
    if TB > T_sat_C + 1:
        T_sup_C = np.linspace(T_sat_C + 0.01, TB + 5, 40)
        s_sup_C = [CP.PropsSI('S', 'H',
                              CP.PropsSI('H', 'T', T, 'P', PC, 'Water'),
                              'P', PC, 'Water') / 1000 for T in T_sup_C]
        h_sup_C = [CP.PropsSI('H', 'T', T, 'P', PC, 'Water') / 1000 for T in T_sup_C]
        s_isob_C = s_bif_C + s_sup_C
        h_isob_C = h_bif_C + h_sup_C
        T_isob_C = T_bif_C + list(T_sup_C)
    else:
        s_isob_C = s_bif_C
        h_isob_C = h_bif_C
        T_isob_C = T_bif_C

    # ── Processos ────────────────────────────────────────────────────────
    # T-s: A→B→C→D→A
    T_cycle = [TA, TB, TC, TD, TA]
    s_cycle = [sA, sB, sC, sD, sA]

    # P-h: A→B→C→D→A
    h_cycle = [hA, hB, hC, hD, hA]
    p_cycle = [PA_bar, PC_bar, PC_bar, PA_bar, PA_bar]

    colors_seg = [GREEN, TEAL, NAVY, ORANGE]
    labels_seg = ['A→B Expansão (turbina)', 'B→C Condensação',
                  'C→D Bomba', 'D→A Caldeira / HRSG']

    # ════════════════════════════════════════════════════════
    # DIAGRAMA T-s
    # ════════════════════════════════════════════════════════
    fig_ts = go.Figure()

    # Domo de saturação
    fig_ts.add_trace(go.Scatter(
        x=np.concatenate([s_liq, s_vap[::-1]]),
        y=np.concatenate([T_dome, T_dome[::-1]]),
        mode='lines',
        line=dict(color='rgba(180,180,180,0.5)', width=1.5),
        name='Curva de saturação',
        fill='toself',
        fillcolor='rgba(100,100,180,0.08)',
    ))

    # Ponto crítico
    s_crit = CP.PropsSI('S', 'T', T_crit - 0.1, 'Q', 1, 'Water') / 1000
    fig_ts.add_trace(go.Scatter(
        x=[s_crit], y=[T_crit],
        mode='markers',
        marker=dict(size=7, color='rgba(180,180,180,0.7)', symbol='diamond'),
        name='Ponto crítico',
        showlegend=False,
    ))

    # Isobaras de referência
    # isobara caldeira removida do T-s — o processo D→A já a representa
    fig_ts.add_trace(go.Scatter(
        x=s_isob_C, y=T_isob_C, mode='lines',
        line=dict(color=TEAL, width=1.2, dash='dot'),
        name=f'Isobara condensador {PC_bar:.3f} bar',
    ))

    # Processos reais
    # A→B, B→C, C→D: segmentos retos
    for i in range(3):
        fig_ts.add_trace(go.Scatter(
            x=[s_cycle[i], s_cycle[i + 1]],
            y=[T_cycle[i], T_cycle[i + 1]],
            mode='lines',
            line=dict(color=colors_seg[i], width=2.5),
            name=labels_seg[i],
        ))
    # D→A: segue a isobara real (líquido → vaporização → superaquecimento)
    # remove a isobara pontilhada do T-s pois o processo já a representa
    fig_ts.add_trace(go.Scatter(
        x=s_isob_A, y=T_isob_A, mode='lines',
        line=dict(color=ORANGE, width=2.5),
        name=labels_seg[3],
    ))

    # Expansão isentrópica ideal (A→Bs tracejado)
    fig_ts.add_trace(go.Scatter(
        x=[sA, sBs], y=[TA, TBs], mode='lines',
        line=dict(color=GREEN, width=1.5, dash='dash'),
        name='A→Bs (ideal)',
    ))

    # Pontos
    pts_s = [sA, sBs, sB, sC, sD]
    pts_T = [TA, TBs, TB, TC, TD]
    pts_lbl = ['A', 'Bs', 'B', 'C', 'D']
    pts_clr = [ORANGE, GREEN, GREEN, TEAL, NAVY]
    pts_pos = ['top right', 'bottom right', 'bottom left',
               'bottom left', 'top left']
    pts_hov = [
        f"<b>Estado A</b><br>T = {TA:.1f} K<br>h = {hA:.1f} kJ/kg<br>s = {sA:.4f} kJ/kg·K",
        f"<b>Estado Bs</b> (ideal)<br>T = {TBs:.1f} K<br>h = {hBs:.1f} kJ/kg<br>s = {sBs:.4f} kJ/kg·K",
        f"<b>Estado B</b> (real)<br>T = {TB:.1f} K<br>h = {hB:.1f} kJ/kg<br>s = {sB:.4f} kJ/kg·K",
        f"<b>Estado C</b><br>T = {TC:.1f} K<br>h = {hC:.1f} kJ/kg<br>s = {sC:.4f} kJ/kg·K",
        f"<b>Estado D</b><br>T = {TD:.1f} K<br>h = {hD:.1f} kJ/kg<br>s = {sD:.4f} kJ/kg·K",
    ]
    fig_ts.add_trace(go.Scatter(
        x=pts_s, y=pts_T,
        mode='markers+text',
        marker=dict(size=10, color=pts_clr, line=dict(width=2, color='white')),
        text=pts_lbl,
        textposition=pts_pos,
        textfont=dict(size=13, color='white'),
        hovertext=pts_hov,
        hoverinfo='text',
        name='Estados',
        showlegend=False,
    ))

    # Anotações de trabalho / calor
    w_turb_v = hA - hB
    w_bomba = hD - hC
    q_in_r = hA - hD
    q_out_r = hB - hC
    eta_r = (w_turb_v - w_bomba) / q_in_r

    fig_ts.add_annotation(
        x=(sA + sBs) / 2 + 0.15, y=(TA + TBs) / 2,
        text=f"w_turb = {w_turb_v:.0f} kJ/kg<br>η_t = {n_tr * 100:.0f}%",
        showarrow=False, font=dict(size=13, color=GREEN),
        bgcolor='rgba(0,0,0,0.45)', bordercolor=GREEN, borderwidth=1,
    )
    fig_ts.add_annotation(
        x=(sC + sD) / 2 - 0.15, y=(TC + TD) / 2,
        text=f"w_bomba = {w_bomba:.1f} kJ/kg<br>η_b = {n_b * 100:.0f}%",
        showarrow=False, font=dict(size=13, color=NAVY),
        bgcolor='rgba(0,0,0,0.45)', bordercolor=NAVY, borderwidth=1,
    )
    fig_ts.add_annotation(
        x=(sB + sC) / 2, y=TC - 8,
        text=f"q_out = {q_out_r:.0f} kJ/kg",
        showarrow=False, font=dict(size=13, color=TEAL),
        bgcolor='rgba(0,0,0,0.45)', bordercolor=TEAL, borderwidth=1,
    )
    fig_ts.add_annotation(
        x=(sD + sA) / 2 + 0.3, y=(TD + TA) / 2,
        text=f"q_in = {q_in_r:.0f} kJ/kg (HRSG)<br>η = {eta_r * 100:.1f}%",
        showarrow=False, font=dict(size=13, color=ORANGE),
        bgcolor='rgba(0,0,0,0.45)', bordercolor=ORANGE, borderwidth=1,
    )

    fig_ts.update_layout(
        title=dict(
            text='Diagrama T-s — Ciclo Rankine (com irreversibilidades)',
            font=dict(size=16, color='white'),
        ),
        xaxis=dict(
            title='Entropia específica  s  (kJ/kg·K)',
            gridcolor='rgba(200,200,200,0.4)',
            color='white',
        ),
        yaxis=dict(
            title='Temperatura  T  (K)',
            gridcolor='rgba(200,200,200,0.4)',
            color='white',
        ),
        legend=dict(orientation='v', x=1.01, y=1, font=dict(color='white')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=520,
    )

    # ════════════════════════════════════════════════════════
    # DIAGRAMA P-h
    # ════════════════════════════════════════════════════════
    fig_ph = go.Figure()

    # Domo de saturação
    fig_ph.add_trace(go.Scatter(
        x=np.concatenate([h_liq, h_vap[::-1]]),
        y=np.concatenate([p_dome, p_dome[::-1]]),
        mode='lines',
        line=dict(color='rgba(180,180,180,0.5)', width=1.5),
        name='Curva de saturação',
        fill='toself',
        fillcolor='rgba(100,100,180,0.08)',
    ))

    # Ponto crítico
    h_crit = CP.PropsSI('H', 'T', T_crit - 0.1, 'Q', 1, 'Water') / 1000
    fig_ph.add_trace(go.Scatter(
        x=[h_crit], y=[P_crit / 1e5],
        mode='markers',
        marker=dict(size=7, color='rgba(180,180,180,0.7)', symbol='diamond'),
        name='Ponto crítico',
        showlegend=False,
    ))

    # Processos reais
    for i in range(4):
        fig_ph.add_trace(go.Scatter(
            x=[h_cycle[i], h_cycle[i + 1]],
            y=[p_cycle[i], p_cycle[i + 1]],
            mode='lines',
            line=dict(color=colors_seg[i], width=2.5),
            name=labels_seg[i],
        ))

    # Ponto Bs no P-h
    fig_ph.add_trace(go.Scatter(
        x=[hA, hBs], y=[PA_bar, PC_bar], mode='lines',
        line=dict(color=GREEN, width=1.5, dash='dash'),
        name='A→Bs (ideal)',
    ))

    # Pontos
    ph_h = [hA, hBs, hB, hC, hD]
    ph_p = [PA_bar, PC_bar, PC_bar, PC_bar, PA_bar]
    ph_lbl = ['A', 'Bs', 'B', 'C', 'D']
    ph_clr = [ORANGE, GREEN, GREEN, TEAL, NAVY]
    ph_pos = ['top right', 'bottom right', 'bottom left',
              'bottom left', 'top left']
    ph_hov = [
        f"<b>Estado A</b><br>h = {hA:.1f} kJ/kg<br>P = {PA_bar:.1f} bar",
        f"<b>Estado Bs</b> (ideal)<br>h = {hBs:.1f} kJ/kg<br>P = {PC_bar:.3f} bar",
        f"<b>Estado B</b> (real)<br>h = {hB:.1f} kJ/kg<br>P = {PC_bar:.3f} bar",
        f"<b>Estado C</b><br>h = {hC:.1f} kJ/kg<br>P = {PC_bar:.3f} bar",
        f"<b>Estado D</b><br>h = {hD:.1f} kJ/kg<br>P = {PA_bar:.1f} bar",
    ]
    fig_ph.add_trace(go.Scatter(
        x=ph_h, y=ph_p,
        mode='markers+text',
        marker=dict(size=10, color=ph_clr, line=dict(width=2, color='white')),
        text=ph_lbl,
        textposition=ph_pos,
        textfont=dict(size=13, color='white'),
        hovertext=ph_hov,
        hoverinfo='text',
        name='Estados',
        showlegend=False,
    ))

    # ── Range do eixo Y: do condensador até 20% acima da caldeira (escala log) ──
    import math
    y_min = math.log10(PC_bar * 0.5)  # meio patamar abaixo do condensador
    y_max = math.log10(PA_bar * 2.0)  # um patamar acima da caldeira

    # Posições das anotações dentro do range visível
    y_mid = 10 ** ((y_min + math.log10(PA_bar)) / 2)  # meio geométrico entre PC e PA

    # # Anotações
    # fig_ph.add_annotation(
    #     x=(hA + hB) / 2, y=y_mid,
    #     text=f"w_turb = {w_turb_v:.0f} kJ/kg",
    #     showarrow=False, font=dict(size=13, color=GREEN),
    #     bgcolor='rgba(0,0,0,0.45)', bordercolor=GREEN, borderwidth=1,
    # )
    # fig_ph.add_annotation(
    #     x=(hC + hD) / 2, y=PA_bar * 1.4,
    #     text=f"w_bomba = {w_bomba:.1f} kJ/kg",
    #     showarrow=False, font=dict(size=13, color=NAVY),
    #     bgcolor='rgba(0,0,0,0.45)', bordercolor=NAVY, borderwidth=1,
    # )
    # fig_ph.add_annotation(
    #     x=(hB + hC) / 2, y=y_mid,
    #     text=f"q_out = {q_out_r:.0f} kJ/kg",
    #     showarrow=False, font=dict(size=13, color=TEAL),
    #     bgcolor='rgba(0,0,0,0.45)', bordercolor=TEAL, borderwidth=1,
    # )
    # fig_ph.add_annotation(
    #     x=(hD + hA) / 2, y=PA_bar * 0.7,
    #     text=f"q_in = {q_in_r:.0f} kJ/kg",
    #     showarrow=False, font=dict(size=13, color=ORANGE),
    #     bgcolor='rgba(0,0,0,0.45)', bordercolor=ORANGE, borderwidth=1,
    # )

    fig_ph.update_layout(
        title=dict(
            text='Diagrama P-h — Ciclo Rankine (com irreversibilidades)',
            font=dict(size=16, color='white'),
        ),
        xaxis=dict(
            title='Entalpia específica  h  (kJ/kg)',
            gridcolor='rgba(200,200,200,0.4)',
            color='white',
        ),
        yaxis=dict(
            title='Pressão  P  (bar)',
            type='log',
            #range=[y_min, y_max],
            gridcolor='rgba(200,200,200,0.4)',
            color='white',
        ),
        legend=dict(orientation='v', x=1.01, y=1, font=dict(color='white')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=520,
    )

    return fig_ts, fig_ph


# **INPUT**

st.title("Case Eneva: Ciclo Combinado", anchor=False)

col1, col2 = st.columns([1, 2])

col1.subheader("Entrada", divider="gray", anchor=False)

if 'active_page' not in st.session_state:
    st.session_state.active_page = 'eneva_cc'

    st.session_state.kT1 = round(300 - 273.15,2)
    st.session_state.kP1 = 101325
    st.session_state.kT3 = round(1500 - 273.15,2)
    st.session_state.krp = 12
    st.session_state.kn_c = 85
    st.session_state.kn_t = 85

    st.session_state.kP_caldeira = 80e5
    st.session_state.kT1_turbina = 500
    st.session_state.kP_condensador = 0.08e5
    st.session_state.kn_tr = 85
    st.session_state.kn_b = 85

    st.session_state.kT_gen_ca = 90 # 85-95°C    (um pouco acima do condensador para ter ΔT de troca)
    st.session_state.kT_cond_ca = 27 + 5 # T_amb + 5  (rejeição de calor para o ambiente, ~30-35°C)
    st.session_state.kT_abs_ca = 27 + 7 # T_amb + 7  (similar ao condensador, ~32-37°C)
    st.session_state.kT_evap_ca = 5 # 2-8°C      (resfria o gás sem risco de condensar água nele)
    st.session_state.kn_b_ca = 85


with col1.expander("Ciclo Brayton (Gás)", expanded=True):

    T1 = st.number_input(
        label='Temperatura de entrada do compressor',
        min_value=0.,
        format="%f",
        step=1.,
        key='kT1',
        help="Temperatura de entrada do compressor [°C]"
    )
    T1 = T1 + 273.15

    P1 = st.number_input(
        label='Pressão na entrada do compressor',
        min_value=101325.,
        format="%f",
        step=1.,
        key='kP1',
        help="Pressão na entrada do compressor [Pa]"
    )

    T3 = st.number_input(
        label='Temperatura de entrada da turbina',
        min_value=0.,
        format="%f",
        step=1.,
        key='kT3',
        help="Temperatura de entrada da turbina [°C]"
    )
    T3 = T3 + 273.15

    rp = st.number_input(
        label='Razão de pressão',
        min_value=1.,
        format="%f",
        step=1.,
        key='krp',
    )

    n_c = st.slider(
        label="% Eficiência isentrópica do compressor",
        min_value=1,
        max_value=100,
        key='kn_c',
    )
    n_c = n_c/100

    n_t = st.slider(
        label="% Eficiência isentrópica da turbina a gás",
        min_value=1,
        max_value=100,
        key='kn_t',
    )
    n_t = n_t/100


with col1.expander("Ciclo Rankine (Vapor)", expanded=True):

    P_caldeira = st.number_input(
        label='Pressão da caldeira',
        min_value=101325.,
        format="%f",
        step=1.,
        key='kP_caldeira',
        help="Pressão da caldeira [Pa]"
    )

    T1_turbina = st.number_input(
        label='Temperatura do vapor na entrada da turbina',
        min_value=0.,
        format="%f",
        step=1.,
        key='kT1_turbina',
        help="Temperatura do vapor na entrada da turbina [°C]"
    )
    T1_turbina = T1_turbina + 273.15

    P_condensador = st.number_input(
        label='Pressão do condensador',
        min_value=101325.,
        format="%f",
        step=1.,
        key='kP_condensador',
        help="Pressão do condensador [Pa]"
    )

    n_tr = st.slider(
        label='% Eficiência isentrópica da turbina a vapor',
        min_value=1,
        max_value=100,
        key='kn_tr',
    )
    n_tr = n_tr / 100

    n_b = st.slider(
        label="% Eficiência isentrópica da bomba",
        min_value=1,
        max_value=100,
        key='kn_b',
    )
    n_b = n_b / 100


# **GERAÇÃO DE ENERGIA BRAYTON**

# Compressor

## Entrada
s1 = CP.PropsSI('S', 'T', T1, 'P', P1, "Air")   # Entropia específica na entrada do compressor [J/kg]
h1 = CP.PropsSI('H', 'T', T1, 'P', P1, "Air")   # Entalpia específica na entrada do compressor [J/kg]

## Saída
P2 = P1 * rp
s2s = s1   # Entropia específica na saída do compressor isentrópico [J/kg]
h2s = CP.PropsSI('H', 'S', s2s, 'P', P2, "Air")
T2s = CP.PropsSI('T', 'H', h2s, 'P', P2, "Air")
h2 = ((h2s - h1) / n_c) + h1
T2 = CP.PropsSI("T", "H", h2, "P", P2, "Air")
s2 = CP.PropsSI('S', 'T', T2, 'P', P2, "Air")   # Entropia específica na saída do compressor real [J/kg]


# Combustão
## Aqui irei usar método iterativo pra ter cp da solução CH4 + ar
## e assim obter o AFR (AIR-FUEL RATIO) também: f = kg CH4 / kg ar (AFR = 1 / f)
cp_ar = 1.005           # estimativa inicial (baseado no ar)
LHV   = 50000           # kJ/kg CH4 (entrada: poder calorífico)
P_alta = 1200000
P_baixa = 101325

for i in range(10):
    q_in = cp_ar * (T3 - T2)
    AFR  = LHV / q_in

    # cp real da mistura pós-combustão na câmara (T médio 2→3)
    T_med = (T2 + T3) / 2
    e_med = estado_pos_combustao(T=T_med, P=P2, AFR=AFR)
    cp_novo = e_med['cp_kJ/kgK']

    erro = abs(cp_novo - cp_ar) / cp_ar
    print(f"iter {i+1}: AFR={AFR:.1f}  cp={cp_novo:.4f}  erro={erro*100:.4f}%")
    cp_ar = cp_novo
    if erro < 1e-6:
        print("Convergiu!")
        break

# AFR derivado
q_in = cp_novo * (T3 - T2)
AFR  = LHV / q_in

e3 = estado_pos_combustao(T=T3, P=P2, AFR=AFR)


# Turbina (com mistura CH4-Ar): Expansão isentrópica ideal (ar-padrão como estimativa inicial)
k3 = e3['k_cp/cv']
cp3 = e3['cp_kJ/kgK']

## Expansão isentrópica com k do estado 3
T4s = T3 * (P1/P2)**((k3-1)/k3)

## T4 real com eficiência da turbina
T4 = T3 - n_t * (T3 - T4s)

## Estado 4
e4 = estado_pos_combustao(T=T4, P=P1, AFR=AFR)


# Performance Brayton
w_comp = (h2 - h1) / 1000            # kJ/kg, trabalho do compressor
w_turb_g = e3['h_kJ/kg'] - e4['h_kJ/kg']  # kJ/kg, trabalho da turbina a gás
w_net_b = w_turb_g - w_comp           # kJ/kg, trabalho líquido Brayton
q_in_b = e3['h_kJ/kg'] - (h2/1000)  # kJ/kg, calor fornecido (combustão)
q_out_b = e4['h_kJ/kg'] - (h1/1000)  # kJ/kg, calor rejeitado (exaustão)
eta_b = w_net_b / q_in_b            # eficiência Brayton
bwr = w_comp  / w_turb_g          # back work ratio
f = 1 / AFR                     # kg CH4 / kg ar

states_brayton = {
    1: {
        'T': T1,
        'P': P1,
        'h': h1 / 1000,       # CoolProp retorna J/kg → converte para kJ/kg
        's': s1 / 1000,       # CoolProp retorna J/kg·K → converte para kJ/kg·K
        'label': '1',
    },
    2: {
        'T': T2,
        'P': P2,
        'h': h2 / 1000,
        's': s2 / 1000,
        'label': '2',
    },
    3: {
        'T': T3,
        'P': P2,
        'h': e3['h_kJ/kg'],   # estado_pos_combustao já retorna kJ/kg
        's': e3['s_kJ/kgK'],
        'label': '3',
    },
    4: {
        'T': T4,
        'P': P1,
        'h': e4['h_kJ/kg'],
        's': e4['s_kJ/kgK'],
        'label': '4',
    },
}

# **GERAÇÃO DE ENERGIA RANKINE**

# **GERAÇÃO DE ENERGIA RANKINE**

## Estado 1: entrada da turbina a vapor (vapor superaquecido)
h1r = CP.PropsSI('H', 'T', T1_turbina, 'P', P_caldeira, 'Water')   # J/kg
s1r = CP.PropsSI('S', 'T', T1_turbina, 'P', P_caldeira, 'Water')   # J/kg·K
T1r = T1_turbina

## Estado 2: saída da turbina a vapor
### Expansão isentrópica ideal: s2r = s1r
h2rs = CP.PropsSI('H', 'S', s1r, 'P', P_condensador, 'Water')
T2rs = CP.PropsSI('T', 'H', h2rs, 'P', P_condensador, 'Water')
### Real com eficiência
h2r = h1r - n_tr * (h1r - h2rs)
T2r = CP.PropsSI('T', 'H', h2r, 'P', P_condensador, 'Water')
#s2r = CP.PropsSI('S', 'T', T2r, 'P', P_condensador, 'Water')
s2r = CP.PropsSI('S', 'H', h2r, 'P', P_condensador, 'Water')

# Título do vapor na saída (x=0 líquido, x=1 vapor saturado)
# Só válido se T2r estiver na região bifásica

try:
    x2r = CP.PropsSI('Q', 'H', h2r, 'P', P_condensador, 'Water')
except:
    x2r = None   # superaquecido ou comprimido

## Estado 3: saída do condensador (líquido saturado)
h3r = CP.PropsSI('H', 'Q', 0, 'P', P_condensador, 'Water')   # líquido saturado
s3r = CP.PropsSI('S', 'Q', 0, 'P', P_condensador, 'Water')
T3r = CP.PropsSI('T', 'Q', 0, 'P', P_condensador, 'Water')

## Estado 4: saída da bomba (entrada da caldeira)
### Compressão isentrópica ideal: s4r = s3r
h4rs = CP.PropsSI('H', 'S', s3r, 'P', P_caldeira, 'Water')

## Real com eficiência
h4r = h3r + (h4rs - h3r) / n_b
T4r = CP.PropsSI('T', 'H', h4r, 'P', P_caldeira, 'Water')
s4r = CP.PropsSI('S', 'T', T4r, 'P', P_caldeira, 'Water')

## ── Performance Rankine (base: 1 kg de vapor) ─────────────────────────────
w_turb_v = (h1r - h2r) / 1000      # kJ/kg vapor — trabalho da turbina a vapor
w_bomba  = (h4r - h3r) / 1000      # kJ/kg vapor — trabalho da bomba
w_net_r  = w_turb_v - w_bomba    # kJ/kg vapor — trabalho líquido Rankine
q_in_r   = (h1r - h4r) / 1000      # kJ/kg vapor — calor fornecido pelo HRSG
q_out_r  = (h2r - h3r) / 1000      # kJ/kg vapor — calor rejeitado no condensador
eta_r    = w_net_r / q_in_r       # eficiência Rankine

## ── Acoplamento HRSG (converte bases: kg gás → kg vapor) ──────────────────
# T_stack: temperatura mínima da chaminé (evita condensação ácida ~420K)
T_stack = 420   # K (input)

## Calor disponível no exausto do Brayton (por kg de gás)
cp_exausto = e4['cp_kJ/kgK']   # cp da mistura pós-combustão na saída
q_hrsg = cp_exausto * (T4 - T_stack)   # kJ/kg gás

## Razão de massa: kg de vapor gerado por kg de gás
m_ratio = q_hrsg / q_in_r      # kg vapor / kg gás

## Performance combinada (base: 1 kg de gás)
W_brayton = w_net_b                    # kJ/kg gás
W_rankine = w_net_r * m_ratio          # kJ/kg gás
W_total = W_brayton + W_rankine      # kJ/kg gás
eta_cc = W_total / q_in_b           # eficiência ciclo combinado

Q_cond    = q_out_r * m_ratio          # kJ/kg gás, calor rejeitado no condensador
Q_stack   = cp_exausto * (T_stack - T1)  # kJ/kg gás, perdas na chaminé

## Balanço: W_total + Q_cond + Q_stack

states_rankine = {
    'A': {'T': T1r, 'P': P_caldeira,    'h': h1r/1000, 's': s1r/1000, 'label': 'A'},
    'B': {'T': T2r, 'P': P_condensador, 'h': h2r/1000, 's': s2r/1000, 'label': 'B'},
    'C': {'T': T3r, 'P': P_condensador, 'h': h3r/1000, 's': s3r/1000, 'label': 'C'},
    'D': {'T': T4r, 'P': P_caldeira,    'h': h4r/1000, 's': s4r/1000, 'label': 'D'},
    'Bs': {'T': T2rs, 'P': P_condensador, 'h': h2rs/1000, 's': s1r/1000, 'label': 'Bs'},
}

# **PLOTS**

with col2:
    ## BRAYTON
    fig_ts, fig_ph = plot_brayton(
        states=states_brayton,
        e3=e3, e4=e4,
        T2s=T2s, T4s=T4s,
        AFR=AFR,
        T1=T1, T2=T2, T3=T3, T4=T4,
        P1=P1, P2=P2,
        n_c=n_c, n_t=n_t,
    )

    tab_ts, tab_ph = col2.tabs(["T-s", "P-h"])
    tab_ts.plotly_chart(fig_ts, use_container_width=True)
    tab_ph.plotly_chart(fig_ph, use_container_width=True)

    ## RANKINE
    fig_ts, fig_ph = plot_rankine(states_rankine, n_tr=n_tr, n_b=n_b)

    tab_ts, tab_ph = col2.tabs(["T-s", "P-h"])
    tab_ts.plotly_chart(fig_ts, use_container_width=True)
    tab_ph.plotly_chart(fig_ph, use_container_width=True)


data_performance = {
    "Parâmetro": [
        "Trabalho Líquido (kJ/kg)",
        "Calor Fornecido (kJ/kg)",
        "Calor Rejeitado (kJ/kg)",
        "Eficiência Térmica (%)",
        "Back Work Ratio (BWR)"
    ],
    "Ciclo Brayton": [
        round(w_net_b, 2),
        round(q_in_b, 2),
        round(q_out_b, 2),
        round(eta_b * 100, 2),
        f"{round(bwr * 100, 2)}%"
    ],
    "Ciclo Rankine": [
        round(w_net_r, 2),
        round(q_in_r, 2),
        round(q_out_r, 2),
        round(eta_r * 100, 2),
        f"{round((w_bomba / w_turb_v) * 100, 2)}%"
    ],
    "Ciclo Combinado": [
        round(W_total, 2),
        round(q_in_b, 2),
        round(Q_cond + Q_stack, 2), # Soma das perdas do condensador + chaminé
        round(eta_cc * 100, 2),
        "-"
    ]
}

df_perf = pd.DataFrame(data_performance)

with col2:
    ## Exibição
    st.table(df_perf)
    st.info(f"Razão de massa: {m_ratio:.4f} kg vapor / kg gás")


# CICLO DE REFRIGERAÇÃO POR ABSORÇÃO (CA)
## Para liquefazer o gás parcialmente, seria necessário ciclo em cascata, mais complexo
## envolvendo tanto o ciclo por absorção, que iria transferir o Qevap pra o outro em um trocador intermediário (Qcond) do outro ciclo.

with col1.expander("Ciclo de Refrigeração por Absorção", expanded=True):

    T_gen_ca = st.number_input(
        label='Temperatura no gerador',
        min_value=0.,
        format="%f",
        step=1.,
        key='kT_gen_ca',
        help="Temperatura no gerador [°C]"
    )
    T_gen_ca = T_gen_ca + 273.15

    T_cond_ca = st.number_input(
        label='Temperatura de condensação',
        min_value=0.,
        format="%f",
        step=1.,
        key='kT_cond_ca',
        help="Temperatura de absorção [°C]"
    )
    T_cond_ca = T_cond_ca + 273.15

    T_abs_ca = st.number_input(
        label='Temperatura de absorção',
        min_value=0.,
        format="%f",
        step=1.,
        key='kT_abs_ca',
        help="Temperatura de absorção [°C]"
    )
    T_abs_ca = T_abs_ca + 273.15

    T_evap_ca = st.number_input(
        label='Temperatura de evaporação',
        min_value=0.,
        format="%f",
        step=1.,
        key='kT_evap_ca',
        help="Temperatura de evaporação [°C]"
    )
    T_evap_ca = T_evap_ca + 273.15

    n_b_ca = st.slider(
        label="% Eficiência isentrópica da bomba (CA)",
        min_value=1,
        max_value=100,
        key='kn_b_ca',
    )
    n_b_ca = n_b_ca/100


