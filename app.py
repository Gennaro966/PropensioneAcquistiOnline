import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# Configurazione generale
st.set_page_config(page_title="Analisi Dataset", layout="wide")

# Titolo
st.title("Dashboard Analisi Dataset Excel 📊")

# Sidebar
st.sidebar.title("Menu")
scelta = st.sidebar.radio("Seleziona una sezione:",
                          ['Load Excel', 'Grafici distribuzione', 'Analisi correlazione', 'Analisi regressione'])

# Inizializzazione variabile di sessione per il DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None

# Sezione: Caricamento Excel
if scelta == 'Load Excel':
    uploaded_file = st.file_uploader("Carica un file Excel (.xlsx)", type=["xlsx"])
    if uploaded_file is not None:
        st.session_state.df = pd.read_excel(uploaded_file)
        st.success("File caricato con successo!")
        st.write("Anteprima del file:")
        st.dataframe(st.session_state.df.head())

# Sezione: Grafici distribuzione
elif scelta == 'Grafici distribuzione':
    if st.session_state.df is None:
        st.warning("⚠️ Carica prima un file Excel nella sezione 'Load Excel'.")
    else:
        st.subheader("Grafico di distribuzione")
        for col in st.session_state.df.columns:
            if st.session_state.df[col].isna().sum() > 0:
                moda = st.session_state.df[col].mode()[0]
                st.session_state.df[col].fillna(moda, inplace=True)
        colonna = st.selectbox("Seleziona la colonna", st.session_state.df.columns)

        # Controllo se la colonna è di tipo oggetto o categoria
        if st.session_state.df[colonna].dtype == 'object' or st.session_state.df[colonna].dtype.name == 'category':
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(data=st.session_state.df, y=colonna, order=st.session_state.df[colonna].value_counts().index, ax=ax)
            ax.set_title(f'Distribuzione di: {colonna}')
            st.pyplot(fig)
        else:

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(st.session_state.df[colonna].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribuzione di: {colonna}')
            st.pyplot(fig)

# Sezione: Analisi correlazione
elif scelta == 'Analisi correlazione':
    if st.session_state.df is None:
        st.warning("⚠️ Carica prima un file Excel nella sezione 'Load Excel'.")
    else:
        st.subheader("Analisi di Correlazione")

        # Selezione variabile dipendente
        target = st.selectbox("Seleziona la variabile dipendente (target)",
                              ['propensione all’acquisto online', 'propensione al rischio'])

        df_corr = st.session_state.df.copy()

        # Costruzione della variabile target numerica
        if target == 'propensione all’acquisto online':
            mapping = {
                'Raramente': 1,
                'Mensile': 2,
                'Settimanale': 3,
                'Giornaliera': 4
            }
            df_corr['target_numerico'] = df_corr['Con quale frequenza fai acquisti online?'].map(mapping)
        elif target == 'propensione al rischio':
            df_corr['target_numerico'] = pd.to_numeric(
                df_corr['Generalmente, negli acquisti online quanto sei disposto a prendere dei rischi in una scala da 1 ( evito rischi) a 5 ( prendo rischi)?'],
                errors='coerce'
            )

        # Selezione variabili indipendenti
        possibili_driver = df_corr.columns.drop(['target_numerico'])
        variabili_indipendenti = st.multiselect("Seleziona le variabili indipendenti (drivers)", options=possibili_driver)

        if variabili_indipendenti:
            try:
                # One-hot encoding delle variabili indipendenti
                df_encoded = pd.get_dummies(df_corr[variabili_indipendenti], drop_first=True)

                # Aggiunta target continuo
                df_encoded['target_numerico'] = df_corr['target_numerico']

                # Calcolo della matrice di correlazione
                corr_matrix = df_encoded.corr(numeric_only=True)

                # Ordinamento righe e colonne in base alla correlazione col target
                if 'target_numerico' in corr_matrix.columns:
                    ordered_cols = corr_matrix['target_numerico'].abs().sort_values(ascending=False).index
                    corr_matrix = corr_matrix.loc[ordered_cols, ordered_cols]

                    # Heatmap completa
                    st.subheader("Matrice di correlazione")
                    fig2, ax2 = plt.subplots(figsize=(12, 10))
                    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, linewidths=0.5, annot=False, ax=ax2)
                    ax2.set_title("Matrice di correlazione tra tutte le variabili")
                    st.pyplot(fig2)

                    # Top correlazioni positive e negative col target
                    st.subheader("🔎 Top 5 variabili più correlate con il target (in positivo e negativo)")

                    corr_target = corr_matrix['target_numerico'].drop('target_numerico').sort_values()

                    st.markdown("#### 📉 Correlazioni negative più forti")
                    st.dataframe(corr_target.head(5).reset_index().rename(columns={'index': 'Variabile', 'target_numerico': 'Correlazione'}))

                    st.markdown("#### 📈 Correlazioni positive più forti")
                    st.dataframe(corr_target.tail(5).sort_values(ascending=False).reset_index().rename(columns={'index': 'Variabile', 'target_numerico': 'Correlazione'}))

                    # Spiegazione coefficiente di Pearson
                    st.markdown("---")
                    st.subheader("📘 Cos'è il coefficiente di correlazione di Pearson?")
                    st.markdown(r"""
Il **coefficiente di correlazione di Pearson (r)** misura la relazione lineare tra due variabili quantitative.

- Il valore di **r** varia tra **-1 e 1**:
  - `+1`: correlazione perfetta positiva
  - ` 0`: nessuna correlazione lineare
  - `-1`: correlazione perfetta negativa

#### 📐 Formula:
$$
r_{xy} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}
$$

Dove:
- \( x_i \), \( y_i \) sono i valori delle due variabili
- \( \bar{x} \), \( \bar{y} \) sono le medie delle due variabili

---

👉 **Quando si usa?**
Il coefficiente di Pearson si usa per:
- valutare la forza e la direzione di una relazione lineare
- supportare la selezione delle feature per modelli predittivi
- individuare possibili multicollinearità tra variabili indipendenti
                    """)

            except Exception as e:
                st.error(f"Errore nell'elaborazione: {e}")
        else:
            st.info("⬅️ Seleziona almeno una variabile indipendente per continuare.")

# Sezione: Analisi regressione
elif scelta == 'Analisi regressione':
    if st.session_state.df is None:
        st.warning("⚠️ Carica prima un file Excel nella sezione 'Load Excel'.")
    else:
        st.subheader("Analisi di Regressione Lineare")

        # Selezione variabile dipendente
        target = st.selectbox("Seleziona la variabile dipendente (target)",
                              ['propensione all’acquisto online', 'propensione al rischio'],
                              key='target_reg')

        df_reg = st.session_state.df.copy()


        # Scelta delle variabili indipendenti
        possibili_driver = df_reg.columns
        variabili_indipendenti = st.multiselect("Seleziona le variabili indipendenti (drivers)", options=possibili_driver, key='indep_reg')

        if variabili_indipendenti:
            try:
                if target == 'propensione all’acquisto online':
                    mapping = {'Raramente': 1,
                        'Mensile': 2,
                        'Settimanale': 3,
                        'Giornaliera': 4}
                    df_reg['target_continuo'] = df_reg['Con quale frequenza fai acquisti online?'].map(mapping)
                elif target == 'propensione al rischio':
                    df_reg['target_continuo'] = pd.to_numeric(df_reg['Generalmente, negli acquisti online quanto sei disposto a prendere dei rischi in una scala da 1 ( evito rischi) a 5 ( prendo rischi)?'],errors='coerce')
                # One-hot encoding
                df_encoded = pd.get_dummies(df_reg[variabili_indipendenti], drop_first=True)
                X = df_encoded.astype(float)
                y = df_reg['target_continuo'].astype(float)

                # Aggiunta dell'intercetta
                X = sm.add_constant(X)

                # Regressione lineare
                model = sm.OLS(y, X).fit()

                # Coefficienti ordinati per valore assoluto
                coeff_summary = model.summary2().tables[1].copy()
                coeff_summary_sorted = coeff_summary.reindex(coeff_summary['Coef.'].abs().sort_values(ascending=False).index)

                st.subheader("📊 Coefficienti ordinati per importanza")
                st.dataframe(coeff_summary_sorted)

                # Metriche
                metrics = {
                    "R-squared": model.rsquared,
                    "Adj. R-squared": model.rsquared_adj,
                    "AIC": model.aic,
                    "BIC": model.bic,
                    "F-statistic": model.fvalue,
                    "Prob (F-statistic)": model.f_pvalue
                }
                st.subheader("📈 Metriche del modello")
                st.table(pd.DataFrame(metrics.items(), columns=["Metrica", "Valore"]))

                # Summary completo
                with st.expander("📄 Vedi summary completo del modello"):
                    st.text(model.summary())

                # Spiegazione
    # Spiegazione dettagliata del summary
                st.markdown("---")
                st.subheader("📖 Interpretazione dettagliata del modello")

                st.markdown(r"""
        Il **summary del modello OLS** (Ordinary Least Squares) restituisce una serie di informazioni statistiche fondamentali:

        ---

        ### 📌 **1. Coefficienti (`Coef.`)**

        Rappresentano l'effetto **atteso sul target** per un incremento unitario della variabile indipendente, **mantenendo le altre costanti**.

        - 🔼 **Valore positivo**: l'aumento del driver è associato a un **aumento** del target.
        - 🔽 **Valore negativo**: l'aumento del driver è associato a una **diminuzione** del target.
        - 🟰 **Vicino a 0**: impatto trascurabile.

        #### Esempio:

        Se il coefficiente di `Età_25-34` è `0.42`, significa che, a parità di altre condizioni, appartenere a questa fascia d’età **aumenta in media di 0.42** la frequenza di acquisto online rispetto alla categoria di riferimento.

        ---

        ### 📌 **2. P-value (`P>|t|`)**

        Il **p-value** misura la probabilità che il coefficiente stimato sia diverso da zero **solo per caso**.

        - ✅ **p < 0.05** → la variabile è **statisticamente significativa**
        - ⚠️ **p > 0.05** → la variabile potrebbe **non contribuire significativamente** al modello

        > Più il p-value è vicino a 0, più è probabile che la variabile sia effettivamente utile nel predire il target.

        ---

        ### 📌 **3. R-squared (`R²`)**

        Indica **quanta parte della varianza** del target è spiegata dal modello.

        - `R² = 0.7` → il 70% della variabilità del target è spiegata dalle variabili indipendenti.

        ### 📌 **4. Adj. R-squared**

        Versione corretta del R² che penalizza l’aggiunta di troppe variabili (più affidabile in modelli complessi).

        ---

        ### 📌 **5. Altre metriche**

        | Metrica           | Significato                                                                 |
        |-------------------|------------------------------------------------------------------------------|
        | **AIC / BIC**     | Misure di bontà del modello penalizzando la complessità (più basso è meglio) |
        | **F-statistic**   | Test globale sulla significatività del modello                               |
        | **Prob (F-stat)** | p-value associato all’F-statistic: indica se il modello è globalmente utile  |

        ---

        ### ✅ Conclusione

        La combinazione di:
        - coefficienti significativi (p < 0.05)
        - R² e Adj. R² elevati
        - AIC/BIC bassi

        ...è indice di un **modello robusto, interpretabile e utile** per capire cosa influenza la variabile target nel contesto dell’analisi.

                """)
            except Exception as e:
                st.error(f"Errore durante la regressione: {e}")