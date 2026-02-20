import streamlit as st
import pandas as pd
from utils import clean_sequence, translate_if_needed, align_and_map
from model import train_model, predict_score
import os

st.set_page_config(page_title="EleProtect", layout="wide")

st.title("🧬 EleProtect v2.0")
st.markdown("### ML-Powered TP53 Hotspot Conservation Tool")
st.markdown("Exploratory Research Tool – Not for Clinical Use")

# Default human TP53
HUMAN_TP53 = """MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDPSDGSLAPPQHLIRVEGNLRAEYLDDSITLRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"""

st.sidebar.header("Sequence Analysis")

query_seq_input = st.text_area("Paste TP53 Sequence (Protein or DNA)")

if st.button("Analyze Sequence"):

    if query_seq_input.strip() == "":
        st.error("Paste a sequence first.")
    else:
        query_seq = clean_sequence(query_seq_input)
        query_seq = translate_if_needed(query_seq)

        df = align_and_map(HUMAN_TP53, query_seq)

        st.subheader("Hotspot Mapping Results")
        st.dataframe(df)

        conservation_score = round((df["Conserved"].sum() / len(df)) * 100, 2)
        st.success(f"Conservation Score: {conservation_score}%")

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "eleprotect_results.csv"
        )

# ML Training Section
st.sidebar.header("Train ML Model")

feature_file = st.sidebar.file_uploader("Upload Feature CSV for ML Training")

if feature_file:
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", feature_file.name)
    with open(file_path, "wb") as f:
        f.write(feature_file.getbuffer())

    message = train_model(file_path)
    st.sidebar.success(message)

st.sidebar.header("Predict with ML Model")

ml_file = st.sidebar.file_uploader("Upload Feature CSV for ML Prediction")

if ml_file:
    df_features = pd.read_csv(ml_file)
    results = predict_score(df_features)
    st.subheader("ML Ranking Results")
    st.dataframe(results)
