# ADME Predictor App
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, Draw, rdMolDescriptors
import io

st.set_page_config(page_title="ADME Predictor", layout="centered")

st.title("ADME Predictor")
st.write("Predict physicochemical properties from a SMILES string using RDKit.")

# -----------------------------
# Helper functions
# -----------------------------

def calculate_properties(smiles):
    if not smiles or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None

    mw   = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    rb   = Lipinski.NumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rings = rdMolDescriptors.CalcNumAromaticRings(mol)

    # FIX: FractionCSP3 belongs to Descriptors, not Lipinski
    ap = Descriptors.FractionCSP3(mol)

    # FIX: Corrected Delaney ESOL formula coefficients
    logS = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap

    return {
        "mol": mol,
        "Molecular Weight (MW)": round(mw, 2),
        "LogP": round(logp, 2),
        "H-Bond Donors (HBD)": hbd,
        "H-Bond Acceptors (HBA)": hba,
        "Rotatable Bonds": rb,
        "TPSA (Å²)": round(tpsa, 2),
        "Aromatic Rings": rings,
        "LogS (ESOL)": round(logS, 2),
    }


def check_lipinski(props):
    violations = sum([
        props["Molecular Weight (MW)"] > 500,
        props["LogP"] > 5,
        props["H-Bond Donors (HBD)"] > 5,
        props["H-Bond Acceptors (HBA)"] > 10,
    ])
    return violations


def mol_to_image(mol):
    img = Draw.MolToImage(mol, size=(300, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# -----------------------------
# Tabs: Single SMILES | Batch CSV
# -----------------------------

tab1, tab2 = st.tabs(["🔬 Single Molecule", "📂 Batch CSV"])

# ── Tab 1: Single molecule ──────────────────────────────────────────────────
with tab1:
    smiles_input = st.text_input(
        "Enter SMILES string",
        value="CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
    )

    if st.button("Predict Properties", key="single"):

        # FIX: guard against empty input
        if not smiles_input.strip():
            st.warning("⚠️ Please enter a SMILES string.")
        else:
            results = calculate_properties(smiles_input)

            if results is None:
                st.error("❌ Invalid SMILES string. Please check and try again.")
            else:
                mol = results.pop("mol")  # separate mol object from display data

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.success("✅ Prediction Successful")
                    df = pd.DataFrame(results.items(), columns=["Property", "Value"])
                    st.table(df)

                with col2:
                    st.markdown("#### 2D Structure")
                    st.image(mol_to_image(mol), caption=smiles_input)

                # Lipinski Ro5 check
                violations = check_lipinski(results)
                if violations <= 1:
                    st.success(f"💊 Lipinski's Rule of Five: {violations} violation(s) — Drug-like ✅")
                else:
                    st.error(f"💊 Lipinski's Rule of Five: {violations} violation(s) — Poor oral bioavailability likely ❌")

                # Interpretation
                st.markdown("### 📖 Interpretation")
                st.write("- **MW < 500 Da** — Favorable for oral absorption")
                st.write("- **LogP 1–5** — Balanced lipophilicity")
                st.write("- **HBD ≤ 5, HBA ≤ 10** — Lipinski's Rule of Five")
                st.write("- **TPSA < 140 Å²** — Good intestinal absorption; < 90 Å² for CNS penetration")
                st.write("- **Rotatable Bonds ≤ 10** — Good oral bioavailability")
                st.write("- **LogS** — More negative = lower aqueous solubility")


# ── Tab 2: Batch CSV ────────────────────────────────────────────────────────
with tab2:
    st.markdown("Upload a CSV file with a column named **`SMILES`**.")
    uploaded = st.file_uploader("Choose CSV file", type="csv")

    if uploaded:
        df_input = pd.read_csv(uploaded)

        if "SMILES" not in df_input.columns:
            st.error("❌ CSV must contain a column named 'SMILES'.")
        else:
            st.info(f"Processing {len(df_input)} molecules...")

            rows = []
            for smi in df_input["SMILES"]:
                props = calculate_properties(str(smi))
                if props:
                    props.pop("mol")
                    props["SMILES"] = smi
                    rows.append(props)
                else:
                    rows.append({"SMILES": smi, "Error": "Invalid SMILES"})

            df_out = pd.DataFrame(rows)
            st.dataframe(df_out)

            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name="adme_results.csv",
                mime="text/csv"
            )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Educational tool for ADME/cheminformatics training · Powered by RDKit")
