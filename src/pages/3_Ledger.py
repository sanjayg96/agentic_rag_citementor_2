import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Royalty Ledger", layout="wide")

st.title("💰 Micro-Royalty Ledger")
st.caption("Transparent, real-time accounting of attribution costs per session.")

# Load Catalog for full titles
def load_titles():
    try:
        with open("catalog.json", "r") as f:
            catalog = json.load(f)
            return {book_id: data["title"] for book_id, data in catalog.items()}
    except Exception:
        return {}

book_titles = load_titles()

# Get session details
ledger_data = st.session_state.get("ledger_details", [])

if not ledger_data:
    st.info("No royalties accumulated yet. Start chatting with the Mentor to generate micro-royalties.")
else:
    # Top-level metric
    total_cost = st.session_state.get("royalties", 0.0)
    st.metric("Total Session Distribution", f"${total_cost:.6f}")
    
    st.markdown("### Transaction Breakdown")
    
    # Process data for display
    df = pd.DataFrame(ledger_data)
    df["Title"] = df["book_id"].map(lambda x: book_titles.get(x, x))
    
    # Group by book for aggregate views
    summary_df = df.groupby(["book_id", "Title"]).agg(
        Snippets_Retrieved=('cost', 'count'),
        Total_Cost=('cost', 'sum')
    ).reset_index()
    
    # Format currency
    summary_df["Total_Cost"] = summary_df["Total_Cost"].apply(lambda x: f"${x:.6f}")
    
    st.dataframe(
        summary_df[["Title", "Snippets_Retrieved", "Total_Cost"]], 
        use_container_width=True, 
        hide_index=True
    )
    
    # Optional CSV Export
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Ledger CSV",
        data=csv,
        file_name='citementor_ledger.csv',
        mime='text/csv',
    )