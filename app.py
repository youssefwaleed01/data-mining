import streamlit as st
import pandas as pd
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# -------------------------------------------------------------------------
# UI Configuration (Beautiful Light Theme)
# -------------------------------------------------------------------------
st.set_page_config(page_title="Data Mining Project", layout="wide")

st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #F8F9FA;
        color: #212529;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Typography */
    h1 { color: #1E3A8A !important; font-weight: 700; }
    h2, h3 { color: #2563EB !important; font-weight: 600; margin-top: 1rem; }
    h4 { color: #3B82F6 !important; }
    
    /* Dataframes/Tables Styling - Card Look */
    .stDataFrame {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        padding: 10px;
        margin-bottom: 2rem;
    }
    
    /* Output Box */
    .freq-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 6px solid #2563EB;
        padding: 20px;
        border-radius: 8px;
        font-size: 18px;
        color: #1E3A8A;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
    }
    
    /* Hide Streamlit Header, Menu, and Deploy Button completely */
    [data-testid="stHeader"] {visibility: hidden !important; display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Custom Step-by-Step Itemset Generation
# -------------------------------------------------------------------------
def get_item_counts(transactions, itemsets):
    counts = {itemset: 0 for itemset in itemsets}
    for transaction in transactions:
        tx_set = set(transaction)
        for itemset in itemsets:
            if set(itemset).issubset(tx_set):
                counts[itemset] += 1
    return counts

def generate_step_by_step_tables(transactions, min_support_count):
    items = set()
    for tx in transactions:
        for item in tx:
            items.add(frozenset([item]))
            
    current_L = items
    k = 1
    steps_data = []
    
    last_passed_itemsets = [] # To store only the itemsets from the final non-empty table
    
    while current_L:
        counts = get_item_counts(transactions, current_L)
        
        step_df_data = []
        next_L = set()
        current_passed_itemsets = []
        
        for itemset, count in counts.items():
            if count >= min_support_count:
                status = "✅ Accept"
                next_L.add(itemset)
                current_passed_itemsets.append(", ".join(list(itemset)))
            else:
                status = "❌ Reject"
                
            step_df_data.append({
                "Itemset": ", ".join(list(itemset)),
                "Count": count,
                "Status": status
            })
                
        step_df_data.sort(key=lambda x: (-x['Count'], x['Itemset']))
        
        if step_df_data:
            steps_data.append((k, pd.DataFrame(step_df_data)))
            
        if current_passed_itemsets:
            last_passed_itemsets = current_passed_itemsets
            
        current_L = next_L
        if not current_L:
            break
            
        next_candidates = set()
        current_L_list = list(current_L)
        for i in range(len(current_L_list)):
            for j in range(i+1, len(current_L_list)):
                union_set = current_L_list[i].union(current_L_list[j])
                if len(union_set) == k + 1:
                    next_candidates.add(union_set)
                    
        current_L = next_candidates
        k += 1
        
    return steps_data, last_passed_itemsets

# -------------------------------------------------------------------------
# Custom FP-Tree Implementation
# -------------------------------------------------------------------------
class FPTreeNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_id = str(id(self)) 

def build_fp_tree(transactions, min_support_count):
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1
            
    item_counts = {k: v for k, v in item_counts.items() if v >= min_support_count}
    
    sorted_transactions = []
    for transaction in transactions:
        filtered_tx = [item for item in transaction if item in item_counts]
        filtered_tx.sort(key=lambda x: (-item_counts[x], x))
        if filtered_tx:
            sorted_transactions.append(filtered_tx)
            
    root = FPTreeNode("Null", 1, None)
    for transaction in sorted_transactions:
        current_node = root
        for item in transaction:
            if item in current_node.children:
                current_node.children[item].count += 1
            else:
                new_node = FPTreeNode(item, 1, current_node)
                current_node.children[item] = new_node
            current_node = current_node.children[item]
            
    return root

def count_tree_nodes(node):
    count = 1
    for child in node.children.values():
        count += count_tree_nodes(child)
    return count

def generate_dot_string(root):
    dot = ['digraph FPTree {']
    dot.append('  rankdir=TB;')
    dot.append('  node [shape=circle, style=filled, fillcolor="#DBEAFE", fontcolor="#1E3A8A", color="#3B82F6", penwidth=2];')
    dot.append('  edge [color="#94A3B8", penwidth=1.5];')
    dot.append(f'  "{root.node_id}" [label="{root.item}"];')
    
    def traverse(node):
        for child in node.children.values():
            dot.append(f'  "{child.node_id}" [label="{child.item}\\n:{child.count}"];')
            dot.append(f'  "{node.node_id}" -> "{child.node_id}";')
            traverse(child)
            
    traverse(root)
    dot.append('}')
    return "\n".join(dot)

# -------------------------------------------------------------------------
# Main Application
# -------------------------------------------------------------------------
st.title("Data Mining Project: Frequent Itemsets & FP-Growth")

st.sidebar.title("Control Panel")
st.sidebar.markdown("---")

input_method = st.sidebar.radio(
    "Choose Input Method",
    ["Upload CSV", "Use Sample Dataset", "Manual Input"]
)

uploaded_file = None
if input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

import csv

@st.cache_data
def load_data(file_path_or_buffer):
    transactions = []
    
    if hasattr(file_path_or_buffer, 'read'):
        file_path_or_buffer.seek(0)
        content = file_path_or_buffer.read().decode('utf-8')
        lines = content.splitlines()
    else:
        with open(file_path_or_buffer, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
    if len(lines) > 0 and 'Items' in lines[0]:
        reader = csv.DictReader(lines)
        for row in reader:
            if 'Items' in row and row['Items']:
                items = str(row['Items']).replace(',', ' ').split()
                transactions.append(items)
    else:
        reader = csv.reader(lines)
        for row in reader:
            cleaned_row = [str(item).strip() for item in row if item.strip()]
            if cleaned_row:
                transactions.append(cleaned_row)
                
    return transactions

transactions = []
if input_method == "Upload CSV":
    if uploaded_file is not None:
        transactions = load_data(uploaded_file)
        st.sidebar.success("Dataset loaded successfully.")
elif input_method == "Use Sample Dataset":
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sample_path = os.path.join(base_dir, "sample_dataset.csv")
        transactions = load_data(sample_path)
        st.sidebar.info("Using default sample dataset.")
    except Exception:
        st.sidebar.error("Sample dataset not found.")
elif input_method == "Manual Input":
    manual_data = st.sidebar.text_area(
        "Enter transactions (one per line, items separated by comma):",
        "bread, milk\nbread, diapers, beer, eggs\nmilk, diapers, beer, cola\nbread, milk, diapers, beer\nbread, milk, diapers, cola"
    )
    if manual_data.strip():
        for line in manual_data.split('\n'):
            if ',' in line:
                cleaned_line = [item.strip() for item in line.split(',') if item.strip()]
            else:
                cleaned_line = [item.strip() for item in line.split() if item.strip()]
            if cleaned_line:
                transactions.append(cleaned_line)
        if transactions:
            st.sidebar.success(f"{len(transactions)} manual transactions loaded.")

if len(transactions) > 0:
    st.markdown("### Dataset Preview")
    st.write(f"**Total Transactions:** {len(transactions)}")
    st.write("First 5 transactions:", transactions[:5])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Algorithm Parameters")
    
    # Changed from ratio slider to absolute count input
    min_support_count = st.sidebar.number_input("Minimum Support (Count)", min_value=1, max_value=len(transactions), value=2, step=1)
    # Calculate ratio for mlxtend which requires ratio
    min_support_ratio = min_support_count / len(transactions)
    
    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("Run Algorithm")
    
    if run_btn:
        st.markdown("---")
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = fpgrowth(df_encoded, min_support=min_support_ratio, use_colnames=True)
        
        if frequent_itemsets.empty:
            st.error(f"No frequent itemsets found with Minimum Support Count = {min_support_count}.")
        else:
            tab1, tab2, tab3 = st.tabs(["Step-by-Step Itemsets", "Final Rules & Confidence", "FP-Growth Tree"])
            
            with tab1:
                st.header(f"Step-by-Step Generation (Min Support = {min_support_count})")
                st.markdown("Calculating the absolute **Count** for individual items, pairs, triplets, etc. Items below minimum support are rejected.")
                
                steps_data, last_passed_itemsets = generate_step_by_step_tables(transactions, min_support_count)
                
                for k, df_step in steps_data:
                    st.divider()
                    st.subheader(f"Step {k}: ({k}-itemsets)")
                    st.dataframe(df_step, use_container_width=True)
                
                st.divider()
                st.markdown("### Final Output")
                formatted_freq_items = " { " + " , ".join([f"({x})" for x in last_passed_itemsets]) + " } "
                st.markdown(f'<div class="freq-box"><b>Largest Frequent Items (Last Table)</b> = <br>{formatted_freq_items}</div>', unsafe_allow_html=True)
                
            with tab2:
                st.subheader("Association Rules & Confidence")
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.01, num_itemsets=len(frequent_itemsets))
                if rules.empty:
                    st.info("No rules can be generated from the current itemsets.")
                else:
                    rules['If (Antecedent)'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules['Then (Consequent)'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    display_rules = rules[['If (Antecedent)', 'Then (Consequent)', 'support', 'confidence']]
                    display_rules.columns = ['If (Antecedent)', 'Then (Consequent)', 'Support Ratio', 'Confidence']
                    
                    display_rules['Support Ratio'] = display_rules['Support Ratio'].apply(lambda x: f"{x:.2f}")
                    display_rules['Confidence'] = display_rules['Confidence'].apply(lambda x: f"{x:.2f}")
                    
                    st.dataframe(display_rules.sort_values(by="Confidence", ascending=False), use_container_width=True)
                    
            with tab3:
                st.subheader("FP-Tree Structure")
                fp_tree_root = build_fp_tree(transactions, min_support_count)
                num_nodes = count_tree_nodes(fp_tree_root)
                
                if num_nodes > 150:
                    st.warning(f"⚠️ **Tree is too massive to visualize!**\nThe generated FP-Tree has **{num_nodes} nodes**. Drawing a tree this huge will freeze the browser and look like a giant ink blot.\n\n👉 **Solution:** Please increase the *Minimum Support* to reduce the number of items, or use a smaller dataset if you want to see the drawing.")
                else:
                    st.markdown("Structural representation of the FP-Tree built from the transactions.")
                    dot_string = generate_dot_string(fp_tree_root)
                    st.graphviz_chart(dot_string, use_container_width=True)
