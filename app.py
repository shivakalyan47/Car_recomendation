import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

# Resolve paths relative to the application directory
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(DIR_PATH, "assets", "app_icon.png")

# Load custom icon if available, fallback to emoji
try:
    app_icon = Image.open(ICON_PATH)
except Exception:
    app_icon = "🚗"

# Set page configurations
st.set_page_config(
    page_title="Car Recommender",
    page_icon=app_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modular components
from utils.preprocessing import CarPreprocessor
from utils.knn_model import recommend_cars
from utils.naive_bayes_model import recommend_cars_nb
from utils.visualizations import (
    plot_fuel_type_distribution,
    plot_price_analysis,
    plot_mileage_comparison,
    plot_best_budget_cars
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title and Header styling */
    .main-title {
        font-size: clamp(1.8rem, 5vw, 2.8rem);
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B 0%, #8E2DE2 50%, #4A00E0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: clamp(0.95rem, 2.5vw, 1.1rem);
        color: #6C7A89;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Card design */
    .car-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border: 1px solid #ECEFF1;
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    .car-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 22px rgba(0,0,0,0.12);
        border-color: #AB63FA;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 20px;
        margin-right: 5px;
        margin-bottom: 8px;
    }
    .badge-petrol { background-color: #E3F2FD; color: #1E88E5; }
    .badge-diesel { background-color: #EFEBE9; color: #6D4C41; }
    .badge-cng { background-color: #FFF8E1; color: #FFB300; }
    .badge-electric { background-color: #E8F5E9; color: #43A047; }
    .badge-manual { background-color: #F3E5F5; color: #8E24AA; }
    .badge-auto { background-color: #EDE7F6; color: #5E35B1; }
    
    /* Price tag styling */
    .price-tag {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2E7D32;
        margin-top: 10px;
    }
    
    /* Metric styling */
    .spec-table {
        width: 100%;
        margin-top: 12px;
        border-collapse: collapse;
    }
    .spec-table td {
        padding: 6px 0;
        border-bottom: 1px solid #F5F7F8;
        font-size: 0.9rem;
    }
    .spec-label {
        color: #7F8C8D;
        font-weight: 500;
    }
    .spec-value {
        font-weight: 600;
        color: #2C3E50;
        text-align: right;
    }
    
    /* Divider and general modifications */
    hr {
        margin: 1.5rem 0;
        border-color: #ECEFF1;
    }

    /* Mobile-specific adjustments */
    @media (max-width: 768px) {
        .main-title {
            margin-bottom: 0.8rem;
            line-height: 1.25;
        }
        .sub-title {
            margin-bottom: 1.5rem;
            line-height: 1.4;
        }
        .car-card {
            padding: 15px;
            margin-bottom: 15px;
        }
        .car-card:hover {
            transform: none; /* Avoid hover shift on touch devices */
            box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        }
        .price-tag {
            font-size: 1.2rem;
        }
        /* Spec tables inside cards on mobile */
        .spec-table td {
            padding: 4px 0;
            font-size: 0.85rem;
        }
        /* Badges styling on mobile */
        .badge {
            font-size: 0.75rem;
            padding: 3px 8px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Resolve paths relative to the application directory (very important for Streamlit Cloud subfolders)
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIR_PATH, "data", "cars.csv")
GENERATE_SCRIPT_PATH = os.path.join(DIR_PATH, "generate_data.py")

# Initialize preprocessor (cached to load only once)
@st.cache_resource
def get_preprocessor():
    try:
        return CarPreprocessor(data_path=DATA_PATH)
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}. Attempting to auto-generate dataset...")
        try:
            # Run generate_data programmatically in the same Python process
            # This is 100% platform-independent and avoids OS subprocess command conflicts
            with open(GENERATE_SCRIPT_PATH) as f:
                code = compile(f.read(), GENERATE_SCRIPT_PATH, "exec")
                exec(code, {"__file__": GENERATE_SCRIPT_PATH})
            return CarPreprocessor(data_path=DATA_PATH)
        except Exception as gen_err:
            st.error(f"Failed to auto-generate dataset: {gen_err}")
            raise gen_err

try:
    preprocessor = get_preprocessor()
    df = preprocessor.df
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Header Section
st.markdown("<h1 class='main-title'>AI-Powered Car Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Advanced Machine Learning Assisted Automobile Discovery (KNN & Naive Bayes)</p>", unsafe_allow_html=True)

# Sidebar - User Inputs
st.sidebar.markdown("### 🛠️ User Preferences")

# 1. Budget slider (based on data min and max)
min_price = float(df['Price'].min())
max_price = float(df['Price'].max())
budget = st.sidebar.slider(
    "Target Budget (Max Price in Lakhs)",
    min_value=round(min_price, 1),
    max_value=round(max_price, 1),
    value=15.0,
    step=0.5,
    format="₹%.1fL"
)

# 2. Fuel Type multiselect/select
fuel_options = ['All'] + preprocessor.fuel_types
selected_fuel = st.sidebar.selectbox("Preferred Fuel Type", fuel_options, index=0)

# 3. Transmission type
transmission_options = ['All'] + preprocessor.transmissions
selected_trans = st.sidebar.selectbox("Transmission Type", transmission_options, index=0)

# 4. Seating Capacity
seat_options = ['All'] + sorted([str(s) for s in df['Seats'].unique()])
selected_seats = st.sidebar.selectbox("Seating Capacity", seat_options, index=0)

# 5. Mileage (Min Mileage slider)
min_mileage_data = float(df['Mileage'].min())
max_mileage_data = float(df['Mileage'].max())
mileage = st.sidebar.slider(
    "Minimum Target Mileage / EV Range (kmpl / eq)",
    min_value=round(min_mileage_data, 1),
    max_value=round(max_mileage_data, 1),
    value=15.0,
    step=1.0
)

# 6. Select ML Algorithm
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Select ML Algorithm")
ml_algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["K-Nearest Neighbors (KNN)", "Naive Bayes Classifier"],
    index=0,
    help="Toggle between the K-Nearest Neighbors (distance-based) and Naive Bayes (probability-based) recommendation engines."
)

if ml_algorithm == "K-Nearest Neighbors (KNN)":
    st.sidebar.markdown("### ⚙️ KNN Parameters")
    recommendation_mode = st.sidebar.radio(
        "Recommendation Mode",
        ["Strict Filtering", "Soft Matching"],
        index=0,
        help="Strict Mode filters by categoricals exactly first, then applies KNN to numbers. Soft Mode matches across all parameters collectively using distance calculations."
    )
    
    distance_metric = st.sidebar.selectbox(
        "Similarity Distance Metric",
        ["Euclidean Distance", "Cosine Similarity"],
        index=0,
        help="Euclidean looks at absolute difference, Cosine looks at vector angle alignments."
    )
    
    metric_key = 'euclidean' if distance_metric == "Euclidean Distance" else 'cosine'
    top_n = st.sidebar.slider("Number of Recommendations", 3, 10, 5)
else:
    st.sidebar.markdown("### ⚙️ Naive Bayes Parameters")
    decay_factor = st.sidebar.slider(
        "Probability Smoothing Weight",
        1.0, 5.0, 2.5,
        step=0.5,
        help="Controls the spread and sensitivity of the match confidence scores. A higher value makes the confidence score higher for marginal matches."
    )
    top_n = st.sidebar.slider("Number of Recommendations", 3, 10, 5)



# Build Query Dictionary
query = {
    'Price': budget,
    'Mileage': mileage,
    'Seats': int(selected_seats) if selected_seats != 'All' else 'All',
    'Fuel Type': selected_fuel,
    'Transmission': selected_trans,
    'Engine': 1200  # Default target engine displacement
}

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Car Recommender", 
    "⚖️ Compare Cars", 
    "📊 Market Analytics", 
    "🔍 Browse Catalog",
    "🤖 Compare Algorithms"
])

# ----------------- TAB 1: RECOMMENDATIONS -----------------
with tab1:
    st.subheader("🤖 Top Recommended Vehicles")
    
    # Process Recommendation and write custom algorithm descriptive texts
    if ml_algorithm == "K-Nearest Neighbors (KNN)":
        st.write(
            f"The **K-Nearest Neighbors (KNN)** model evaluated **{len(df)}** vehicles using **{distance_metric}** under **{recommendation_mode}** and identified the following top matches based on your inputs:"
        )
        mode_key = 'Strict' if recommendation_mode == "Strict Filtering" else 'Soft'
        recommendations = recommend_cars(preprocessor, query, top_n=top_n, mode=mode_key, metric=metric_key)
    else:
        st.write(
            f"The **Naive Bayes Classifier** model calculated the continuous **Gaussian Probability Density** (for Price, Mileage, and Engine) and Laplace-smoothed discrete categorical probabilities for **{len(df)}** vehicles, identifying the following top matches:"
        )
        recommendations = recommend_cars_nb(preprocessor, query, top_n=top_n, decay_factor=decay_factor)
    
    if recommendations.empty:
        st.warning(
            "⚠️ No exact matches found matching your strict criteria (Fuel Type, Transmission, and Seating combined). "
            "Please switch the **Recommendation Mode** to **'Soft Matching'** in the sidebar to view closely matched options, or loosen your constraints!"
        )
    else:
        # Layout recommendations beautifully
        # Use columns in groups of 3 to list the cards
        cols_per_row = 3
        recs_list = recommendations.to_dict('records')
        
        for i in range(0, len(recs_list), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(recs_list):
                    car = recs_list[i + j]
                    with cols[j]:
                        # Fuel type class
                        ft_class = f"badge-{car['Fuel Type'].lower()}"
                        trans_class = f"badge-manual" if car['Transmission'] == "Manual" else "badge-auto"
                        
                        # Engine details string
                        engine_display = f"{int(car['Engine'])} cc" if car['Engine'] > 0 else "Electric Drive (EV)"
                        
                        # Set confidence color
                        conf = car['Confidence Score']
                        if conf >= 85:
                            conf_color = "green"
                        elif conf >= 70:
                            conf_color = "orange"
                        else:
                            conf_color = "red"
                            
                        # Card HTML
                        st.markdown(f"""
                        <div class="car-card">
                            <span style="font-size:0.8rem; font-weight:600; color:#8E2DE2; text-transform:uppercase;">{car['Company Name']}</span>
                            <h3 style="margin:2px 0 10px 0; color:#2C3E50; font-size:1.3rem;">{car['Car Name']}</h3>
                            <div>
                                <span class="badge {ft_class}">{car['Fuel Type']}</span>
                                <span class="badge {trans_class}">{car['Transmission']}</span>
                            </div>
                            <table class="spec-table">
                                <tr>
                                    <td class="spec-label">⛽ Efficiency / Range</td>
                                    <td class="spec-value">{car['Mileage']} {'km/charge' if car['Fuel Type'] == 'Electric' else 'kmpl'}</td>
                                </tr>
                                <tr>
                                    <td class="spec-label">⚙️ Engine displacement</td>
                                    <td class="spec-value">{engine_display}</td>
                                </tr>
                                <tr>
                                    <td class="spec-label">💺 Seating Capacity</td>
                                    <td class="spec-value">{int(car['Seats'])} Seater</td>
                                </tr>
                            </table>
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:15px;">
                                <div class="price-tag">₹{car['Price']:.2f} L</div>
                                <span style="font-size:0.9rem; font-weight:600; color:{conf_color};">⚡ {conf}% Match</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Dynamic Streamlit progress bar for confidence
                        st.progress(min(max(float(conf) / 100.0, 0.0), 1.0))
        
        # Display extra visualizations corresponding to recommended subset
        st.markdown("---")
        st.subheader("📈 Recommendations Analysis")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            mileage_chart = plot_mileage_comparison(recommendations)
            if mileage_chart:
                st.plotly_chart(mileage_chart, use_container_width=True)
        
        with chart_col2:
            st.markdown("#### 🤖 Recommendation Diagnostics")
            if ml_algorithm == "K-Nearest Neighbors (KNN)":
                st.write(
                    "Below are details regarding the mathematical distances calculated by the **KNN** algorithm for each recommendation. "
                    "The **Distance** indicates absolute spatial vector difference from your scaled input vector (smaller is better)."
                )
            else:
                st.write(
                    "Below are details regarding the mathematical log-likelihood metrics calculated by the **Naive Bayes** algorithm. "
                    "The **Distance** indicates the log-likelihood difference ($\\log L_{\\text{max}} - \\log L_i$) from a theoretical perfect match (smaller is better)."
                )
            diagnostics_df = recommendations[['Car Name', 'Price', 'Mileage', 'Fuel Type', 'Transmission', 'Distance', 'Confidence Score']]
            st.dataframe(
                diagnostics_df.style.background_gradient(subset=['Distance'], cmap='Reds')
                                   .background_gradient(subset=['Confidence Score'], cmap='Greens'),
                use_container_width=True
            )

# ----------------- TAB 2: COMPARE CARS -----------------
with tab2:
    st.subheader("⚖️ Side-by-Side Car Specifications Comparison")
    st.write("Select any two cars from our dataset to compare their details and view structured analytics.")
    
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        car_a_name = st.selectbox("Select Car A:", df['Car Name'].sort_values().unique(), index=0)
    with comp_col2:
        # Set a default index for Car B so they aren't identical
        all_cars = df['Car Name'].sort_values().unique()
        b_idx = min(1, len(all_cars) - 1)
        car_b_name = st.selectbox("Select Car B:", all_cars, index=b_idx)
        
    if car_a_name == car_b_name:
        st.warning("Please select two different vehicles to compare.")
    else:
        car_a = df[df['Car Name'] == car_a_name].iloc[0]
        car_b = df[df['Car Name'] == car_b_name].iloc[0]
        
        # Layout columns
        col_spec1, col_spec2 = st.columns(2)
        
        # Highlight logic function helper
        def get_highlight_span(val_a, val_b, lower_is_better=False, is_numeric=True):
            if not is_numeric:
                return f"<span>{val_a}</span>", f"<span>{val_b}</span>"
            
            diff = val_a - val_b
            if diff == 0:
                return f"<span>{val_a}</span>", f"<span>{val_b}</span>"
                
            a_is_better = (diff < 0) if lower_is_better else (diff > 0)
            
            if a_is_better:
                return f"<span style='color:#2E7D32; font-weight:700;'>{val_a} ✔</span>", f"<span style='color:#7F8C8D;'>{val_b}</span>"
            else:
                return f"<span style='color:#7F8C8D;'>{val_a}</span>", f"<span style='color:#2E7D32; font-weight:700;'>{val_b} ✔</span>"

        # Generate compared highlights
        price_a, price_b = get_highlight_span(car_a['Price'], car_b['Price'], lower_is_better=True)
        mileage_a, mileage_b = get_highlight_span(car_a['Mileage'], car_b['Mileage'], lower_is_better=False)
        seats_a, seats_b = get_highlight_span(car_a['Seats'], car_b['Seats'], lower_is_better=False)
        engine_a, engine_b = get_highlight_span(car_a['Engine'], car_b['Engine'], lower_is_better=False)
        
        st.markdown(f"""
        <div style="overflow-x: auto; -webkit-overflow-scrolling: touch; width: 100%;">
        <table style="width:100%; border-collapse:collapse; margin-top:20px; font-size:1.05rem; min-width: 600px;">
            <thead>
                <tr style="background-color:#F5F7F8; border-bottom:2px solid #DCE4EC;">
                    <th style="padding:15px; text-align:left; width:30%;">Feature Specification</th>
                    <th style="padding:15px; text-align:center; width:35%;">{car_a['Car Name']}</th>
                    <th style="padding:15px; text-align:center; width:35%;">{car_b['Car Name']}</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid #ECEFF1;">
                    <td style="padding:15px; font-weight:600; color:#5D6D7E;">Brand Name</td>
                    <td style="padding:15px; text-align:center; font-weight:600; color:#2C3E50;">{car_a['Company Name']}</td>
                    <td style="padding:15px; text-align:center; font-weight:600; color:#2C3E50;">{car_b['Company Name']}</td>
                </tr>
                <tr style="border-bottom:1px solid #ECEFF1; background-color:#FAFCFD;">
                    <td style="padding:15px; font-weight:600; color:#5D6D7E;">Price (Lakhs)</td>
                    <td style="padding:15px; text-align:center; font-size:1.15rem;">₹{price_a}</td>
                    <td style="padding:15px; text-align:center; font-size:1.15rem;">₹{price_b}</td>
                </tr>
                <tr style="border-bottom:1px solid #ECEFF1;">
                    <td style="padding:15px; font-weight:600; color:#5D6D7E;">Mileage / Efficiency</td>
                    <td style="padding:15px; text-align:center;">{mileage_a} {'km/charge' if car_a['Fuel Type'] == 'Electric' else 'kmpl'}</td>
                    <td style="padding:15px; text-align:center;">{mileage_b} {'km/charge' if car_b['Fuel Type'] == 'Electric' else 'kmpl'}</td>
                </tr>
                <tr style="border-bottom:1px solid #ECEFF1; background-color:#FAFCFD;">
                    <td style="padding:15px; font-weight:600; color:#5D6D7E;">Fuel Type</td>
                    <td style="padding:15px; text-align:center; font-weight:600;"><span class="badge badge-{car_a['Fuel Type'].lower()}">{car_a['Fuel Type']}</span></td>
                    <td style="padding:15px; text-align:center; font-weight:600;"><span class="badge badge-{car_b['Fuel Type'].lower()}">{car_b['Fuel Type']}</span></td>
                </tr>
                <tr style="border-bottom:1px solid #ECEFF1;">
                    <td style="padding:15px; font-weight:600; color:#5D6D7E;">Transmission</td>
                    <td style="padding:15px; text-align:center; font-weight:600;"><span class="badge badge-{'manual' if car_a['Transmission'] == 'Manual' else 'auto'}">{car_a['Transmission']}</span></td>
                    <td style="padding:15px; text-align:center; font-weight:600;"><span class="badge badge-{'manual' if car_b['Transmission'] == 'Manual' else 'auto'}">{car_b['Transmission']}</span></td>
                </tr>
                <tr style="border-bottom:1px solid #ECEFF1; background-color:#FAFCFD;">
                    <td style="padding:15px; font-weight:600; color:#5D6D7E;">Seating Capacity</td>
                    <td style="padding:15px; text-align:center;">{seats_a} Seats</td>
                    <td style="padding:15px; text-align:center;">{seats_b} Seats</td>
                </tr>
                <tr style="border-bottom:1px solid #ECEFF1;">
                    <td style="padding:15px; font-weight:600; color:#5D6D7E;">Engine Capacity (displacement)</td>
                    <td style="padding:15px; text-align:center;">{engine_a} { 'cc' if car_a['Engine'] > 0 else 'EV Motor' }</td>
                    <td style="padding:15px; text-align:center;">{engine_b} { 'cc' if car_b['Engine'] > 0 else 'EV Motor' }</td>
                </tr>
            </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary analysis of which is better
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Quick spec note:** Highlighted features (in green with checkmarks) represent superior characteristics (e.g. higher mileage, lower pricing, larger capacity) according to basic industry standards.")

# ----------------- TAB 3: MARKET ANALYTICS -----------------
with tab3:
    st.subheader("📊 Market Trends & Database Insights")
    st.write("Understand catalog distributions, average prices, and optimal models in our vehicle dataset.")
    
    # Standard stats widgets
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("Total Models Available", len(df), delta="Multi-Brand")
    with stat_col2:
        st.metric("Average Price", f"₹{df['Price'].mean():.2f} Lakhs", "Competitive market scale")
    with stat_col3:
        best_mil_car = df.loc[df['Mileage'].idxmax()]
        st.metric("Best Fuel/EV Efficiency", f"{best_mil_car['Mileage']} kmpl/eq", f"{best_mil_car['Car Name']}")
    with stat_col4:
        cheapest_car = df.loc[df['Price'].idxmin()]
        st.metric("Most Budget-Friendly", f"₹{cheapest_car['Price']:.2f} Lakhs", f"{cheapest_car['Car Name']}")
        
    st.markdown("---")
    
    # Plots layout
    an_col1, an_col2 = st.columns(2)
    with an_col1:
        st.plotly_chart(plot_fuel_type_distribution(df), use_container_width=True)
    with an_col2:
        st.plotly_chart(plot_price_analysis(df), use_container_width=True)
        
    st.markdown("---")
    budget_limit = st.slider("Filter Budget Limit for Budget Chart (₹ Lakhs)", 5.0, 140.0, 30.0, step=5.0)
    st.plotly_chart(plot_best_budget_cars(df, max_budget=budget_limit), use_container_width=True)

# ----------------- TAB 4: BROWSE & SEARCH -----------------
with tab4:
    st.subheader("🔍 Explore Vehicle Specifications Catalog")
    st.write("Use filters or search bar queries to explore complete specifications of all vehicles.")
    
    # Setup search inputs
    se_col1, se_col2, se_col3 = st.columns([2, 1, 1])
    with se_col1:
        search_query = st.text_input("🔍 Search by Car or Brand Name", "", placeholder="e.g. Swift, Toyota, EV")
    with se_col2:
        sort_by = st.selectbox("Sort Data By", ["Price (Low to High)", "Price (High to Low)", "Mileage (High to Low)"])
    with se_col3:
        rows_num = st.selectbox("Display Count", [10, 25, 50, 100], index=1)
        
    # Filtering dataset based on inputs
    search_df = df.copy()
    if search_query:
        search_df = search_df[
            search_df['Car Name'].str.contains(search_query, case=False) | 
            search_df['Company Name'].str.contains(search_query, case=False)
        ]
        
    # Sorting
    if sort_by == "Price (Low to High)":
        search_df = search_df.sort_values(by='Price', ascending=True)
    elif sort_by == "Price (High to Low)":
        search_df = search_df.sort_values(by='Price', ascending=False)
    else:
        search_df = search_df.sort_values(by='Mileage', ascending=False)
        
    # Display table
    st.write(f"Showing **{min(len(search_df), rows_num)}** of **{len(search_df)}** matching cars:")
    
    st.dataframe(
        search_df.head(rows_num).style.background_gradient(subset=['Price'], cmap='Blues')
                                 .background_gradient(subset=['Mileage'], cmap='Greens'),
        use_container_width=True
    )

# ----------------- TAB 5: ALGORITHM COMPARISON -----------------
with tab5:
    st.subheader("🤖 Algorithm Accuracy & Overlap Analysis")
    st.write(
        "Compare the recommendations, match confidence, and constraint satisfaction accuracy "
        "between **K-Nearest Neighbors (KNN)** and **Naive Bayes Classifier** side-by-side on your target preferences."
    )

    # 1. Run both models concurrently
    # Get parameters safely with defaults in case they are not in scope (e.g. Naive Bayes active)
    try:
        knn_mode = 'Strict' if recommendation_mode == "Strict Filtering" else 'Soft'
    except NameError:
        knn_mode = 'Strict'
        
    try:
        knn_metric_key = 'euclidean' if distance_metric == "Euclidean Distance" else 'cosine'
    except NameError:
        knn_metric_key = 'euclidean'
    
    # KNN Recommendations
    knn_recs = recommend_cars(preprocessor, query, top_n=top_n, mode=knn_mode, metric=knn_metric_key)
    
    # Naive Bayes Recommendations
    try:
        nb_decay = decay_factor
    except NameError:
        nb_decay = 2.5
    nb_recs = recommend_cars_nb(preprocessor, query, top_n=top_n, decay_factor=nb_decay)

    # 2. Calculate Overlap
    if not knn_recs.empty and not nb_recs.empty:
        knn_names = set(knn_recs['Car Name'])
        nb_names = set(nb_recs['Car Name'])
        overlap_names = knn_names.intersection(nb_names)
        overlap_percent = (len(overlap_names) / top_n) * 100 if top_n > 0 else 0
    else:
        overlap_names = set()
        overlap_percent = 0.0

    # 3. Helper function for Constraint Accuracy
    def get_car_constraint_accuracy(car, q):
        price_ok = 1.0 if car['Price'] <= q['Price'] else 0.0
        mileage_ok = 1.0 if car['Mileage'] >= q['Mileage'] else 0.0
        fuel_ok = 1.0 if q['Fuel Type'] == 'All' or car['Fuel Type'] == q['Fuel Type'] else 0.0
        trans_ok = 1.0 if q['Transmission'] == 'All' or car['Transmission'] == q['Transmission'] else 0.0
        seats_ok = 1.0 if q['Seats'] == 'All' or int(car['Seats']) == int(q['Seats']) else 0.0
        
        satisfied = price_ok + mileage_ok + fuel_ok + trans_ok + seats_ok
        return (satisfied / 5.0) * 100.0

    # 4. Calculate average confidence and constraint satisfaction accuracy
    knn_avg_conf = 0.0
    knn_avg_acc = 0.0
    if not knn_recs.empty:
        knn_avg_conf = knn_recs['Confidence Score'].mean()
        knn_recs['Constraint Accuracy'] = knn_recs.apply(lambda r: get_car_constraint_accuracy(r, query), axis=1)
        knn_avg_acc = knn_recs['Constraint Accuracy'].mean()
        
    nb_avg_conf = 0.0
    nb_avg_acc = 0.0
    if not nb_recs.empty:
        nb_avg_conf = nb_recs['Confidence Score'].mean()
        nb_recs['Constraint Accuracy'] = nb_recs.apply(lambda r: get_car_constraint_accuracy(r, query), axis=1)
        nb_avg_acc = nb_recs['Constraint Accuracy'].mean()

    # 5. Display KPI Cards
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.metric(
            label="Recommendations Overlap",
            value=f"{overlap_percent:.0f}%",
            delta=f"{len(overlap_names)} of {top_n} cars common"
        )
    with kpi_col2:
        st.metric(
            label="KNN Avg Constraint Accuracy",
            value=f"{knn_avg_acc:.1f}%",
            delta="Matches target preferences"
        )
    with kpi_col3:
        st.metric(
            label="NB Avg Constraint Accuracy",
            value=f"{nb_avg_acc:.1f}%",
            delta="Matches target preferences"
        )

    st.markdown("---")

    # 6. Side-by-side Recommendations Columns
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("### 🎯 K-Nearest Neighbors (KNN)")
        if knn_recs.empty:
            st.warning("No KNN recommendations returned under strict constraints.")
        else:
            for idx, car in knn_recs.iterrows():
                ft_class = f"badge-{car['Fuel Type'].lower()}"
                trans_class = "badge-manual" if car['Transmission'] == "Manual" else "badge-auto"
                
                # Check target matches
                price_ok = car['Price'] <= query['Price']
                mileage_ok = car['Mileage'] >= query['Mileage']
                fuel_ok = query['Fuel Type'] == 'All' or car['Fuel Type'] == query['Fuel Type']
                trans_ok = query['Transmission'] == 'All' or car['Transmission'] == query['Transmission']
                seats_ok = query['Seats'] == 'All' or int(car['Seats']) == int(query['Seats'])
                
                satisfied_count = sum([price_ok, mileage_ok, fuel_ok, trans_ok, seats_ok])
                acc = (satisfied_count / 5.0) * 100.0
                
                st.markdown(f"""
                <div class="car-card" style="border: 1px solid #AB63FA; border-left: 5px solid #AB63FA; padding: 15px; border-radius: 12px; margin-bottom: 15px;">
                    <span style="font-size:0.75rem; font-weight:700; color:#AB63FA; text-transform:uppercase;">{car['Company Name']}</span>
                    <h4 style="margin:2px 0 10px 0; color:#2C3E50; font-size:1.15rem;">{car['Car Name']}</h4>
                    <span class="badge {ft_class}">{car['Fuel Type']}</span>
                    <span class="badge {trans_class}">{car['Transmission']}</span>
                    <div style="font-size:0.8rem; margin:10px 0; border-bottom:1px solid #ECEFF1; padding-bottom:5px;">
                        <div style="display:flex; justify-content:space-between;"><span>Price:</span> <strong>₹{car['Price']:.2f}L</strong></div>
                        <div style="display:flex; justify-content:space-between;"><span>Efficiency:</span> <strong>{car['Mileage']} kmpl/eq</strong></div>
                    </div>
                    <div style="font-size:0.75rem; color:#6C7A89; background:#FAFAFA; padding:6px; border-radius:6px;">
                        <div style="display:flex; justify-content:space-between; font-weight:600;">
                            <span>Constraint Accuracy:</span>
                            <span style="color:#8E24AA;">{acc:.0f}% ({satisfied_count}/5)</span>
                        </div>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px;">
                        <div class="price-tag" style="font-size:1.1rem; margin:0; color:#2E7D32;">₹{car['Price']:.2f} L</div>
                        <span style="font-size:0.85rem; font-weight:600; color:#AB63FA;">⚡ {car['Confidence Score']}% Match</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(car['Confidence Score']) / 100.0)

    with rec_col2:
        st.markdown("### 📊 Naive Bayes Classifier")
        if nb_recs.empty:
            st.warning("No Naive Bayes recommendations returned.")
        else:
            for idx, car in nb_recs.iterrows():
                ft_class = f"badge-{car['Fuel Type'].lower()}"
                trans_class = "badge-manual" if car['Transmission'] == "Manual" else "badge-auto"
                
                # Check target matches
                price_ok = car['Price'] <= query['Price']
                mileage_ok = car['Mileage'] >= query['Mileage']
                fuel_ok = query['Fuel Type'] == 'All' or car['Fuel Type'] == query['Fuel Type']
                trans_ok = query['Transmission'] == 'All' or car['Transmission'] == query['Transmission']
                seats_ok = query['Seats'] == 'All' or int(car['Seats']) == int(query['Seats'])
                
                satisfied_count = sum([price_ok, mileage_ok, fuel_ok, trans_ok, seats_ok])
                acc = (satisfied_count / 5.0) * 100.0
                
                st.markdown(f"""
                <div class="car-card" style="border: 1px solid #00C853; border-left: 5px solid #00C853; padding: 15px; border-radius: 12px; margin-bottom: 15px;">
                    <span style="font-size:0.75rem; font-weight:700; color:#00C853; text-transform:uppercase;">{car['Company Name']}</span>
                    <h4 style="margin:2px 0 10px 0; color:#2C3E50; font-size:1.15rem;">{car['Car Name']}</h4>
                    <span class="badge {ft_class}">{car['Fuel Type']}</span>
                    <span class="badge {trans_class}">{car['Transmission']}</span>
                    <div style="font-size:0.8rem; margin:10px 0; border-bottom:1px solid #ECEFF1; padding-bottom:5px;">
                        <div style="display:flex; justify-content:space-between;"><span>Price:</span> <strong>₹{car['Price']:.2f}L</strong></div>
                        <div style="display:flex; justify-content:space-between;"><span>Efficiency:</span> <strong>{car['Mileage']} kmpl/eq</strong></div>
                    </div>
                    <div style="font-size:0.75rem; color:#6C7A89; background:#FAFAFA; padding:6px; border-radius:6px;">
                        <div style="display:flex; justify-content:space-between; font-weight:600;">
                            <span>Constraint Accuracy:</span>
                            <span style="color:#2E7D32;">{acc:.0f}% ({satisfied_count}/5)</span>
                        </div>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px;">
                        <div class="price-tag" style="font-size:1.1rem; margin:0; color:#2E7D32;">₹{car['Price']:.2f} L</div>
                        <span style="font-size:0.85rem; font-weight:600; color:#2E7D32;">⚡ {car['Confidence Score']}% Match</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(car['Confidence Score']) / 100.0)

    # 7. Chart Comparison & Diagnostics Table
    st.markdown("---")
    st.subheader("📈 Quantitative Accuracy & Match Summary")
    
    chart_col, diag_col = st.columns(2)
    
    with chart_col:
        # Plotly grouped bar chart
        import plotly.graph_objects as go
        fig = go.Figure(data=[
            go.Bar(
                name='Average Match Confidence',
                x=['KNN Model', 'Naive Bayes'],
                y=[float(knn_avg_conf), float(nb_avg_conf)],
                marker_color='#AB63FA'
            ),
            go.Bar(
                name='Avg Constraint Satisfaction',
                x=['KNN Model', 'Naive Bayes'],
                y=[float(knn_avg_acc), float(nb_avg_acc)],
                marker_color='#00C853'
            )
        ])
        fig.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Outfit, Arial, sans-serif"),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor='rgba(0, 0, 0, 0.05)', title="Percentage (%)", range=[0, 105]),
            margin=dict(t=30, b=30, l=10, r=10),
            height=340,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

    with diag_col:
        st.markdown("#### Side-by-Side Diagnostic Summary")
        st.write("Detailed metrics breakdown for the top recommended cars from each model.")
        
        diag_rows = []
        if not knn_recs.empty:
            for idx, car in knn_recs.iterrows():
                diag_rows.append({
                    'Algorithm': f'KNN #{len(diag_rows)+1}',
                    'Vehicle Model': car['Car Name'],
                    'Confidence': f"{car['Confidence Score']:.1f}%",
                    'Metric Log': f"{car['Distance']:.4f}",
                    'Constraint Acc.': f"{car['Constraint Accuracy']:.0f}%"
                })
        nb_count = 0
        if not nb_recs.empty:
            for idx, car in nb_recs.iterrows():
                nb_count += 1
                diag_rows.append({
                    'Algorithm': f'NB #{nb_count}',
                    'Vehicle Model': car['Car Name'],
                    'Confidence': f"{car['Confidence Score']:.1f}%",
                    'Metric Log': f"{car['Distance']:.4f}",
                    'Constraint Acc.': f"{car['Constraint Accuracy']:.0f}%"
                })
                
        if diag_rows:
            st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)
        else:
            st.write("No diagnostic data available.")
