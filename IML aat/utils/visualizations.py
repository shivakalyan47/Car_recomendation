import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Curated premium color palettes
PRIMARY_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
BRAND_COLORS = {
    'Petrol': '#4A90E2',   # Vibrant Blue
    'Diesel': '#8B572A',   # Bronze/Brown
    'CNG': '#F5A623',      # Gold/Orange
    'Electric': '#00C853'  # Electric Green
}

def plot_fuel_type_distribution(df):
    """
    Creates a premium donut chart showing the distribution of cars by fuel type.
    """
    fuel_counts = df['Fuel Type'].value_counts().reset_index()
    fuel_counts.columns = ['Fuel Type', 'Count']
    
    # Map colors to ensure consistent themes
    colors = [BRAND_COLORS.get(ft, '#7F8C8D') for ft in fuel_counts['Fuel Type']]
    
    fig = px.pie(
        fuel_counts, 
        values='Count', 
        names='Fuel Type', 
        hole=0.45,
        title='Market Share by Fuel Type',
        color_discrete_sequence=colors
    )
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hoverinfo='label+value',
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        title_x=0.5,
        title_font=dict(size=18, family="Inter, Arial, sans-serif"),
        margin=dict(t=50, b=50, l=20, r=20),
        height=380
    )
    
    return fig

def plot_price_analysis(df):
    """
    Generates a premium bar chart showing average pricing by brand (Company Name).
    """
    avg_price = df.groupby('Company Name')['Price'].mean().reset_index()
    avg_price = avg_price.sort_values(by='Price', ascending=False)
    
    fig = px.bar(
        avg_price,
        x='Price',
        y='Company Name',
        orientation='h',
        title='Average Car Price by Brand (Lakhs)',
        labels={'Price': 'Average Price (in ₹ Lakhs)', 'Company Name': 'Brand'},
        color='Price',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=18, family="Inter, Arial, sans-serif"),
        xaxis=dict(showgrid=True, gridcolor='rgba(0, 0, 0, 0.05)'),
        yaxis=dict(categoryorder='total ascending'),
        coloraxis_showscale=False,
        margin=dict(t=50, b=50, l=20, r=20),
        height=400
    )
    
    fig.update_traces(
        hovertemplate="<b>Brand:</b> %{y}<br><b>Avg Price:</b> ₹%{x:.2f} Lakhs<extra></extra>"
    )
    
    return fig

def plot_mileage_comparison(recommended_df):
    """
    Generates a comparative horizontal bar chart showing fuel efficiency (mileage) of recommended cars.
    """
    if recommended_df.empty:
        return None
        
    df_sorted = recommended_df.sort_values(by='Mileage', ascending=True)
    
    fig = px.bar(
        df_sorted,
        x='Mileage',
        y='Car Name',
        orientation='h',
        title='Fuel Efficiency / Range Comparison',
        labels={'Mileage': 'Mileage (kmpl or km/kWh equivalent)', 'Car Name': 'Car Model'},
        color='Mileage',
        color_continuous_scale=px.colors.sequential.Emrld
    )
    
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=18, family="Inter, Arial, sans-serif"),
        xaxis=dict(showgrid=True, gridcolor='rgba(0, 0, 0, 0.05)'),
        coloraxis_showscale=False,
        margin=dict(t=50, b=50, l=20, r=20),
        height=320
    )
    
    fig.update_traces(
        hovertemplate="<b>Car:</b> %{y}<br><b>Efficiency:</b> %{x} kmpl/eq<extra></extra>"
    )
    
    return fig

def plot_best_budget_cars(df, max_budget=None):
    """
    Lists the top 8 budget-friendly cars in the database. Optionally filtered by a max budget.
    """
    filtered = df.copy()
    if max_budget:
        filtered = filtered[filtered['Price'] <= max_budget]
        
    best_budget = filtered.sort_values(by='Price', ascending=True).head(8)
    
    fig = px.bar(
        best_budget,
        x='Car Name',
        y='Price',
        title=f"Best Budget Cars {'(under ₹' + str(max_budget) + ' Lakhs)' if max_budget else ''}",
        labels={'Price': 'Price (₹ Lakhs)', 'Car Name': 'Car Model'},
        color='Price',
        color_continuous_scale=px.colors.sequential.Sunsetdark
    )
    
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=18, family="Inter, Arial, sans-serif"),
        yaxis=dict(showgrid=True, gridcolor='rgba(0, 0, 0, 0.05)'),
        coloraxis_showscale=False,
        margin=dict(t=50, b=50, l=20, r=20),
        height=350
    )
    
    fig.update_traces(
        hovertemplate="<b>Car:</b> %{x}<br><b>Price:</b> ₹%{y:.2f} Lakhs<extra></extra>"
    )
    
    return fig
