# Streamlit Logistics Analytics Dashboard

This repository provides an interactive Streamlit dashboard for cross-analyzing logistics delivery reliability, customer experience feedback, cost breakdowns, and operational risks using consolidated analytics from multiple CSV datasets.

🚀 ## Features
* Unified analytics from orders, delivery performance, routes, costs, feedback, inventory, and fleet.
* Dynamic filtering by date, customer segment, carrier, route, origin/destination.
* Multiple interactive visualizations: histograms, line charts, box plots (delivery time by priority), stacked bar charts (cost breakdown).
* Automated KPI summary and insights section.
* Downloadable filtered data and recommendations.
* Modular, well-commented code base for easy extension.

📦 ## Repository Structure
.
├── app.py # Main Streamlit dashboard application
├── scripts/
│   └── load_data.py # Data loading and transformation logic
├── data/ # Folder containing all CSV datasets
├── requirements.txt # Python dependencies

📓 ## Setup Instructions

1.  **Clone this repository**
    ```bash
    git clone [https://github.com/vedantnarang/ai_intern.git](https://github.com/vedantnarang/ai_intern.git)
    cd ai_intern
    ```

2.  **(Recommended) Setup Python virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate # On macOS/Linux
    venv\Scripts\activate # On Windows
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data Setup**
    Ensure your data CSV files are placed inside the `data/` folder. If you're extending or customizing, edit/add CSVs as needed.

5.  **Run the Streamlit application**
    ```bash
    streamlit run app.py
    ```
    By default, the dashboard launches at `http://localhost:8501` in your browser.

    If the default port is busy, specify a new one:
    ```bash
    streamlit run app.py --server.port 8502
    ```

🔍 ## How to Use
* Use sidebar filters to view and analyze delivery performance, feedback, or cost drivers across any segment, carrier, or route.
* Expand visualization and insights sections to see operational bottlenecks, cost structure, and actionable recommendations.
* Export filtered data and insights for further analysis or sharing.

📝 ## Customization
* To add more derived metrics, extend logic in `load_data.py`.
* New visualizations or analytics panels can be easily added in `app.py` using Plotly or Altair.
