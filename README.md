# Kohelet Forum Twitter Analysis

This project analyzes Twitter data related to the Kohelet Forum, focusing on changes in narratives and discourse before and after the October 7th war.

## Project Structure

```
.
├── analyzer.py              # Core analysis functionality
├── analyzer_enhanced.py     # Enhanced analysis with LLM integration
├── analyzer_group.py        # Group-level analysis functions
├── html_exporter.py        # HTML report generation
├── report_generator.py      # Report generation logic
├── cross_periods_analysis.ipynb  # Main analysis notebook
├── test_llm.ipynb          # LLM testing notebook
├── docs/                   # Generated reports
│   ├── index.html         # Group analysis report
│   └── users/             # Individual user reports
├── data/                  # Data directory
│   ├── raw/              # Raw Twitter data
│   ├── cleaned/          # Cleaned Twitter data
│   └── analysis/         # Analysis results
└── requirements.txt       # Project dependencies

## Analysis Process

1. **Data Collection**
   - Extract Twitter data for specified users
   - Split data into pre-war and post-war periods

2. **Data Cleaning**
   - Remove duplicates and irrelevant content
   - Standardize text format
   - Save cleaned data to CSV files

3. **Analysis**
   - Group Analysis:
     * Volume changes
     * Toxicity analysis
     * Narrative evolution
     * Metrics comparison
   - Individual User Analysis:
     * Personal metrics
     * Narrative changes
     * Behavioral shifts

4. **Report Generation**
   - Group-level report with overall trends
   - Individual user reports with detailed analysis
   - Interactive navigation between reports

## Key Features

- LLM-powered narrative analysis
- Toxicity and sentiment tracking
- Pre/post war comparison
- Interactive HTML reports
- Cross-report navigation

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   - Create .env file with AWS credentials
   - Configure AWS region

3. Run analysis:
   - Open cross_periods_analysis.ipynb
   - Run cells in order
   - Reports will be generated in docs/

## Reports

- Group Analysis: https://oferweintraub.github.io/jonna_reports/
- User Reports: https://oferweintraub.github.io/jonna_reports/users/

## Dependencies

- Python 3.8+
- AWS Bedrock (Claude 3.5 Haiku)
- Pandas
- Matplotlib
- Other requirements in requirements.txt 