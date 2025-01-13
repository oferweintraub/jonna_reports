import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Markdown, clear_output, HTML
from individual_users_report import UserAnalysisReport

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 10]
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Find the most recent enhanced analysis files
pre_war_pattern = os.path.join('data', 'analysis', 'pre_war', 'merged_analysis_pre_war_enhanced_*.csv')
post_war_pattern = os.path.join('data', 'analysis', 'post_war', 'merged_analysis_post_war_enhanced_*.csv')

pre_war_files = glob.glob(pre_war_pattern)
post_war_files = glob.glob(post_war_pattern)

if not pre_war_files or not post_war_files:
    raise FileNotFoundError("Enhanced analysis files not found. Please run the enhanced analysis first.")

# Get the most recent files
pre_war_file = max(pre_war_files, key=os.path.getctime)
post_war_file = max(post_war_files, key=os.path.getctime)

# Read the enhanced analysis results
pre_war_enhanced = pd.read_csv(pre_war_file)
post_war_enhanced = pd.read_csv(post_war_file)

# Define test users
test_users = ['ptr_dvd', 'SagiBarmak']

# Create results dictionary
period_results = {
    'pre_war': pre_war_enhanced,
    'post_war': post_war_enhanced
}

# Initialize report generator
report_generator = UserAnalysisReport()

# Generate report and get the figures
report_path, figures = report_generator.generate_report(
    period_results=period_results,
    test_users=test_users
)

# Clear any previous output and close any existing figures
clear_output(wait=True)
plt.close('all')

# Read the report content
with open(report_path, 'r', encoding='utf-8') as f:
    report_content = f.read()

# Split into sections (header and user sections)
sections = report_content.split('## ')

# Display the header (first section)
display(Markdown(sections[0].strip()))

# Process each user section
for i, section in enumerate(sections[1:], 0):
    if not section.strip():  # Skip empty sections
        continue
        
    # Split the section into lines
    lines = section.split('\n')
    section_title = lines[0]
    
    # Start building the markdown content
    markdown_content = f"## {section_title}\n\n"
    
    # Add all content up to any existing visualization marker
    content_before_viz = []
    for line in lines[1:]:
        if line.strip() == "### Data Visualization":
            break
        content_before_viz.append(line)
    
    markdown_content += '\n'.join(content_before_viz)
    
    # Display the content before visualization
    display(Markdown(markdown_content))
    
    # Display the corresponding figure if available
    if i < len(figures):
        display(Markdown("### Data Visualization"))
        display(figures[i])
    
    # Add any remaining content after the visualization
    remaining_content = []
    found_viz = False
    for line in lines[1:]:
        if found_viz:
            remaining_content.append(line)
        if line.strip() == "### Data Visualization":
            found_viz = True
    
    if remaining_content:
        display(Markdown('\n'.join(remaining_content)))
    
    # Add separator between users (except for last user)
    if i < len(sections) - 2:
        display(Markdown("---"))
    
    plt.close('all')  # Close figures after each section

print(f"\nReport generated successfully: {report_path}") 