import base64
import io
from datetime import datetime
import markdown
import matplotlib.pyplot as plt

class HTMLExporter:
    def __init__(self):
        """Initialize the HTML exporter with custom styling."""
        self.style = """
        <style>
            body { 
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 { 
                color: #2c3e50;
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 30px;
            }
            h2 { 
                color: #34495e;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
                margin-top: 30px;
            }
            h3 { 
                color: #455a64;
                margin-top: 25px;
            }
            ul { 
                padding-left: 20px;
            }
            li {
                margin: 10px 0;
            }
            .figure {
                text-align: center;
                margin: 30px 0;
            }
            .figure img {
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            pre {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 4px;
                overflow-x: auto;
            }
            code {
                font-family: 'Courier New', Courier, monospace;
            }
        </style>
        """

    def _figure_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def export_report(self, report_text: str, figures: list[plt.Figure], output_path: str = None) -> str:
        """
        Export the report to HTML.
        
        Args:
            report_text: Markdown formatted report text
            figures: List of matplotlib figures to include
            output_path: Optional path to save the HTML file
            
        Returns:
            Path to the generated HTML file
        """
        # Convert markdown to HTML
        html_content = markdown.markdown(report_text)
        
        # Generate base64 strings for figures
        figure_htmls = []
        for i, fig in enumerate(figures, 1):
            img_base64 = self._figure_to_base64(fig)
            figure_html = f'<div class="figure"><img src="data:image/png;base64,{img_base64}" alt="Figure {i}"></div>'
            figure_htmls.append(figure_html)
        
        # Combine everything into final HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Analysis Report</title>
            {self.style}
        </head>
        <body>
            <div class="container">
                {html_content}
                {''.join(figure_htmls)}
            </div>
        </body>
        </html>
        """
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'analysis_report_{timestamp}.html'
        
        # Save the HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path 