import base64
import io
from datetime import datetime
import markdown
import matplotlib.pyplot as plt
import os

class HTMLExporter:
    def __init__(self):
        self.style = """
        <style>
            body {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background-color: #2d2d2d;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            h1, h2, h3 {
                color: #ffffff;
                margin-top: 1.5em;
                margin-bottom: 0.5em;
            }
            h1 { font-size: 2.2em; }
            h2 { font-size: 1.8em; }
            h3 { font-size: 1.5em; }
            p, ul, ol {
                margin-bottom: 1em;
            }
            code {
                background-color: #363636;
                padding: 2px 5px;
                border-radius: 3px;
                font-family: 'Consolas', 'Monaco', monospace;
                display: block;
                padding: 10px;
                margin: 10px 0;
                white-space: pre-wrap;
            }
            img {
                max-width: 100%;
                height: auto;
                margin: 20px 0;
                border-radius: 4px;
            }
            .figure {
                text-align: center;
                margin: 30px 0;
            }
            span[style*="color: #3498DB"] {
                color: #5dade2 !important;
                font-weight: bold;
            }
            span[style*="color: #9B59B6"] {
                color: #af7ac5 !important;
                font-weight: bold;
            }
            span[style*="color: #2ECC71"] {
                color: #52be80 !important;
                font-weight: bold;
            }
            span[style*="color: #E74C3C"] {
                color: #ec7063 !important;
                font-weight: bold;
            }
        </style>
        """

    def _figure_to_base64(self, fig):
        """Convert matplotlib figure to base64 string."""
        # Set the figure and axes background to dark theme
        fig.patch.set_facecolor('#2d2d2d')
        for ax in fig.get_axes():
            ax.patch.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            # Make spines white
            for spine in ax.spines.values():
                spine.set_color('white')
            # Make grid lines lighter
            ax.grid(True, alpha=0.15)
        
        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', 
                   facecolor='#2d2d2d', edgecolor='none',
                   pad_inches=0.1)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def export_report(self, report_text: str, figures: list, output_path: str = None):
        """Export report to HTML with embedded figures."""
        # Convert markdown to HTML
        html_content = markdown.markdown(report_text)

        # Create figure HTML
        figure_html = ""
        for fig in figures:
            if fig.get_axes():  # Only process figures that have content
                base64_fig = self._figure_to_base64(fig)
                figure_html += f'<div class="figure"><img src="data:image/png;base64,{base64_fig}"/></div>'
                plt.close(fig)  # Close the figure to free memory

        # Combine everything into HTML document
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Analysis Report</title>
            {self.style}
        </head>
        <body>
            <div class="container">
                {html_content}
                {figure_html}
            </div>
        </body>
        </html>
        """

        # Set output path
        if output_path is None:
            os.makedirs('docs', exist_ok=True)
            if "# Group Analysis Report" in report_text:
                output_path = 'docs/index.html'
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f'docs/report_{timestamp}.html'

        # Save HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_doc)

        return output_path 