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
                padding: 0;
            }
            .nav {
                background-color: #2d2d2d;
                padding: 15px 20px;
                position: sticky;
                top: 0;
                z-index: 1000;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
                border-bottom: 1px solid #404040;
            }
            .nav-content {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .nav a {
                color: #ffffff;
                text-decoration: none;
                padding: 8px 16px;
                border-radius: 6px;
                transition: all 0.3s ease;
                margin: 0 5px;
                font-weight: 500;
            }
            .nav a:hover {
                background-color: #404040;
                transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            .nav .current {
                background-color: #404040;
                font-weight: bold;
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            .container {
                max-width: 800px;
                margin: 20px auto;
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

    def _generate_nav(self, page_type='group'):
        """Generate navigation HTML based on the current page type."""
        nav_html = """
        <div class="nav">
            <div class="nav-content">
                <div>
                    <a href="/jonna_reports/index.html" class="{group_class}">📊 Group Analysis</a>
                    <a href="/jonna_reports/users/index.html" class="{user_class}">👥 User Analysis</a>
                </div>
                {back_link}
            </div>
        </div>
        """
        
        group_class = 'current' if page_type == 'group' else ''
        user_class = 'current' if page_type in ['user', 'user_detail'] else ''
        back_link = '<a href="/jonna_reports/users/index.html">⬅️ Back to User List</a>' if page_type == 'user_detail' else ''
        
        return nav_html.format(
            group_class=group_class,
            user_class=user_class,
            back_link=back_link
        )

    def export_report(self, report_text: str, figures: list, output_path: str = None):
        """Export report to HTML file."""
        if output_path is None:
            output_path = 'docs/index.html'
        
        # Determine page type based on output path
        if output_path == 'docs/index.html':
            page_type = 'group'
        elif output_path == 'docs/users/index.html':
            page_type = 'user'
        else:
            page_type = 'user_detail'
        
        # Convert markdown to HTML
        html_content = markdown.markdown(report_text)
        
        # Create figure HTML
        figures_html = ''
        for fig in figures:
            figures_html += f'<img src="data:image/png;base64,{self._figure_to_base64(fig)}" style="max-width: 100%; height: auto; margin: 20px 0;"><br>'
        
        # Create HTML document
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Analysis Report</title>
            {self.style}
        </head>
        <body>
            {self._generate_nav(page_type)}
            <div class="container">
                {html_content}
                {figures_html}
            </div>
        </body>
        </html>
        """
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write HTML to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

    def generate_user_index(self, usernames):
        """Generate an index page for user analyses."""
        # Create markdown content
        content = [
            "# User Analysis Reports\n",
            "Click on a user below to view their detailed analysis:\n",
            "## Available Reports"
        ]
        
        # Add links to each user's report
        for username in sorted(usernames):
            content.append(f"- [@{username}]({username}_analysis.html)")
        
        # Convert to HTML
        html_content = markdown.markdown("\n".join(content))
        
        # Generate navigation
        nav_html = self._generate_nav('user')
        
        # Create HTML document
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>User Analysis Reports</title>
            {self.style}
        </head>
        <body>
            {nav_html}
            <div class="container">
                {html_content}
            </div>
        </body>
        </html>
        """
        
        # Create the index file
        os.makedirs('docs/users', exist_ok=True)
        with open('docs/users/index.html', 'w', encoding='utf-8') as f:
            f.write(html_doc)
        
        return 'docs/users/index.html' 