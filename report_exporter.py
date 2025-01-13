from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
import os
import matplotlib.pyplot as plt

class ReportExporter:
    def __init__(self):
        """Initialize the Google Docs exporter."""
        self.SCOPES = [
            'https://www.googleapis.com/auth/documents',
            'https://www.googleapis.com/auth/drive.file'
        ]
        self.creds = None
        self._authenticate()
        self.docs_service = build('docs', 'v1', credentials=self.creds)
        self.drive_service = build('drive', 'v3', credentials=self.creds)

    def _authenticate(self):
        """Handle Google OAuth authentication."""
        if os.path.exists('token.json'):
            self.creds = Credentials.from_authorized_user_file('token.json', self.SCOPES)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', self.SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            with open('token.json', 'w') as token:
                token.write(self.creds.to_json())

    def _upload_image(self, figure: plt.Figure) -> str:
        """Upload a matplotlib figure to Google Drive and return its ID."""
        buf = io.BytesIO()
        figure.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        file_metadata = {'name': f'report_figure_{id(figure)}.png'}
        media = MediaIoBaseUpload(buf, mimetype='image/png', resumable=True)
        file = self.drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        return file.get('id')

    def export_group_report(self, report_text: str, figures: list[plt.Figure]) -> str:
        """
        Export the group analysis report to Google Docs.
        
        Args:
            report_text: Markdown formatted report text
            figures: List of matplotlib figures to include
            
        Returns:
            URL of the created Google Doc
        """
        # Create new document
        doc = self.docs_service.documents().create(
            body={'title': 'Group Analysis Report'}
        ).execute()
        doc_id = doc.get('id')
        
        # Convert markdown to Google Docs format
        requests = []
        
        # Process text and add formatting
        sections = report_text.split('\n')
        current_pos = 1  # Start after title
        
        for section in sections:
            if section.startswith('# '):  # Main title
                requests.append({
                    'insertText': {
                        'location': {'index': current_pos},
                        'text': section[2:] + '\n'
                    }
                })
                requests.append({
                    'updateParagraphStyle': {
                        'range': {
                            'startIndex': current_pos,
                            'endIndex': current_pos + len(section) - 1
                        },
                        'paragraphStyle': {
                            'namedStyleType': 'HEADING_1',
                            'alignment': 'CENTER'
                        },
                        'fields': 'namedStyleType,alignment'
                    }
                })
            elif section.startswith('## '):  # Section title
                requests.append({
                    'insertText': {
                        'location': {'index': current_pos},
                        'text': section[3:] + '\n'
                    }
                })
                requests.append({
                    'updateParagraphStyle': {
                        'range': {
                            'startIndex': current_pos,
                            'endIndex': current_pos + len(section) - 2
                        },
                        'paragraphStyle': {
                            'namedStyleType': 'HEADING_2'
                        },
                        'fields': 'namedStyleType'
                    }
                })
            elif section.startswith('### '):  # Subsection title
                requests.append({
                    'insertText': {
                        'location': {'index': current_pos},
                        'text': section[4:] + '\n'
                    }
                })
                requests.append({
                    'updateParagraphStyle': {
                        'range': {
                            'startIndex': current_pos,
                            'endIndex': current_pos + len(section) - 3
                        },
                        'paragraphStyle': {
                            'namedStyleType': 'HEADING_3'
                        },
                        'fields': 'namedStyleType'
                    }
                })
            elif section.startswith('- '):  # Bullet points
                requests.append({
                    'insertText': {
                        'location': {'index': current_pos},
                        'text': section[2:] + '\n'
                    }
                })
                requests.append({
                    'createParagraphBullets': {
                        'range': {
                            'startIndex': current_pos,
                            'endIndex': current_pos + len(section) - 1
                        },
                        'bulletPreset': 'BULLET_DISC_CIRCLE_SQUARE'
                    }
                })
            else:  # Regular text
                requests.append({
                    'insertText': {
                        'location': {'index': current_pos},
                        'text': section + '\n'
                    }
                })
            
            current_pos += len(section) + 1
        
        # Insert figures
        for i, fig in enumerate(figures):
            image_id = self._upload_image(fig)
            requests.append({
                'insertText': {
                    'location': {'index': current_pos},
                    'text': '\n'  # Add space before image
                }
            })
            current_pos += 1
            
            requests.append({
                'insertInlineImage': {
                    'location': {'index': current_pos},
                    'uri': f'https://drive.google.com/uc?id={image_id}',
                    'objectSize': {
                        'height': {'magnitude': 400, 'unit': 'PT'},
                        'width': {'magnitude': 600, 'unit': 'PT'}
                    }
                }
            })
            
            requests.append({
                'insertText': {
                    'location': {'index': current_pos + 1},
                    'text': '\n'  # Add space after image
                }
            })
            current_pos += 2
        
        # Apply all changes
        self.docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={'requests': requests}
        ).execute()
        
        return f'https://docs.google.com/document/d/{doc_id}/edit' 