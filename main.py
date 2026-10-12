"""
main.py - Entry point for the Household Income Prediction application.

Usage:
    python main.py                  # Start the web server (default port 8080)
    python main.py --port 3000      # Start on a custom port
    python main.py --train          # Re-train models before starting the server
"""

import argparse
import http.server
import socketserver
import os
import sys
import webbrowser


def run_training():
    """Run the model training pipeline."""
    print("Running model training pipeline...")
    from src.train_and_evaluate_all import train_and_evaluate
    train_and_evaluate()
    print()


def start_server(port: int = 8080, open_browser: bool = True):
    """
    Start a static HTTP server serving the web app.

    Parameters
    ----------
    port : int
        Port number to serve on.
    open_browser : bool
        Whether to auto-open the browser.
    """
    static_dir = os.path.join(os.path.dirname(__file__), 'static')

    if not os.path.isdir(static_dir):
        print(f"Error: '{static_dir}' directory not found.")
        print("Please ensure the project structure is set up correctly.")
        sys.exit(1)

    os.chdir(static_dir)

    handler = http.server.SimpleHTTPRequestHandler
    handler.extensions_map.update({
        '.js': 'application/javascript',
        '.css': 'text/css',
        '.json': 'application/json',
    })

    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}"
        print("=" * 60)
        print(f"  Household Income Predictor - Web App")
        print(f"  Server running at: {url}")
        print(f"  Press Ctrl+C to stop")
        print("=" * 60)

        if open_browser:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


def main():
    parser = argparse.ArgumentParser(
        description='Household Income Prediction - AI Web App'
    )
    parser.add_argument(
        '--port', type=int, default=8080,
        help='Port for the web server (default: 8080)'
    )
    parser.add_argument(
        '--train', action='store_true',
        help='Re-train models before starting the server'
    )
    parser.add_argument(
        '--no-browser', action='store_true',
        help='Do not auto-open the browser'
    )
    args = parser.parse_args()

    if args.train:
        run_training()

    start_server(port=args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    main()
